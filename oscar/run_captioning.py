# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.

import argparse
import base64
import numpy as np
import os
import os.path as op
import random
import time
import json
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from oscar.utils.logger import setup_logger
from oscar.utils.tsv_file import TSVFile
from oscar.utils.tsv_file_ops import (tsv_writer, concat_tsv_files, delete_tsv_files, reorder_tsv_keys)
from oscar.utils.misc import (mkdir, set_seed, load_from_yaml_file, find_file_path_in_yaml)
from oscar.utils.caption_evaluate import (evaluate_on_coco_caption, ScstRewardCriterion)
from oscar.utils.cbs import ConstraintFilter, ConstraintBoxesReader
from oscar.utils.cbs import FiniteStateMachineBuilder
from oscar.modeling.modeling_bert import BertForImageCaptioning, BertForImageCaptioningOCR
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule


class CaptionTSVDataset(Dataset):
    def __init__(
            self,
            yaml_file,
            tokenizer=None,
            add_od_labels=True,
            add_ocr_labels=True,
            max_img_seq_length=50,
            max_seq_length=70,
            max_seq_a_length=40,
            max_ocr_seq_length=10,
            is_train=True,
            mask_prob=0.15,
            max_masked_tokens=3,
            **kwargs
    ):
        """Constructor.
        Args:
            yaml file with all required data (image feature, caption, labels, etc)
            tokenizer: tokenizer for text processing.
            add_od_labels: whether to add labels from yaml file to BERT. 
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            kwargs: other arguments.
        """
        self.yaml_file = yaml_file
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = op.dirname(yaml_file)
        self.label_file = find_file_path_in_yaml(self.cfg['label'], self.root)
        self.feat_file = find_file_path_in_yaml(self.cfg['feature'], self.root)
        self.caption_file = find_file_path_in_yaml(self.cfg.get('caption'), self.root)
        self.ocr_file = find_file_path_in_yaml(self.cfg.get('ocr'), self.root)

        assert op.isfile(self.feat_file)
        if add_od_labels:
            assert op.isfile(self.label_file)
        if add_ocr_labels:
            assert op.isfile(self.ocr_file)
        if is_train:
            assert op.isfile(self.caption_file) and tokenizer is not None

        self.label_tsv = None if not self.label_file else TSVFile(self.label_file)
        self.feat_tsv = TSVFile(self.feat_file)
        self.captions = []
        if self.caption_file and op.isfile(self.caption_file):
            with open(self.caption_file, 'r') as f:
                self.captions = json.load(f)
        self.ocr_blocks = {}
        if self.ocr_file and op.isfile(self.ocr_file):
            ob_list = []
            with open(self.ocr_file, 'r') as f:
                ob_list = json.load(f)
            self.ocr_blocks = {el['image_id']: el['data'] for el in ob_list}

        self.tokenizer = tokenizer
        tensorizer_class = CaptionTensorizerOCR if add_ocr_labels else CaptionTensorizer
        self.tensorizer = tensorizer_class(
            self.tokenizer, max_img_seq_length, max_seq_length, max_seq_a_length, max_ocr_seq_length,
            mask_prob=mask_prob, max_masked_tokens=max_masked_tokens, is_train=is_train,
        )
        self.add_od_labels = add_od_labels
        self.add_ocr_labels = add_ocr_labels
        self.is_train = is_train
        self.kwargs = kwargs
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.key2captions = self.prepare_image_key_to_captions()

    def get_valid_tsv(self):
        # based on the order of file size
        if self.label_tsv:
            return self.label_tsv
        if self.feat_tsv:
            return self.feat_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.seek(i)[0] for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.seek(i)[0]: i for i in range(tsv.num_rows())}

    def prepare_image_key_to_captions(self):
        if self.captions:
            key2captions = {key: [] for key in self.image_keys}
            for cap in self.captions:
                key2captions[cap['image_id']].append(cap['caption'])
            return key2captions

    def get_image_index(self, idx):
        if self.is_train:
            img_cap_pair = self.captions[idx]
            img_key = img_cap_pair['image_id']
            return self.key2index[img_key]
        return idx

    def get_image_key(self, idx):
        img_idx = self.get_image_index(idx)
        return self.image_keys[img_idx]

    def get_image_features(self, img_idx):
        feat_info = json.loads(self.feat_tsv.seek(img_idx)[1])
        num_boxes = feat_info['num_boxes']
        features = np.frombuffer(
            base64.b64decode(feat_info['features']), np.float32
        ).reshape((num_boxes, -1))
        return torch.Tensor(features.copy())  # clone array
        # feat_tensor = torch.as_tensor(features)
        # return feat_tensor

    def get_caption(self, idx):
        if self.is_train:
            img_cap_pair = self.captions[idx]
            return img_cap_pair['caption']
        return ''

    def get_ocr_labels(self, img_key):
        ocr_labels = None
        if self.add_ocr_labels:
            img_ocr_blocks = self.ocr_blocks[img_key]  # list of [box, text, conf]
            # Get only concatenated text without any processing
            # TODO: processing of OCR blocks: boxes, conf
            for block in img_ocr_blocks:
                block_text = block[1]
                # print(block_text, end=' + ')
                # if ' ' in block_text:
                #     print(block_text)
            # print()
            # ocr_labels = ' '.join([b[1] for b in img_ocr_blocks])
            ocr_labels = [b[1] for b in img_ocr_blocks]
        return ocr_labels

    def get_ocr_boxes(self, img_key):
        ocr_boxes = None
        if self.add_ocr_labels:
            img_ocr_blocks = self.ocr_blocks[img_key]  # list of [box, text, conf]
            ocr_boxes = []
            for block in img_ocr_blocks:
                # x1, y1, x2, y2, w, h = block[0]
                # w, h = x2 - x1, y2 - y1
                # Make extended bbox with width and height
                # ocr_boxes.append([x1, y1, x2, y2, w, h])
                ocr_boxes.append(block[0])
            # ocr_boxes = np.array(ocr_boxes)  # CHECK?
        return ocr_boxes

    def get_od_labels(self, img_idx):
        od_labels = None
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels = ' '.join([el['class'] for el in label_info])
        return od_labels

    def get_caption_file_in_coco_format(self):
        cap_file = op.splitext(self.caption_file)[0] + '_coco_format.json'
        return cap_file

    def get_captions_by_key(self, key):
        return self.key2captions[key]

    def __getitem__(self, idx):
        img_idx = self.get_image_index(idx)
        img_key = self.image_keys[img_idx]
        features = self.get_image_features(img_idx)
        caption = self.get_caption(idx)
        od_labels = self.get_od_labels(img_idx)
        ocr_labels = self.get_ocr_labels(self.get_image_key(idx))
        ocr_boxes = self.get_ocr_boxes(self.get_image_key(idx))
        # print()
        # print(img_key)
        # print(caption)
        # print(od_labels[:80])
        # print(ocr_labels)

        # print('FEAT x1y1x2y2wh:', features[0, -6:])
        # print('LABELrand:', od_labels.split(' ')[0])

        example = self.tensorizer.tensorize_example_v2(
            text_a=caption, img_feat=features, text_b=od_labels, text_c=ocr_labels, text_c_pos=ocr_boxes,
        )
        # print(ocr_labels)
        # print(ocr_boxes)
        # print('\n', example[6], flush=True)
        # print('\n', example[7], flush=True)

        return img_key, example

    def __len__(self):
        if self.is_train:
            return len(self.captions)
        return self.get_valid_tsv().num_rows()


class CaptionTSVDatasetWithConstraints(CaptionTSVDataset):
    r"""
    Providing inputs for inference with Constraint Beam Search

    nms_threshold: float, optional (default = 0.85)
        NMS threshold for suppressing generic object class names during constraint filtering,
        for two boxes with IoU higher than this threshold, "dog" suppresses "animal".
    max_given_constraints: int, optional (default = 3)
        Maximum number of constraints which can be specified for CBS decoding. Constraints are
        selected based on the prediction confidence score of their corresponding bounding boxes.
    """

    def __init__(self, yaml_file, nms_threshold=0.85, max_given_constraints=3, **kwargs):
        super().__init__(yaml_file, **kwargs)
        boxes_tsvpath = find_file_path_in_yaml(self.cfg['cbs_box'], self.root)
        constraint2tokens_tsvpath = find_file_path_in_yaml(self.cfg['cbs_constraint'], self.root)
        tokenforms_tsvpath = find_file_path_in_yaml(self.cfg['cbs_tokenforms'], self.root)
        hierarchy_jsonpath = find_file_path_in_yaml(self.cfg['cbs_hierarchy'], self.root)

        self._boxes_reader = ConstraintBoxesReader(boxes_tsvpath)
        self._constraint_filter = ConstraintFilter(
            hierarchy_jsonpath, nms_threshold, max_given_constraints
        )
        self._fsm_builder = FiniteStateMachineBuilder(
            self.tokenizer, constraint2tokens_tsvpath, tokenforms_tsvpath, max_given_constraints
        )

    def __getitem__(self, index):
        img_key, example = super().__getitem__(index)

        # Apply constraint filtering to object class names.
        constraint_boxes = self._boxes_reader[img_key]

        candidates = self._constraint_filter(
            constraint_boxes['boxes'],
            constraint_boxes['class_names'],
            constraint_boxes['scores']
        )

        num_constraints = len(candidates)
        fsm, nstates = self._fsm_builder.build(candidates)

        return img_key, example + (fsm, num_constraints, )


class CaptionTensorizer(object):
    def __init__(self,
                 tokenizer,
                 max_img_seq_length=50,
                 max_seq_length=70,
                 max_seq_a_length=40,
                 max_ocr_seq_length=10,
                 mask_prob=0.15,
                 max_masked_tokens=3,
                 is_train=True):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
        """
        # TO-DO: zzz
        if is_train:
            raise RuntimeError('You must use CaptionTensorizerOCR class')

        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.max_ocr_seq_length = max_ocr_seq_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(
            torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.long)
        )

    def tensorize_example_v1(self, text_a, img_feat, text_b=None, text_c=None,
                             cls_token_segment_id=0, pad_token_segment_id=0,
                             sequence_a_segment_id=0,  sequence_c_segment_id=1, sequence_b_segment_id=1):
        # v1: sentence > ocr > od > img_feats
        # v1:    30    >  10 > 30 >    50
        # text_a - caption
        # text_c - ocr_labels
        # text_b - od_labels
        # a > c > b > f

        max_cap_len = self.max_seq_a_len - self.max_ocr_seq_length  # 40-10=30

        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)  # All tokens
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (max_cap_len - 2)  # [MASK] * 28

        if len(tokens_a) > max_cap_len - 2:
            tokens_a = tokens_a[:(max_cap_len - 2)]  # Keep only first 28 tokens

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]  # [[CLS], Senence, [SEP]]  <= 30
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)  # [0, 0, ..., 0] <= 30
        seq_a_len = len(tokens)  # <= 30

        if text_c or text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = max_cap_len - seq_a_len  # pad to 30
            tokens += [self.tokenizer.pad_token] * padding_a_len  # [[CLS], Sentence, [SEP], [PAD]*n] = 30
            segment_ids += ([pad_token_segment_id] * padding_a_len)  # [0, 0, ..., 0] = 30

        ocr_len = 0
        if text_c:
            tokens_c = self.tokenizer.tokenize(text_c)  # Tokenize OCR labels string
            # TODO: tokenizer for text not in dictionary?
            # if len(tokens_b) > 70 - 30 - 10 - 1 = 29
            if len(tokens_c) > self.max_ocr_seq_length:
                tokens_c = tokens_c[: self.max_ocr_seq_length]  # [ocr_tokens] = 10
            ocr_len = len(tokens_c)
            tokens += tokens_c  # [[CLS], Sentence, [SEP]] + [ocr_tokens] <= 30 + 10
            padding_c_len = self.max_ocr_seq_length - len(tokens_c)  # pad to 10
            tokens += [self.tokenizer.pad_token] * padding_c_len  # [[CLS], Sentence, [SEP], [PAD]s, OCR, [PAD]s] = 40
            segment_ids += [sequence_c_segment_id] * self.max_ocr_seq_length  # [0, 0, ..., 0, 1, 1, ...,  1] = 30 + 10 = 40

        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)  # Tokenize od labels string
            # if len(tokens_b) > 70 - 30 - 10 - 1 = 29
            if len(tokens_b) > self.max_seq_len - len(tokens) - self.max_ocr_seq_length - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]  # [od_tokens] = 29
            tokens += tokens_b + [self.tokenizer.sep_token]  # [[CLS], Sentence, [SEP], [PAD]s, OCR, [PAD]s] + [od_tokens] + [SEP] <= 40 + 30 = 70
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)  # [0, 0, ..., 0, 1, 1, ...,  1] <= 40 + 30 = 70

        seq_len = len(tokens)  # <= 70
        masked_ids = None
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1, seq_a_len))  # only mask text_a  <= 30
            random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] * (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len  # <= 70
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)  # [[CLS], Sentence, [SEP], [PAD]s, OCR, [PAD]s] + [od_tokens] + [SEP] + [PAD]s] = 40 + 30 = 70
        segment_ids += ([pad_token_segment_id] * padding_len)  # [0, 0, ..., 0, 1, 1, ...,  1] = 70
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]  # Keep only first 50
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # prepare attention mask:
        # note that there is no attention from caption to image (??? WUT ???)
        # because otherwise it will violate the triangle attention
        # for caption as caption will have full attention on image.
        max_len = self.max_seq_len + self.max_img_seq_len  # 70 + 50 = 120
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)  # 120x120
        # C: caption, O: ocr, L: label, R: image region
        c_start, c_end = 0, seq_a_len  # 0, 0..30
        o_start, o_end = max_cap_len, max_cap_len + ocr_len  # 30, 30..40
        l_start, l_end = self.max_seq_a_len, seq_len  # 40, 40..70
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len  # 70, 70..120
        # triangle mask for caption to caption
        attention_mask[c_start: c_end, c_start: c_end].copy_(self._triangle_mask[0: seq_a_len, 0: seq_a_len])
        # full attention for O-O, L-L, R-R
        attention_mask[o_start: o_end, o_start: o_end] = 1
        attention_mask[l_start: l_end, l_start: l_end] = 1
        attention_mask[r_start: r_end, r_start: r_end] = 1
        # full attention for C-O, C-L, C-R  # and no O-C, L-C, R-C
        attention_mask[c_start: c_end, o_start: o_end] = 1
        attention_mask[c_start: c_end, l_start: l_end] = 1
        attention_mask[c_start: c_end, r_start: r_end] = 1
        # full attention for O-L & L-O; O-R & R-O; L-R & R-L:
        attention_mask[o_start: o_end, l_start: l_end] = 1
        attention_mask[l_start: l_end, o_start: o_end] = 1
        attention_mask[o_start: o_end, r_start: r_end] = 1
        attention_mask[r_start: r_end, o_start: o_end] = 1
        attention_mask[l_start: l_end, r_start: r_end] = 1
        attention_mask[r_start: r_end, l_start: l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            return input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids
        return input_ids, attention_mask, segment_ids, img_feat, masked_pos

    def tensorize_example(self, text_a, img_feat, text_b=None,
                          cls_token_segment_id=0, pad_token_segment_id=0,
                          sequence_a_segment_id=0, sequence_b_segment_id=1):
        raise RuntimeError('For OscarOCR Use tenzorize_example_v1 or v2!')
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        masked_ids = None
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1, seq_a_len))  # only mask text_a
            random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] * (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention
        # for caption as caption will have full attention on image.
        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # triangle mask for caption to caption
        attention_mask[c_start: c_end, c_start: c_end].copy_(self._triangle_mask[0: seq_a_len, 0: seq_a_len])
        # full attention for L-L, R-R
        attention_mask[l_start: l_end, l_start: l_end] = 1
        attention_mask[r_start: r_end, r_start: r_end] = 1
        # full attention for C-L, C-R  # and no L-C, R-C
        attention_mask[c_start: c_end, l_start: l_end] = 1
        attention_mask[c_start: c_end, r_start: r_end] = 1
        # full attention for L-R, R-L:
        attention_mask[l_start: l_end, r_start: r_end] = 1
        attention_mask[r_start: r_end, l_start: l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            return input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids
        return input_ids, attention_mask, segment_ids, img_feat, masked_pos


class CaptionTensorizerOCR(object):
    def __init__(self,
                 tokenizer,
                 max_img_seq_length=50,
                 max_seq_length=70,
                 max_seq_a_length=40,
                 max_ocr_seq_length=50,
                 mask_prob=0.15,
                 max_masked_tokens=3,
                 is_train=True):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.max_ocr_seq_length = max_ocr_seq_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(
            torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.long)
        )

    def tensorize_example_v2(self, text_a, img_feat, text_b=None, text_c=None,
                             cls_token_segment_id=0, pad_token_segment_id=0, text_c_pos=None,
                             sequence_a_segment_id=0, sequence_b_segment_id=1, sequence_c_segment_id=2):
        # v2: sentence > od > img_feats > ocr
        # v1:    40    > 30 >    50     > 50
        # text_a - caption
        # text_b - od_labels
        # text_c - ocr_labels
        # a > b > f > c

        max_cap_len = self.max_seq_a_len  # = 40   #- self.max_ocr_seq_length  # 40-10=30

        # TEXT TOKENS (CAPTION)
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)  # All tokens
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (max_cap_len - 2)  # [MASK] * 38

        if len(tokens_a) > max_cap_len - 2:
            tokens_a = tokens_a[:(max_cap_len - 2)]  # Keep only first 38 tokens

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]  # [[CLS], Senence, [SEP]]  <= 40
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)  # [0, 0, ..., 0] <= 40
        seq_a_len = len(tokens)  # <= 40

        # IMAGE TAGS (TOKENS)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = max_cap_len - seq_a_len  # pad to 40
            tokens += [self.tokenizer.pad_token] * padding_a_len  # [[CLS], Sentence, [SEP], [PAD]*n] = 40
            segment_ids += ([pad_token_segment_id] * padding_a_len)  # [0, 0, ..., 0] = 40
            # Tokenize od labels string
            tokens_b = self.tokenizer.tokenize(text_b)
            # if len(tokens_b) > 70 - 40 - 1 = 29
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]  # [od_tokens] = 29
            tokens += tokens_b + [self.tokenizer.sep_token]  # [[CLS], Sentence, [SEP], [PAD]s, OCR, [PAD]s] + [od_tokens] + [SEP] <= 40 + 30 = 70
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)  # [0, 0, ..., 0, 1, 1, ...,  1] <= 40 + 30 = 70

        seq_len = len(tokens)  # <= 70
        masked_ids = None
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1, seq_a_len))  # only mask text_a  <= 40
            random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] * (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # Pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len  # <= 70
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)  # [[CLS], Sentence, [SEP], [PAD]s, OCR, [PAD]s] + [od_tokens] + [SEP] + [PAD]s] = 40 + 30 = 70
        segment_ids += ([pad_token_segment_id] * padding_len)  # [0, 0, ..., 0, 1, 1, ...,  1] = 70
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # IMAGE FEATURES
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]  # Keep only first 50
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # OCR TOKENS
        ocr_len = 0
        tokens_c_pos = []
        if text_c is not None:
            # tokens_c = self.tokenizer.tokenize(text_c)  # Tokenize OCR labels string
            tokens_c = []
            tokens_c_pos = []
            for text_block, text_pos in zip(text_c, text_c_pos):
                tokens_block = self.tokenizer.tokenize(text_block)  # Tokenize OCR labels string
                tokens_c.extend(tokens_block)
                tokens_c_pos.extend([text_pos] * len(tokens_block))  # Repeat bbox N times
                # print(f'TEXT: {text_block}, TOKENS: {tokens_block}, BBOXES: {[text_pos] * len(tokens_block)}')

            # TODO: tokenizer for text not in dictionary?
            if len(tokens_c) > self.max_ocr_seq_length:
                tokens_c = tokens_c[:self.max_ocr_seq_length]  # [ocr_tokens] 0:50
                tokens_c_pos = tokens_c_pos[:self.max_ocr_seq_length]  # 0:50
            ocr_tokens = tokens_c  # [ocr_tokens] <= 50
            ocr_len = len(tokens_c)
            padding_c_len = self.max_ocr_seq_length - len(tokens_c)  # pad to <= 50
            ocr_tokens += [self.tokenizer.pad_token] * padding_c_len  # [OCR, [PAD]s] = 50
            tokens_c_pos.extend([[0, 0, 0, 0, 0, 0]] * padding_c_len)  # PAD to 50
            input_ocr_ids = self.tokenizer.convert_tokens_to_ids(ocr_tokens)
            # print(input_ocr_ids, flush=True)
        else:
            ocr_tokens = [self.tokenizer.pad_token] * self.max_ocr_seq_length  # [[PAD]s] = 50
            input_ocr_ids = self.tokenizer.convert_tokens_to_ids(ocr_tokens)
            # print(input_ocr_ids, flush=True)

        # tokens_c_pos = np.zeros((len(ocr_tokens), 6))  # Fake positions
        tokens_c_pos = np.array(tokens_c_pos)
        # print(tokens_c)
        # print(tokens_c_pos.shape)

        # TODO: FILL segments IF NEEDED
        ocr_segment_ids = [sequence_c_segment_id] * self.max_ocr_seq_length  # [2, 2, ..., 2] = 50

        # prepare attention mask:
        # note that there is no attention from caption to image (no L-C, R-C, O-C, ...)
        # because otherwise it will violate the triangle attention
        # for caption as caption will have full attention on image.
        ocr_start_pos = self.max_seq_len + self.max_img_seq_len
        max_len = self.max_seq_len + self.max_img_seq_len + self.max_ocr_seq_length  # 70 + 50 + 50 = 170
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)  # 170x170
        # C: caption, L: label, R: image region, O: ocr
        c_start, c_end = 0, seq_a_len  # 0, 1..40
        l_start, l_end = self.max_seq_a_len, seq_len  # 40, 40..70
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len  # 70, 70..120
        o_start, o_end = ocr_start_pos, ocr_start_pos + ocr_len  # 120, 120..170
        # triangle mask for C-C (caption to caption)
        attention_mask[c_start: c_end, c_start: c_end].copy_(self._triangle_mask[0: seq_a_len, 0: seq_a_len])
        # full attention for O-O, L-L, R-R
        attention_mask[l_start: l_end, l_start: l_end] = 1
        attention_mask[r_start: r_end, r_start: r_end] = 1
        attention_mask[o_start: o_end, o_start: o_end] = 1
        # full attention for C-L, C-R, C-O  # and no L-C, R-C, O-C
        attention_mask[c_start: c_end, l_start: l_end] = 1
        attention_mask[c_start: c_end, r_start: r_end] = 1
        attention_mask[c_start: c_end, o_start: o_end] = 1
        # full attention for L-R, R-L; L-O, O-L; R-O, O-R;
        attention_mask[l_start: l_end, r_start: r_end] = 1
        attention_mask[r_start: r_end, l_start: l_end] = 1
        attention_mask[l_start: l_end, o_start: o_end] = 1
        attention_mask[o_start: o_end, l_start: l_end] = 1
        attention_mask[r_start: r_end, o_start: o_end] = 1
        attention_mask[o_start: o_end, r_start: r_end] = 1

        # for row in attention_mask:
        #     for col in row:
        #         print(col.item, end=' ')
        #     print(flush=True)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # TODO: input_ocr_ids should work with no OCR too
        input_ocr_ids = torch.tensor(input_ocr_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        input_ocr_posits = torch.tensor(tokens_c_pos, dtype=torch.float32)  # Convert to float from int

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            return input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids, input_ocr_ids, input_ocr_posits
        return input_ids, attention_mask, segment_ids, img_feat, masked_pos, input_ocr_ids, input_ocr_posits


def build_dataset(yaml_file, tokenizer, args, is_train=True):
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file)

    dataset_class = CaptionTSVDatasetWithConstraints if args.use_cbs and not is_train else CaptionTSVDataset

    return dataset_class(
        yaml_file,
        tokenizer=tokenizer,
        add_od_labels=args.add_od_labels,
        add_ocr_labels=args.add_ocr_labels,
        max_img_seq_length=args.max_img_seq_length,
        max_seq_length=args.max_seq_length,
        max_seq_a_length=args.max_seq_a_length,
        max_ocr_seq_length=args.max_ocr_seq_length,
        is_train=is_train,
        mask_prob=args.mask_prob,
        max_masked_tokens=args.max_masked_tokens if is_train else 3,
    )


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args, yaml_file, tokenizer, is_distributed=True, is_train=True):
    dataset = build_dataset(yaml_file, tokenizer, args, is_train=(is_train and not args.scst))

    if is_train:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_epoch = len(dataset) // images_per_batch   # num of batches per all dataset
        num_iters = iters_per_epoch * args.num_train_epochs
        logger.info(f'Train with {images_per_gpu} images per GPU.')
        logger.info(f'Total batch size {images_per_batch}')
        logger.info(f'Total training steps {num_iters}')
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        sampler=sampler,
        batch_size=images_per_gpu,
        pin_memory=True,
    )

    return data_loader


def save_checkpoint(model, tokenizer, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, f'checkpoint-{epoch}-{iteration}')
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f'Save checkpoint to {checkpoint_dir}')
            break
        except:
            pass
    else:
        logger.info(f'Failed to save checkpoint after {num_trial} trails.')
    return checkpoint_dir


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data  # argmax
    scores = logits == labels
    return scores


def train(args, train_dataloader, val_dataset, model, tokenizer):
    # Main train procedure

    scaler = None
    if args.fp16:
        logger.info('// Running FP16 training //')
        scaler = torch.cuda.amp.GradScaler()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.scheduler == 'constant':
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == 'linear':
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError(f'Unknown scheduler type: {args.scheduler}')

    total_tbs = args.per_gpu_train_batch_size * get_world_size() * args.gradient_accumulation_steps

    logger.info('***** Running training *****')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(f'  Batch size per GPU = {args.per_gpu_train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, & accumulation) = {total_tbs}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {t_total}')

    scst_criterion = None
    if args.scst:
        scst_criterion = ScstRewardCriterion(
            cider_cached_tokens=op.join(args.data_dir, args.cider_cached_tokens),
            baseline_type=args.sc_baseline_type,
        )
        logger.info('  SCST training...')

    checkpoint_dir = None
    global_step, global_loss, global_acc = 0, 0.0, 0.0
    eval_log = []
    best_score = 0

    model.zero_grad()
    for epoch in range(int(args.num_train_epochs)):
        for step, (img_keys, batch) in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)

            if not args.scst:
                model.train()
                inputs = {
                    'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2],
                    'img_feats': batch[3], 'masked_pos': batch[4], 'masked_ids': batch[5],
                    'input_ocr_ids': batch[6], 'input_ocr_posits': batch[7],
                }

                # *** RUN MODEL *** #
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = model(**inputs)
                        loss, logits = outputs[:2]
                else:
                    outputs = model(**inputs)
                    loss, logits = outputs[:2]
                # print(outputs)
                # input('PRESS ANY KEY:')

                masked_ids = inputs['masked_ids']
                masked_ids = masked_ids[masked_ids != 0]
                batch_score = compute_score_with_logits(logits, masked_ids)
                batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])
            else:
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        loss = scst_train_iter(args, train_dataloader, model, scst_criterion, img_keys, batch, tokenizer)
                        batch_acc = scst_criterion.get_score()
                else:
                    loss = scst_train_iter(args, train_dataloader, model, scst_criterion, img_keys, batch, tokenizer)
                    batch_acc = scst_criterion.get_score()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16 and not args.scst:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            global_loss += loss.item()
            global_acc += batch_acc

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                # scheduler.step()

                if args.fp16 and not args.scst:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # if step > 100:   # HARDCODE!111111
                #    scheduler.step()

                model.zero_grad()

                scheduler.step()

                if global_step % args.logging_steps == 0:
                    logger.info(
                        f'Epoch: {epoch}, '
                        f'global_step: {global_step}, '
                        f'lr: {optimizer.param_groups[0]["lr"]:.6f}, '
                        f'loss: {loss:.4f} ({global_loss / global_step:.4f}), '
                        f'score: {batch_acc:.4f} ({global_acc / global_step:.4f})'
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or global_step == t_total:
                    checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step)
                    # evaluation
                    if args.evaluate_during_training:
                        logger.info('Perform evaluation at step: %d' % (global_step))
                        evaluate_file = evaluate(args, val_dataset, model, tokenizer, checkpoint_dir)
                        with open(evaluate_file, 'r') as f:
                            res = json.load(f)
                        best_score = max(best_score, res['CIDEr'])
                        res['epoch'] = epoch
                        res['global_step'] = step
                        res['best_CIDEr'] = best_score
                        eval_log.append(res)
                        with open(args.output_dir + '/eval_logs.json', 'w') as f:
                            json.dump(eval_log, f)
    return checkpoint_dir


def scst_train_iter(args, train_dataloader, model, scst_criterion, img_keys, batch, tokenizer):
    # SCST train iteration

    cls_token_id, sep_token_id, pad_token_id, mask_token_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token]
    )
    inputs = {
        'is_decode': True,
        'input_ids': batch[0],
        'attention_mask': batch[1],
        'token_type_ids': batch[2],
        'img_feats': batch[3],
        'masked_pos': batch[4],
        'do_sample': False,
        'bos_token_id': cls_token_id,
        'pad_token_id': pad_token_id,
        'eos_token_ids': [sep_token_id],
        'mask_token_id': mask_token_id,
        # for adding od labels
        'add_od_labels': args.add_od_labels,
        'od_labels_start_posid': args.max_seq_a_length,
        # hyperparameters of beam search
        'max_length': args.max_gen_length,
        'num_beams': args.sc_beam_size,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
        'length_penalty': args.length_penalty,
        'num_return_sequences': 1,
        'num_keep_best': 1,
    }

    def _ids_to_captions(all_ids):
        captions = []
        for ids in all_ids:
            c = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            captions.append(c)
        return captions

    if args.sc_baseline_type == 'greedy':  # Default: 'greedy'
        model.eval()
        with torch.no_grad():
            # print('Run model once - greedy')
            greedy_res_raw, _ = model(**inputs)
            greedy_res_raw.squeeze_(1)  # batch_size * max_len
        greedy_res = _ids_to_captions(greedy_res_raw)
    else:
        greedy_res = None

    model.train()
    inputs['do_sample'] = True
    inputs['num_return_sequences'] = args.sc_train_sample_n  # Default arg: 5

    # *** RUN MODEL *** #
    # print('Run model - sample', flush=True)
    sample_res_raw, sample_logprobs = model(**inputs)
    # print(sample_res_raw, sample_logprobs, flush=True)

    sample_res_raw.squeeze_(1)
    sample_logprobs.squeeze_(1)
    assert sample_logprobs.requires_grad is True
    assert sample_res_raw.requires_grad is False
    sample_res = _ids_to_captions(sample_res_raw)
    # print(sample_res)
    # input('PRESS ANY KEY_SCST:')
    gt_res = [train_dataloader.dataset.get_captions_by_key(k) for k in img_keys]
    loss = scst_criterion(gt_res, greedy_res, sample_res, sample_logprobs)
    return loss


def get_predict_file(output_dir, yaml_file, args):
    cc = ['pred']
    # make sure it works with/without / in end of the path.
    data = op.basename(op.join(args.data_dir, '')[:-1])
    split = op.basename(yaml_file)
    assert split.endswith('.yaml')
    split = split[:-5]
    cc.append(data)
    cc.append(split)
    cc.append('beam{}'.format(args.num_beams))
    cc.append('max{}'.format(args.max_gen_length))
    if args.add_od_labels:
        cc.append('odlabels')
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.use_cbs:
        cc.append('cbs{}'.format(args.min_constraints_to_satisfy))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))


def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    fpath = op.splitext(predict_file)[0]
    return fpath + '.eval.json'


def get_evaluate_method(predict_file):
    if 'nocaps' in op.basename(predict_file):
        return 'nocaps'
    else:
        return 'coco'


def evaluate(args, val_dataloader, model, tokenizer, output_dir):
    predict_file = get_predict_file(output_dir, val_dataloader.dataset.yaml_file, args)
    test(args, val_dataloader, model, tokenizer, predict_file)

    if get_world_size() > 1:
        torch.distributed.barrier()

    evaluate_file = get_evaluate_file(predict_file)

    if is_main_process():
        caption_file = val_dataloader.dataset.get_caption_file_in_coco_format()
        data = val_dataloader.dataset.yaml_file.split('/')[-2]
        if 'nocaps' not in data:
            result = evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file)
            logger.info(f'evaluation result: {str(result)}')
            logger.info(f'evaluation result saved to {evaluate_file}')

    if get_world_size() > 1:
        torch.distributed.barrier()

    return evaluate_file


def test(args, test_dataloader, model, tokenizer, predict_file):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token, '.']
    )
    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        cache_file = op.splitext(predict_file)[0] + f'_{get_rank()}_{world_size}' + op.splitext(predict_file)[1]

    model.eval()
    inputs_param = {
        'is_decode': True,
        'do_sample': False,
        'bos_token_id': cls_token_id,
        'pad_token_id': pad_token_id,
        'eos_token_ids': [sep_token_id],
        'mask_token_id': mask_token_id,
        # for adding od labels
        'add_od_labels': args.add_od_labels,
        'od_labels_start_posid': args.max_seq_a_length,
        # hyperparameters of beam search
        'max_length': args.max_gen_length,
        'num_beams': args.num_beams,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
        'length_penalty': args.length_penalty,
        'num_return_sequences': args.num_return_sequences,
        'num_keep_best': args.num_keep_best,
    }
    if args.use_cbs:
        inputs_param.update({
            'use_cbs': True,
            'min_constraints_to_satisfy': args.min_constraints_to_satisfy,
        })

    def gen_rows():
        time_meter = 0

        with torch.no_grad():
            for step, (img_keys, batch) in tqdm(enumerate(test_dataloader)):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'img_feats': batch[3],
                    'masked_pos': batch[4],
                    'input_ocr_ids': batch[5],
                    'input_ocr_posits': batch[6],
                }
                if args.use_cbs:
                    raise NotImplementedError
                    inputs.update({
                        'fsm': batch[5],
                        'num_constraints': batch[6],
                    })
                inputs.update(inputs_param)
                tic = time.time()
                # captions, logprobs
                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    if isinstance(img_key, torch.Tensor):
                        img_key = img_key.item()
                    yield img_key, json.dumps(res)

        logger.info(f'Inference model computing time: {time_meter / (step+1)} seconds per batch')

    tsv_writer(gen_rows(), cache_file)
    if world_size > 1:
        torch.distributed.barrier()

    if world_size > 1 and is_main_process():
        name_begin = op.splitext(predict_file)[0]
        name_end = op.splitext(predict_file)[1]
        cache_files = [name_begin + f'_{i}_{world_size}' + name_end for i in range(world_size)]
        concat_tsv_files(cache_files, predict_file)
        delete_tsv_files(cache_files)
        reorder_tsv_keys(predict_file, test_dataloader.dataset.image_keys, predict_file)
    if world_size > 1:
        torch.distributed.barrier()


def restore_training_settings(args):
    if args.do_train:
        if not args.scst:
            return args
        checkpoint = args.model_name_or_path
    else:
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(checkpoint, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        if hasattr(train_args, 'scst') and train_args.scst:
            max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
        else:
            max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning(
            f'Override max_seq_length to {max_seq_length} = '
            f'max_gen_length:{args.max_gen_length} + '
            f'od_labels_len:{max_od_labels_len}'
        )

    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels', 'max_img_seq_length']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning(f'Override {param} with train args: {test_v} -> {train_v}')
                setattr(args, param, train_v)

    return args


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print(f'Init distributed training on local rank {local_rank}')
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    return local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='datasets/coco_caption', type=str, required=False, help='The input data dir with all required files.')
    parser.add_argument('--train_yaml', default='train.yaml', type=str, required=False, help='yaml file for training.')
    parser.add_argument('--test_yaml', default='test.yaml', type=str, required=False, help='yaml file for testing.')
    parser.add_argument('--val_yaml', default='val.yaml', type=str, required=False, help='yaml file used for validation during training.')
    parser.add_argument('--model_name_or_path', default=None, type=str, required=False, help='Path to pre-trained model or model type.')
    parser.add_argument('--output_dir', default='output/', type=str, required=False, help='The output directory to save checkpoint and test results.')
    parser.add_argument('--config_name', default='', type=str, help='Pretrained config name or path if not the same as model_name.')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_test', action='store_true', help='Whether to run inference.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run evaluation.')
    parser.add_argument('--add_od_labels', default=False, action='store_true', help='Whether to add object detection labels or not')

    # OCR
    parser.add_argument('--add_ocr_labels', default=False, action='store_true', help='Whether to add OCR labels or not')
    parser.add_argument('--max_ocr_seq_length', default=10, type=int, help='The maximum sequence length for OCR caption.')
    parser.add_argument('--ocr_dim', default=768+6, type=int, help='embedding + pos (xyxywh)')

    parser.add_argument('--tokenizer_name', default='', type=str, help='Pretrained tokenizer name or path if not the same as model_name.')
    parser.add_argument('--do_lower_case', action='store_true', help='Set this flag if you are using an uncased model.')
    parser.add_argument('--mask_prob', default=0.15, type=float, help='Probability to mask input sentence during training.')
    parser.add_argument('--max_masked_tokens', type=int, default=3, help='The max number of masked tokens per sentence.')
    parser.add_argument('--drop_out', default=0.1, type=float, help='Drop out in BERT.')
    parser.add_argument('--max_seq_length', default=70, type=int, help='The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--max_seq_a_length', default=40, type=int, help='The maximum sequence length for caption.')
    parser.add_argument('--max_img_seq_length', default=50, type=int, help='The maximum input image features sequence length.')
    parser.add_argument('--img_feature_dim', default=2054, type=int, help='The Image Feature Dimension. 2048+xyxywh')
    parser.add_argument('--img_feature_type', default='frcnn', type=str, help='Image feature type.')
    parser.add_argument('--tie_weights', default=False, action='store_true', help='Whether to tie decoding weights to that of encoding')
    parser.add_argument('--freeze_embedding', default=False, action='store_true', help='Whether to freeze word embeddings in Bert')
    parser.add_argument('--label_smoothing', default=0, type=float, help='.')
    parser.add_argument('--drop_worst_ratio', default=0, type=float, help='.')
    parser.add_argument('--drop_worst_after', default=0, type=int, help='.')
    parser.add_argument('--per_gpu_train_batch_size', default=64, type=int, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--per_gpu_eval_batch_size', default=64, type=int, help='Batch size per GPU/CPU for evaluation.')
    parser.add_argument('--output_mode', default='classification', type=str, help='output mode, support classification or regression.')
    parser.add_argument('--loss_type', default='sfmx', type=str, help='Loss function types: support kl, x2, sfmx')
    parser.add_argument('--num_labels', default=2, type=int, help='num_labels is 2 for classification and 1 for regression.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before backward.')
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='The initial lr.')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Epsilon for Adam.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max gradient norm.')
    parser.add_argument('--warmup_steps', default=0, type=int, help='Linear warmup.')
    parser.add_argument('--scheduler', default='linear', type=str, help='constant or linear or')
    parser.add_argument('--num_workers', default=4, type=int, help='Workers in dataloader.')
    parser.add_argument('--num_train_epochs', default=40, type=int, help='Total number of training epochs to perform.')
    parser.add_argument('--max_steps', default=-1, type=int, help='Total number of training steps. Override num_train_epochs.')
    parser.add_argument('--logging_steps', type=int, default=20, help='Log every X steps.')
    parser.add_argument('--save_steps', type=int, default=-1, help='Save checkpoint every X steps. Will also perform evaluation.')
    parser.add_argument('--evaluate_during_training', action='store_true', help='Run evaluation during training at each save_steps.')

    parser.add_argument(
        '--fp16', action='store_true',
        help='Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit'
    )

    parser.add_argument(
        '--fp16_opt_level', type=str, default='O1',
        help="For FP16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html"
    )

    parser.add_argument('--no_cuda', action='store_true', help='Avoid using CUDA.')
    parser.add_argument('--local_rank', type=int, default=0, help='For distributed training.')
    parser.add_argument('--seed', type=int, default=88, help='random seed for initialization.')

    # for self-critical sequence training
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')
    parser.add_argument('--sc_train_sample_n', type=int, default=5, help='number of sampled captions for sc training')
    parser.add_argument('--sc_baseline_type', type=str, default='greedy', help='baseline type of REINFORCE algorithm')
    parser.add_argument('--sc_beam_size', type=int, default=1, help='beam size for scst training')
    parser.add_argument('--cider_cached_tokens', type=str, default='coco-train-words.p', help='path to cached cPickle file used to calculate CIDEr scores')

    # for generation
    parser.add_argument('--eval_model_dir', type=str, default='', help='Model directory for evaluation.')
    parser.add_argument('--max_gen_length', type=int, default=20, help='Max length of generated sentences.')
    parser.add_argument('--output_hidden_states', action='store_true', help='Turn on for fast decoding.')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='Repeating times per image.')
    parser.add_argument('--num_beams', type=int, default=1, help='Beam search width')
    parser.add_argument('--num_keep_best', type=int, default=1, help='Number of hypotheses to keep in beam search')
    parser.add_argument('--temperature', type=float, default=1, help='Temperature in softmax for sampling')
    parser.add_argument('--top_k', type=int, default=0, help='Filter distribution for sampling')
    parser.add_argument('--top_p', type=float, default=1, help='Filter distribution for sampling')
    parser.add_argument('--repetition_penalty', type=int, default=1, help='Repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)')
    parser.add_argument('--length_penalty', type=int, default=1, help='Beam search length penalty')

    # for Constrained Beam Search
    parser.add_argument('--use_cbs', action='store_true', help='Use constrained beam search for decoding')
    parser.add_argument('--min_constraints_to_satisfy', type=int, default=2, help='minimum number of constraints to satisfy')

    args = parser.parse_args()

    global logger

    # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.num_gpus = get_world_size()
    args.distributed = args.num_gpus > 1
    args.device = torch.device('cuda')
    synchronize()

    output_dir = args.output_dir
    print('OUT_DIR:', output_dir)
    mkdir(output_dir)

    logger = setup_logger('vlp', output_dir, args.local_rank)
    logger.propagate = False  # Disable stderr logging
    logger.warning(f'Device: {args.device}, n_gpu: {args.num_gpus}')
    set_seed(args.seed, args.num_gpus)
    args = restore_training_settings(args)

    # Load pretrained model and tokenizer
    checkpoint = None
    config_class = BertConfig
    model_class = BertForImageCaptioningOCR if args.add_ocr_labels else BertForImageCaptioning
    tokenizer_class = BertTokenizer

    if args.do_train:
        assert args.model_name_or_path is not None

        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=args.num_labels,
            finetuning_task='image_captioning'
        )

        config.add_ocr_labels = args.add_ocr_labels
        config.ocr_dim = args.ocr_dim
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.tie_weights = args.tie_weights
        config.freeze_embedding = args.freeze_embedding
        config.label_smoothing = args.label_smoothing
        config.drop_worst_ratio = args.drop_worst_ratio
        config.drop_worst_after = args.drop_worst_after

        if args.scst:
            # avoid using too much memory
            config.output_hidden_states = True

        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case
        )

        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path),
            config=config
        )
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        config.output_hidden_states = args.output_hidden_states
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info(f'Evaluate the following checkpoint: {checkpoint}')
        model = model_class.from_pretrained(checkpoint, config=config)

    model.to(args.device)
    logger.info(f'Training/evaluation parameters {args}', )
    if args.do_train:
        train_dataloader = make_data_loader(args, args.train_yaml, tokenizer, args.distributed, is_train=True)
        val_dataloader = None
        if args.evaluate_during_training:
            val_dataloader = make_data_loader(args, args.val_yaml, tokenizer, args.distributed, is_train=False)
        last_checkpoint = train(args, train_dataloader, val_dataloader, model, tokenizer)

        # test the last checkpoint after training
        if args.do_test:
            logger.info('Evaluate on dataset: ' + args.test_yaml)
            test_dataloader = make_data_loader(args, args.test_yaml, tokenizer, args.distributed, is_train=False)
            evaluate(args, test_dataloader, model, tokenizer, last_checkpoint)

    # inference and evaluation
    elif args.do_test or args.do_eval:
        logger.info('Evaluate on dataset: ' + args.test_yaml)
        test_dataloader = make_data_loader(args, args.test_yaml, tokenizer, args.distributed, is_train=False)

        if not args.do_eval:
            predict_file = get_predict_file(checkpoint, test_dataloader.dataset.yaml_file, args)
            test(args, test_dataloader, model, tokenizer, predict_file)
            logger.info('Prediction results saved to: {}'.format(predict_file))
        else:
            evaluate_file = evaluate(args, test_dataloader, model, tokenizer, checkpoint)
            logger.info('Evaluation results saved to: {}'.format(evaluate_file))


if __name__ == '__main__':
    main()
