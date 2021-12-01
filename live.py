import os
import os.path as op
import time
import json
import glob
import importlib
import argparse

import numpy as np
from datetime import datetime

from PIL import Image
from tqdm import tqdm

import cv2

import torch
import torchvision

from v_maskrcnn_benchmark.config import cfg
from v_maskrcnn_benchmark.modeling.detector import build_detection_model
from v_maskrcnn_benchmark.structures.image_list import to_image_list
from v_maskrcnn_benchmark.utils.model_serialization import load_state_dict

os.environ['CUDA_VISIBLE_DEVICES']='0'
print('CUDA_VISIBLE_DEVICES set to "0"')
import easyocr

from pythia.common.registry import registry
from pythia.utils.configuration import Configuration

from oscar.run_cap_eval_only import OscarLive

import warnings
warnings.filterwarnings("ignore")

os.environ['TRANSFORMERS_CACHE'] = '../nmt_cache/'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# from transformers import FSMTForConditionalGeneration, FSMTTokenizer
# mname = "facebook/wmt19-en-ru"
# hf_tokenizer = FSMTTokenizer.from_pretrained(mname)#, torch_dtype=torch.float16)
# hf_model = FSMTForConditionalGeneration.from_pretrained(mname)#, torch_dtype=torch.float16)


class NMT:
    def __init__(self, lang='ru'):
        mname = f'Helsinki-NLP/opus-mt-en-{lang}'
        self.tokenizer = AutoTokenizer.from_pretrained(mname)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(mname)
        
    def translate(self, sentence):
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
        hf_outputs = self.model.generate(input_ids)
        translated = self.tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
        return translated


class OCRReader:
    def __init__(self, ocr_thresh=0.2, bbox_thresh=15):
        self.reader = easyocr.Reader(['en'])
        self.threshold = ocr_thresh
        self.max_bboxes = bbox_thresh  # If bboxes is greater then bbox_thresh, skip OCR recognition

    @classmethod
    def area(cls, bbox):
        assert len(bbox) == 6
        return bbox[4] * bbox[5]

    @classmethod
    def int_area(cls, a, b):  # returns None if rectangles don't intersect
        #   0    1    2    3   4  5
        # xmin ymin xmax ymax  w  h
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0

    @classmethod
    def normalize_bbox(cls, bbox, im_size):
        # Normalize bbox coords from (absolute --> relative 0..1)
        assert len(bbox) == 4 and len(bbox[0]) == 2
        w, h = im_size
        xmin = bbox[0][0] / w
        xmax = bbox[2][0] / w
        ymin = bbox[0][1] / h
        ymax = bbox[2][1] / h
        return (xmin, ymin, xmax, ymax)

    @classmethod
    def get_rel_bbox(cls, x_s, y_s, w, h):
        return [
            min(x_s) / w,
            min(y_s) / h,
            max(x_s) / w,
            max(y_s) / h,
            (max(x_s) - min(x_s)) / w,
            (max(y_s) - min(y_s)) / h,
        ]

    def get_ocr_features(self, im_filepath):
            im = Image.open(im_filepath)
            if len(im.split()) > 3:
                background = Image.new("RGB", im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
                im = background

            w, h = im.width, im.height
            # im.load() # required for png.split()

            max_wh = 750
            max_width = max_wh if im.width >= im.height else max_wh * im.width / im.height

            # *** OCR DETECT

            ocr_horiz_dets, ocr_freeform_dets = self.reader.detect(
                np.asarray(im),
                canvas_size=max_width,  # For detection
                # text_threshold=0.75,
                add_margin=0.05,
                # link_threshold=0.3
                width_ths=0.1,
                height_ths=0.5,
                ycenter_ths=0.5,
                min_size=15,
            )

            boxes_cnt = len(ocr_horiz_dets[0]) + len(ocr_freeform_dets[0])
            
            # We don't count side boxes?
            # for det in ocr_horiz_dets
            
            if boxes_cnt > self.max_bboxes or boxes_cnt == 0:
                # print(boxes_cnt)
                return

            # *** OCR RECOGNIZE

            ocr_text = self.reader.recognize(
                np.asarray(im),
                horizontal_list=ocr_horiz_dets[0], 
                free_list=ocr_freeform_dets[0],
                # decoder='beamsearch',
                decoder='greedy',
                # batch_size=16,
            )
            im.close()

            if not ocr_text:
                # print('No OCR text')
                return

            # *** OCR POSTPROCESS

            # ocr_tokens = list(zip(*ocr_text))[1]      

            # Filter by threshold
            ocr_text = [item for item in ocr_text if item[2] > self.threshold]
            if not ocr_text:
                return

            ocr_text_rel = []
            for (bbox, text, conf) in ocr_text:
                x_s, y_s = tuple(zip(*bbox))
                uni_bbox_rel = OCRReader.get_rel_bbox(x_s, y_s, w, h)

                # Remove small text
                if OCRReader.area(uni_bbox_rel) / len(text) < 0.0001:
                    # print('DEL: ', text)
                    # conf *= 0.1
                    continue

                # Remove text near edges of image
                main_area = (0.05, 0.1, 0.95, 0.9)       
                if OCRReader.int_area(main_area, uni_bbox_rel) < (OCRReader.area(uni_bbox_rel) * 0.5):
                    # print('DEL NEAR EDGE:', text)
                    continue

                # Remove strange symbols
                text = text.strip('*-"\\,;~[](){}`^|_ :&@')
                if text == '':
                    # print('EMPTY TOKEN')
                    continue

                ocr_text_rel.append((uni_bbox_rel[:4], text, 0.0))

            if not ocr_text_rel:
                return

            ocr_boxes, ocr_tokens, ocr_confidence = list(zip(*ocr_text_rel))        
            ocr_normalized_boxes = np.asarray(ocr_boxes, dtype=np.float32)

            # print(ocr_tokens)
            # ocr_normalized_boxes = np.asarray([normalize_bbox(t, (w, h)) for t in ocr_boxes], dtype=np.float32)
            return {'im_size': (w, h), 'ocr_results': (ocr_tokens, ocr_normalized_boxes, ocr_confidence)}


class FeatureExtractor:
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self, image_dir):
        with open('../models/visual_genome_categories.json') as f:
            cats = json.load(f)
            cats = cats['categories']

        self.id2cat = {el['id']: el['name'] for el in cats}
        self.model_file = '../models/detectron_model.pth'
        self.config_file = '../models/detectron_model.yaml'
        self.confidence_threshold = 0.2
        self.num_features = 100
        self.batch_size = 1
        self.feature_name = 'fc6'
        self.image_dir = image_dir
        
        self.detection_model = self._build_detection_model()
        self.image_sizes = dict()

    def _build_detection_model(self):
        cfg.merge_from_file(self.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.model_file, map_location=torch.device('cpu'))

        load_state_dict(model, checkpoint.pop('model'))

        model.to('cuda')
        model.eval()
        return model

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)

        if im.shape[-1] > 3:  # RGBA => RGB
            im = np.array(img.convert('RGB')).astype(np.float32)

        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:  # gray => RGB
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)

        im = im[:, :, ::-1]  # RGB => BGR
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {'width': im_width, 'height': im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name='fc6', conf_thresh=0.0
    ):
        batch_size = len(output[0]['proposals'])
        #print(output[0])  # {'fc6', 'fc7', 'proposals', 'pooled', 'scores', 'bbox_deltas'}
        out0, out1 = output  # RPN, ROI

        fc6 = out0['fc6']  # 1000 x 2048
        fc7 = out0['fc7']  # 1000 x 2048
        pooled = out0['pooled']       # 1000 x 512x7x7
        scores = out0['scores']       # 1000 x 1601
        bbox_d = out0['bbox_deltas']  # 1000 x 6404
        props = out0['proposals'][0]  # BoxList 1000 x 4
        # print(props.bbox.shape)

        #print(output[1])  # [BoxList]
        bbox_l = out1[0] # BoxList(num_boxes=100, im_w, im_h, mode)  [100 x 4]
        #print(bbox_l.bbox.shape)

        n_boxes_per_image = [len(boxes) for boxes in output[0]['proposals']]
        #print(n_boxes_per_image) # rpn boxes
        score_list = output[0]['scores'].split(n_boxes_per_image)  # rpn scores for 1 image
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image) # rpn features for 1 image  1000x2048???
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]['proposals'][i].bbox / im_scales[i]   # Scaled bboxes for 1 image [1000x4]
            scores = score_list[i]  # softmaxed scores for 1 image for all classes [1000x1601]
            max_conf = torch.zeros(scores.shape[0]).to(cur_device)  # 1000
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1

            t0 = time.time()
                        
            obj_scores, obj_ids = torch.max(scores[:, start_index:], 1) 
            #                                  [N x 4]   [ N ]    [ N ]
            keep = torchvision.ops.batched_nms(dets, obj_scores, obj_ids, 0.5)

            keeped_scores = obj_scores[keep]
            keeped_dets = dets[keep]
            keeped_obj_ids = obj_ids[keep]
            keeped_feats = feats[i][keep]

            sorted_scores, sorted_indices = torch.sort(keeped_scores, descending=True)
            sorted_ids_by_score = keeped_obj_ids[sorted_indices]
            sorted_dets_by_score = keeped_dets[sorted_indices]
            sorted_feats_by_score = keeped_feats[sorted_indices]

            mask = (sorted_scores > conf_thresh)
            max_feats = self.num_features
            
            res_scores = sorted_scores[mask][:max_feats].cpu().numpy()
            res_obj_ids = sorted_ids_by_score[mask][:max_feats].cpu().numpy()
            res_dets = sorted_dets_by_score[mask][:max_feats].cpu().numpy()
            res_feats = sorted_feats_by_score[mask][:max_feats]
            num_boxes = len(res_scores)

            obj_normalized_boxes = res_dets.copy()
            obj_normalized_boxes[:, ::2] /= im_infos[i]['width']
            obj_normalized_boxes[:, 1::2] /= im_infos[i]['height']

            feat_list.append(res_feats)
            info_list.append(
                {
                    'bbox': res_dets,
                    'bbox_norm': obj_normalized_boxes,
                    'num_boxes': num_boxes,
                    'objects': res_obj_ids,
                    'cls_prob': res_scores,
                    'image_width': im_infos[i]['width'],
                    'image_height': im_infos[i]['height'],
                    'image_classes': set([self.id2cat[obj_id] for obj_id in res_obj_ids]),
                }
            )
                                               
        return feat_list, info_list

    def get_detectron_features(self, image_paths):
        img_tensor, im_scales, im_infos = [], [], []

        for image_path in image_paths:
            im, im_scale, im_info = self._image_transform(image_path)
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to('cuda')

        with torch.no_grad():
            output = self.detection_model(current_img_list)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.feature_name,
            self.confidence_threshold,
        )

        return feat_list

    def extract_features(self, filename):
        im_filepath = filename #op.join(self.image_dir, filename)
        features, infos = self.get_detectron_features([im_filepath])
        # self._save_feature(im_filepath, features[0], infos[0])
        return features[0].cpu().numpy(), infos[0]
    
    def extract_ocr_features(self, im_filepath, ocr_normalized_boxes):
        im, im_scale, im_info = self._image_transform(im_filepath)
        w, h = im_info['width'], im_info['height']

        ocr_boxes = ocr_normalized_boxes.reshape(-1, 4) * [w, h, w, h]
        # ocr_info = {'ocr_boxes': ocr_boxes, 'ocr_tokens': ocr_tokens}

        input_boxes = torch.from_numpy(ocr_boxes.copy()).float()
        input_boxes *= im_scale

        img_tensor, im_scales = [im], [im_scale]
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to('cuda')
        with torch.no_grad():
            output = self.detection_model(current_img_list, input_boxes=input_boxes)

        extracted_feat = output[0][self.feature_name].cpu().numpy()
        return extracted_feat

    def process_image_for_mmf(self, im_filepath, ocr_out):
        w, h = ocr_out['im_size']
        ocr_tokens, ocr_normalized_boxes, ocr_confidence = ocr_out['ocr_results']
        
        im_filename = op.basename(im_filepath)
        img_id = im_filename.split('.')[0]

        self.image_sizes[img_id] = (w, h)
            
        img_info = {
            'image_id': img_id,
            'image_name': im_filename,
            'image_width': w,
            'image_height': h,
            'feature_path': img_id + '.npy',
            'image_path': im_filename,
            'ocr_tokens': ocr_tokens,
            'ocr_normalized_boxes': ocr_normalized_boxes,
            'ocr_confidence': ocr_confidence,
            'caption_id': 300000000,
            'set_name': 'test',
        }

        # 1. Extract ROI feats
        
        roi_features, roi_infos = self.extract_features(im_filepath)
        
        # 2. Add ROI box info into IMDB
        
        assert roi_infos['image_width'] == w and roi_infos['image_height'] == h

        img_info['obj_normalized_boxes'] = roi_infos['bbox_norm']  # N x [xmix, ymin, xmax, ymax]
        img_info['image_classes'] = roi_infos['image_classes']
        img_info['num_boxes'] = roi_infos['num_boxes']
        img_info['features'] = roi_features
        
        # 3. Extract OCR ROI feats
        
        # TODO: Replace multiple image open to one.
        # OPTIMIZE!
        
        ocr_features = self.extract_ocr_features(im_filepath, ocr_normalized_boxes)
        img_info['ocr_features'] = ocr_features
        
        return img_info
    

def setup_pythia_imports():
    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("pythia_root", no_warning=True)

    if root_folder is None:
        root_folder = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.join(root_folder, "CNMT")

        environment_pythia_path = os.environ.get("PYTHIA_PATH")

        if environment_pythia_path is not None:
            root_folder = environment_pythia_path

        root_folder = os.path.join(root_folder, "pythia")
        registry.register("pythia_path", root_folder)

    trainer_folder = os.path.join(root_folder, "trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    datasets_folder = os.path.join(root_folder, "datasets")
    datasets_pattern = os.path.join(datasets_folder, "**", "*.py")
    model_folder = os.path.join(root_folder, "models")
    model_pattern = os.path.join(model_folder, "**", "*.py")

    importlib.import_module("pythia.common.meter")

    files = glob.glob(datasets_pattern, recursive=True) + \
            glob.glob(model_pattern, recursive=True) + \
            glob.glob(trainer_pattern, recursive=True)

    for f in files:
        if f.find("models") != -1:
            splits = f.split(os.sep)
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("pythia.models." + module_name)
        elif f.find("trainer") != -1:
            splits = f.split(os.sep)
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("pythia.trainers." + module_name)
        elif f.endswith("builder.py"):
            splits = f.split(os.sep)
            task_name = splits[-3]
            dataset_name = splits[-2]
            if task_name == "datasets" or dataset_name == "datasets":
                continue
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module(
                "pythia.datasets." + task_name + "." + dataset_name + "." + module_name
            )


class MMFInstance:
    def __init__(self):
        setup_pythia_imports()

        args = argparse.Namespace(
            batch_size=None, clip_gradients=None, 
            config='CNMT/configs/cnmt_rt.yml', config_override=None, 
            config_overwrite=None, data_parallel=None, datasets='m4c_textcaps', 
            device=None, distributed=None, evalai_inference=True, 
            experiment_name=None, fast_read=None, force_restart=False, 
            load_pretrained=None, local_rank=None, log_dir=None, 
            log_interval=None, logger_level=None, lr_scheduler=None, 
            max_epochs=None, max_iterations=None, model='cnmt', 
            num_workers=None, opts=[], patience=None, 
            resume=None, resume_file='../models/best.ckpt', run_type='inference', 
            save_dir='../save/pred/', seed=None, should_not_log=False, 
            snapshot_interval=None, tasks='captioning', verbose_dump=None,
        )
        self.args = args
        
        self.trainer = self.build_trainer()
        self.trainer.load()
        
    def build_trainer(self):
        b_args = vars(self.args)
        configuration = Configuration(b_args['config'])

        configuration.update_with_args(self.args)
        configuration.freeze()

        config = configuration.get_config()

        registry.register("config", config)
        registry.register("configuration", configuration)

        trainer_type = config.training_parameters.trainer
        trainer_cls = registry.get_trainer_class(trainer_type)
        trainer_obj = trainer_cls(config)

        # Set args as an attribute for future use
        setattr(trainer_obj, 'args', self.args)

        return trainer_obj
    
    def run(self, sample):
        result = self.trainer.predict_live(sample)
        return result


def main():
    print('Load 1/5')
    ocr_reader = OCRReader(args.ocr_thresh, args.bbox_thresh) if args.with_ocr else None
    print('Load 2/5')
    feature_extractor = FeatureExtractor(args.image_dir) if args.with_ocr else None
    print('Load 3/5')
    mmf_inst = MMFInstance() if args.with_ocr else None
    print('Load 4/5')
    oscar_inst = OscarLive()
    print('Load 5/5')
    nmt_inst = NMT(lang=args.lang) if args.translate else None
    print('Predicting')
    
    im_list = sorted(os.listdir(args.image_dir))

    images_info = [{'date created': datetime.today().strftime('%Y-%m-%d')}]
    
    gen_results = []
    for im_filename in im_list:  #[:1]: #tqdm(im_list):
        im_filepath = op.join(args.image_dir, im_filename)
        
        # OCR DETECTION
        
        t0 = time.time()
        ocr_results = None
        if args.with_ocr:
            ocr_results = ocr_reader.get_ocr_features(im_filepath)
        print(f" {'*' if ocr_results else ' '} ", end='')
        print(f'OCR: {time.time() - t0:.3f}  ', end=''); t0 = time.time()

        if ocr_results is None:
            # NO OCR FOUND
            res_sentence = oscar_inst.inference(im_filepath) #'../../CNMT/images/COCO_val2014_000000000641.jpg')
        else:     
            img_info = feature_extractor.process_image_for_mmf(im_filepath, ocr_results)
            print(f'Dtrn2: {time.time() - t0:.3f}  ', end=''); t0 = time.time()

            # RUN MMF
            t0 = time.time()
            gen_result = mmf_inst.run(img_info)
            print(f' MMF: {time.time() - t0:.3f}  ', end=''); t0 = time.time()

            # gen_result['image_id']
            caption = gen_result['caption'].split(' ')
            word_types = gen_result['pred_source']

            res_words = [word.upper() if w_type == "OCR" else word for (word, w_type) in zip(caption, word_types)]
            res_sentence = ' '.join(res_words)    
            # print(word_types)

            # RUN LANGUAGE TRANSLATION
            res_words = [f'"{word.upper()}"' if w_type == "OCR" else word for (word, w_type) in zip(caption, word_types)]
            res_sentence = ' '.join(res_words)    

        if not args.translate:
            res_sentence = res_sentence.replace('" "', ' ')
            res_sentence = res_sentence[0].upper() + res_sentence[1:]        
            print((res_sentence + ' ' * 100)[:100])
            gen_results.append({'img_id': im_filename, 'caption': res_sentence, 'translated': ''})
            continue
            
        # else TRANSLATE NMT
        
        post_sentence = res_sentence.replace('a group of ', '').replace('a close up of ', '').replace('aurora ', 'aurora borealis ')
        post_sentence = post_sentence.replace('television ', 'tv ').replace('a couple of ', '').replace('sitting ', '')
        post_sentence = post_sentence.replace('sits ', ' ').replace('on it.', '.')
        
        t0 = time.time()
        translated = nmt_inst.translate(post_sentence)
        print(f'NMT: {time.time() - t0:.3f}  ', end=''); t0 = time.time()

        translated = translated.replace('" "', ' ')
        translated = translated[0].upper() + translated[1:]        
        res_sentence = res_sentence.replace('" "', ' ')
        res_sentence = res_sentence[0].upper() + res_sentence[1:]        
        print((res_sentence + ' ' * 30)[:52], '', translated[:55])        
        #print(ocr_results)
        gen_results.append({'img_id': im_filename, 'caption': res_sentence, 'ocr_tokens': [] if ocr_results is None else ocr_results['ocr_results'][0], 'translated': translated})
    

    os.makedirs(args.save_dir, exist_ok=True)
    
    with open(op.join(args.save_dir, 'gen_result.txt'), 'w') as f:
        [f.writelines(s['caption'] + '\n') for s in gen_results]

    with open(op.join(args.save_dir, 'preds.json'), 'w') as f:
        json.dump({'annotations': gen_results}, f, ensure_ascii=False)

    print('Done.')

    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='../images', type=str, required=False, help='The images folder.')
    parser.add_argument('--save_dir', default='../preds/', type=str, required=False, help='The output directory to save results.')
    parser.add_argument('--with_ocr', action='store_true')
    parser.add_argument('--translate', action='store_true')
    parser.add_argument('--ocr_thresh', default=0.2, type=float, required=False, help='OCR confidence threshold')
    parser.add_argument('--bbox_thresh', default=15, type=int, required=False, help='If OCR founds too many bboxes, we skip OCR recognition')
    parser.add_argument('--lang', default='ru', type=str, required=False, help='ru, fr, es')

    args = parser.parse_args()
    
    assert args.lang in ['ru', 'es', 'fr', 'de']
    
    main()
