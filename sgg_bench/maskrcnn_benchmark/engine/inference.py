# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import imp
import logging
import time
import os
import json
import base64

import numpy as np
import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.structures.tsv_file_ops import tsv_writer
from maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from scene_graph_benchmark.scene_parser import SceneParserOutputs

from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather, gather_on_master
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2

# # STOPMOSK
# ds_root = '/media/stopmosk/data/huawei/datasets/my'
# img_list = os.listdir(ds_root)


def compute_on_dataset(model, data_loader, device, bbox_aug, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, scales = batch[0], batch[1], batch[2], batch[3:]

        # images_ = cv2.imread(os.path.join(ds_root, img_list[i]))
        # print(images.tensors)
        # print(images.tensors.size(), flush=True)
        # print(images.tensors[0])

        # a = images.tensors[0].permute(1, 2, 0).numpy()
        # cv2.imshow('window', a)
        # cv2.waitKey(0)
        # exit()
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                try:
                    output = model(images.to(device), targets)
                except RuntimeError as e:
                    image_ids_str = [str(img_id) for img_id in image_ids]
                    print("Runtime error occurred in Image Ids: {}"
                          .format(','.join(image_ids_str)))
                    print(e)
                    continue
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, gather_on_cpu=False):
    if gather_on_cpu:
        all_predictions = gather_on_master(predictions_per_gpu)
    else:
        all_predictions = all_gather(predictions_per_gpu)

    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    return predictions


def convert_predictions_to_tsv(predictions, dataset, output_folder,
                               data_subset, labelmap_file=None,
                               relation_on=False,
                               output_tsv_name='predictions.tsv'):
    # convert the prediction results to tsv format and save
    # for easier visualization and post-processing.
    if 'class' in data_subset:
        if os.path.isfile(labelmap_file):
            labelmap = load_labelmap_file(labelmap_file)
            labelmap = {labelmap[key] + 1: key for key in labelmap}
        elif hasattr(dataset, 'ind_to_class'):
            labelmap = dataset.ind_to_class
        else:
            raise ValueError("object labelmap is required, but was not provided")
    if 'attr_labels' in data_subset:
        if os.path.isfile(labelmap_file):
            attr_labelmap = json.load(open(labelmap_file, 'r'))['attribute_to_idx']
            attr_labelmap['__no_attribute__'] = 0
            attr_labelmap = {v:k for k, v in attr_labelmap.items()}
        elif hasattr(dataset, 'ind_to_attribute'):
            attr_labelmap = dataset.ind_to_attribute
        else:
            raise ValueError("attribute labelmap is required, but was not provided")
    if 'relations' in data_subset:
        if os.path.isfile(labelmap_file):
            relation_labelmap = json.load(open(labelmap_file, 'r'))['predicate_to_idx']
            relation_labelmap['__no_relation__'] = 0
            relation_labelmap = {relation_labelmap[key]: key for key in relation_labelmap}
        elif hasattr(dataset, 'ind_to_relation'):
            relation_labelmap = dataset.ind_to_relation
        else:
            raise ValueError("relation labelmap is required, but was not provided")
    
    def gen_rows():
        for idx, prediction in sorted(predictions.items()):
            image_key = dataset.get_img_key(idx)
            image_width = dataset.get_img_info(idx)['width']
            image_height = dataset.get_img_info(idx)['height']

            if isinstance(prediction, SceneParserOutputs):
                prediction_pred = prediction.prediction_pairs
                prediction = prediction.predictions

                relations = prediction_pred.get_field("idx_pairs").numpy()
                relation_scores = prediction_pred.get_field("scores").numpy()
                predicates = prediction_pred.get_field("labels").numpy()
                if 'relation_scores_all' in data_subset:
                    relation_scores_all = prediction_pred.get_field("scores_all").numpy()
                if 'relation_feature' in data_subset:
                    relation_features = prediction_pred.get_field("pred_features").numpy()

            prediction = prediction.resize((image_width, image_height))
            boxes = prediction.bbox.tolist()

            if 'conf' in data_subset:
                scores = prediction.get_field('scores').tolist()
            if 'class' in data_subset:
                labels = prediction.get_field('labels').tolist()
            if 'feature' in data_subset:
                features = prediction.get_field('box_features').numpy()
            if 'scores_all' in data_subset:
                scores_all = prediction.get_field('scores_all').numpy()
            if 'boxes_all' in data_subset:
                boxes_all = prediction.get_field('boxes_all').numpy()
            if "attr_labels" in data_subset:
                attr_labels = prediction.get_field("attr_labels").tolist()
            if "attr_scores" in data_subset:
                attr_scores = prediction.get_field("attr_scores").tolist()
            if "attr_scores_all" in data_subset:
                attr_scores_all = prediction.get_field("attr_scores_all").numpy()
            if 'relations' in data_subset:
                relations = relations.tolist()
                predicates = [relation_labelmap[rel+1] for rel in predicates.tolist()]
            if 'relation_scores' in data_subset:
                relation_scores = relation_scores.tolist()
            if 'relation_scores_all' in data_subset:
                relation_scores_all = [base64.b64encode(relation_scores_all[i]).decode('utf-8') for i in range(len(relations))]

            objects = []
            for i in range(len(boxes)):
                cur_d = {}
                for name in data_subset:
                    if name == 'rect':
                        cur_d['rect'] = boxes[i]
                        cur_d['bbox_id'] = i
                    if name == 'class':
                        cur_d['class'] = labelmap[labels[i]]
                    if name == 'conf':
                        cur_d['conf'] = scores[i]
                    if name == 'feature':
                        cur_d['feature'] = base64.b64encode(features[i]).decode('utf-8')

                    if name == 'scores_all':
                        cur_d['scores_all'] = base64.b64encode(scores_all[i]).decode('utf-8')
                    if name == 'boxes_all':
                        cur_d['boxes_all'] = base64.b64encode(boxes_all[i]).decode('utf-8')
                    if name == 'attr_labels':
                        cur_d['attributes'] = []
                        for attr in attr_labels[i]:
                            cur_d['attributes'].append(attr_labelmap[attr])
                    if name == 'attr_scores':
                        cur_d['attr_scores'] = []
                        for attr_score in attr_scores[i]:
                            cur_d['attr_scores'].append(attr_score)
                    if name == 'attr_scores_all':
                        cur_d['attr_scores_all'] = base64.b64encode(attr_scores_all[i]).decode('utf-8')
                objects.append(cur_d)
            
            triplets = None
            if relation_on:
                triplets = []
                for i in range(len(relations)):
                    cur_d = {}
                    for name in data_subset:
                        if name == 'relations':
                            cur_d['subj_id'] = relations[i][0]
                            cur_d['obj_id'] = relations[i][1]
                            cur_d['class'] = predicates[i]
                        if name == 'relation_scores':
                            cur_d['conf'] = relation_scores[i]
                        if name == 'relation_scores_all':
                            cur_d['scores_all'] = relation_scores_all[i]
                        if name == 'relation_feature':
                            cur_d['relation_feature'] = base64.b64encode(relation_features[i]).decode('utf-8')
                    triplets.append(cur_d)
            
            yield image_key, json.dumps({'objects': objects, 'relations':triplets})
    
    tsv_writer(gen_rows(), os.path.join(output_folder, output_tsv_name))


def convert_predi_to_tsv_oscar(predictions, dataset, output_folder, data_subset,
                               labelmap_file=None, relation_on=False, output_tsv_name='predictions.tsv'):
    if 'class' in data_subset:
        if os.path.isfile(labelmap_file):
            labelmap = load_labelmap_file(labelmap_file)
            labelmap = {labelmap[key] + 1: key for key in labelmap}
        elif hasattr(dataset, 'ind_to_class'):
            labelmap = dataset.ind_to_class
        else:
            raise ValueError("object labelmap is required, but was not provided")

    def gen_rows_1():
        for idx, prediction in sorted(predictions.items()):
            image_key = dataset.get_img_key(idx)
            image_width = dataset.get_img_info(idx)['width']
            image_height = dataset.get_img_info(idx)['height']
            prediction = prediction.resize((image_width, image_height))

            boxes = prediction.bbox.tolist()
            scores = prediction.get_field('scores').tolist()
            labels = prediction.get_field('labels').tolist()

            objects = []
            for i in range(len(boxes)):
                cur_d = {}
                cur_d['rect'] = boxes[i]
                # cur_d['bbox_id'] = i
                cur_d['class'] = labelmap[labels[i]]
                cur_d['conf'] = scores[i]
                objects.append(cur_d)

            yield image_key, json.dumps(objects)

    def gen_rows_2():
        for idx, prediction in sorted(predictions.items()):
            # print(type(prediction))
            image_key = dataset.get_img_key(idx)
            image_width = dataset.get_img_info(idx)['width']
            image_height = dataset.get_img_info(idx)['height']
            im = dataset.get_image(idx)
            im: Image.Image

            # im.show()
            # print(prediction.bbox)

            prediction = prediction.resize((image_width, image_height))

            boxes = prediction.bbox.tolist()

            features = prediction.get_field('box_features').numpy()  # [n, 2048]

            # Add positional info
            features_pos = np.zeros((len(features), 2048 + 6), dtype=np.float32)
            features_pos[:, :-6] = features
            assert features.dtype == features_pos.dtype

            # DRAW IMG BOXES
            plt.figure(figsize=(16, 10))
            draw = ImageDraw.Draw(im)

            for i, box in enumerate(boxes):
                # box: [x1, y1, x2, y2]
                x1, y1, x2, y2 = box
                # print(i, box, image_width, image_height)
                # x1, y1 = max(0, x1), max(0, y1)
                # x2, y2 = min(x2, image_width), min(y2, image_height)
                # assert x1 < x2 and y1 < y2

                box_relative = [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]
                w, h = box_relative[2] - box_relative[0], box_relative[3] - box_relative[1]
                features_pos[i, -6:] = np.array(box + [w, h])

                draw.rectangle((x1, y1, x2, y2), outline=255)

            plt.axis('off')

            plt.imshow(im)
            # plt.savefig(sample_img['image_id'] + '.png', bbox_inches='tight')
            plt.show()

            # Unsqueeze #
            all_features = features_pos.reshape(-1)

            cur_d = {
                'features': base64.b64encode(all_features).decode('utf-8'),
                'num_boxes': len(boxes),
            }

            yield image_key, json.dumps(cur_d)


    def gen_rows_3():
        for idx, prediction in sorted(predictions.items()):
            # print(type(prediction))
            image_key = dataset.get_img_key(idx)
            image_width = dataset.get_img_info(idx)['width']
            image_height = dataset.get_img_info(idx)['height']
            im = dataset.get_image(idx)
            im: Image.Image

            # im.show()

            prediction = prediction.resize((image_width, image_height))

            # ['labels', 'scores', 'box_features', 'scores_all', 'boxes_all', 'attr_labels', 'attr_scores']
            labels = prediction.get_field('labels')
            scores = prediction.get_field('scores')
            print(labels, scores)
            labels_text = [labelmap[el.item()] for el in labels]
            print(labels_text)


            boxes = prediction.bbox.tolist()
            features = prediction.get_field('box_features').numpy()  # [n, 2048]

            # Add positional info
            features_pos = np.zeros((len(features), 2048 + 6), dtype=np.float32)
            features_pos[:, :-6] = features
            assert features.dtype == features_pos.dtype

            # DRAW IMG BOXES
            plt.figure(figsize=(16, 10))
            draw = ImageDraw.Draw(im)

            for i, box in enumerate(boxes):
                # box: [x1, y1, x2, y2]
                x1, y1, x2, y2 = box
                print(i, box, image_width, image_height)
                # x1 = max(0, x1)
                # y1 = max(0, y1)
                # x2 = min(x2, image_width)
                # y2 = min(y2, image_height)
                # x2 = max(x2, x1)
                # y2 = max(y2, y1)
                # print(i, (x1, y1, x2, y2))
                # assert x1 <= x2 and y1 <= y2
                box_relative = [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]
                w, h = box_relative[2] - box_relative[0], box_relative[3] - box_relative[1]
                features_pos[i, -6:] = np.array(box + [w, h])

                bbox = box
                # text = block['word']
                # print(text, end=' | ')
                w, h = x2 - x1, y2 - y1
                # x1, x2 = x1 * im.width, x2 * im.width
                # y1, y2 = y1 * im.height, y2 * im.height
                # w, h = w * im.width, h * im.height
                draw.rectangle((x1, y1, x2, y2), outline=255)


            # text_cap = f"GENERATED CAP: {res_dict[sample_img['image_id']]}"
            # print('\n' + text_cap)

            # max_str_len = im.width // 6  # max string length in letters
            # strings_num = len(text_cap) // max_str_len + 1
            #
            # b_text = [0, 0, im.width, 12 * strings_num]
            # draw.rectangle(b_text, fill=255)
            #
            # text_cap_chunks = [text_cap[seek: seek + max_str_len] for seek in range(0, len(text_cap), max_str_len)]
            # for i, chunk in enumerate(text_cap_chunks):
            #     draw.text((0, i * 12), chunk, stroke_width=40, stroke_fill=255)

            plt.axis('off')

            plt.imshow(im)
            # plt.savefig(sample_img['image_id'] + '.png', bbox_inches='tight')
            plt.show()



            # Unsqueeze #
            all_features = features_pos.reshape(-1)

            cur_d = {
                'features': base64.b64encode(all_features).decode('utf-8'),
                'num_boxes': len(boxes),
            }

            yield image_key, json.dumps(cur_d)


    tsv_writer(gen_rows_1(), os.path.join(output_folder, output_tsv_name[:-4] + '.label.tsv'))
    tsv_writer(gen_rows_2(), os.path.join(output_folder, output_tsv_name[:-4] + '.feature.tsv' ))


def inference(
        model,
        cfg,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        eval_attributes=False,
        save_predictions=False,
        skip_performance_eval=False,
        labelmap_file='',
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions, cfg.TEST.GATHER_ON_CPU)

    if not is_main_process():
        return

    if output_folder and save_predictions:
        torch.save(predictions, os.path.join(output_folder, 'predictions.pth'))
    
    if output_folder and cfg.TEST.SAVE_RESULTS_TO_TSV:
        logger.info('Convert prediction results to tsv format and save.')
        output_tsv_name = 'predictions_forcebox.tsv' if eval_attributes else 'predictions.tsv'
        # convert_predictions_to_tsv(
        convert_predi_to_tsv_oscar(
            predictions, dataset, output_folder,
            data_subset=cfg.TEST.TSV_SAVE_SUBSET,
            labelmap_file=labelmap_file,
            output_tsv_name=output_tsv_name,
            relation_on=cfg.MODEL.RELATION_ON,
        )
    
    if skip_performance_eval:
        logger.info('Skip performance evaluation and return.')
        return

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        save_predictions=save_predictions
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
