import imp
import logging
import time
import os
import os.path as op
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


def compute_on_dataset(model, data_loader, device, bbox_aug, output_folder, labelmap=None):
    dataset = data_loader.dataset
    cpu_device = torch.device("cpu")

    model.eval()

    split_name = op.basename(output_folder)[:-5]
    split_path = op.join(output_folder, split_name) 
    
    label_tsv = split_path +'.label.tsv'
    label_lineidx = split_path +'.label.lineidx'
    feature_tsv = split_path + '.feature.tsv'
    feature_lineidx = split_path +'.feature.lineidx'

    label_idx, feature_idx = 0, 0
    
    with open(label_tsv, 'w') as l_fp, open(label_lineidx, 'w') as l_fpidx, open(feature_tsv, 'w') as f_fp, open(feature_lineidx, 'w') as f_fpidx:
    
        for i, batch in enumerate(tqdm(data_loader)):
            images, targets, image_ids, scales = batch[0], batch[1], batch[2], batch[3:]

            with torch.no_grad():
                output = model(images.to(device), targets)
                output = [o.to(cpu_device) for o in output]

            for img_id, result in zip(image_ids, output):
                idx = img_id
                prediction = result

                image_key = dataset.get_img_key(idx)
                image_width = dataset.get_img_info(idx)['width']
                image_height = dataset.get_img_info(idx)['height']
                prediction = prediction.resize((image_width, image_height))

                boxes_int = prediction.bbox.int().tolist()
                scores = prediction.get_field('scores').tolist()
                labels = prediction.get_field('labels').tolist()

                objects_props = []
                for i in range(len(boxes_int)):
                    cur_d = {}
                    cur_d['class'] = labelmap[labels[i]]
                    cur_d['conf'] = scores[i]
                    cur_d['rect'] = boxes_int[i]
                    objects_props.append(cur_d)

                boxes = prediction.bbox.tolist()
                features = prediction.get_field('box_features').numpy()  # [n, 2048]

                # Add positional info
                features_pos = np.zeros((len(features), 2048 + 6), dtype=np.float32)
                features_pos[:, :-6] = features
                assert features.dtype == features_pos.dtype

                for i, box in enumerate(boxes):
                    # box: [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box
                    box_relative = [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]
                    w, h = box_relative[2] - box_relative[0], box_relative[3] - box_relative[1]
                    features_pos[i, -6:] = np.array(box_relative + [w, h])

                # Unsqueeze #
                all_features = features_pos.reshape(-1)

                objects_feats = {
                    'num_boxes': len(boxes),
                    'features':  base64.b64encode(all_features).decode('utf-8'),
                }

                row = [image_key, json.dumps(objects_props)]
                value = [v if type(v)!=bytes else v.decode('utf-8') for v in row]
                v = '\t'.join(map(str, value)) + '\n'
                l_fp.write(v)
                l_fpidx.write(str(label_idx) + '\n')
                label_idx += len(v)

                row = [image_key, json.dumps(objects_feats)]
                value = [v if type(v)!=bytes else v.decode('utf-8') for v in row]
                v = '\t'.join(map(str, value)) + '\n'
                f_fp.write(v)
                f_fpidx.write(str(feature_idx) + '\n')
                feature_idx += len(v)


def inference(model, cfg, data_loader, dataset_name, iou_types=("bbox",), box_only=False, bbox_aug=False, device="cuda", expected_results=(),
              expected_results_sigma_tol=4, output_folder=None, eval_attributes=False, save_predictions=False, skip_performance_eval=False, labelmap_file=''):
    device = torch.device(device)
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    #print(dir(dataset))

    if os.path.isfile(labelmap_file):
        labelmap = load_labelmap_file(labelmap_file)
        labelmap = {labelmap[key] + 1: key for key in labelmap}
    elif hasattr(dataset, 'ind_to_class'):
        labelmap = dataset.ind_to_class
    else:
        raise ValueError("object labelmap is required, but was not provided")

    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))

    compute_on_dataset(model, data_loader, device, bbox_aug, output_folder, labelmap=labelmap)
    
    logger.info('Done.')