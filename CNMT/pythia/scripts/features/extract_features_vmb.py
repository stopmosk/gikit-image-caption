# Copyright (c) Facebook, Inc. and its affiliates.

# Requires vqa-maskrcnn-benchmark (https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark)
# to be built and installed. Category mapping for visual genome can be downloaded from
# https://dl.fbaipublicfiles.com/pythia/data/visual_genome_categories.json
# When the --background flag is set, the index saved with key "objects" in
# info_list will be +1 of the Visual Genome category mapping above and 0
# is the background class. When the --background flag is not set, the
# index saved with key "objects" in info list will match the Visual Genome
# category mapping.
import argparse
import os
import glob
import time
import json

import cv2
import numpy as np
import torch
import torchvision

from v_maskrcnn_benchmark.config import cfg
from v_maskrcnn_benchmark.layers import nms
from v_maskrcnn_benchmark.modeling.detector import build_detection_model
from v_maskrcnn_benchmark.structures.image_list import to_image_list
from v_maskrcnn_benchmark.utils.model_serialization import load_state_dict

#from mmf.utils.download import download
from PIL import Image
from tqdm import tqdm
#from tools.scripts.features.extraction_utils import chunks, get_image_files


def get_image_files(
    image_dir,
    exclude_list=None,
    partition=None,
    max_partition=None,
    start_index=0,
    end_index=None,
    output_folder=None,
):
    files = glob.glob(os.path.join(image_dir, "*.png"))
    files.extend(glob.glob(os.path.join(image_dir, "*.jpg")))
    files.extend(glob.glob(os.path.join(image_dir, "*.jpeg")))

    files = set(files)
    exclude = set()

    if os.path.exists(exclude_list):
        with open(exclude_list) as f:
            lines = f.readlines()
            for line in lines:
                exclude.add(line.strip("\n").split(os.path.sep)[-1].split(".")[0])
    output_ignore = set()
    if output_folder is not None:
        output_files = glob.glob(os.path.join(output_folder, "*.npy"))
        for f in output_files:
            file_name = f.split(os.path.sep)[-1].split(".")[0]
            output_ignore.add(file_name)

    for f in list(files):
        file_name = f.split(os.path.sep)[-1].split(".")[0]
        if file_name in exclude: # or file_name in output_ignore:
            files.remove(f)

    files = list(files)
    files = sorted(files)

    if partition is not None and max_partition is not None:
        interval = math.floor(len(files) / max_partition)
        if partition == max_partition:
            files = files[partition * interval :]
        else:
            files = files[partition * interval : (partition + 1) * interval]

    if end_index is None:
        end_index = len(files)

    files = files[start_index:end_index]

    return files


def chunks(array, chunk_size):
    for i in range(0, len(array), chunk_size):
        yield array[i : i + chunk_size], i



class FeatureExtractor:

    MODEL_URL = {
        "X-101": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model.pth",
        "X-152": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model_x152.pth",
    }
    CONFIG_URL = {
        "X-101": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model.yaml",
        "X-152": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model_x152.yaml",
    }

    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self._try_downloading_necessities(self.args.model_name)
        self.detection_model = self._build_detection_model()

        os.makedirs(self.args.output_folder, exist_ok=True)

    def _try_downloading_necessities(self, model_name):
        if self.args.model_file is None and model_name is not None:
            model_url = self.MODEL_URL[model_name]
            config_url = self.CONFIG_URL[model_name]
            self.args.model_file = model_url.split("/")[-1]
            self.args.config_file = config_url.split("/")[-1]
            if os.path.exists(self.args.model_file) and os.path.exists(
                self.args.config_file
            ):
                print(f"model and config file exists in directory: {os.getcwd()}")
                return
            print("Downloading model and configuration")
            download(model_url, ".", self.args.model_file)
            download(config_url, ".", self.args.config_file)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_name", default="X-152", type=str, help="Model to use for detection"
        )
        parser.add_argument(
            "--model_file",
            default=None,
            type=str,
            help="Detectron model file. This overrides the model_name param.",
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Detectron config file"
        )
        parser.add_argument(
            "--start_index", default=0, type=int, help="Index to start from "
        )
        parser.add_argument("--end_index", default=None, type=int, help="")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--output_folder", type=str, default="./output", help="Output folder"
        )
        parser.add_argument("--image_dir", type=str, help="Image directory or file")
        parser.add_argument(
            "--feature_name",
            type=str,
            help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--exclude_list",
            type=str,
            help="List of images to be excluded from feature conversion. "
            + "Each image on a new line",
            default="./list",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        return parser

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)

        if im.shape[-1] > 3:
            im = np.array(img.convert("RGB")).astype(np.float32)

        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)

        im = im[:, :, ::-1]
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

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        batch_size = len(output[0]["proposals"])
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

        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        #print(n_boxes_per_image) # rpn boxes
        score_list = output[0]["scores"].split(n_boxes_per_image)  # rpn scores for 1 image
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image) # rpn features for 1 image  1000x2048???
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]   # Scaled bboxes for 1 image [1000x4]
            scores = score_list[i]  # softmaxed scores for 1 image for all classes [1000x1601]
            max_conf = torch.zeros(scores.shape[0]).to(cur_device)  # 1000
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0

            t0 = time.time()
                        
            obj_scores, obj_ids = torch.max(scores[:, start_index:], 1)  # находим максимальные скоры в каждом боксе из 1000 и получаем соотв id классов
            #                                  [N x 4]   [ N ]    [ N ]
            keep = torchvision.ops.batched_nms(dets, obj_scores, obj_ids, 0.5)  # фильтруем боксы, keep возвращает номера боксов, которые нужно оставить

            keeped_scores = obj_scores[keep]
            keeped_dets = dets[keep]
            keeped_obj_ids = obj_ids[keep]
            keeped_feats = feats[i][keep]

            sorted_scores, sorted_indices = torch.sort(keeped_scores, descending=True)
            sorted_ids_by_score = keeped_obj_ids[sorted_indices]
            sorted_dets_by_score = keeped_dets[sorted_indices]
            sorted_feats_by_score = keeped_feats[sorted_indices]

            mask = (sorted_scores > conf_thresh)
            max_feats = self.args.num_features
            
            res_scores = sorted_scores[mask][:max_feats]
            res_obj_ids = sorted_ids_by_score[mask][:max_feats]
            res_dets = sorted_dets_by_score[mask][:max_feats]
            res_feats = sorted_feats_by_score[mask][:max_feats]
            num_boxes = len(res_scores)

            feat_list.append(res_feats)
            info_list.append(
                {
                    "bbox": res_dets.cpu().numpy(),
                    "num_boxes": num_boxes,
                    "objects": res_obj_ids.cpu().numpy(),
                    "cls_prob": res_scores.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    "image_classes": set([id2cat[obj_id] for obj_id in res_obj_ids.cpu().numpy()]),
                }
            )
                                               
            # print(' '.join([id2cat[obj_id] for obj_id in res_obj_ids.cpu().numpy()]))
            t1 = time.time()
            
#             for cls_ind in range(start_index, scores.shape[1]):  # 1..1601
#                 cls_scores = scores[:, cls_ind]  # scores for current class for all boxes [1000]
#                 # keep = torchvision.ops.nms(dets, cls_scores, 0.5)
#                 keep = nms(dets, cls_scores, 0.5)  # NMS  list of keeped boxes numbers
#                 max_conf[keep] = torch.where(
#                     # Better than max one till now and minimally greater
#                     # than conf_thresh
#                     (cls_scores[keep] > max_conf[keep])
#                     & (cls_scores[keep] > conf_thresh_tensor[keep]),
#                     cls_scores[keep],
#                     max_conf[keep],
#                 )
            
#             sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
#             #print(sorted_scores[: self.args.num_features])
#             num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
#             keep_boxes = sorted_indices[: num_boxes] #self.args.num_features]
#             #print(max_conf.shape)
#             feat_list.append(feats[i][keep_boxes])
#             bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
#             # Predict the class label using the scores
#             objects = torch.argmax(scores[keep_boxes][:, start_index:], dim=1)

#             info_list.append(
#                 {
#                     "bbox": bbox.cpu().numpy(),
#                     "num_boxes": num_boxes.item(),
#                     "objects": objects.cpu().numpy(),
#                     "cls_prob": scores[keep_boxes][:, start_index:].cpu().numpy(),
#                     "image_width": im_infos[i]["width"],
#                     "image_height": im_infos[i]["height"],
#                 }
#             )

            t2 = time.time()
            # print('batched_nms:', t1-t0) #, 'nms_in:', t2-t1)

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
        current_img_list = current_img_list.to("cuda")

        t_srt = time.time()

        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            output = self.detection_model(current_img_list)

        t_mid = time.time()

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )
        t_end = time.time()

        # print('detectron:', t_mid - t_srt)
        # print('nms:', t_end - t_mid)
        # print('other:', t_ - t_srt)


        return feat_list

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info_file_base_name = file_base_name + "_info.npy"
        file_base_name = file_base_name + ".npy"

        np.save(
            os.path.join(self.args.output_folder, file_base_name), feature.cpu().numpy()
        )
        np.save(os.path.join(self.args.output_folder, info_file_base_name), info)

    def extract_features(self):
        image_dir = self.args.image_dir
        if os.path.isfile(image_dir):
            features, infos = self.get_detectron_features([image_dir])
            self._save_feature(image_dir, features[0], infos[0])
        else:
            files = get_image_files(
                self.args.image_dir,
                exclude_list=self.args.exclude_list,
                start_index=self.args.start_index,
                end_index=self.args.end_index,
                output_folder=self.args.output_folder,
            )

            total = len(files)
            for chunk, begin_idx in tqdm(chunks(files, self.args.batch_size), total=total):
                features, infos = self.get_detectron_features(chunk)
                for idx, file_name in enumerate(chunk):
                    self._save_feature(file_name, features[idx], infos[idx])

                    
if __name__ == "__main__":
    with open('visual_genome_categories.json') as f:
        cats = json.load(f)
        cats = cats['categories']

    id2cat = {el['id']: el['name'] for el in cats}

    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
