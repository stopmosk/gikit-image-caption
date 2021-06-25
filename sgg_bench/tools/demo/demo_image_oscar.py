import os
import os.path as op
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import cv2
import torch

from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import config_dataset_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from tools.demo.visual_utils import draw_bb, draw_rel


def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def detect_objects_on_single_image(model, transforms, cv2_img):
    # cv2_img is the original input, so we can get the height and width information to scale the output boxes.
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    img_input = img_input.to(model.device)

    # with torch.cuda.amp.autocast():
    with torch.no_grad():
        prediction = model(img_input)
        prediction = prediction[0].to(torch.device('cpu'))

    img_height = cv2_img.shape[0]
    img_width = cv2_img.shape[1]

    # Filter small bboxes
    pred_rel = prediction.resize((1.0, 1.0))
    squares = (pred_rel.bbox[:, 2] - pred_rel.bbox[:, 0]) * (pred_rel.bbox[:, 3] - pred_rel.bbox[:, 1])
    # square_threshold = cfg.MODEL.ROI_HEADS.SQR_THRESH
    square_threshold = 0.005
    filtered_box_ids = (squares > square_threshold).nonzero(as_tuple=False).squeeze(1).tolist()

    prediction = prediction.resize((1, 1)) #((img_width, img_height))
    boxes = prediction.bbox.tolist()
    classes = prediction.get_field('labels').tolist()
    scores = prediction.get_field('scores').tolist()
    features = prediction.get_field('box_features').numpy()  # [n, 2048]
    # Positional info will be added in next steps

    if 'attr_scores' in prediction.extra_fields:
        attr_scores = prediction.get_field("attr_scores")
        attr_labels = prediction.get_field("attr_labels")
        return [
            {'rect': box, 'class': cls, 'conf': score, 'feat': feat,
             'attr': attr[attr_conf > 0.01].tolist(), 'attr_conf': attr_conf[attr_conf > 0.01].tolist()}
            for i, (box, cls, score, feat, attr, attr_conf) in
            enumerate(zip(boxes, classes, scores, features, attr_labels, attr_scores)) if i in filtered_box_ids
        ]

    return [
        {'rect': box, 'class': cls, 'conf': score, 'feat': feat}
        for i, (box, cls, score, feat) in enumerate(zip(boxes, classes, scores, features)) if i in filtered_box_ids
    ]


def postprocess_attr(dataset_attr_labelmap, label_list, conf_list):
    common_attributes = {
        'white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'silver', 'wooden',
        'wood', 'orange', 'gray', 'grey', 'metal', 'pink', 'tall', 'long', 'dark', 'purple'
    }
    common_attributes_thresh = 0.1
    attr_alias_dict = {'blonde': 'blond'}
    attr_dict = {}
    for label, conf in zip(label_list, conf_list):
        label = dataset_attr_labelmap[label]
        if label in common_attributes and conf < common_attributes_thresh:
            continue
        if label in attr_alias_dict:
            label_target = attr_alias_dict[label]
        else:
            label_target = label
        if label_target in attr_dict:
            attr_dict[label_target] += conf
        else:
            attr_dict[label_target] = conf
    if len(attr_dict) > 0:
        # the most confident one comes the last
        sorted_dic = sorted(attr_dict.items(), key=lambda kv: kv[1])
        return list(zip(*sorted_dic))
    else:
        return [[], []]


class VinVLDetector:
    def __init__(self):
        # parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
        #                     help='Modify config options using the command-line')

        # config_file = './sgg_bench/sgg_configs/vgattr/vinvl_imcap.yaml'
        config_file = './sgg_bench/models/vinvl/vg/for_live.yaml'
        # args.visualize_attr = True

        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.set_new_allowed(False)
        cfg.merge_from_file(config_file)
        # cfg.merge_from_list(args.opts)
        cfg.freeze()

        self.input_dir = cfg.DATA_DIR
        self.output_dir = cfg.OUTPUT_DIR
        mkdir(self.output_dir)

        self.model = AttrRCNN(cfg)
        self.model.to(cfg.MODEL.DEVICE)
        self.model.eval()

        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=self.output_dir)
        checkpointer.load(cfg.MODEL.WEIGHT)

        # dataset labelmap is used to convert the prediction to class labels
        dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR, cfg.DATASETS.LABELMAP_FILE)
        assert dataset_labelmap_file
        dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))

        self.dataset_labelmap = {int(val): key for key, val in dataset_allmap['label_to_idx'].items()}
        self.transforms = build_transforms(cfg, is_train=False)

        # print(torch.cuda._initialized)
        # x = torch.randn(10).cuda()
        # print(torch.cuda._initialized)

    def infer_file(self, filename):
        cv2_img = cv2.imread(filename)
        dets = detect_objects_on_single_image(self.model, self.transforms, cv2_img)

        for obj in dets:
            obj['class'] = self.dataset_labelmap[obj['class']]

        rects = [d['rect'] for d in dets]
        feats = [d['feat'] for d in dets]
        labels = [d['class'] for d in dets]
        scores = [d['conf'] for d in dets]

        # draw_bb(cv2_img, rects, labels, scores)

        # cv2.imshow('image', cv2_img)
        # cv2.waitKey(0)

        # save_file = op.splitext(filename)[0] + '.detect.jpg'
        # save_file = op.join(self.output_dir, save_file)
        # cv2.imwrite(save_file, cv2_img)

        return rects, feats, labels, scores

    def infer_dir(self):
        file_list = sorted(os.listdir(self.input_dir))
        for filename in tqdm(file_list):
            self.infer_file(op.join(self.input_dir, filename))


def main():
    parser = argparse.ArgumentParser(description='Object Detection Demo')
    parser.add_argument('--config_file', metavar='FILE', help='path to config file')
    # parser.add_argument('--img_file', metavar='FILE', help='image path')
    parser.add_argument('--img_dir', metavar='FILE', help='image path')
    parser.add_argument('--save_file', required=False, type=str, default=None, help='filename to save the proceed image')
    parser.add_argument('--visualize_attr', action='store_true', help='visualize the object attributes')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    # dataset labelmap is used to convert the prediction to class labels
    dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR, cfg.DATASETS.LABELMAP_FILE)
    assert dataset_labelmap_file
    dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))
    dataset_labelmap = {int(val): key for key, val in dataset_allmap['label_to_idx'].items()}

    if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
        dataset_attr_labelmap = {
            int(val): key for key, val in
            dataset_allmap['attribute_to_idx'].items()}
    
    transforms = build_transforms(cfg, is_train=False)

    file_list = os.listdir(args.img_dir)
    for filename in tqdm(file_list):
        cv2_img = cv2.imread(op.join(args.img_dir, filename))
        dets = detect_objects_on_single_image(model, transforms, cv2_img)

        for obj in dets:
            obj['class'] = dataset_labelmap[obj['class']]

        if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
            for obj in dets:
                obj['attr'], obj['attr_conf'] = postprocess_attr(dataset_attr_labelmap, obj['attr'], obj['attr_conf'])

        rects = [d['rect'] for d in dets]
        scores = [d['conf'] for d in dets]
        feats = [d['feat'] for d in dets]

        if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
            attr_labels = [','.join(d['attr']) for d in dets]
            attr_scores = [d['attr_conf'] for d in dets]
            labels = [attr_label+' '+d['class']
                      for d, attr_label in zip(dets, attr_labels)]
        else:
            labels = [d['class'] for d in dets]

        draw_bb(cv2_img, rects, labels, scores)

        # cv2.imshow('image', cv2_img)
        # cv2.waitKey(0)

        # img2 = cv2_img[:, :, ::-1]
        # plt.imshow(img2)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        save_file = op.splitext(filename)[0] + '.detect.jpg'
        save_file = op.join(cfg.OUTPUT_DIR, save_file)
        cv2.imwrite(save_file, cv2_img)
        # print("save results to: {}".format(save_file))

        # save results in text
        if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
            result_str = ''
            for label, score, attr_score in zip(labels, scores, attr_scores):
                result_str += label+'\n'
                result_str += ','.join([str(conf) for conf in attr_score])
                result_str += '\t'+str(score)+'\n'
            text_save_file = op.splitext(save_file)[0] + '.txt'
            with open(text_save_file, 'w') as fid:
                fid.write(result_str)


def main2():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--config_file', metavar='FILE', help='path to config file')
    parser.add_argument('--visualize_attr', action='store_true', help='visualize the object attributes')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')

    # args.config_file = '../../sgg_configs/vgattr/vinvl_imcap.yaml'
    # args.visualize_attr = True

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    detector = VinVLDetector(cfg)
    detector.infer_dir()


if __name__ == "__main__":
    main2()
