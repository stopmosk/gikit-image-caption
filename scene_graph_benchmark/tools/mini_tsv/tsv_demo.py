# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import os
import os.path as op
import json
import cv2
import base64
from tqdm import tqdm

from maskrcnn_benchmark.structures.tsv_file_ops import tsv_reader, tsv_writer
from maskrcnn_benchmark.structures.tsv_file_ops import generate_linelist_file
from maskrcnn_benchmark.structures.tsv_file_ops import generate_hw_file
from maskrcnn_benchmark.structures.tsv_file import TSVFile
from maskrcnn_benchmark.data.datasets.utils.image_ops import img_from_base64

# To generate a tsv file:
orig_root = '/media/stopmosk/data/huawei/datasets/textcaps_orig'
data_path = op.join(orig_root, 'train_images')
cap_orig = op.join(orig_root, 'TextCaps_0.1_train.json')
img_list = os.listdir(data_path)

exp_root = '/media/stopmosk/data/huawei/datasets/textcaps_nn'
tsv_file = op.join(exp_root, 'train.tsv')
label_file = op.join(exp_root, 'train.label.tsv')
hw_file = op.join(exp_root, 'train.hw.tsv')
linelist_file = op.join(exp_root, 'train.linelist.tsv')
cap_exp = op.join(exp_root, 'train_caption.json')

with open(cap_orig) as fp:
    ds = json.load(fp)
ds_labels = ds['data']
# print(len(ds_labels))

# sample = ds_labels[2]
# print(sample.keys())
# print(sample)
# print(sample['set_name'], sample['image_id'], sample['image_width'], sample['image_height'])
# ref_strs = sample['reference_strs']
# print(len(ref_strs))
# ref_sample = ref_strs[0]
# print(ref_sample)

captions = []
cap_idx = 0
for sample in ds_labels:
    image_id = sample['image_id']
    ref_strs = sample['reference_strs']
    for ref in ref_strs:
        captions.append(
            {
                'image_id': image_id,
                'id': cap_idx,
                'caption': ref
            }
        )
        cap_idx +=1
print(captions[:10])

with open(cap_exp, 'w') as fp:
    json.dump(captions, fp)
exit()

rows = []
rows_label = []
rows_hw = []
for img_p in tqdm(img_list):
    img_key = img_p.split('.')[0]
    img_path = op.join(data_path, img_p)
    img = cv2.imread(img_path)
    img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])
    row = [img_key, img_encoded_str]
    rows.append(row)

    # Here is just a toy example of labels.
    # The real labels can be generated from the annotation files
    # given by each dataset. The label is a list of dictionary 
    # where each box with at least "rect" (xyxy mode) and "class"
    # fields. It can have any other fields given by the dataset.
    # labels = []
    # labels.append({"rect": [1, 1, 30, 40], "class": "Dog"})
    # labels.append({"rect": [2, 3, 100, 100], "class": "Cat"})
    # row_label = [img_key, json.dumps(labels)]
    # rows_label.append(row_label)

    height = img.shape[0]
    width = img.shape[1]
    row_hw = [img_key, json.dumps([{"height": height, "width": width}])]
    rows_hw.append(row_hw)

tsv_writer(rows, tsv_file)
# tsv_writer(rows_label, label_file)
tsv_writer(rows_hw, hw_file)
input('PRESS CTRL+C TO EXIT:')

# generate linelist file
generate_linelist_file(label_file, save_file=linelist_file)

# To access a tsv file:
# 1) Use tsv_reader to read dataset in given order
rows = tsv_reader("tools/mini_tsv/data/train.tsv")
rows_label = tsv_reader("tools/mini_tsv/data/train.label.tsv")
for row, row_label in zip(rows, rows_label):
    img_key = row[0]
    labels = json.loads(row_label[1])
    img = img_from_base64(row[1])

# 2) use TSVFile to access dataset at any given row.
tsv = TSVFile("tools/mini_tsv/data/train.tsv")
row = tsv.seek(1) # to access the second row 
img_key = row[0]
img = img_from_base64(row[1])



