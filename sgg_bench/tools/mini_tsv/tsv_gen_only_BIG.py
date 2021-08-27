#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import os.path as op
import json
import cv2
import base64
import random
import numpy as np
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from PIL import Image, ImageDraw

from maskrcnn_benchmark.structures.tsv_file import TSVFile


# In[ ]:


orig_root = '/mnt/Toshiba2TB/dataset_ImageCaption'
exp_root = '../../../../datasets_proc/big_nn'


# In[ ]:


img_paths = dict()

folder_list = [f for f in os.listdir(orig_root) if '.' not in f]

for folder in folder_list[:5]:
    print(folder, flush=True)
    img_list = [f for f in os.listdir(op.join(orig_root, folder))]
    for img_filename in tqdm(img_list):
        img_paths[img_filename] = op.join(orig_root, folder, img_filename)


# In[ ]:


# sample
img_paths[list(img_paths.keys())[0]]


# In[ ]:


with open(op.join(orig_root, 'Final_EN.json')) as f:
    dataset_captions = json.load(f)

dataset_captions = dataset_captions['annotations']
len(dataset_captions)


# In[ ]:


img_keys = set(c['file_name'] for c in dataset_captions)
len(img_keys)


# In[ ]:


captions_by_num = dict()

for caption in tqdm(dataset_captions):
    cap = caption['file_name']
    if cap not in captions_by_num:
        captions_by_num[cap] = 1
    else:
        captions_by_num[cap] += 1


# In[ ]:


# calculate counts of captions per image over dataset

counts = dict()

for count in captions_by_num.values():
    if count not in counts:
        counts[count] = 1
    else:
        counts[count] += 1

print(sorted(counts.items(), key=lambda x:x[0]))

# remove images with more than 16 captions

img_keys_filtered = [name for (name, count) in captions_by_num.items() if count <= 16]
len(img_keys_filtered)


# In[ ]:


# filter keys

json_keys = set(img_keys_filtered)
file_keys = set(img_paths.keys())

img_keys_filtered = json_keys.intersection(file_keys)
img_keys_filtered = list(img_keys_filtered)


# In[ ]:


# Split dataset to train/val/test

random.shuffle(img_keys_filtered)
split_cnt = len(img_keys_filtered) // 100
print(split_cnt)

splits = {
    'train': img_keys_filtered[split_cnt * 2:],
    'val':img_keys_filtered[split_cnt: split_cnt * 2],
    'test': img_keys_filtered[: split_cnt],
}


# In[ ]:


# Shrink image if big

def scale_img(img):
    height, width = img.shape[:2]
    max_height = 1000
    max_width = 1000

    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img


# In[ ]:


def tsv_writer(values, tsv_file):
    #mkdir(op.dirname(tsv_file))
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    idx = 0
    tsv_file_tmp = tsv_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'
    with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value is not None
            value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
            v = '{0}\n'.format('\t'.join(map(str, value)))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)
    os.rename(tsv_file_tmp, tsv_file)
    os.rename(lineidx_file_tmp, lineidx_file)


# In[ ]:


splits.keys()


# In[ ]:


for split in splits:
    exp_encoded_img_file = op.join(exp_root, f'{split}.img.tsv')
    exp_hw_file = op.join(exp_root, f'{split}.hw.tsv')
    
    lineidx_file = op.splitext(exp_encoded_img_file)[0] + '.lineidx'
    idx = 0
    
    tsv_file_tmp = exp_encoded_img_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'

    rows_hw = []
    
    with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
        for img_p in tqdm(splits[split]):
            img_key = img_p.split('.')[0]
            img_path = img_paths[img_p]

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = scale_img(img)
            height = img.shape[0]
            width = img.shape[1]

            img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])
            row = [img_key, img_encoded_str]
            value = [v if type(v)!=bytes else v.decode('utf-8') for v in row]
            v = '\t'.join(map(str, value)) + '\n'
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)
            
            row_hw = [img_key, json.dumps([{'height': height, 'width': width}])]
            rows_hw.append(row_hw)
    
    os.rename(tsv_file_tmp, exp_encoded_img_file)
    os.rename(lineidx_file_tmp, lineidx_file)

    tsv_writer(rows_hw, exp_hw_file)

print('Done.')


# In[ ]:


# for split in splits:
#     exp_hw_file = op.join(exp_root, f'{split}.hw.tsv')
    
#     rows_hw = []
#     for img_p in tqdm(splits[split]):
#         img_key = img_p.split('.')[0]
#         img_path = img_paths[img_p]

#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#         img = scale_img(img)
#         height = img.shape[0]
#         width = img.shape[1]
#         row_hw = [img_key, json.dumps([{'height': height, 'width': width}])]
#         rows_hw.append(row_hw)
    
#     tsv_writer(rows_hw, exp_hw_file)

# print('Done.')


# In[ ]:


k = splits['val'][0]
dataset_captions[0]
#k


# In[ ]:


# Run cell only if we need to re-read keys from generated TSVs

# img_splits = dict()

# for split in splits:
#     tsv = TSVFile(f'/mnt/Toshiba2TB/big_vinvl_oscar/{split}.label.tsv')
#     keys = [tsv.seek(i)[0] for i in tqdm(range(tsv.num_rows()))]
#     img_splits[split] = keys

# len(img_splits['train'])


# In[ ]:


# Make dict with img_key : {img_key, id, caption}

ds_caps_by_key = {
    s['file_name'][:-4]: {
        'image_id': s['file_name'][:-4],
        'id': i + 1,
        'caption': s['caption'],
    } for (i, s) in enumerate(dataset_captions)
}

ds_caps_by_key['a1650e00b6261e99a6bbe6fe13919302']


# In[ ]:


# Filter captions for splits

for split in tqdm(splits):
    
    # Get current split samples from dataset
    
    if False: #img_splits is not None:  # ???????
        out_captions = [ds_caps_by_key[img_key] for img_key in img_splits[split]]
    else:
        out_captions = [ds_caps_by_key[img_key[:-4]] for img_key in splits[split]]              
    
    # Generate captions in COCO format

    idim = []
    for cap in out_captions:
        idim.append({'id': cap['image_id'], 'file_name': cap['image_id']})

    out_captions_coco_fmt = {'annotations': out_captions, 'images': idim, 'type': 'captions', 'info': 'dummy', 'licenses': 'dummy'}

    # Save JSON

    with open(os.path.join(exp_root, f'{split}_caption.json'), 'w') as fp:
        json.dump(out_captions, fp)

    with open(os.path.join(exp_root, f'{split}_caption_coco_format.json'), 'w') as f:
        json.dump(out_captions_coco_fmt, f)

out_captions[:3]


# In[ ]:




