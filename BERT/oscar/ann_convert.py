import argparse
import numpy as np
import os
import os.path as op
import json
import pandas as pd


def main():
            
    if True:

        # Conver txt to json
    
        cap_source = '../../datasets/huawei_5/anns'
        ann_all = []
        for i, ann_filename in enumerate(sorted(os.listdir(cap_source))):
            df = pd.read_csv(op.join(cap_source, ann_filename), sep='|', header=None)
            df.columns = ['image_id', 'caption']
            ann_json = json.loads(df.to_json(orient='records'))
            for j, ann in enumerate(ann_json):
                ann['id'] = i * len(ann_json) + j
                ann_all.append(ann)
        print(ann_all)

        
        # Convert original annotations
        # Convert predictions

        new_pred_filename = op.join(cap_source, 'ann.json')
        #with open(args.pred_file) as f:
        #    pred_ann = json.load(f)

        #for pred in pred_ann:
        #    img_id = int(pred['image_id'].split('.')[0])
        #    pred['image_id'] = img_id

        with open(new_pred_filename, 'w') as f:
            json.dump(ann_all, f)

    
if __name__ == '__main__':
    #  python ./oscar/compute_cider.py --pred_file=../pred.json --caption_file=../val_caption_coco_format.json
    main()
