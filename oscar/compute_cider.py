import argparse
import numpy as np
import os
import os.path as op
import json

from oscar.utils.caption_evaluate import evaluate_on_coco_caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--caption_file', type=str, required=True)

    args = parser.parse_args()
    
#     with open(args.caption_file) as f:
#         val_ann = json.load(f)
        
#     new_cap_filename = args.caption_file[:-5] + '_exp.json'
    
#     idim = []
#     for ann in val_ann['annotations']:
#         idim.append({'id': str(ann['image_id']), 'file_name': str(ann['image_id'])})
    
#     with open(new_cap_filename, 'w') as f:
#         json.dump({'annotations': val_ann['annotations'],
#                   'images': idim,
#                   'type': 'captions', 'info': 'dummy', 'licenses': 'dummy'}, f)
#         #json.dump(val_ann, f)

        
#     with open(args.pred_file) as f:
#         pred_ann = json.load(f)
        
#     for pred in pred_ann:
#         img_id = str(int(pred['image_id'][-12:-4]))
#         pred['image_id'] = img_id
        
#     new_pred_filename = args.pred_file[:-5] + '_exp.json'

#     with open(new_pred_filename, 'w') as f:
#         json.dump(pred_ann, f)

    new_pred_filename = args.pred_file
    new_cap_filename = args.caption_file
    result = evaluate_on_coco_caption(new_pred_filename, new_cap_filename)
    print('Done.')
    
    
if __name__ == '__main__':
    #  python ./oscar/compute_cider.py --pred_file=../pred.json --caption_file=../val_caption_coco_format.json
    main()
