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
    
    with open(args.caption_file) as f:
        val_ann = json.load(f)
        
    new_cap_filename = args.caption_file[:-5] + '_exp.json'
    
    with open(new_cap_filename), 'w') as f:
        json.dump(val_ann['annotations'], f)

    result = evaluate_on_coco_caption(args.pred_file, new_cap_filename)
    print('Done.')
    
    
if __name__ == '__main__':
    #  python ./oscar/compute_cider.py --pred_file=../pred.json --caption_file=../val_caption_coco_format.json
    main()
