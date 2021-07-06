import argparse
import numpy as np
import os
import os.path as op
import json

from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap


def evaluate_on_coco_caption(res_file, label_file, orig_coco, outfile=None):
    assert label_file.endswith('.json')
    res_file_coco = res_file

    coco = COCO(label_file)
    print(res_file_coco)
    cocoRes = coco.loadRes(res_file_coco)
    cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    if not orig_coco:
        cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--caption_file', type=str, required=True)
    parser.add_argument('--orig_coco', default=False, action='store_true', help='True for Original COCO, False for karpathy split')

    args = parser.parse_args()
    
    if args.orig_coco:
        
        # Convert original annotations

        new_cap_filename = args.caption_file[:-5] + '_exp.json'
        with open(args.caption_file) as f:
            val_ann = json.load(f)

        idim = []
        for ann in val_ann['annotations']:
            idim.append({'id': ann['image_id'], 'file_name': ann['image_id']})

        with open(new_cap_filename, 'w') as f:
            json.dump({'annotations': val_ann['annotations'], 'images': idim, 'type': 'captions', 'info': 'dummy', 'licenses': 'dummy'}, f)

        # Convert predictions

        new_pred_filename = args.pred_file[:-5] + '_exp.json'
        with open(args.pred_file) as f:
            pred_ann = json.load(f)

        for pred in pred_ann:
            img_id = int(pred['image_id'][-12:-4])
            pred['image_id'] = img_id

        with open(new_pred_filename, 'w') as f:
            json.dump(pred_ann, f)
            
    else:
        new_pred_filename = args.pred_file
        new_cap_filename = args.caption_file
        
    result = evaluate_on_coco_caption(new_pred_filename, new_cap_filename, orig_coco=args.orig_coco)
    print('Done.')

    
if __name__ == '__main__':
    #  python ./oscar/compute_cider.py --pred_file=../pred.json --caption_file=../val_caption_coco_format.json
    main()
