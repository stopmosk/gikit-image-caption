#!/bin/bash

python ./oscar/compute_cider.py \
--orig_coco \
--pred_file=results/coco2014/pred.json \
--caption_file=results/coco2014/captions_val2014.json


