#!/bin/bash

python ./oscar/compute_cider.py \
--pred_file=../datasets/huawei_5/pred.json \
--caption_file=../datasets/huawei_5/ann.txt \
--huawei_fmt
