#!/bin/bash

# Oscar+ evaluate pretrained model

python oscar/run_captioning.py \
--do_eval \
--add_ocr_labels \
--data_dir=../datasets/coco_caption \
--per_gpu_eval_batch_size=1 \
--num_workers=4 \
--num_beams=1 \
--max_gen_length=20 \
--eval_model_dir=../pretrained_models/image_captioning/coco_captioning_base_xe/checkpoint-60-66360
