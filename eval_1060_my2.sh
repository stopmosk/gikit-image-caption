#!/bin/bash

# Oscar+ evaluation on custom checkpoint

python oscar/run_captioning.py \
--do_eval \
--data_dir=../datasets/coco_cap_4000 \
--per_gpu_eval_batch_size=1 \
--num_workers=4 \
--num_beams=1 \
--max_gen_length=20 \
--output_dir=../output/coco_4000_results \
--eval_model_dir=../output/coco_4000_base_xe/checkpoint-79-40000 \
--test_yaml='train.yaml'
