#!/bin/bash

# Oscar+ evaluate pretrained model

python oscar/run_captioning.py \
--do_eval \
--do_test \
--data_dir=../datasets/coco_oscar_preexacted_vinvl \
--per_gpu_eval_batch_size=16 \
--num_workers=4 \
--num_beams=5 \
--max_gen_length=20 \
--eval_model_dir=../output_dl2/coco_ru_base_xe_3/checkpoint-28-250000 \
--fp16
