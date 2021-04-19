#!/bin/bash

# Oscar+ evaluation on custom checkpoint

python oscar/run_captioning.py \
--do_eval \
--data_dir=../datasets/coco_caption \
--per_gpu_eval_batch_size=32 \
--num_workers=4 \
--num_beams=1 \
--max_gen_length=20 \
--output_dir=../output/scst_results \
--eval_model_dir=../output/scst/checkpoint-2-74000
