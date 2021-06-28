#!/bin/bash

python oscar/run_cap_eval_only.py \
--do_test \
--data_dir=../datasets/my \
--per_gpu_eval_batch_size=1 \
--num_workers=0 \
--num_beams=5 \
--max_gen_length=20 \
--output_dir=../output/my_results \
--eval_model_dir=../models/checkpoint-22-628000 \
--test_yaml='val.yaml'
