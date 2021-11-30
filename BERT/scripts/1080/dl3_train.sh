#!/bin/bash

# Train Oscar+ with CrossEntropyLoss
python oscar/run_captioning.py \
--do_train \
--max_seq_length=40 \
--data_dir=/home/dlserver2/tosha2tb/sergeishutov/coco_oscar_vvl_my \
--model_name_or_path=bert-base-uncased \
--do_lower_case \
--learning_rate=0.00005 \
--per_gpu_train_batch_size=96 \
--per_gpu_eval_batch_size=120 \
--gradient_accumulation_steps=2 \
--num_train_epochs=80 \
--num_workers=6 \
--freeze_embedding \
--tie_weights \
--logging_steps=200 \
--save_steps=2000 \
--output_dir=../../output/coco_my_vvl_orig_xe3/ \
--fp16
