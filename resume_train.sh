#!/bin/bash

# Train Oscar+ with CrossEntropyLoss
python oscar/run_captioning.py \
--do_train \
--evaluate_during_training \
--data_dir=../datasets/coco_caption \
--model_name_or_path=../output/coco_base_ce/checkpoint-51-4060000 \
--do_lower_case \
--add_od_labels \
--learning_rate=0.000007 \
--per_gpu_train_batch_size=72 \
--per_gpu_eval_batch_size=128 \
--gradient_accumulation_steps=1 \
--num_train_epochs=60 \
--num_workers=6 \
--warmup_steps=0 \
--tie_weights \
--freeze_embedding \
--label_smoothing=0.1 \
--drop_worst_ratio=0.2 \
--drop_worst_after=20000 \
--save_steps=2000 \
--logging_steps=50 \
--output_dir=../output/coco_base_ce_resume/ \
--fp16
