#!/bin/bash

# Train Oscar+ with CrossEntropyLoss
python oscar/run_captioning.py \
--do_train \
--evaluate_during_training \
--data_dir=../datasets/textcaps \
--model_name_or_path=../output/scst/checkpoint-31-896000 \
--do_lower_case \
--add_od_labels \
--add_ocr_labels \
--learning_rate=0.00001 \
--per_gpu_train_batch_size=60 \
--per_gpu_eval_batch_size=100 \
--gradient_accumulation_steps=1 \
--num_train_epochs=80 \
--num_workers=6 \
--warmup_steps=2000 \
--tie_weights \
--freeze_embedding \
--label_smoothing=0.1 \
--drop_worst_ratio=0.2 \
--drop_worst_after=20000 \
--logging_steps=100 \
--save_steps=2000 \
--output_dir=../output/txtcps_base_xe_rosetta02/ \
--fp16
