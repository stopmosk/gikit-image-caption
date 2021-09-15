#!/bin/bash

# Train Oscar+ with CrossEntropyLoss
python oscar/run_captioning.py \
--do_train \
--evaluate_during_training \
--data_dir=/mnt/Toshiba2TB/cc_vvl_nms1 \
--model_name_or_path=../pretrained_models/image_captioning/pretrained_large/checkpoint-1410000 \
--do_lower_case \
--add_od_labels \
--learning_rate=0.00001 \
--per_gpu_train_batch_size=14 \
--per_gpu_eval_batch_size=96 \
--gradient_accumulation_steps=1 \
--num_train_epochs=1 \
--num_workers=6 \
--warmup_steps=5000 \
--tie_weights \
--freeze_embedding \
--label_smoothing=0.1 \
--drop_worst_ratio=0.2 \
--drop_worst_after=20000 \
--logging_steps=500 \
--save_steps=10000 \
--output_dir=../output/cc_large_xe/ \
--fp16


#--evaluate_during_training \