#!/bin/bash

# Train Oscar+ with CrossEntropyLoss
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_captioning.py \
--do_train \
--evaluate_during_training \
--data_dir=/mnt/d/coco_preexacted \
--model_name_or_path=../pretrained_models/image_captioning/pretrained_large/checkpoint-1410000 \
--do_lower_case \
--add_od_labels \
--learning_rate=1e-5 \
--per_gpu_train_batch_size=128 \
--per_gpu_eval_batch_size=1024 \
--gradient_accumulation_steps=1 \
--num_train_epochs=60 \
--num_workers=4 \
--tie_weights \
--freeze_embedding \
--label_smoothing=0.1 \
--drop_worst_ratio=0.2 \
--drop_worst_after=20000 \
--logging_steps=100 \
--save_steps=1000 \
--output_dir=../output/coco_large_xe \
--fp16

#--warmup_steps=2000 \
#--fp16
#--evaluate_during_training \