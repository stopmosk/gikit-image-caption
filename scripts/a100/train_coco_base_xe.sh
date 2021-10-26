#!/bin/bash

# Train Oscar+ with CrossEntropyLoss
#python -m torch.distributed.launch --nproc_per_node=8 
python oscar/run_captioning.py \
--do_train \
--evaluate_during_training \
--data_dir=/mnt/d/coco \
--model_name_or_path=../pretrained_models/image_captioning/pretrained_base/checkpoint-2000000 \
--do_lower_case \
--add_od_labels \
--learning_rate=3e-5 \
--per_gpu_train_batch_size=64 \
--per_gpu_eval_batch_size=256 \
--gradient_accumulation_steps=1 \
--num_train_epochs=60 \
--num_workers=16 \
--tie_weights \
--freeze_embedding \
--label_smoothing=0.1 \
--drop_worst_ratio=0.2 \
--drop_worst_after=20000 \
--logging_steps=100 \
--save_steps=2000 \
--output_dir=../output/coco_base_xe \
--warmup_steps=2000

#--fp16
#--evaluate_during_training \