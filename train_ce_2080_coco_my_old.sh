#!/bin/bash

# Train Oscar+ with CrossEntropyLoss
python oscar/run_captioning.py \
--do_train \
--evaluate_during_training \
--data_dir=../datasets/coco_caption_vvl_feats \
--model_name_or_path=../pretrained_models/image_captioning/coco_captioning_base_xe/checkpoint-60-66360 \
--do_lower_case \
--add_od_labels \
--max_ocr_seq_length=10 \
--learning_rate=0.00005 \
--per_gpu_train_batch_size=72 \
--per_gpu_eval_batch_size=128 \
--gradient_accumulation_steps=1 \
--num_train_epochs=80 \
--num_workers=6 \
--warmup_steps=2000 \
--tie_weights \
--freeze_embedding \
--label_smoothing=0.1 \
--drop_worst_ratio=0.2 \
--drop_worst_after=20000 \
--logging_steps=200 \
--save_steps=2000 \
--output_dir=../output/coco_my_vvl_xe/ \
--fp16
