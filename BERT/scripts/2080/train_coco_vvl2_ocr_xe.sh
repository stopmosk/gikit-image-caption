#!/bin/bash

# Train Oscar+ with CrossEntropyLoss
python oscar/run_captioning.py \
--do_train \
--evaluate_during_training \
--data_dir=/mnt/Toshiba2TB/coco_my_oscar_vvl_tags_nms2 \
--model_name_or_path=../models/checkpoint-13-256000 \
--do_lower_case \
--add_od_labels \
--add_ocr_labels \
--max_ocr_seq_length=40 \
--learning_rate=0.00005 \
--per_gpu_train_batch_size=72 \
--per_gpu_eval_batch_size=128 \
--gradient_accumulation_steps=4 \
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
--output_dir=../output/coco_my_vvl_xe4/ \
--fp16

#--add_ocr_labels \
#--model_name_or_path=../output/txtcps_xe_clro_posenc_ros_pos_xywh/checkpoint-79-182960 \
#--model_name_or_path=../pretrained_models/image_captioning/coco_captioning_base_scst/checkpoint-15-66405 \