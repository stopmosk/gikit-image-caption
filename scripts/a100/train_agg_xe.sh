#!/bin/bash

# Train Oscar+ with CrossEntropyLoss
python -m torch.distributed.launch --nproc_per_node=8 \
oscar/run_captioning.py \
--do_train \
--data_dir=/mnt/d/ \
--train_yaml=agg.yaml \
--agg_dataset \
--model_name_or_path=../pretrained_models/image_captioning/pretrained_base/checkpoint-2000000 \
--do_lower_case \
--add_od_labels \
--learning_rate=5e-5 \
--per_gpu_train_batch_size=512 \
--per_gpu_eval_batch_size=512 \
--gradient_accumulation_steps=1 \
--num_train_epochs=25 \
--num_workers=8 \
--warmup_steps=1000 \
--tie_weights \
--freeze_embedding \
--label_smoothing=0.1 \
--drop_worst_ratio=0.2 \
--drop_worst_after=20000 \
--logging_steps=50 \
--save_steps=500 \
--output_dir=../output/agg_1cc_10coco_base_xe/ \
--fp16 \
--evaluate_during_training \
--val_yaml='coco_vvl_nms1/val.yaml'