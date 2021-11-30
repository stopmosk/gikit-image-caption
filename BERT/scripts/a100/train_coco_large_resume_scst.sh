#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8 \
oscar/run_captioning.py \
--data_dir=/mnt/d/coco_preexacted \
--model_name_or_path=../output/coco_large_xe_1.274_scst_1/checkpoint-24-17725 \
--do_train \
--evaluate_during_training \
--do_lower_case \
--add_od_labels \
--learning_rate=1e-7 \
--per_gpu_train_batch_size=20 \
--per_gpu_eval_batch_size=512 \
--num_train_epochs=10 \
--num_workers=10 \
--tie_weights \
--freeze_embedding \
--scst \
--output_dir=../output/coco_large_xe_1.274_scst_1_17725/ \
--save_steps=250 \
--fp16
