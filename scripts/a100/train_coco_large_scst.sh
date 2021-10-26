#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8 \
oscar/run_captioning.py \
--data_dir=/mnt/d/coco_preexacted \
--model_name_or_path=../output/coco_large_xe_1.274/checkpoint-59-33000 \
--do_train \
--evaluate_during_training \
--do_lower_case \
--add_od_labels \
--learning_rate=8e-7 \
--per_gpu_train_batch_size=20 \
--per_gpu_eval_batch_size=512 \
--num_train_epochs=25 \
--num_workers=10 \
--tie_weights \
--freeze_embedding \
--scst \
--output_dir=../output/coco_large_xe_1.274_scst/ \
--save_steps=350 \
--fp16
