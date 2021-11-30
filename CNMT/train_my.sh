#!/bin/bash

python -m torch.distributed.launch --nproc_per_node 1 \
tools/run.py \
--tasks captioning \
--datasets m4c_textcaps \
--model cnmt \
--config configs/cnmt_config.yml \
--save_dir save/cnmt \
training_parameters.distributed True

# You can also specify a different path to --save_dir to save to a location you prefer.