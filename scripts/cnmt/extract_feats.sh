#!/bin/bash

python CNMT/pythia/scripts/features/extract_features_vmb.py \
--model_file=detectron_model.pth \
--config_file=detectron_model.yaml \
--image_dir=./images \
--output_folder=CNMT/data/my_data/features \
--confidence_threshold=0.2 \

#--num_features=100
