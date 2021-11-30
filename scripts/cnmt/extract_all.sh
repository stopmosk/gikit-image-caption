#!/bin/bash

echo 'Build IMDB for image folder and run OCR engine'
python CNMT/build_imdb.py

echo 'Extract ROI features'
python CNMT/pythia/scripts/features/extract_features_vmb.py \
--model_file=detectron_model.pth \
--config_file=detectron_model.yaml \
--image_dir=./images \
--output_folder=CNMT/data/my_data/features \
--confidence_threshold=0.2 \
#--num_features=100

echo 'Add extracted ROI features to IMDB'
python CNMT/imdb_add_frcn_feats.py

echo 'Extract OCR ROI features'
python CNMT/pythia/scripts/features/extract_ocr_frcn_feature.py \
--detection_model=detectron_model.pth \
--detection_cfg=detectron_model.yaml \
--imdb_file=./CNMT/data/my_data/imdb/imdb_test_my.npy \
--image_dir=./images \
--save_dir=./CNMT/data/my_data/ocr_feats

echo 'Finished!'
