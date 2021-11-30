#!/bin/bash

python CNMT/build_imdb.py

python CNMT/pythia/scripts/features/extract_ocr_frcn_feature.py \
--detection_model=detectron_model.pth \
--detection_cfg=detectron_model.yaml \
--imdb_file=./CNMT/data/my_data/imdb/imdb_test_my.npy \
--image_dir=./images \
--save_dir=./CNMT/data/my_data/ocr_feats
