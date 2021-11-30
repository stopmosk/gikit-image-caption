#!/bin/bash

pip install gdown

gdown https://drive.google.com/uc?id=12345678

mkdir ../models
tar -xzfv models.tar.gz -C ../models
cp -a ../models/cnmt_data/. CNMT/data

sudo apt update && sudo apt install ffmpeg libsm6 libxext6

conda create --name hic python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate hic
