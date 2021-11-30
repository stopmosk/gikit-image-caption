#!/bin/bash

export MAX_JOBS=6

conda create --name hic_pipe python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate hic_pipe

#conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
#conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch -y
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y

#Oscar
cd huawei-image-caption
pip install -r requirements.txt
#cd coco_caption && ./get_stanford_models.sh && cd ..
#cd huawei-image-caption
python setup.py build develop

#VinVL
#sudo apt update && sudo apt install ffmpeg libsm6 libxext6
pip install ninja yacs==0.1.8 cityscapesscripts opencv-python
conda install -c conda-forge pycocotools -y

cd sgg_bench
python setup.py build develop
cd ..

# EasyOCR
pip install easyocr

# Translate
conda install -c huggingface tokenizers=0.10.1 transformers=4.6.1 -y
conda install sentencepiece -y
