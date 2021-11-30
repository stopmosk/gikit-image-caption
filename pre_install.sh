#!/bin/bash

CUR_DIR=$PWD

sudo apt update && sudo apt install unzip aria2 ffmpeg libsm6 libxext6 &&

pip install gdown &&

cd .. &&
if [ ! -f ./models.tar.gz ]; then
    gdown https://drive.google.com/uc?id=1_PqHZkeGc1zEsgqU63uIcIJ8o2Q5qoEu
fi &&

mkdir -p models &&
#tar -xzvf models.tar.gz -C ./models && cd $CUR_DIR && 
unzip models.zip -d ./models &&

aria2c -x 16 -s 16 -d ./models https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip &&
unzip wiki.en.zip -d ./models && cd $CUR_DIR &&

cp -a ../models/cnmt_data/. CNMT/data
