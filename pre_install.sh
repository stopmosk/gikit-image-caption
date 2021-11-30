#!/bin/bash

CUR_DIR=$PWD

sudo apt update && sudo apt install unzip ffmpeg libsm6 libxext6 &&

pip install gdown &&

cd .. &&
if [ ! -f ./models.tar.gz ]; then
    gdown https://drive.google.com/uc?id=19rI7ARKZAuhBUDIR-kFq3xTwcNa5yJn_
fi &&

mkdir -p models &&
unzip models.zip -d ./models && cd $CUR_DIR &&
#tar -xzvf models.tar.gz -C ./models && cd $CUR_DIR && 

cp -a ../models/cnmt_data/. CNMT/data
