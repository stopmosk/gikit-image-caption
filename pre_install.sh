#!/bin/bash

CUR_DIR=$PWD

sudo apt update && sudo apt install unzip aria2 ffmpeg libsm6 libxext6 &&

pip install gdown && mkdir -p models &&

cd ../models &&
if [ ! -f ./models.zip ]; then
    gdown https://drive.google.com/uc?id=1_PqHZkeGc1zEsgqU63uIcIJ8o2Q5qoEu
fi &&

unzip models.zip &&

if [ ! -f ./wiki.en.zip ]; then
    aria2c -x 16 -s 16 -d ./models https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
fi &&

unzip wiki.en.zip && cd $CUR_DIR &&

cp -a ../models/cnmt_data/. CNMT/data

echo 'Done.'
