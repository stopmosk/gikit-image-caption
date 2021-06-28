# Clone repo
```bash
git clone https://github.com/servacc/HuaweiImageCaptionCode.git
cd HuaweiImageCaptionCode
```

# Create a new environment
```bash
conda create --name image_caption python=3.7
conda activate image_caption

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

# Oscar
```bash
pip install -r requirements.txt
cd Oscar/coco_caption && ./get_stanford_models.sh && cd ..
python setup.py build develop
```

# VinVL
```bash
pip install ninja yacs==0.1.8 cityscapesscripts opencv-python
conda install -c conda-forge pycocotools

cd sgg_bench
python setup.py build develop
cd ..
```

# Download datasets (COCO features or your own images)

# Download models
```
link
```

# Inference
```bash
./inference_oscar_vvl.sh
```
