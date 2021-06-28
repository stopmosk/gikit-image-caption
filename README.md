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
cd coco_caption && ./get_stanford_models.sh && cd ..
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

[Detector model](https://drive.google.com/file/d/11YdV_4yLx3W0oKDFgk0yzAd-bP2fY-EZ/view?usp=sharing)
Unzip and put in ```sgg_bench/models/vinvl/vg/```.

[Main model checkpoint]()
Unzip and put in ```../models/```

# Inference
```bash
./inference_oscar_vvl.sh
```
