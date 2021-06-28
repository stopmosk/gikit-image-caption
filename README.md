# clone repo
git clone https://github.com/servacc/HuaweiImageCaptionCode.git
cd HuaweiImageCaptionCode

# create a new environment
conda create --name image_caption python=3.7
conda activate image_caption

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# oscar
pip install -r requirements.txt
cd Oscar/coco_caption && ./get_stanford_models.sh && cd ..
python setup.py build develop

# vinvl
pip install ninja yacs==0.1.8 cityscapesscripts opencv-python
conda install -c conda-forge pycocotools

cd sgg_bench
python setup.py build develop
cd ..

# Download datasets (COCO features or your own images)

# Download models

# inference
./inference_oscar_vvl.sh