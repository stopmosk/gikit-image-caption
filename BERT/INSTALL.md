conda create -n hic python=3.8 -y
conda activate hic
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

cd /media/stopmosk/data/huawei/huawei-image-caption/
pip install -e .
pip install -r requirements.txt

pip install ninja yacs==0.1.8 cityscapesscripts opencv-python
conda install -c conda-forge pycocotools
cd sgg_bench/
MAX_JOBS=4 pip install -e .
