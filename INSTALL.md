conda create -n pt181cu111 python=3.8
conda activate pt181cu111
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

cd /media/stopmosk/data/huawei/huawei-image-caption/
python setup.py build develop
pip install -r requirements.txt

pip install ninja yacs==0.1.8 cityscapesscripts opencv-python
conda install -c conda-forge pycocotools
cd sgg_bench/
MAX_JOBS=4 python setup.py build develop

