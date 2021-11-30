### OLD README

# Clone repo
```bash
git clone https://github.com/servacc/HuaweiImageCaptionCode.git
cd HuaweiImageCaptionCode
```

# Run install scripts

```bash
./pre_install.sh
```

```bash
./install.sh
```

```bash
conda activate hic
```

# Inference

Put your images in image_dir and run:

```bash
python live.py --image_dir=image_dir
```

```bash
Keys:
--with_ocr
--translate

and other
```

Output ```.tsv``` and ```.json``` with the results will be in model folder.
