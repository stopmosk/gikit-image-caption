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
  --image_dir IMAGE_DIR   # The images folder.
  --save_dir SAVE_DIR   # The output directory to save results.
  --with_ocr    # Enable OCR
  --translate    #Enable Translation
  --ocr_thresh OCR_THRESH   # OCR confidence threshold
  --bbox_thresh BBOX_THRESH   # If OCR founds too many bboxes, we skip OCR recognition for speed
  --lang LANG   # ru, fr, es, de
```

Output ```.tsv``` and ```.json``` with the results will be in model folder.
