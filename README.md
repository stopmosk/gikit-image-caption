### OLD README

# Clone repo
```bash
git clone https://github.com/servacc/HuaweiImageCaptionCode.git
cd HuaweiImageCaptionCode
```

# Run install scripts from repo folder

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

Put your images in image_dir folder and run:

```bash
python live.py --image_dir=image_dir --save_dir=save_dir
```

Run with OCR support:
```bash
python live.py --image_dir=image_dir --save_dir=save_dir --with_ocr
```

Run with translation to different languages:
```bash
python live.py --image_dir=image_dir --save_dir=save_dir --translate --lang=LANG
```
where LANG can be 'ru', 'fr', 'es' or 'de'

You can combine keys:
```bash
python live.py --image_dir=image_dir --save_dir=save_dir --with_ocr --translate --lang='es'
```


All script keys:

```bash
Keys:
  --image_dir IMAGE_DIR   # The images input folder.
  --save_dir SAVE_DIR   # The output directory to save results.
  --with_ocr    # Enable OCR support
  --translate    #Enable Translation
  --ocr_thresh OCR_THRESH   # OCR confidence threshold, default 0.2
  --bbox_thresh BBOX_THRESH   # If OCR founds too many bboxes, we skip OCR recognition for speed
  --lang LANG   # ru, fr, es, de. Default is 'ru'
```

Output ```.txt``` and ```.json``` with the results will be in save_dir folder.
