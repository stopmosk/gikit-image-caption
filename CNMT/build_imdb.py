import os
import os.path as op
import numpy as np
from datetime import datetime

from PIL import Image
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='0'
print('CUDA_VISIBLE_DEVICES set to "0"')

import easyocr


def area(bbox):
    assert len(bbox) == 6
    return bbox[4] * bbox[5]


def int_area(a, b):  # returns None if rectangles don't intersect
    #   0    1    2    3   4  5
    # xmin ymin xmax ymax  w  h
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0


def normalize_bbox(bbox, im_size):
    # Normalize bbox coords from (absolute --> relative 0..1)
    assert len(bbox) == 4 and len(bbox[0]) == 2
    w, h = im_size
    xmin = bbox[0][0] / w
    xmax = bbox[2][0] / w
    ymin = bbox[0][1] / h
    ymax = bbox[2][1] / h
    return (xmin, ymin, xmax, ymax)


def get_rel_bbox(x_s, y_s, w, h):
    return [
        min(x_s) / w,
        min(y_s) / h,
        max(x_s) / w,
        max(y_s) / h,
        (max(x_s) - min(x_s)) / w,
        (max(y_s) - min(y_s)) / h,
    ]

def get_ocr_features(im_filepath, reader):
        im = Image.open(im_filepath)
        if len(im.split()) > 3:
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
            im = background

        w, h = im.width, im.height
        # im.load() # required for png.split()

        threshold = 0.2
        max_wh = 750
        max_words_to_recognize = 10

        max_width = max_wh if im.width >= im.height else max_wh * im.width / im.height

        # *** OCR DETECT
        
        ocr_horiz_dets, ocr_freeform_dets = reader.detect(
            np.asarray(im),
            canvas_size=max_width,  # For detection
            # text_threshold=0.75,
            add_margin=0.05,
            # link_threshold=0.3
            width_ths=0.1,
            height_ths=0.5,
            ycenter_ths=0.5,
            min_size=15,
        )

        boxes_cnt = len(ocr_horiz_dets[0]) + len(ocr_freeform_dets[0])
        if boxes_cnt > max_words_to_recognize or boxes_cnt == 0:
            # print('')
            return
            
        # *** OCR RECOGNIZE
        
        ocr_text = reader.recognize(
            np.asarray(im),
            horizontal_list=ocr_horiz_dets[0], 
            free_list=ocr_freeform_dets[0],
            # decoder='beamsearch',
            decoder='greedy',
            # batch_size=16,
        )
        im.close()
        
        if not ocr_text:
            # print('No OCR text')
            return

        # *** OCR POSTPROCESS
        
        # ocr_tokens = list(zip(*ocr_text))[1]      
        # print(ocr_tokens)
        
        # Filter by threshold
        ocr_text = [item for item in ocr_text if item[2] > threshold]
        if not ocr_text:
            return
            
        ocr_text_rel = []
        for (bbox, text, conf) in ocr_text:
            x_s, y_s = tuple(zip(*bbox))
            uni_bbox_rel = get_rel_bbox(x_s, y_s, w, h)
            
            # Remove small text
            if area(uni_bbox_rel) / len(text) < 0.0001:
                # print('DEL: ', text)
                # conf *= 0.1
                continue
            
            # Remove text near edges of image
            main_area = (0.05, 0.1, 0.95, 0.9)       
            if int_area(main_area, uni_bbox_rel) < (area(uni_bbox_rel) * 0.5):
                # print('DEL NEAR EDGE:', text)
                continue
            
            # Remove strange symbols
            text = text.strip('*-"\\,;~[](){}`^|_ :&@')
            if text == '':
                # print('EMPTY TOKEN')
                continue
            
            ocr_text_rel.append((uni_bbox_rel[:4], text, 0.0))
        
        if not ocr_text_rel:
            return
            
        ocr_boxes, ocr_tokens, ocr_confidence = list(zip(*ocr_text_rel))        
        ocr_normalized_boxes = np.asarray(ocr_boxes, dtype=np.float32)
        
        print(ocr_tokens)
        # ocr_normalized_boxes = np.asarray([normalize_bbox(t, (w, h)) for t in ocr_boxes], dtype=np.float32)
        return (w, h), (ocr_tokens, ocr_normalized_boxes, ocr_confidence)


def main():
    ocr_reader = easyocr.Reader(['en'])

    img_dir = op.join(root_dir, 'images')
    im_list = sorted(os.listdir(img_dir))

    images_info = [{'date created': datetime.today().strftime('%Y-%m-%d')}]

    for im_filename in im_list: #tqdm(im_list):
        im_filepath = op.join(img_dir, im_filename)

        ocr_results = get_ocr_features(im_filepath, ocr_reader)
        if ocr_results is None:
            # NO OCR FOUND
            continue
            
        w, h = ocr_results[0] # im_size
        ocr_tokens, ocr_normalized_boxes, ocr_confidence = ocr_results[1]
        
        img_id = im_filename.split('.')[0]

        img_info = {
            'image_id': img_id,
            'image_name': im_filename,
            'image_width': w,
            'image_height': h,
            'feature_path': img_id + '.npy',
            'image_path': im_filename,
            'ocr_tokens': ocr_tokens,
            'ocr_normalized_boxes': ocr_normalized_boxes,
            'ocr_confidence': ocr_confidence,
            'obj_normalized_boxes': None,  #obj_normalized_boxes,
            'caption_id': 300000000,
            'set_name': 'test',
        }
        images_info.append(img_info)

    save_dir = op.join(root_dir, data_path, 'imdb')
    os.makedirs(save_dir, exist_ok=True)
    np.save(op.join(save_dir, 'imdb_test_my.npy'), images_info)


if __name__ == '__main__':
    root_dir = './'
    data_path = 'CNMT/data/my_data'
    main()
