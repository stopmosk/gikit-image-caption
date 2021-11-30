import os
import os.path as op
import numpy as np

from PIL import Image
from tqdm import tqdm


def main():
    imdb_file = op.join(root_dir, data_path, 'imdb/imdb_test_my.npy')
    img_dir = op.join(root_dir, 'images')
    im_list = sorted(os.listdir(img_dir))

    imdb = np.load(imdb_file, allow_pickle=True)  #.item()

    for imdb_item in tqdm(imdb[1:]):
        im_id = imdb_item['image_id']

        feat_npy_name = im_id + '_info.npy'
        feat_npy_filepath = op.join(root_dir, data_path, 'features', feat_npy_name)

        feat_item = np.load(feat_npy_filepath, allow_pickle=True).item()
        w, h = imdb_item['image_width'], imdb_item['image_height']
        assert feat_item['image_width'] == w and feat_item['image_height'] == h

        obj_boxes = feat_item['bbox']  # N x [xmix, ymin, xmax, ymax]
        obj_boxes[:, ::2] /= w
        obj_boxes[:, 1::2] /= h
        obj_normalized_boxes = obj_boxes

        imdb_item['obj_normalized_boxes'] = obj_normalized_boxes
        imdb_item['image_classes'] = feat_item['image_classes']
        imdb_item['num_boxes'] = feat_item['num_boxes']

    np.save(imdb_file, imdb)


if __name__ == '__main__':
    root_dir = './'
    data_path = 'CNMT/data/my_data'
    main()
