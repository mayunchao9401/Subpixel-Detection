import random
from random import randint
from opencv_marker.subpixel import cv_subpixel
from utils.widget import *
from opencv_marker.harris import cv_harris
import os
import cv2
from common import *


def create_label_file(root_dir, verbose=False, dst_h=h_for_train, dst_w=w_for_train):
    """
    create a corresponding label file using opencv subpixel
    for all the image files in root_dir

    :param root_dir:
    :return:
    """
    exts_allowed = ('JPEG', 'jpg')
    root_pre = None
    MAX_IMG_SHOWN = 10
    count = 0
    for root, dirs, files in os.walk(root_dir):
        if root != root_pre:
            root_pre = root
            print('marking files in directory:', root)
        for img_file_name in files:
            is_img = False
            img_file_ext = img_file_name.split('.')[-1]
            for ext in exts_allowed:
                is_img = is_img or img_file_ext == ext
            if is_img:
                count += 1
                file_name_prefix = '.'.join(img_file_name.split('.')[:-1])
                txt_file_name = '.'.join([file_name_prefix, 'txt'])
                img_file_path = os.path.join(root, img_file_name)
                txt_file_path = os.path.join(root, txt_file_name)

                if verbose and count <= MAX_IMG_SHOWN:
                    my_imshow(img)

                img = cv2.imread(img_file_path)
                resized_img = np.array(Image.fromarray(img).resize((dst_w, dst_h)))
                centroids = cv_harris(resized_img, thres=harris_thres)

                img = cv2.imread(img_file_path)
                img_h, img_w = img.shape[:2]
                centroids_for_subpix = centroids.copy()
                centroids_for_subpix[:, 0] *= (img_w / dst_w)
                centroids_for_subpix[:, 1] *= (img_h / dst_h)
                corners = cv_subpixel(img, centroids=centroids_for_subpix)
                corners[:, 0] *= (dst_w / img_w)
                corners[:, 1] *= (dst_h / img_h)

                res = np.hstack((corners, centroids))
                with open(txt_file_path, 'w') as f:
                    for corner_x, corner_y, centoird_x, centroid_y in res:
                        f.write(str(corner_x))
                        f.write('\t')
                        f.write(str(corner_y))
                        f.write('\t')
                        f.write(str(centoird_x))
                        f.write('\t')
                        f.write(str(centroid_y))
                        f.write('\n')


# create a random mark file for each image
if __name__ == "__main__":
    # data_dir = '../data/tiny-imagenet-100-A'
    # create_label_file(data_dir)
    from utils.img_show import *

    # test marker coordinates
    MAX_IMG_SHOWN = 30
    data_dir = '../data/tiny-imagenet-100-A/train/n02769748/images'
    count = 0
    flag = False
    for root, dirs, files in os.walk(data_dir):
        exts_allowed = ('JPEG',)
        if flag:
            break
        for file_name in files:
            is_img = False
            file_ext = file_name.split('.')[-1]
            for ext in exts_allowed:
                is_img = is_img or file_ext == ext
            is_txt = file_ext == '.txt'
            file_name_prefix = '.'.join(file_name.split('.')[:-1])
            if is_img:
                img_file_path = os.path.join(root, file_name)
                img = get_train_img(img_file_path)
                txt_file_path = os.path.join(root, '.'.join((file_name_prefix, 'txt')))
                with open(txt_file_path, 'r') as f:
                    for line in f:
                        print(line, end='')
                        corner_x, corner_y, centroid_x, centroid_y = line.split()
                        corner_x = float(corner_x)
                        corner_y = float(corner_y)
                        centroid_x = float(centroid_x)
                        centroid_y = float(centroid_y)
                        img = draw_marker(img, [(centroid_x, centroid_y)], color=np.array([0, 0, 255]))
                        img = draw_marker(img, [(corner_x, corner_y)], color=np.array([0, 255, 0]))
                my_imshow(img, title='{}\t{}'.format(count, file_name))
                print('-' * 40)
                count += 1
                if count >= MAX_IMG_SHOWN:
                    flag = True
                    break
