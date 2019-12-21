import cv2
import numpy as np
import os
from utils.img_show import *
from matplotlib import pyplot as plt
from common import *
from opencv_marker.harris import cv_harris


def cv_subpixel(img, require_centroids=False,
                max_iter=100, min_eps=0.001,
                harris_thres=harris_thres,
                zero_zone=(-1, -1),
                win_size=(5, 5), centroids=None):
    """
    :param img: input image (RGB)
    :param require_centroids: True: return tupel (corners, centroids)
    :param max_iter: max num of iterations for subpix
    :param min_eps: min eps for subpix
    :param verbose: whether to print some hint
    :param harris_thres: R > ? * dst.max() -> corner point
    :param zero_zone: ignore for subpixel-level detection
    :param win_size: window size for subpixel-level detection
    :return:<np.ndarray> coordinates of corners detected (x, y)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if centroids is None:
        centroids = cv_harris(img, harris_thres)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, min_eps)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), win_size, zero_zone, criteria)
    return corners


if __name__ == '__main__':
    file_name = 'triangle.png'
    image_path = os.path.join('tiny_test_images', file_name)
    img = cv2.imread(image_path)
    corners = cv_subpixel(img, harris_thres=harris_thres)
    print(corners)
    img = draw_marker(img, corners, thickness=1)

    dst_file_name = '.'.join(file_name.split('.')[:-1]) + '-subpixel.' + file_name.split('.')[-1]
    my_imshow(img, title='cv_subpixel')
