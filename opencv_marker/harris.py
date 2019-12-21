import cv2
import numpy as np
from utils.img_show import my_imshow, draw_marker
from common import *


def cv_harris(img, thres=harris_thres):
    """
    :param
    :return: <np.array> N * 2 centroids of corners detected by cv2.cornerHarris
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    ret, dst = cv2.threshold(dst, thres * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    return centroids[1:]


if __name__ == '__main__':
    import os

    image_dir = 'tiny_test_images'
    file_name = 'triangle.png'
    file_path = os.path.join(image_dir, file_name)
    img = cv2.imread(file_path)

    centroids = cv_harris(img, thres=harris_thres)
    print(centroids)

    img = draw_marker(img, centroids, thickness=1)

    my_imshow(img, title='cv_harris')
