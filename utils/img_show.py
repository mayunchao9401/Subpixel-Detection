import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.widget import np2tensor, tensor2np, cut
from math import sin, cos, tan
import cv2


def my_imshow(img, title=None, normalize=False, show_in_func=True):
    """
    show an image
    :param img: <np.ndarray/torch.tensor>
    :param title: <str>
    :param normalize: <boolean> whether normalize image value into 0-255
    :param show_in_func: <boolean> whether to call plt.show()
    :return: None
    """
    try:
        assert img is not None
    except AssertionError as e:
        print("Parameter <img> shouldn't be None")
        raise e

    if torch.is_tensor(img):
        img = tensor2np(img)

    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)

    plt.imshow(img.astype('uint8').squeeze())
    plt.gca().axis('off')
    if (title):
        plt.title(title)

    if show_in_func:
        plt.show()
        # cv2.waitKey(300)


def draw_marker(img, corners, thickness=0, color=np.array([0, 255, 0])):
    """
    :param img:<np.ndarray/torch.tensor>
    :param corners:anything with shape=(N, 2) N=len(corners)
    :param thickness:<int> length of side = 2 * thickness + 1
    :param color: <np.ndarray> (3,)
    :return: <np.ndarray/torch.tensor> with marker drawn
    """
    n = None
    try:
        n = len(corners)
    except Exception as e:
        print('wrong type of corners')
        raise e
    if img is None:
        print("Parameter <img> shouldn't be None")
        raise TypeError

    output_img = None
    if torch.is_tensor(img):
        output_img = tensor2np(img).copy()
    elif isinstance(img, np.ndarray):
        output_img = img.copy()
    else:
        print('Wrong type of parameter <img>')
        raise TypeError

    img_w = output_img.shape[1]
    img_h = output_img.shape[0]
    for ii in range(n):
        corner = corners[ii]
        x, y = np.round(corner)
        left = x - thickness
        right = x + thickness + 1
        top = y - thickness
        bottom = y + thickness + 1

        left = cut(left, 0, img_w)
        right = cut(right, 0, img_w)
        top = cut(top, 0, img_h)
        bottom = cut(bottom, 0, img_h)

        output_img[top:bottom, left:right] = color
    return output_img


if __name__ == '__main__':
    import cv2
    from common import *
    import os
    # my_imshow
    ori_img = cv2.imread(os.path.join(data_dir, 'test', 'images', 'test_0.JPEG'))
    my_imshow(ori_img)
    img = np2tensor(ori_img)
    img = img / 2
    my_imshow(img, normalize=True, show_in_func=False, title='my_imshow')
    plt.show()

    # draw_marker
    corners = [(32, 32), (63, 63.2)]
    img = draw_marker(ori_img, corners=corners, thickness=2, color=np.array((255, 0, 255)))
    my_imshow(img, title='draw_marker')

