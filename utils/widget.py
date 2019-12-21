import torch
import numpy as np
from PIL import Image
import cv2
import common
import pickle
import os


def np2tensor(img):
    if len(img.shape) == 3:
        img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    return img


def tensor2np(img):
    img = img.numpy()
    img = img.transpose((1, 2, 0))
    return img


def cut(x, low, high):
    """
    :param x:<float/int/torch.tensor> float or int or 1*1 tensor
    :return:<int> round x, and not beyond [low, high]
    """
    if (torch.is_tensor(x)):
        x = x.item()
    return int(min(max(round(x), low), high))


def get_train_img(img_path):
    """
    We can only access resized images in training and testing
    This function is the interface to read those images
    :param img_path:
    :return:<np.ndarray> resized image
    """
    from common import h_for_train, w_for_train
    img = cv2.imread(img_path)
    img = np.array(Image.fromarray(img).resize((w_for_train, h_for_train)))
    return img


def get_img_around(img, centroid, win_size, padding=True):
    """
    :param img:<np.ndarray/torch.tensor> RGB image H * W * C
    :param centroid:<tuple> (float, float)
    :param win_size:<int> 2 * win_size + 1 is the targeted width and height
    :param padding True -> pad img, False -> returned size will vary
    :return:<np.ndarray/torch.tensor> such a image with 128 padding
    """
    gray = False
    if len(img.shape) == 2: # gray img
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        gray = True
    is_tensor = False
    if torch.is_tensor(img):
        img = tensor2np(img)
        is_tensor = True

    side = win_size * 2 + 1
    img_out = np.zeros(shape=(side, side, 3),
                       dtype=img.dtype)
    img_w = img.shape[1]
    img_h = img.shape[0]
    centroid_x, centroid_y = centroid
    center_x = int(round(centroid_x, None))
    center_y = int(round(centroid_y, None))
    left = center_x - win_size
    right = center_x + win_size + 1
    top = center_y - win_size
    bottom = center_y + win_size + 1
    # right bottom not included 128 padding
    img_left, img_top = 0, 0
    img_right = side  # not included
    img_bottom = side  # not included
    padding_value = np.array([255 // 2] * 3)

    if left < 0:
        img_out[:, :-left, :] = padding_value
        img_left = -left
        left = 0
    if top < 0:
        img_out[:-top, :, :] = padding_value
        img_top = -top
        top = 0
    if right > img_w:
        padding_num = right - img_w
        img_out[:, -padding_num:, :] = padding_value
        img_right = img_right - padding_num
        right = img_w
    if bottom > img_h:
        padding_num = bottom - img_h
        img_out[-padding_num:, :, :] = padding_value
        img_bottom = img_bottom - padding_num
        bottom = img_h

    img_out[img_top: img_bottom, img_left: img_right, :] = \
        img[top: bottom, left: right, :]

    if not padding:
        img_out = img[top: bottom, left: right, :]

    if gray:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2GRAY)

    if is_tensor:
        img_out = np2tensor(img_out)
    return img_out


def load_model(model_num):
    """
    :param model_num: the model trained in ?
    :return: (model, win_size)
    """
    cache_dir = os.path.join(common.cache_dir, 'train' + str(model_num))
    model_name = 'model.pth'
    win_size_name = 'win_size.pkl'
    model_path = os.path.join(cache_dir, model_name)
    win_size_path = os.path.join(cache_dir, win_size_name)
    model = torch.load(model_path)
    with open(win_size_path, 'rb') as f:
        win_size = pickle.load(f)
    return model, win_size


if __name__ == '__main__':
    from common import *
    import cv2

    img_path = os.path.join(data_dir, 'test', 'images', 'test_0.JPEG')

    # np2tensor tensor2np
    img = cv2.imread(img_path)
    print(img.shape, type(img))
    img = np2tensor(img)
    print(img.shape, type(img))
    img = tensor2np(img)
    print(img.shape, type(img))

    # cut
    x = 3
    print(cut(x, 1, 5))
    print(cut(x, 4, 5))
    print(cut(x, 1, 2))

    # get_img_around
    from utils.img_show import my_imshow

    ori_img = img
    my_imshow(ori_img, title='original')
    img = get_img_around(ori_img, (70.2, 50.6), 100)
    my_imshow(img, title='shape:{} type:{}'.format(img.shape, type(img)))
    img = np2tensor(ori_img)
    img = get_img_around(img, (70.2, 50.6), 100)
    my_imshow(img, title='shape:{} type:{}'.format(img.shape, type(img)))

    # get_img
    img = get_train_img(img_path)
    my_imshow(img, title='resized image')
