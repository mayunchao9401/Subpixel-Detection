from math import sqrt

import torch
import numpy as np

from experiment.my_subpixel import my_subpixel
from experiment.test_dataset import TestDataset

from opencv_marker.subpixel import cv_subpixel
from common import *
from training.train import Net
import pickle
from utils.widget import load_model
from opencv_marker.harris import cv_harris
import time
import common

def evaluate_dist(test_corners, ref_corners):
    """
    :param test_corners:<np.ndarray>
    :param ref_corners:<np.ndarray>
    :return: dist_sum
    """
    dist_sum = 0
    dist = (test_corners - ref_corners) ** 2
    dist = dist.sum(axis=1)
    dist = np.sqrt(dist)
    dist_sum = dist.sum()
    return dist_sum


if __name__ == '__main__':
    model, win_size = load_model(model_num)
    use_cuda = False
    if not torch.cuda.is_available():
        use_cuda = False
    if use_cuda:
        model = model.cuda()

    data_dir = os.path.join(data_dir, 'test')
    dataset = TestDataset(data_dir, new_size=(h_for_train, w_for_train))
    MAX_SHOWN = 100

    my_dist_sum = 0.0
    cv_dist_sum = 0.0
    num_corners = 0
    my_time = 0.0
    cv_time = 0.0
    for i, sample in enumerate(dataset):
        img = sample['img']
        corners = sample['corners']

        time_start = time.time()  # time in second
        my_corners = my_subpixel(img, model, win_size, harris_thres=harris_thres, use_cuda=use_cuda)
        time_end = time.time()
        my_time += (time_end - time_start)

        time_start = time.time()  # time in second
        cv_corners = cv_subpixel(img, harris_thres=harris_thres)
        time_end = time.time()
        cv_time += (time_end - time_start)

        my_dist = evaluate_dist(test_corners=my_corners, ref_corners=corners)
        my_dist_sum += my_dist
        cv_dist = evaluate_dist(test_corners=cv_corners, ref_corners=corners)
        cv_dist_sum += cv_dist
        num_corners += len(corners)
        if i % 100 == 0:
            print('%d images processed' % i)
    my_dist_mean = my_dist_sum / num_corners
    cv_dist_mean = cv_dist_sum / num_corners

    print('-' * 40)
    print('my subpixel:')
    print('mean dist:\t{}'.format(my_dist_mean))
    print('time:\t{}\n'.format(my_time))

    print('-' * 40)
    print('cv subpixel:')
    print('mean dist:\t{}'.format(cv_dist_mean))
    print('time\t{}\n'.format(cv_time))

    log_dir = os.path.join(log_dir, 'train' + str(model_num), 'evaluation.txt')
    with open(log_dir, 'w') as f:
        f.write('my subpixel:\n')
        f.write('mean dist:\t{}\n'.format(my_dist_mean))
        f.write('time:\t{}\n'.format(my_time))

        f.write('-' * 40 + '\n')

        f.write('cv subpixel:\n')
        f.write('mean dist:\t{}\n'.format(cv_dist_mean))
        f.write('time\t{}\n'.format(cv_time))
