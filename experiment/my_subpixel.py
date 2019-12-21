import pickle
from opencv_marker.subpixel import cv_subpixel
from opencv_marker.harris import *
from common import *
from utils.widget import *
from training.train import Net


def my_subpixel(img, model, win_size, harris_thres=harris_thres, use_cuda=False, verbose=False):
    """

    :param img: numpy
    :param model: cpu model
    :param win_size:
    :return: corner coordinates np.ndarray
    """
    win_size_actual = 2 * win_size + 1
    centroids = cv_harris(img, harris_thres)

    if verbose:
        my_imshow(img, 'original')
        img_mk_drawn = draw_marker(img, centroids, 3, (255, 0, 255))
        my_imshow(img_mk_drawn)
    img_arounds = torch.zeros((len(centroids), 1, win_size_actual, win_size_actual))
    img = cv2.Canny(img, 50, 200, 3)
    if verbose:
        my_imshow(img, 'edged')
    with torch.no_grad():
        for i, centroid in enumerate(centroids):
            img_arounds[i] = np2tensor(get_img_around(img, centroid, win_size))
            if verbose:
                my_imshow(img_arounds[i], f'cutted-{i}', normalize=True)

        if use_cuda:
            img_arounds = img_arounds.cuda()
        shift = model(img_arounds)
        if use_cuda:
            shift = shift.cpu()

        # print(shift)

        corners = centroids + shift.numpy()
    return corners


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import os.path as osp

    # model, win_size = load_model(model_num)
    # ori_img = cv2.imread(osp.join('..', 'data', 'mscoco', 'tennis.jpg'))[:, :, ::-1]
    # corners = my_subpixel(ori_img, model, 5, harris_thres=harris_thres, verbose=True)
    # img_my = draw_marker(ori_img, corners, 1, color=np.array([0, 255, 0]))
    # print(len(corners))
    # print('my_subpix')
    # print(corners)
    # my_imshow(img_my, title='neural network regression', show_in_func=True)


    model, win_size = load_model(model_num)

    img_dir = os.path.join(data_dir, 'test', 'images')
    img_name = 'test_92.JPEG'
    txt_name = '.'.join(['.'.join(img_name.split('.')[:-1]), 'txt'])
    img_path = os.path.join(img_dir, img_name)
    txt_path = os.path.join(img_dir, txt_name)
    ori_img = get_train_img(img_path)

    thickness = 0
    groundtruth_corners = []
    with open(txt_path, 'r') as f:
        for line in f:
            corner_x, corner_y, centroid_x, centroid_y = line.split()
            groundtruth_corners.append((float(corner_x), float(corner_y)))
    my_imshow(ori_img, show_in_func=True)
    save_dir = "results"
    plt.savefig(osp.join(save_dir, "ori_img.png"))
    img = draw_marker(ori_img, groundtruth_corners, thickness=1, color=np.array([255, 0, 255]))
    print(len(groundtruth_corners))
    print(groundtruth_corners)

    corners_cv_sub = cv_subpixel(ori_img, harris_thres=harris_thres)
    img_cvsub = draw_marker(img, corners_cv_sub, thickness, color=np.array([0, 0, 255]))
    print(len(corners_cv_sub))
    print('cv_subpix')
    print(corners_cv_sub)
    my_imshow(img_cvsub, title='cv subpixel', show_in_func=True)
    plt.savefig(osp.join(save_dir, "cv-subpixel.png"))

    corners_harris = cv_harris(ori_img, thres=harris_thres)
    img_cvharris = draw_marker(img, corners_harris, thickness, color=np.array([255, 0, 0]))
    print(len(corners_harris))
    print('cv_harris')
    print(corners_harris)
    my_imshow(img_cvharris, title='cv harris', show_in_func=True)
    plt.savefig(osp.join(save_dir, "cv-harris.png"))

    corners = my_subpixel(ori_img, model, win_size, harris_thres=harris_thres)
    img_my = draw_marker(img, corners, thickness, color=np.array([0, 255, 0]))
    print(len(corners))
    print('my_subpix')
    print(corners)
    my_imshow(img_my, title='neural network regression', show_in_func=True)
    plt.savefig(osp.join(save_dir, "my-subpixel.png"))

    print('pink:groundtruth')
    print('blue:cv_subpix')
    print('red:cv_harris')
    print('green:my_subpix')