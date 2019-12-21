import os
import cv2
from opencv_marker.subpixel import cv_subpixel
from utils.widget import cut, get_img_around
from utils.img_show import draw_marker, my_imshow
from common import harris_thres
import numpy as np
from lines.lines_par import *
import os
from lines.union_find import UnionFind
import matplotlib.pyplot as plt
import traceback
from math import cos, sin, tan
from copy import deepcopy
from utils.img_show import *


class MarkerFinder:
    def __init__(self, img, win_size=20, pars=None, verbose=False):
        self.pars = {'common': dict(canny_thres1=50, canny_thres2=200, canny_apertureSize=3, harris_thres=0.1),
                     '+': dict(max_dist=40, rho_thres=1, theta_thres=5 * PI / 180, parallel_dist=10,
                               hough_thres=int(1.5 * win_size), noise_thres=10, merge_thres=3.0)}
        try:
            if pars is not None:
                for outer_key in pars:
                    if outer_key not in self.pars:
                        print("Unacceptable parameter {}".format(outer_key))
                        raise IndexError
                    for inner_key in pars[outer_key]:
                        if inner_key not in self.pars[outer_key]:
                            print("Unacceptable parameter {}-{}".format(outer_key, inner_key))
                            raise IndexError
                        self.pars[outer_key][inner_key] = pars[outer_key][inner_key]
        except IndexError as e:
            traceback.print_exc()
            raise e
        self.img = img
        self.win_size = win_size
        self.img_edged = cv2.Canny(img, self.pars['common']['canny_thres1'],
                                   self.pars['common']['canny_thres2'],
                                   self.pars['common']['canny_apertureSize'])  # edge the img once
        self.verbose = verbose
        self.centroids = cv_subpixel(img, harris_thres)
        self.proc_idx = None
        self.proc_centroid = None
        if verbose:
            self.target_window = dict(left=200, right=300, top=200, bottom=300)
            cache_num = 0
            while os.path.exists(os.path.join('.', 'cache' + str(cache_num))):
                cache_num += 1
            cache_dir = os.path.join('.', 'cache' + str(cache_num))
            os.makedirs(cache_dir)
            self.cache_dir = cache_dir
            my_imshow(self.img_edged, title='img-edged')
        else:
            self.cache_dir = None

    def find_corners(self, modes=('+',)):
        img = self.img
        win_size = self.win_size
        pars = self.pars
        edges = self.img_edged

        corners = {key: [] for key in modes}
        modes = set(modes)

        for ii, centroid in enumerate(centroids):
            self.proc_idx = ii
            self.proc_centroid = centroid
            img_around = get_img_around(img, centroid, win_size, padding=True)
            edges_around = get_img_around(edges, centroid, win_size, padding=True)
            if '+' in modes:
                is_found = self.__mark_cross(img_around, edges_around, corners)
                if is_found:
                    continue
        for mode in corners:
            merge_thres = pars[mode]['merge_thres']
            if len(corners[mode]) > 0:
                corners[mode] = MarkerFinder.__union(corners[mode],
                                                     func=lambda pt1, pt2: MarkerFinder.__rel(pt1, pt2) < merge_thres)
        return corners

    #             # if len(lines) == 0:
    #             #     new_point = tuple(centroid)
    #             #     corners[0].append(new_point)
    #             #     continue
    #             # elif len(lines) == 1:
    #             #     full_lines = my_hough(edges, 2 * win_size)
    #             #     if full_lines is None:
    #             #         proj_point = proj_pt2line(init_point, lines[0])
    #             #         new_point = map2img(proj_point, centroid, win_size)
    #             #         corners[1].append(new_point)
    #             # elif len(lines) == 2:
    #             #     crossing = cross(lines[0], lines[1])
    #             #     if rel(crossing, init_point) < max_dist:
    #             #         corners[2].append(map2img(crossing, centroid, win_size))
    #             # elif len(lines) == 3:
    #             #     thres = 3.0
    #             #     crossings = [None] * 3
    #             #     crossings[0] = cross(lines[0], lines[1])
    #             #     crossings[1] = cross(lines[0], lines[2])
    #             #     crossings[2] = cross(lines[1], lines[2])
    #             #     if rel(crossings[0], crossings[1]) < thres and \
    #             #             rel(crossings[0], crossings[2]) < thres and \
    #             #             rel(crossings[1], crossings[2]) < thres:
    #             #         crossing = np.array(crossings).mean(axis=0)
    #             #         if rel(crossing, init_point) < max_dist:
    #             #             corners[3].append(map2img(crossing, centroid, win_size))
    #         return corners
    #
    #
    def __mark_cross(self, img_around, edges_around, corners):
        mode = '+'  # line cross like +
        pars = self.pars[mode]
        rho_thres, theta_thres, hough_thres = pars['rho_thres'], pars['theta_thres'], pars['hough_thres']
        parallel_dist, max_dist = pars['parallel_dist'], pars['max_dist']
        noise_thres, merge_thres = pars['noise_thres'], pars['merge_thres']

        # if self.__is_in_window():
        #     pause = True

        ori_lines = cv2.HoughLines(edges_around, rho_thres, theta_thres, hough_thres, 0, 0)
        if ori_lines is None:
            ori_lines = np.zeros((0, 2))
        else:
            ori_lines = ori_lines.squeeze(axis=1)
        lines = MarkerFinder.__union(ori_lines,
                                     func=MarkerFinder.__line_union_func(parallel_dist, theta_thres))
        if len(ori_lines) >= noise_thres:  # too much lines means noise
            return False
        init_point = np.array((self.win_size, self.win_size))
        if self.verbose:
            show = self.__is_in_window()
            # show = True
            # show = (len(lines) == 2) and \
            #        MarkerFinder.__rel(MarkerFinder.__cross(lines[0], lines[1]), init_point) < max_dist
            if show:
                cache_dir_per_mode = os.path.join(self.cache_dir, mode)
                if not os.path.exists(cache_dir_per_mode):
                    os.makedirs(cache_dir_per_mode)
                my_imshow(img_around, title='{}-centroid={}'.format(self.proc_idx, self.proc_centroid),
                          show_in_func=True)
                # plt.savefig(os.path.join(selected_dir, '{}-centroid={}.png'.format(ii, centroid)))
                my_imshow(edges_around, title='{}-edges'.format(self.proc_idx), show_in_func=True)
                # plt.savefig(os.path.join(selected_dir, '{}-edges.png'.format(ii)))
                img_lines = MarkerFinder.__draw_lines(img_around, ori_lines)
                my_imshow(img_lines, title='{}-ori_lines-num={}'.format(self.proc_idx, len(ori_lines)),
                          show_in_func=True)
                # plt.savefig(os.path.join(selected_dir, '{}-ori_lines-num={}.png'.format(ii, len(ori_lines))))
                img_lines = MarkerFinder.__draw_lines(img_around, lines)
                my_imshow(img_lines, title='{}-lines-num={}'.format(self.proc_idx, len(lines)), show_in_func=True)
                # plt.savefig(os.path.join(selected_dir, '{}-lines-num={}.png'.format(ii, len(lines))))
        if len(lines) == 2:
            crossing = MarkerFinder.__cross(lines[0], lines[1])
            if MarkerFinder.__rel(crossing, init_point) < max_dist:
                corners[mode].append(self.__map2img(crossing, self.proc_centroid))
                return True
        return False

    def __is_in_window(self):
        """
        :param point: x, y
        :return: point is in self.target_window?
        """
        right = self.target_window['right']
        left = self.target_window['left']
        top = self.target_window['top']
        bottom = self.target_window['bottom']
        x, y = self.proc_centroid
        return right > x > left and bottom > y > top

    def __map2img(self, point, src_center):
        """
        :param point: point in src to be mapped (x, y)
        :param src_center: src img center in dst img (x:int, y:int)
        :return: mapped point coordinate (x, y)
        """
        left = int(round(src_center[0])) - self.win_size
        top = int(round(src_center[1])) - self.win_size
        return point[0] + left, point[1] + top

    @staticmethod
    def __draw_lines(img, lines):
        """
        :param img: np.array
        :param lines: np.array(num_lines * 2)
        :return: img with lines drawn on it
        """
        img = img.copy()
        color = (0, 255, 0)
        for line in lines:
            k, b = MarkerFinder.__polar2kb(line)
            x1 = 0
            y1 = k * x1 + b
            x2 = img.shape[1]  # w
            y2 = k * x2 + b
            pt1 = (int(round(x1)), int(round(y1)))
            pt2 = (int(round(x2)), int(round(y2)))
            cv2.line(img, pt1, pt2, color)
        return img

    @staticmethod
    def __polar2kb(line):
        rho, theta = line
        k = - 1 / (tan(theta) + 1e-7)
        b = rho / (sin(theta) + 1e-7)
        return k, b

    @staticmethod
    def __cross(line1, line2, polar=True):
        """
        :param line1: rho, theta / k, b
        :return: crossing in (x, y) of the 2 points
        """
        k1, b1, k2, b2 = (None,) * 4
        if polar:
            k1, b1 = MarkerFinder.__polar2kb(line1)
            k2, b2 = MarkerFinder.__polar2kb(line2)
        else:
            k1, b1 = line1
            k2, b2 = line2
        x = (b2 - b1) / (k1 - k2 + 1e-7)
        y = (k1 * b2 - k2 * b1) / (k1 - k2 + 1e-7)
        return x, y

    @staticmethod
    def __proj_pt2line(point, line):
        """
        :param point: (x, y)
        :param line: (rho, theta)
        :return: point's projection on line
        """
        rho, theta = line
        x, y = point
        line_k, line_b = MarkerFinder.__polar2kb(line)
        point_k = tan(theta)
        point_b = y - tan(theta) * x
        proj_x, proj_y = MarkerFinder.__cross((line_k, line_b), (point_k, point_b), polar=False)
        return proj_x, proj_y

    @staticmethod
    def __is_on_line(point, line, thres=3.0):
        """
        :param point: np.array(2) x, y
        :param line: np.array(2) rho, theta
        :param thres: if dist < thres : on line
        :return: judge whether point is on line
        """
        x, y = point
        rho, theta = line
        dist = abs(x * cos(theta) + y * sin(theta) - rho)
        if dist < thres:
            return True
        else:
            return False

    @staticmethod
    def __rel(vec1, vec2):
        """
        :param vec1:
        :param vec2:
        :return: L2 distance
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.sqrt(((vec1 - vec2) ** 2).sum())

    @staticmethod
    def __line_union_func(rho_thres, theta_thres):
        return lambda line1, line2: abs(line1[0] - line2[0]) < rho_thres and abs(line1[1] - line2[1]) < theta_thres

    @staticmethod
    def __union(ver_set, func):
        """
        :param ver_set: N * m
        :param func: func(ver_set[i], ver_set[j]) = True if they are judged close
        :return: compressed ver_set
        """
        if len(ver_set) == 0:
            return ver_set
        from lines.union_find import UnionFind
        union = UnionFind(range(len(ver_set)))
        for ii, obj_line in enumerate(ver_set):
            for jj, ref_line in enumerate(ver_set[ii + 1:], ii + 1):
                if func(obj_line, ref_line):
                    if not union.connected(ii, jj):
                        union.union(ii, jj)
        components = union.components()
        comp_set = []
        for component in components:
            set_t = np.array(ver_set)[list(component)]
            comp_set.append(set_t.mean(axis=0))
        return np.array(comp_set).reshape(-1, 2)


if __name__ == "__main__":
    thickness = 2
    file_name = 'cross.jpg'
    img = cv2.imread(os.path.join(data_dir, file_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    centroids = cv_subpixel(img, 0.1)
    all_centroids = draw_marker(img, centroids, thickness=thickness)
    my_imshow(all_centroids, title='all centroids')
    print('centroids:')
    print(centroids)

    thickness = 10
    finder = MarkerFinder(img, win_size=20, verbose=False)
    corners = finder.find_corners()
    print('corners:')
    print(corners)
    img_drawn = img
    img_drawn = draw_marker(img_drawn, corners['+'], thickness=thickness, color=np.array((255, 0, 0)))
    # # img_drawn = draw_marker(img_drawn, corners[1], thickness=thickness, color=np.array((0, 255, 0)))
    # # img_drawn = draw_marker(img_drawn, corners[2], thickness=thickness, color=np.array((0, 0, 255)))
    # # img_drawn = draw_marker(img_drawn, corners[3], thickness=thickness, color=np.array((255, 0, 255)))
    my_imshow(img_drawn, title=file_name)
