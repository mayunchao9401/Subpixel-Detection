from torch.utils.data import Dataset
import cv2
import pickle
from utils.widget import *
import os
from common import *
from utils.img_show import my_imshow


class SubpixelDataset(Dataset):

    def __init__(self, root_dir,
                 win_size=5, new_size=(32, 32),
                 exts_allowed=('JPEG',)):
        """
        :param root_dir:<str> the root directory files saved in
        :param win_size:<int> half of the window size for training
        :param new_size:<tuple> resize the high-precision image into lower (h, w)
        :param exts_allowed:<tuple> ext string allowed for input image files
        """
        self.root_dir = root_dir
        self.exts_allowed = exts_allowed
        self.new_size = new_size
        self.win_size = win_size
        index_file_name = 'SubpixelDataset-' + os.path.split(root_dir)[-1] + '.pkl'
        index_file_path = os.path.join(cache_dir, index_file_name)
        if os.path.exists(index_file_path):  # read the file information directly from the file
            with open(index_file_path, 'rb') as f:
                self.index_entries = pickle.load(f)
            print('Index file loaded from {}'.format(index_file_path))
        else:
            self.index_entries = []
            # [i]: entry i, dict type
            # img_file_path, corner_x, corner_y, centroid_x, centroid_y
            self.__get_file_names(root_dir=self.root_dir)
            with open(index_file_path, 'wb') as f:
                pickle.dump(self.index_entries, f)

    def __len__(self):
        return len(self.index_entries)

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
            res['img']: train img;
            res['corner'] torch.tensor (x, y), groudtruth corner coordinate
        """
        entry = self.index_entries[idx]
        img_file_path = entry['img_file_path']
        img = get_train_img(img_file_path)

        corner_x, corner_y = entry['corner_x'], entry['corner_y']
        centroid_x, centroid_y = entry['centroid_x'], entry['centroid_y']

        sample = {}

        # print(idx, img_file_path, img.shape)
        # transform gray image
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # imshow_no_ax(img)

        # my_imshow(img)
        img = cv2.Canny(img, 50, 200, 3)
        # my_imshow(img)

        img = get_img_around(img, (centroid_x, centroid_y), self.win_size)
        # my_imshow(img)
        img_out = np2tensor(img)

        center_x = round(centroid_x)
        center_y = round(centroid_y)
        shift_x = corner_x - center_x
        shift_y = corner_y - center_y

        sample['img'] = img_out
        sample['corner'] = torch.tensor([shift_x, shift_y])
        return sample

    def __get_file_names(self, root_dir):
        """
        The result is sorted in my system.
        And it should be sorted for subsequent process.
        But I'm not sure whether this always hold

        *** The directory containing images should be named as images ***

        :param file_dir: directory of the data:
        :return: None
        """
        root_pre = None
        for root, dirs, files in os.walk(root_dir):
            if os.path.split(root)[-1] == 'images':
                try:
                    assert self.__file_check(root)
                except AssertionError as e:
                    print("Image file and label file with the same name should alternate in the list filenames")
                    print("Problem happened in directory:", root)
                    raise e
                if root != root_pre:
                    root_pre = root
                    print('processing files in directory', root)
                img_file_name = None
                txt_file_name = None
                for i, file_name in enumerate(files):
                    is_img = False
                    file_ext = file_name.split('.')[-1]
                    file_prefix = '.'.join(file_name.split('.')[:-1])
                    for ext in self.exts_allowed:
                        is_img = is_img or file_ext == ext
                    is_txt = file_ext == 'txt'
                    if is_img:
                        img_file_name = file_name
                    elif is_txt:
                        txt_file_name = file_name
                    else:
                        print('File structure not expected in directory {}'.format(root))
                        raise AssertionError
                    if i % 2 == 1:  # one pair is readed
                        img_file_path = os.path.join(root, img_file_name)
                        with open(os.path.join(root, txt_file_name), 'r') as f:
                            for line in f:
                                corner_x, corner_y, centroid_x, centroid_y = line.split()
                                self.index_entries.append({})
                                self.index_entries[-1]['img_file_path'] = img_file_path
                                self.index_entries[-1]['corner_x'] = float(corner_x)
                                self.index_entries[-1]['corner_y'] = float(corner_y)
                                self.index_entries[-1]['centroid_x'] = float(centroid_x)
                                self.index_entries[-1]['centroid_y'] = float(centroid_y)

    def __file_check(self, root):
        """
        Image file and label file with the same name should alternate in the list filenames
        Split file_names into img_file_names and txt_file_names
        :return: True if the permutation is correct
        """
        file_names = list(os.walk(root))[0][-1]
        for_check_alternate_img = False
        for_check_alternate_txt = False
        img_file_name = ""
        txt_file_name = ""
        self.img_file_names = []
        self.txt_file_names = []
        if len(file_names) % 2 != 0:
            return False
        for i, file_name in enumerate(file_names):
            is_img = False
            file_ext = file_name.split('.')[-1]
            file_name_prefix = ".".join(file_name.split('.')[:-1])
            for ext in self.exts_allowed:
                is_img = is_img or file_ext == ext
            is_txt = file_ext == "txt"
            if is_img:
                for_check_alternate_img = True
                img_file_name = file_name_prefix
                self.img_file_names.append(file_name)
            if is_txt:
                for_check_alternate_txt = True
                txt_file_name = file_name_prefix
                self.txt_file_names.append(file_name)
            if i % 2 == 1:  # a pair is scanned
                if not for_check_alternate_img or not for_check_alternate_txt:
                    return False
                if img_file_name != txt_file_name:
                    return False
        return True


if __name__ == "__main__":
    from common import *
    from utils.img_show import *

    MAX_IMG_SHOWN = 30

    data_dir = os.path.join(data_dir, 'val')
    IMG_W = w_for_train  # 1280 for practical usage
    IMG_H = h_for_train  # 720 for practical usage

    win_size = 5
    dataset = SubpixelDataset(data_dir, win_size=win_size)
    print('len(dataset):{}'.format(len(dataset)))
    for i, entry in enumerate(dataset.index_entries):
        print('idx:', i)
        print('entry:', entry)
        corner_x, corner_y = entry['corner_x'], entry['corner_y']
        title = 'corner_x:{}\tcorner_y:{}'.format(corner_x, corner_y)
        ori_img = get_train_img(entry['img_file_path'])
        ori_img_drawn = draw_marker(ori_img, [(corner_x, corner_y)])
        my_imshow(ori_img_drawn, title=title)

        img = dataset[i]['img']
        corner = dataset[i]['corner']
        img_w = img.shape[2]
        img_h = img.shape[1]
        shift_x_int = cut(win_size + corner[0], 0, img_w)
        shift_y_int = cut(win_size + corner[1], 0, img_h)
        img = draw_marker(img, [(shift_x_int, shift_y_int)])
        my_imshow(img, title='entry:{},x:{},y:{}'.format(i, corner[0], corner[1]))

        if i >= MAX_IMG_SHOWN:
            break
