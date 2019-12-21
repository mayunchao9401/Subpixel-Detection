import os

from torch.utils.data import Dataset

import pickle
import os
from skimage import io, transform
from utils.widget import *
from common import *


class TestDataset(Dataset):
    """Trush dataset"""

    def __init__(self, root_dir,
                 new_size=(h_for_train, w_for_train),
                 exts_allowed=('JPEG',)):
        """

        :param root_dir: the root directory files saved in
        :param transform: transform an image for process
        :param new_size: resize the high-precision image into lower (h, w)
        :param dtype: only for input image (not for label)  (not used)
        :param exts_allowed: ext string allowed for input image files
        """
        self.root_dir = root_dir
        self.exts_allowed = exts_allowed
        self.new_size = new_size
        index_file_name = 'TestDataset-' + os.path.split(root_dir)[-1] + '.pkl'
        index_file_path = os.path.join(cache_dir, index_file_name)
        if os.path.exists(index_file_path):  # read the file information directly from the file
            with open(index_file_path, 'rb') as f:
                self.index_entries = pickle.load(f)
            print('Index file loaded from {}'.format(index_file_path))
        else:
            self.index_entries = []
            self.__get_file_names(root_dir=self.root_dir)
            with open(index_file_path, 'wb') as f:
                pickle.dump(self.index_entries, f)

    def __len__(self):
        return len(self.index_entries)

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
            res['img']: numpy, image, H * W * C
            res['corner'] np.array (x, y), groudtruth corner coordinate
        """
        entry = self.index_entries[idx]
        img_file_path = entry['img_file_path']
        img = get_train_img(img_file_path)
        corners = entry['corners']

        sample = {}

        # transform gray image
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB);
            # imshow_no_ax(img)
        sample['img'] = img
        sample['corners'] = corners
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
            if root.split('\\')[-1] == 'images':
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
                            self.index_entries.append({})
                            self.index_entries[-1]['img_file_path'] = img_file_path
                            corners = []
                            for line in f:
                                corner_x, corner_y, centroid_x, centroid_y = line.split()
                                corners.append((float(corner_x), float(corner_y)))
                            self.index_entries[-1]['corners'] = np.array(corners)

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
    from utils.img_show import *
    MAX_IMG_SHOWN = 10

    data_dir = "..\\data\\tiny-imagenet-100-A\\test"

    dataset = TestDataset(data_dir, new_size=(h_for_train, w_for_train))
    print(len(dataset))
    for i, entry in enumerate(dataset.index_entries):
        img = dataset[i]['img']
        corners = dataset[i]['corners']
        corner = dataset[i]['corners']
        img_w = img.shape[2]
        img_h = img.shape[1]
        print(corners)
        img = draw_marker(img, corners, thickness=0)
        my_imshow(img, title='entry {}'.format(i))
        print('-' * 40)
        if i >= 10:
            break
