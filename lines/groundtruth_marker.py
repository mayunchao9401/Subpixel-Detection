from opencv_marker.subpixel_marker import create_label_file
from lines.lines_par import *
import cv2
from PIL import Image
import os

if __name__ == '__main__':
    create_label_file(data_dir, dst_h=dst_h, dst_w=dst_w)
    from utils.img_show import *

    # test marker coordinates
    MAX_IMG_SHOWN = 30
    count = 0
    flag = False
    for root, dirs, files in os.walk(data_dir):
        exts_allowed = ('JPEG', 'jpg')
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
                img = cv2.imread(img_file_path)
                img = np.array(Image.fromarray(img).resize((dst_w, dst_h)))
                txt_file_path = os.path.join(root, '.'.join((file_name_prefix, 'txt')))
                with open(txt_file_path, 'r') as f:
                    for line in f:
                        print(line, end='')
                        corner_x, corner_y, centroid_x, centroid_y = line.split()
                        corner_x = float(corner_x)
                        corner_y = float(corner_y)
                        centroid_x = float(centroid_x)
                        centroid_y = float(centroid_y)
                        thickness = 2
                        img = draw_marker(img, [(centroid_x, centroid_y)], thickness=2, color=np.array([0, 0, 255]))
                        img = draw_marker(img, [(corner_x, corner_y)], thickness=2, color=np.array([0, 255, 0]))
                my_imshow(img, title='{}\t{}'.format(count, file_name))
                print('-' * 40)
                count += 1
                if count >= MAX_IMG_SHOWN:
                    flag = True
                    break
