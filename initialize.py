import os
import random as rd

from PIL import Image


def divide_dataset(img_path: str = 'data/custom/images/'):
    # -------------------
    #   divide data set
    # -------------------
    split_ratio = 0.8  # split the dataset
    file_list = os.listdir(img_path)
    file_nums = len(file_list)
    rd.shuffle(file_list)
    # train set
    with open('data/custom/train.txt', 'w', encoding='utf8') as f:
        for i in range(int(split_ratio * file_nums)):
            f.write(img_path + file_list[i] + '\n')
    # validation set
    with open('data/custom/valid.txt', 'w', encoding='utf8') as f:
        for i in range(int(split_ratio * file_nums), file_nums):
            f.write(img_path + file_list[i] + '\n')


def remake_labels(lbl_path: str = 'data/custom/labels/', img_path: str = 'data/custom/images/'):
    # -----------------
    #   remake labels
    # -----------------
    base_path = 'data/custom/labels/'
    os.makedirs(base_path, exist_ok=True)
    file_list = os.listdir(lbl_path)
    for _file in file_list:
        with open(lbl_path + _file, 'r', encoding='utf8') as f:
            rows = [row for row in f.read().split('\n') if row]
        w, h = Image.open(img_path + _file.replace('txt', 'jpg')).size
        for i in range(len(rows)):
            rows[i] = rows[i].split(' ')[1:]
            if rows[i][0] == '未指定类别': continue
            rows[i][0] = 0 if rows[i][0] == '带电芯充电宝' else 1
            lu_x, lu_y, rb_x, rb_y = map(int, rows[i][1:])
            rows[i][1], rows[i][2] = (lu_x + rb_x) / 2 / w, (lu_y + rb_y) / 2 / h  # center_x, center_y
            rows[i][3], rows[i][4] = (rb_x - lu_x) / w, (rb_y - lu_y) / h  # width, height
        with open(base_path + _file, 'w') as f:
            for row in rows:
                if isinstance(row[0], int): f.write(' '.join(list(map(str, row)) + ['\n']))


if __name__ == "__main__":
    divide_dataset()
    remake_labels()
