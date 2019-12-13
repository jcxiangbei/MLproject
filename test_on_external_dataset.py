from initialize import remake_labels
from models import Darknet
from test import evaluate
from utils.utils import load_classes

import os
import torch as tc


def testpath_to_subtest():
    base_path = 'data/custom/'
    with open(base_path + 'test.txt', 'r', encoding='utf8') as _old:
        with open(base_path + 'battery_sub_test.txt', 'w') as _new:
            for s in _old.readlines():
                _new.write(s.split('/')[-1].split('.')[0].split('_')[-1] + '\n')


def subtest_to_testpath(imagesetfile: str, imgpath: str):
    base_path = 'data/custom/'
    imgs = os.listdir(imgpath)
    with open(imagesetfile, 'r') as _old:
        with open(base_path + 'test.txt', 'w', encoding='utf8') as _new:
            for name in _old.readlines():
                name = name.rstrip('\n')
                for img in imgs:
                    if name in img:
                        _new.write(f'{imgpath}{img}\n')
                        imgs.remove(img)
                        break


if __name__ == "__main__":
    # -----------------------------------------------------
    #   edit these THREE paths to test on other datasets:
    #   using absolute path (outside the code folder)
    #      or relative path (inside the code folder)
    # -----------------------------------------------------
    imagesetfile = r'D:\BUAA\aaa/battery_sub_test.txt'.replace('\\', '/')
    anno_path = r'D:\BUAA\aaa/annos/'.replace('\\', '/')  # trailing '/' is needed
    img_path = r'D:\BUAA\aaa/images/'.replace('\\', '/')  # trailing '/' is needed

    # -----------------------------------
    #   DO NOT EDIT THE FOLLOWING CODES
    # -----------------------------------

    # other paths
    test_path = 'data/custom/test.txt'
    weights_path = 'checkpoints/yolov3.pth'
    model_def = 'config/custom.cfg'
    class_path = 'data/custom/classes.names'

    # transform something
    subtest_to_testpath(imagesetfile, img_path)
    remake_labels(anno_path, img_path)

    # load model
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    class_names = load_classes(class_path)
    model = Darknet(model_def).to(device)
    model.load_state_dict(tc.load(weights_path))

    # test, then compute APs and mAP
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=test_path,
        iou_thres=0.5,
        conf_thres=0.001,
        nms_thres=0.5,
        img_size=416,
        batch_size=8,
        device=device,
    )

    # print results
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP = {AP[i]}")
    print(f" ---- mAP = {AP.mean()}")
