from __future__ import  absolute_import
import os


import matplotlib
import argparse

import torch
import numpy as np
from PIL import Image
import cv2
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.board_tool import tensorboard_bbox
from utils.eval_tool import eval_detection_voc
from data.dataset import preprocess

def parse_args():
    parser = argparse.ArgumentParser(description='images detection')
    parser.add_argument('--img_dir', default='./imgs', help='the dir to the images')
    parser.add_argument('--save_dir', default='./imgs', help='dir to save detected imgs')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    imgs = list()
    for roots, dirs, files in os.walk(args.img_dir):
        for name in files:
            img = Image.open(os.path.join(roots, name))
            img = img.convert('RGB')
            img = np.asarray(img, dtype='float32')
            imgs.append(img)
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    load_path = './checkpoints/fasterrcnn_05042345_0.7958382759319134'
    trainer.load(load_path)
    print('load pretrained model from ', load_path)
    for i, img in enumerate(imgs):
        img = img.transpose((2, 0, 1))
        img = preprocess(img)
        ori_img_ = inverse_normalize(img)
        show_bboxes_, show_labels_, show_scores_ = trainer.faster_rcnn.predict([ori_img_], visualize=True)
        show_bbox = np.array(show_bboxes_[0])
        show_label = np.array(show_labels_[0])
        show_score = np.array(show_scores_[0])
        pred_img = tensorboard_bbox(ori_img_,
                                    show_bbox,
                                    show_label,
                                    show_score)
        matplotlib.pyplot.imsave(args.save_dir +'/' + str(i) + 'inf.jpg', pred_img.transpose((1,2,0)))


if __name__ == '__main__':
    main()


