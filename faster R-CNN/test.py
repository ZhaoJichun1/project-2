from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

import torch
import time
import numpy as np
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.board_tool import tensorboard_bbox
from utils.eval_tool import eval_detection_voc

matplotlib.use('agg')


def test(**kwargs):
    opt._parse(kwargs)

    print('load data')
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
                                       pin_memory=False
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    load_path = './checkpoints/fasterrcnn_05042345_0.7958382759319134'
    trainer.load(load_path)
    print('load pretrained model from ', load_path)
#    lr_ = opt.lr
#    trainer.reset_meters()
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, scale) in tqdm(enumerate(test_dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
#        img, bbox, label = imgs.cuda().float(), gt_bboxes_.cuda(), gt_labels_.cuda()
        ori_img_ = inverse_normalize(at.tonumpy(imgs[0]))
        pred_bboxes_, pred_labels_, pred_scores_ = trainer.faster_rcnn.predict(imgs, [sizes])
        if (ii + 1) % 100 == 0:
            rois = trainer.faster_rcnn.predict_rpn(imgs, [sizes])
            ori_img_ = inverse_normalize(at.tonumpy(imgs[0]))
            rpn_img = tensorboard_bbox(ori_img_, at.tonumpy(rois))
            trainer.board.img('rpn_img', rpn_img, 'Test', (ii+1)//100)

        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        torch.cuda.empty_cache()
    eval_result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=False)
    print("mAP = {}".format(eval_result['map']))


if __name__ == '__main__':
    import fire

    fire.Fire()
