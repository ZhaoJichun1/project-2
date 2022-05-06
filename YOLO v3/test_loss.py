import logging
import utils.gpu as gpu
from model.yolov3 import Yolov3
from model.loss.yolo_loss import YoloV3Loss
import torch
import utils.datasets as data
import argparse
from eval.evaluator import *
from utils.tools import *
from torch.utils.tensorboard import SummaryWriter
import config.yolov3_config_voc as cfg


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='2'
#


def loss_test(model, epoch, device, writer):
    print("testing loss")
    test_dataset = data.VocDataset(anno_file_type="test", img_size=cfg.TEST["TEST_IMG_SIZE"])
    test_dataloader = DataLoader(test_dataset,
                                      batch_size=1,
                                      num_workers=2,
                                      shuffle=False)
    criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

    mloss = torch.zeros(4)
    for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)  in enumerate(test_dataloader):
        if i%200==0:
            print(i)
        imgs = imgs.cuda()
        label_sbbox = label_sbbox.to(device)
        label_mbbox = label_mbbox.to(device)
        label_lbbox = label_lbbox.to(device)
        sbboxes = sbboxes.to(device)
        mbboxes = mbboxes.to(device)
        lbboxes = lbboxes.to(device)

        p, p_d = model(imgs)

        loss, loss_giou, loss_conf, loss_cls = criterion(p, p_d, label_sbbox, label_mbbox,
                                          label_lbbox, sbboxes, mbboxes, lbboxes)


        # Update running mean of tracked metrics
        loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
        mloss = (mloss * i + loss_items) / (i + 1)


        # Print batch results
    writer.add_scalar('\\test/loss_giou\\', mloss[0], epoch)
    writer.flush()
    writer.add_scalar('\\test/loss_conf\\', mloss[1], epoch)
    writer.flush()
    writer.add_scalar('\\test/cls\\', mloss[2], epoch)
    writer.flush()
    writer.add_scalar('\\test/loss_total\\', mloss[3], epoch)
    writer.flush()
    s = ('loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
             ) % (mloss[0],mloss[1], mloss[2], mloss[3])
    print(s)


