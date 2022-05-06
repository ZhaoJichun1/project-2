from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

import torch
import time
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from data import util
from utils.board_tool import tensorboard_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
#import resource

#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, trainer, test_num=250):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    trainer.reset_meters()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, scale) in tqdm(enumerate(dataloader)):
        scale = at.scalar(scale)
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        H, W = imgs.shape[-2:]
        bbox_resize = at.totensor(util.resize_bbox(at.tonumpy(gt_bboxes_[0]), (H, W), (H*scale, W*scale)).reshape(1,-1,4))
        trainer.test_step(imgs.cuda().float(), bbox_resize.cuda(), gt_labels_.cuda(), scale)
        pred_bboxes_, pred_labels_, pred_scores_ = trainer.faster_rcnn.predict(imgs.cuda().float(), [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break
    trainer.board.plot_many(trainer.get_meter_data(), 'Test')

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=False
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        print('epoch {}'.format(epoch))
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.board.plot_many(trainer.get_meter_data(), 'Train')

        torch.cuda.empty_cache()
        print("testing")

        eval_result = eval(test_dataloader, trainer, test_num=opt.test_num)
        trainer.board.plot('test_map', eval_result['map'], 'Test')
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']


        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()
