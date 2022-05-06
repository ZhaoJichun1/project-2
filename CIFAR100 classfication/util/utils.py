import numpy as np
import logging
import math
import os
import shutil

import torch
import torchvision
import torchvision.transforms as transforms


def mixup_data(x, y, alpha):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(img, label, beta):
    bts, _, h, w = img.size()
    lamb = np.random.beta(beta, beta)
    rat = np.sqrt(1.-lamb)
    w_len = w * rat
    h_len = h * rat
    y_center = np.random.randint(h)
    x_center = np.random.randint(w)
    y1 = np.clip(y_center - h_len // 2, 0, h).astype(int)
    y2 = np.clip(y_center + h_len // 2, 0, h).astype(int)
    x1 = np.clip(x_center - w_len // 2, 0, w).astype(int)
    x2 = np.clip(x_center + w_len // 2, 0, w).astype(int)
    idx = torch.randperm(bts)
    img[:, :, y1:y2, x1:x2] = img[idx, :, y1:y2, x1:x2]
    label_mix = label[idx]
    lamb = 1 - (y2-y1)*(x2-x1) / (h*w)
    return img, label, label_mix.cuda(), lamb


def cutmix_criterion(criterion, pred, y, y_mix, lamb):
    return lamb * criterion(pred, y) + (1 - lamb) * criterion(pred, y_mix)