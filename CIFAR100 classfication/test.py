import argparse
import numpy as np
from tqdm import tqdm

import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from model.resnet import ResNet18
from model.wide_resnet import WideResNet


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    correct_top5 = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        _, pred_top5 = pred.topk(5, 1, True, True)
        label_resize = labels.view(-1, 1)
        pred = torch.max(pred.data, 1)[1]

        total += labels.size(0)
        correct += (pred == labels).sum().item()
        correct_top5 += torch.eq(pred_top5, label_resize).sum().float().item()

    val_acc = correct / total
    val_acc_top5 = correct_top5 / total
    return val_acc, val_acc_top5


if __name__ == '__main__':
    model_options = ['resnet18', 'wideresnet']
    dataset_options = ['cifar10', 'cifar100', 'svhn']

    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--dataset', '-d', default='cifar100',
                        choices=dataset_options)
    parser.add_argument('--model', '-a', default='resnet18',
                        choices=model_options)
    parser.add_argument('--weight_path', type=str, default=None,
                        help='path of the weight to test')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True  # Should make training should go faster for large models

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    test_id = args.dataset + '_' + args.model

    print(args)

    # Image Preprocessing
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)


    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    num_classes = 100

    test_dataset = datasets.CIFAR100(root='data/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)

    # Data Loader (Input Pipeline)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)


    if args.model == 'resnet18':
        cnn = ResNet18(num_classes=num_classes)
        cnn.load_state_dict(torch.load(args.weight_path))

    cnn = cnn.cuda()
    test_acc, test_acc_top5 = test(test_loader)
    tqdm.write('test_acc: %.4f \n test_acc_top5: %.4f' % (test_acc, test_acc_top5))



