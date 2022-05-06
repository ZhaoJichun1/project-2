import os
import numpy as np
import random
import shutil
from utils.config import opt


data_dir = opt.voc_data_dir
path_07 = os.path.join(data_dir, 'VOC2007')
path_12 = os.path.join(data_dir, 'VOC2012')
for root, dir, files in os.walk(os.path.join(path_12, 'Annotations')):
    for file_name in files:
        shutil.move(os.path.join(root, file_name), os.path.join(os.path.join(path_07, 'Annotations')))

for root, dir, files in os.walk(os.path.join(path_12, 'JPEGImages')):
    for file_name in files:
        shutil.move(os.path.join(root, file_name), os.path.join(os.path.join(path_07, 'JPEGImages')))

with open(os.path.join(path_12, 'ImageSets', 'Main', 'trainval.txt'), "r") as f:
    data = f.readlines()

f = open(os.path.join(path_07, 'ImageSets', 'Main', 'trainval.txt'), "a")

for x in data:
    f.write(x)
f.close()