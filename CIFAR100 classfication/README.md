---
title: CIFAR 100 classification task
renderNumberedHeading: true
grammar_cjkRuby: true
---



## Result and Usage
### Result
The result is based on ResNet18


| Method       | Top 1 accuracy | Top 5  accuracy |
| ------------ | -------------- | --------------- |
| baseline     | 78.02          | 92.96           |
| Cutout       | 78.51          | 93.95           |
| mixup        | 79.68          | 95.06           |
| cutout+mixup | 79.94          | 94.24           |
| CutMix       | 80.20          | 95.23           |
### Usage
 **train**
``` javascript
python train.py --data_augmentation
```
some key arguments

 - `--cutout`:use cutout augmentation. 
 - `--n_holes`:n holes per image for cutout. `default=1`
 - `--length`:length of holes for cutout. `default=8`
 - `--mixup`:use mixup augmentation
 - `--alpha`:alpha of mixup. `default=0.4`
 - `--cutmix`:use cutmix augmentation.
 - `--beta`:beta of cutmix. `default=1`
   
  **test**
  `python test.py --weight_path /path/weight_file/`

## 
  The code is based on https://github.com/uoguelph-mlrg/Cutout 
	 