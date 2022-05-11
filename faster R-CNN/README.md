# Faster R-CNN
---
***百度网盘：https://pan.baidu.com/s/17ggibl6sFx-txYlP-kmYmw 提取码：06wi***
## 1. Performance
### 1.1 mAP

| Implementation | Data sets | mAP   |
| -------------- | --------- | ----- |
| origin paper   | VOC 2007  | 69.9  |
| origin paper   | VOC 07+12 | 73.2  |
| Mine           | VOC 2007  | 71.17 |
| Mine           | VOC 07+12 | 77.30 |
### 1.2 Speed

| Implementation | GPU      | FPS    |
| -------------- | -------- | ------ |
| origin paper   | K40      | 5 FPS  |
| Mine           | RTX 3060 | 10 FPS |

## 2. preparation

 - pytorch 1.11
 - CUDA 11.3
 - python 3.8
 - NVIDIA Geforce RTX3060  
 ```
 #install packages
 pip install -r requirements.txt --user
 ```
  
## 3. Train
  ### 3.1 Prepare data
  1.Download the training, validation, test data and VOCdevkit
``` javascript
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

2.Extract all of these tars into one directory named ***VOCdevikit***
```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

 3.It should have this basic structure
```
$VOCdevkit/
$VOCdevkit/VOCcode/
$VOCdevkit/VOC2007/
$VOCdevkit/VOC2012
```
4.If you want to train on VOC 07+12, run
`python voc.py`
If you just want to train on VOC 2007, test or inference, ignore it.

5.modify `voc_data_dir=/path/VOCdevkit/` cfg item in `utils/config.py`
### 3.2 begin training
```python train.py train```

Some Key arguments:
- `--plot-every=n`:visualize prediction, loss etc every n iterations
- `--use-drop`:use dropout in RoI head, default False
- `--use-Adam`:use Adam instead of SGD
- `load-path`:pretrained model path, default None
### 3.3 Visualize  
```
cd plot
tensorboard --logdir=./
```
The loss and mAP curve have already existed in dir`plot`, you can cheak it directly
## 4. Test
1. modify `load_path` in test.py
2. `python test.py test`
3.  If you want to visualize RPN proposal box, refer 3.3

## 5. Inference
If you want to detect on your own image, run the following command
```python inference.py --img_dir /path/to/image --save_dir /path/to/save/image```
## 6. Reference
The code if based on https://github.com/chenyuntc/simple-faster-rcnn-pytorch

 

