
# YOLO v3
***百度网盘:https://pan.baidu.com/s/1X8uuRLME3BNsX5tNLyFlxQ 提取码：wqlx***
## 1. Result
### 1.1 mAP

| Model                | Train dataset      | Test dataset  | mAP   | Notes                                       |
| -------------------- | ------------------ | ------------- | ----- | ------------------------------------------- |
| YOLO v3<br/> 448-544 | VOC 07+12 trainval | VOC 2007 test | 78.85 | Baseline                                    |
| YOLO v3  <br/>*-544  | VOC 2007 trainval  | VOC 2007 test | 74.55 | +multi-scale  <br/>+mixup  <br/> +cosine lr |
| YOLO v3  <br/>*-544  | VOC 07+12 trainval | VOC 2007 test | 80.46 |                                             |
| YOLO v3<br/>*-544    | VOC 07+12 trainval | VOC 2007 test | 81.35 | +Gradient Accumulation                      |
| YOLO v3<br/>*-544    | VOC 07+12 trainval | VOC 2007 test | 83.18 | +multi-scale and flip test                  |
### 1.2 FPS

| Implementation            | GPU      | FPS  |
| ------------------------- | -------- | ---- |
| origin paper(YOLO v3-416) | Titan X  | 34.5 |
| origin paper(YOLO v3-608) | Titan X  | 20   |
| Mine(YOLO v3-544)         | RTX 3060 | 20   |
---
## 2. Environment
 - pytorch 1.11
 - CUDA 11.3
 - python 3.8
 - NVIDIA Geforce RTX3060
```
#install packages
pip install -r requirements.txt --user
```
---
## 3. Dataset
- Download VOC dataset:[VOC 2012 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)，[VOC 2007 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar), [VOC2007 test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar). Put them in the same dir, modify `DATA_PATH` in `yolov3_config_voc.py`
- Convert data format:
  `python voc.py`
 ## 4. Train
  ### 4.1 Run the following command to start training
  ```
  python train.py --weight_path /path/weight file/
 ```
- Add `--resume` for resume training, it will load `last.pt`

### 4.2 Visualize
```
cd log
tensorboard --logdir=./
```
Loss and mAP curve have already existed in dir `log`, you can cheak it directly
## 5.Test
### 5.1 Evaluate mAP
```
python test.py --weight_path /path/weight file/ --eval
```
### 5.2 Detect your own image
modify `PROJECT_PATH` in `yolov3_config_voc.py` as the path you want to save the detected image, run
```
python test.py --visiual /path/to/image/
```
## 6. Code base
The code is based on https://github.com/Peterisfar/YOLOV3
