# Dense Events Grounding in Video (AAAI 2021 oral)

## Introduction
This is a pytorch implementation of **Dense Events Propagation Network (DepNet)**  on ActivityNet Captions for the AAAI 2021 oral paper "Dense Events Grounding in Video" .

- [pdf](https://peijunbao.github.io/files/PeijunBao_AAAI21_DenseEventsGrounding.pdf)
- [code](https://github.com/baopj/DenseEventsGrounding/tree/main/DepNet_ANet_Release)

## Dataset
Please download the visual features from the official website of ActivityNet: [Official C3D Feature](http://activity-net.org/download.html). And you can download preprocessed annotation files [here](https://github.com/baopj/DenseEventsGrounding/blob/main/DepNet_ANet_Release/files_/acnet_annot.zip). 

## Prerequisites
- python 3.5
- pytorch 1.4.0
- torchtext
- easydict
- terminaltables

## Training
Use the following commands for training:
```
cd moment_localization && export CUDA_VISIBLE_DEVICES=0
python dense_train.py --verbose --cfg ../experiments/dense_activitynet/acnet.yaml
```
You may get better results than that reported in our paper thanks to the code updates.


### Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{densegrounding,
  author    = {Peijun Bao and
               Qian Zheng and
               Yadong Mu},
  title     = {Dense Events Grounding in Video,
  booktitle = {AAAI},
  year      = {2021}
}
```
