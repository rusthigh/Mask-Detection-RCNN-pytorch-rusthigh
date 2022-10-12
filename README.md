# Mask-Detection-RCNN-pytorch-rusthigh

This project represents a Pytorch implementation of [Mask-RCNN](https://arxiv.org/abs/1703.06870), adapted to work with the VOC dataset format. The model generates segmentation masks and their respective scores for each instance of an object within the image. This repository is based on the [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

![Network Structure](results/network_structure.png)

## Training

Collect and label your data with [labelme](https://github.com/wkentaro/labelme) and then export the VOC-format dataset from json files using [labelme2voc](https://github.com/wkentaro/labelme/tree/master/examples/instance_segmentation).

Prepare your dataset following this