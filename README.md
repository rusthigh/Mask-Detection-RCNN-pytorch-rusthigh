# Mask-Detection-RCNN-pytorch-rusthigh

This project represents a Pytorch implementation of [Mask-RCNN](https://arxiv.org/abs/1703.06870), adapted to work with the VOC dataset format. The model generates segmentation masks and their respective scores for each instance of an object within the image. This repository is based on the [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

![Network Structure](results/network_structure.png)

## Training

Collect and label your data with [labelme](https://github.com/wkentaro/labelme) and then export the VOC-format dataset from json files using [labelme2voc](https://github.com/wkentaro/labelme/tree/master/examples/instance_segmentation).

Prepare your dataset following this format:

```
my_dataset
      ├── labels.txt
      │
      ├── JPEGImages
      │       ├── image1.jpg
      │       └── image2.jpg
      │
      ├── SegmentationObject
      │       ├── image1.png
      │       └── image2.png
      │
      └── SegmentationClass
              ├── image1.png
              └── image2.png
```

Once your repository is cloned and you have your `my_dataset` folder ready within the `Mask-Detection-RCNN-pytorch-rusthigh` directory, you can execute this command to start training:

```
$ python3 train.py --data my_datase