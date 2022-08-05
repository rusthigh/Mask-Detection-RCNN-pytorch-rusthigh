import os
import torch
import argparse

import utils.utils
from utils.engine import train_one_epoch, evaluate
from utils.dataset import maskrcnn_Dataset, get_transform
from utils.model import get_instance_segmentation_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='my_dataset', help='dataset path')
    parser.add_argument('--num_classes', type=int, default=11, help='number of classes (background as a class)')
    parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs')
    parser.a