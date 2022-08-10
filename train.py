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
    parser.add_argument('--batchsize', type=int, default=4, help='batchsize')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = parser.parse_args()
    
    DATASET_PATH = args.data
    num_classes = args.num_classes
    num_epochs = args.num_epochs
    batchsize = args.batchsize
    workers = args.workers
    
    
    #DATASET
    # use our dataset and defined transformations
    dataset = maskrcnn_Dataset(DATASET_PATH, get_transform(train=True))
    dataset_test = maskrcnn_Dataset(DATASET_PATH, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-int(0.3*len(dataset))])
  