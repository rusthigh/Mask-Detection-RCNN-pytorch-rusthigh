
import os
import numpy as np
import cv2
import torch
import torch.utils.data
import utils.transforms as T
from PIL import Image

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class maskrcnn_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "SegmentationObject"))))
        self.class_masks = list(sorted(os.listdir(os.path.join(root, "SegmentationClass"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "SegmentationObject", self.masks[idx])
        class_mask_path = os.path.join(self.root, "SegmentationClass", self.class_masks[idx])
        
        #read and convert image to RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance