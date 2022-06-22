###############################################
# pytorch Mask-RCNN based on torchvision model
# Amirhossein Heydarian
###############################################

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
import cv2
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 2
thickness = 3


def get_instance_segmentation_model(