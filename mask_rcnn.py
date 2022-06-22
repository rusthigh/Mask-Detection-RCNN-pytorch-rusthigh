###############################################
# pytorch Mask-RCNN based on torchvision model
# Amirhossein Heydarian
###############################################

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as 