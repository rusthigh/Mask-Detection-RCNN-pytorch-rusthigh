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


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

class segmentation_model():
    def __init__(self, model_path, num_classes):
        self.model = get_instance_segmentation_model(num_classes).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def detect_masks(self,image,rgb_image):
        if not(rgb_image):
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(image)
        with torch.no_grad():
            prediction = self.model([img.to(device)