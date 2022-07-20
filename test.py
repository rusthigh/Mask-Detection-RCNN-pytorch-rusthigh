import os
import cv2
import argparse
import matplotlib.pyplot as plt
from mask_rcnn import segmentation_model, plot_masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='test_image.jpg', help='path to your test image')
    parser.add_argument('--labels', type=str, default='./my_dataset/labels.txt', help='path to labels list text file (labels.txt)')
    parser.add_argument('--model', type=str, default='./maskrcnn_saved_models/mask_rcnn_model.pt', help='path to saved model')

    args = parser.parse_args()
    
    IM