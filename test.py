import os
import cv2
import argparse
import matplotlib.pyplot as plt
from mask_rcnn import segmentation_model, plot_masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='test_image.jpg', help='path to your test image')
    parser.add_argument('--labels', type=str, 