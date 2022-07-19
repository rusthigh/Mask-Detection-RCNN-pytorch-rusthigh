import os
import cv2
import argparse
import matplotlib.pyplot as plt
from mask_rcnn import segmentation_model, plot_masks

if __name__ == '__main__':
    parser = argpar