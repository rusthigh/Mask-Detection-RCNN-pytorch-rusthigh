import random
import torch

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):