# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Loading SAM3...")
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import utils
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label

import subprocess, sys

# Clean uninstall first
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'sam3', '-y'], capture_output=True)

# Fresh install
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-e', '/content/sam3', '--break-system-packages'],
    capture_output=True, text=True
)


# Verify
import importlib.util
spec = importlib.util.find_spec("sam3")
print("\nsam3 location:", spec.origin if spec else "NOT FOUND")

import importlib
import sam3
importlib.reload(sam3)
from sam3.model_builder import build_sam3_image_model
print("✓ SAM3 architecture loaded")