# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Loading YOLOv11 and YOLO26...")

from ultralytics import YOLO

#### Monkey Patch when saving in NOLOSS #############
import ultralytics.data.loaders as loaders

# Step 1: Save the original YOLO normalization function
_orig = loaders.LoadTensor._single_check

# Step 2: Define a patch that skips extra normalization
def _patched(im, stride=32):
    return _orig(im, stride=stride)  # Do shape check only, no undo/redo of normalization

# Step 3: Apply the patch
loaders.LoadTensor._single_check = staticmethod(_patched)


print("✓ YOLOv11 and YOLO26 model architectures loaded")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Loading U-Net...")
########################### Define the U-NET architecture #############################
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


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contracting path (encoder)
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.enc5 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Expanding path (decoder)
        self.dec5 = self.conv_block(1024 + 512, 512)
        self.dec4 = self.conv_block(512 + 256, 256)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 +32, 32)

        # Final layer
        self.final = nn.Conv2d(32, 2, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.1),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))

    def forward(self, x):
        # Encoder (Downsampling with Max Pooling)
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        e5 = self.enc5(F.max_pool2d(e4, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e5, 2))

        # Decoder (Upsampling with Transpose Convolutions)
        d5 = self.dec5(torch.cat([F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True), e5], dim=1))
        d4 = self.dec4(torch.cat([F.interpolate(d5, scale_factor=2, mode='bilinear', align_corners=True),e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True), e1], dim=1))

        # Final output
        #out = torch.sigmoid(self.final(d1))  # Sigmoid activation for binary segmentation
        out = self.final(d1) # output model LOGITS!
        return out


########### Reading input Images and masks processing block #######################

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class CocoBacteriaDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=True):
        self.coco = COCO(annotation_file)  # Load COCO dataset annotations
        self.image_dir = image_dir  # Directory containing images
        self.transform = transform  # Transformations to apply
        self.image_ids = self.coco.getImgIds()  # List of image IDs in the dataset
        # Get category IDs for OM and IM.
        self.cat_ids = self.coco.getCatIds(catNms=['IM', 'OM'])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_metadata['file_name'])
        image = Image.open(img_path).convert("L")  # Load as grayscale

        # Convert image to tensor
        image = transforms.ToTensor()(image)  # Convert the image to tensor.ToTensor() is not simple conversion to tensor. It does PIL to Tensor + channel reordering + 0–255→0–1 scaling

        # Load multilabel segmentation masks
        h, w = img_metadata['height'], img_metadata['width']
        # Create a mask with 2 channels: [2, H, W]
        multi_mask = np.zeros((2, h, w), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        for ann in annotations:
            cat_id = ann['category_id']
            channel = 0 if cat_id == self.cat_ids[0] else 1
            segmentation = ann['segmentation']
            for poly in segmentation:
                poly = np.array(poly).reshape((int(len(poly) / 2), 2)).astype(np.int32)
                cv2.fillPoly(multi_mask[channel], [poly], color=1)

        # Convert to tensor [2, H, W]
        mask = torch.tensor(multi_mask, dtype=torch.float32)
        raw_image, raw_mask = image.clone(), mask.clone()

        #if self.transform:
            # Note: 320x320 resize should be consistent
            #image = TF.resize(image, (320, 320))
            #mask = TF.resize(mask, (320, 320))

        #    angle, translations, scale, shear = transforms.RandomAffine.get_params(
        #        degrees=(-180, 180),
        #        translate=(0.1, 0.1),
        #        scale_ranges=(0.9, 1.1),
        #        shears=(-5, 5),
        #        img_size=(320, 320))

        #    image = TF.affine(image, angle=angle, translate=translations, scale=scale, shear=shear)
        #    mask = TF.affine(mask, angle=angle, translate=translations, scale=scale, shear=shear)

        return raw_image, raw_mask, image, mask
print("✓ U-Net architecture loaded")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Loading Detectron2...")
########################## Define the Detectron 2 architecture #############################

# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions

# Properly install detectron2. (Please do not install twice in both ways)
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

print("✓ Detectron2 architecture loaded")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Loading SAM3...")
########################## Define the SAM3 architecture #############################

import os

os.chdir("/content/sam3")

import subprocess, sys

# Clean uninstall first
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'sam3', '-y'], capture_output=True)

# Fresh install
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-e', '/content/sam3', '--break-system-packages'],
    capture_output=True, text=True
)
print(result.stdout[-2000:])
print(result.stderr[-2000:])

# Verify
import importlib.util
spec = importlib.util.find_spec("sam3")
print("\nsam3 location:", spec.origin if spec else "NOT FOUND")

import importlib
import sam3
importlib.reload(sam3)
from sam3.model_builder import build_sam3_image_model
print("✓ SAM3 architecture loaded")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("✓ All model architectures loaded")