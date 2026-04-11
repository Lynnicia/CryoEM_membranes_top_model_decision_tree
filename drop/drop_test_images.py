# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%cd /content/CryoEM_membranes_top_model_decision_tree

#from drop import drop_test_images
from drop.coco_bacteria_dataset import CocoBacteriaDataset
from torch.utils.data import DataLoader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

if custom_dataset == "YES":
    from drop import custom_test_images

import pandas as pd
import numpy as np
import cv2
import glob
import os
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

if custom_dataset == "NO":
    from drop import LD_test_images, ULD_test_images
    from drop.LD_test_images import TEST_MAP as LD_TEST_MAP
    from drop.ULD_test_images import TEST_MAP as ULD_TEST_MAP

    TEST_MAP = {
        **LD_TEST_MAP,
        **ULD_TEST_MAP,
    }

    print(TEST_MAP)

if custom_dataset == "YES":
    from drop import LD_custom_test_images, ULD_custom_test_images
    from drop.LD_custom_test_images import TEST_MAP as LD_TEST_MAP
    from drop.ULD_custom_test_images import TEST_MAP as ULD_TEST_MAP

    TEST_MAP = {
        **LD_TEST_MAP,
        **ULD_TEST_MAP,
    }

    print(TEST_MAP)


