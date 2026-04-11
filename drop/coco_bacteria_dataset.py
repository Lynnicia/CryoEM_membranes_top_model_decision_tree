import os
import numpy as np
import cv2
import torch

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as transforms


class CocoBacteriaDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir

        self.image_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds(catNms=['IM', 'OM'])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img_id = self.image_ids[idx]

        img_metadata = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(
            self.image_dir,
            img_metadata['file_name']
        )

        image = Image.open(img_path).convert("L")
        image = transforms.ToTensor()(image)

        if self.transform:
            image = self.transform(image)

        h, w = img_metadata['height'], img_metadata['width']

        multi_mask = np.zeros((2, h, w), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        for ann in annotations:

            cat_id = ann['category_id']

            channel = 0 if cat_id == self.cat_ids[0] else 1

            for poly in ann['segmentation']:

                poly = np.array(poly).reshape(-1, 2).astype(np.int32)

                cv2.fillPoly(
                    multi_mask[channel],
                    [poly],
                    color=1
                )

        mask = torch.tensor(
            multi_mask,
            dtype=torch.float32
        )

        return image, mask