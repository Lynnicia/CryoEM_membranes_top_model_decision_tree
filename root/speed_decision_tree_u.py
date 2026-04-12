# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


# U-Net
def run_model_speed_pipeline_u(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_u, test_loader, test_image_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import scipy.spatial
    import pandas as pd
    import os

    import glob
    import cv2
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize


    MODEL_PATH = MODEL_PATH_u
    TARGET_SIZE = Test_Image_Size

    csv_path = "/content/CryoEM_membranes_top_model_decision_tree/top_model_table.csv"
    df = pd.read_csv(csv_path)

    #convert Image size values to integers
    df["Model_Image_Size"] = pd.to_numeric(df["Model_Image_Size"], errors="coerce").astype("Int64")
    df["Test_Image_Size"] = pd.to_numeric(df["Test_Image_Size"], errors="coerce").astype("Int64")

    all_condition = (
        (df["Model"] == Model) &
        (df["Model_Electron_Dose"] == Model_Electron_Dose) &
        (df["Model_Image_Size"] == Model_Image_Size) &
        (df["Test_Electron_Dose"] == Test_Electron_Dose) &
        (df["Test_Image_Size"] == Test_Image_Size) &
        (df["Class"] == "All")
    )



    im_condition = (
        (df["Model"] == Model) &
        (df["Model_Electron_Dose"] == Model_Electron_Dose) &
        (df["Model_Image_Size"] == Model_Image_Size) &
        (df["Test_Electron_Dose"] == Test_Electron_Dose) &
        (df["Test_Image_Size"] == Test_Image_Size) &
        (df["Class"] == "IM")
    )

    om_condition = (
        (df["Model"] == Model) &
        (df["Model_Electron_Dose"] == Model_Electron_Dose) &
        (df["Model_Image_Size"] == Model_Image_Size) &
        (df["Test_Electron_Dose"] == Test_Electron_Dose) &
        (df["Test_Image_Size"] == Test_Image_Size) &
        (df["Class"] == "OM")
    )

    # =========================
    # Loop through folder
    # =========================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)  # Ensure your UNet final layer is nn.Conv2d(..., 2, ...)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    torch.backends.cudnn.benchmark = True

    # -------------------------
    # WARMUP
    # -------------------------
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            _ = model(images)
            if i == 9:   # 10 warmup batches
                break

    torch.cuda.synchronize()

    # -------------------------
    # TIMING
    # -------------------------
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    total_time = 0
    total_images = 0


    with torch.no_grad():
        for images, masks in test_loader:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            starter.record()
            outputs = model(images)
            ender.record()

            torch.cuda.synchronize()
            batch_time = starter.elapsed_time(ender)  # ms

            total_time += batch_time
            total_images += images.size(0)

    avg_time_per_batch = total_time / len(test_loader)
    avg_time_per_image = total_time / total_images
    fps = 1000 / avg_time_per_image

    print(f"Avg batch time per batch: {avg_time_per_batch:.2f} ms")
    print(f"Avg image time per image: {avg_time_per_image:.2f} ms")
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "Avg Time_Image"] = avg_time_per_image
    print(f"FPS: {fps:.2f}")
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "FPS"] = fps
    #-----------------------------------------------

    import torch

    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
        print("GPU Count:", torch.cuda.device_count())
        print("Current Device:", torch.cuda.current_device())
    else:
        print("No CUDA GPU detected.")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)

        print("Total Memory (GB):", props.total_memory / 1e9)
        print("Multiprocessors:", props.multi_processor_count)
        print("Compute Capability:", f"{props.major}.{props.minor}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory (GB): {props.total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Compute Capability: {props.major}.{props.minor}")

    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")

    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_u, test_loader, test_image_dir