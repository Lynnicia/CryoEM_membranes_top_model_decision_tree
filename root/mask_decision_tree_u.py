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
def run_model_pipeline_u(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_u, test_loader, test_image_dir):
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

    def split_instances(binary_mask):
        num_labels, labels = cv2.connectedComponents(binary_mask)
        masks = []
        for label_id in range(1, num_labels):
            masks.append((labels == label_id).astype(np.uint8) * 255)
        return masks

    model_path = MODEL_PATH_u
    input_folder = test_image_dir

    output_folder = os.path.join(TARGET_FOLDER, f"Results_{Model}-{Model_Electron_Dose}")
    os.makedirs(output_folder, exist_ok=True)


    image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + \
                  glob.glob(os.path.join(input_folder, "*.png"))

    total_bacteria = 0

    for img_path in image_files:

        base = os.path.splitext(os.path.basename(img_path))[0]
        save_img = os.path.join(output_folder, f"{base}_thickness.png")
        save_csv = os.path.join(output_folder, f"{base}_thickness.csv")
        save_angles_csv = os.path.join(output_folder, f"{base}_angles.csv")
        
        used_ims = set()

        print("Processing:", img_path)

        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]

        # Resize for model
        img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

        image_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        image_gray_float = image_gray / 255.0

        img_tensor = torch.tensor(image_gray_float, dtype=torch.float32)\
                        .unsqueeze(0).unsqueeze(0).to(device)

        # 🔥 MODEL PREDICTION
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.sigmoid(outputs)

        pred_masks = (probs > 0.5).cpu().numpy()[0]

        im_mask = (pred_masks[0] * 255).astype(np.uint8)
        om_mask = (pred_masks[1] * 255).astype(np.uint8)

        def split_instances(binary_mask):
            num_labels, labels = cv2.connectedComponents(binary_mask)
            masks = []
            for label_id in range(1, num_labels):
                masks.append((labels == label_id).astype(np.uint8) * 255)
            return masks

        im_mask = cv2.resize(im_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        om_mask = cv2.resize(om_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        im_masks = split_instances(im_mask)
        om_masks = split_instances(om_mask)

        for om_mask_bin in om_masks:

            best_overlap, best_im_idx = 0, None

            for j, im_mask_bin in enumerate(im_masks):
                if j in used_ims:
                    continue
                overlap = np.sum((om_mask_bin > 0) & (im_mask_bin > 0))
                if overlap > best_overlap:
                    best_overlap, best_im_idx = overlap, j

            # 🚨 FILTER
            if best_im_idx is None:
                continue

            # ✅ VALID bacterium
            total_bacteria += 1
            used_ims.add(best_im_idx)

        print(f"{img_path}: {total_bacteria} bacteria")

        # Skip if OM or IM missing
        if not om_masks or not im_masks:
            print(f"Skipping {img_path} (missing OM or IM contour)")
            continue

        # Save combined masks
        im_combined = np.maximum.reduce(im_masks)
        om_combined = np.maximum.reduce(om_masks)

        cv2.imwrite(os.path.join(output_folder, f"{base}_IM_mask.png"), im_combined)
        cv2.imwrite(os.path.join(output_folder, f"{base}_OM_mask.png"), om_combined)


    print("Found images:", len(image_files))

    print(image_files[:3])

    print(f"\nTotal predicted bacteria with U-Net across all images: {total_bacteria}")
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "total_bacteria"] = total_bacteria

    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")

    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_u, test_loader, test_image_dir