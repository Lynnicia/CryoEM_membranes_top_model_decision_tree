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
def run_model_metrics_pipeline_u(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_u, test_loader, test_image_dir, test_annotation_file):
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


    from pycocotools.coco import COCO
    from pycocotools import mask as coco_mask

    # Load COCO annotations once outside the loop
    coco_gt = COCO(test_annotation_file)

    # Build a lookup: filename -> image_id
    filename_to_id = {img['file_name']: img['id'] for img in coco_gt.imgs.values()}

    # Category mapping — adjust names to match your COCO JSON
    CATEGORY_NAME_TO_ID = {cat['name']: cat['id'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    IM_CAT_ID = CATEGORY_NAME_TO_ID['IM']  # or whatever your class is named
    OM_CAT_ID = CATEGORY_NAME_TO_ID['OM']

    def coco_anns_to_mask(coco, image_id, cat_id, h, w):
        """Convert COCO polygon/RLE annotations to a binary mask."""
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=[cat_id])
        anns = coco.loadAnns(ann_ids)

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            rle = coco_mask.frPyObjects(ann['segmentation'], h, w)
            m = coco_mask.decode(rle)  # [H, W, N] or [H, W]
            if m.ndim == 3:
                m = m.max(axis=2)      # merge multiple polygons
            mask



    # Define IOU
    def calculate_iou(pred_mask, true_mask):
        """
        Args:
            pred_mask (np.array): shape [2, H, W] (0 or 1)
            true_mask (np.array): shape [2, H, W] (0 or 1)
        """
        ious = []
        # Loop through each channel (0=IM, 1=OM)
        for c in range(pred_mask.shape[0]):
            p = pred_mask[c]
            t = true_mask[c]

            # --- Your original logic applied to one channel ---
            intersection = np.logical_and(t, p).sum()
            union = np.logical_or(t, p).sum()
            iou = intersection / (union + 1e-6)
            ious.append(iou)
        return np.mean(ious)  # Return the average across both membranes

    # Define DICE
    def calculate_dice(pred_mask, true_mask):
        """
        Args:
            pred_mask (np.array): shape [2, H, W] (0 or 1)
            true_mask (np.array): shape [2, H, W] (0 or 1)
        """
        dices = []
        # Loop through each channel (0=IM, 1=OM)
        for c in range(pred_mask.shape[0]):
            p = pred_mask[c]
            t = true_mask[c]

            # --- Your original logic applied to one channel ---
            intersection = np.logical_and(t, p).sum()
            dice = (2 * intersection) / (p.sum() + t.sum() + 1e-6)
            dices.append(dice)

        return np.mean(dices)

    #Define Precision_Recall_F1

    def calculate_precision_recall_f1(pred_mask, true_mask):
        """
        Args:
            pred_mask (np.array): shape [2, H, W] (0 or 1)
            true_mask (np.array): shape [2, H, W] (0 or 1)
        """
        precisions, recalls, f1s = [], [], []

        for c in range(pred_mask.shape[0]):
            p = pred_mask[c].flatten()
            t = true_mask[c].flatten()

            # True Positive: Both are 1
            tp = np.sum((p == 1) & (t == 1))
            # False Positive: Predicted 1, but actually 0
            fp = np.sum((p == 1) & (t == 0))
            # False Negative: Predicted 0, but actually 1
            fn = np.sum((p == 0) & (t == 1))

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return np.mean(precisions), np.mean(recalls), np.mean(f1s)


    def calculate_auprc(pred_probs, true_mask):
        """
        Args:
            pred_probs (np.array): shape [C, H, W] (continuous 0–1 values)
            true_mask (np.array): shape [C, H, W] (0 or 1)
        """
        auprcs = []

        for ch in [0, 1]:

            # Make sure arrays are 1D
            y_true = np.array(all_true[ch]).flatten()
            y_prob = np.array(all_probs[ch]).flatten()

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            auprc = auc(recall, precision)

            auprcs.append(auprc)

            print(f"Class {ch} AUPRC: {auprc:.4f}")

        return np.mean(auprcs)


    # METRICS

    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------


    from sklearn.metrics import precision_recall_curve, average_precision_score
    import pandas as pd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)  # Ensure your UNet final layer is nn.Conv2d(..., 2, ...)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    all_probs = {0: [], 1: []}
    all_true  = {0: [], 1: []}



    def best_threshold(y_true, y_prob):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return thresholds[np.argmax(f1[:-1])]

    def compute_metrics_at_best_f1(all_true, all_probs):
        results = {}
        labels = {0: "IM", 1: "OM"}

        for ch in [0, 1]:
            y_true = all_true[ch]
            y_prob = all_probs[ch]
            thresh = best_threshold(y_true, y_prob)
            y_pred = (y_prob >= thresh).astype(np.uint8)

            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            ap50      = average_precision_score(y_true, y_prob)

            results[labels[ch]] = {
                "Mask Precision": (precision),
                "Mask Recall":    (recall),
                "mAP50":          (ap50),
                "F1 Score":       (f1),
                "Best Threshold": (float(thresh)),
            }

        # Overall
        y_true_all = np.concatenate([all_true[0], all_true[1]])
        y_prob_all  = np.concatenate([all_probs[0], all_probs[1]])
        thresh = best_threshold(y_true_all, y_prob_all)
        y_pred_all = (y_prob_all >= thresh).astype(np.uint8)

        tp = np.sum((y_pred_all == 1) & (y_true_all == 1))
        fp = np.sum((y_pred_all == 1) & (y_true_all == 0))
        fn = np.sum((y_pred_all == 0) & (y_true_all == 1))

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        ap50      = average_precision_score(y_true_all, y_prob_all)

        results["all"] = {
            "Mask Precision": (precision),
            "Mask Recall":    (recall),
            "mAP50":          (ap50),
            "F1 Score":       (f1),
            "Best Threshold": (float(thresh)),
        }

        return pd.DataFrame(results).T


    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)
            probs  = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            true_np  = masks.cpu().numpy()

            for ch in [0, 1]:
                all_probs[ch].append(probs_np[:, ch].ravel())
                all_true[ch].append(true_np[:, ch].ravel())

    for ch in [0, 1]:
        all_probs[ch] = np.concatenate(all_probs[ch])
        all_true[ch]  = np.concatenate(all_true[ch])

    print("all_true[0] positives:", all_true[0].sum())
    print("all_true[1] positives:", all_true[1].sum())

    # Now compute metrics
    metrics_df = compute_metrics_at_best_f1(all_true, all_probs)
    print("\n── U-Net Validation Metrics @Best F1 ──")
    print(metrics_df.to_string())

    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "O.Mask_Precision"] = metrics_df.loc["all", "Mask Precision"]
    df.loc[all_condition, "O.Mask_Recall"]    = metrics_df.loc["all", "Mask Recall"]
    df.loc[all_condition, "O.mAP50"]          = np.nan
    df.loc[all_condition, "O.F1-Score"]       = metrics_df.loc["all", "F1 Score"]
    df.loc[im_condition, "O.Mask_Precision"] = metrics_df.loc["IM", "Mask Precision"]
    df.loc[im_condition, "O.Mask_Recall"]    = metrics_df.loc["IM", "Mask Recall"]
    df.loc[im_condition, "O.mAP50"]          = np.nan
    df.loc[im_condition, "O.F1-Score"]       = metrics_df.loc["IM", "F1 Score"]
    df.loc[om_condition, "O.Mask_Precision"] = metrics_df.loc["OM", "Mask Precision"]
    df.loc[om_condition, "O.Mask_Recall"]    = metrics_df.loc["OM", "Mask Recall"]
    df.loc[om_condition, "O.mAP50"]          = np.nan
    df.loc[om_condition, "O.F1-Score"]       = metrics_df.loc["OM", "F1 Score"]

    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")


    # METRICS - U-Net

    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------


    # overall pixel segmentation metrics


    print("\n=== Overall Pixel Segmentation Metrics ===\n")

    model.eval()

    iou_scores = []
    dice_scores = []
    precisions, recalls, f1s = [], [], []

    with torch.no_grad():
        for images, masks in test_loader:

            images = images.to(device)
            masks = masks.to(device)   # shape: [B, 2, H, W]

            outputs = model(images)    # shape: [B, 2, H, W]

            # Convert logits → probabilities
            probs = torch.sigmoid(outputs)

            # Threshold to binary
            preds = (probs > 0.5).float()

            # Move to CPU numpy
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # Loop per image in batch
            for b in range(preds_np.shape[0]):

                pred_mask = preds_np[b]      # [2, H, W]
                true_mask = masks_np[b]      # [2, H, W]

                iou = calculate_iou(pred_mask, true_mask)
                dice = calculate_dice(pred_mask, true_mask)
                precision, recall, f1 = calculate_precision_recall_f1(pred_mask, true_mask)

                iou_scores.append(iou)
                dice_scores.append(dice)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)




    from sklearn.metrics import precision_recall_curve, auc

    model.eval()

    all_true = [[], []]
    all_probs = [[], []]

    with torch.no_grad():
        for images, masks in test_loader:

            images = images.to(device)
            masks = masks.to(device)      # shape [B, 2, H, W]

            outputs = model(images)       # logits [B, 2, H, W]

            # Convert logits → probabilities
            probs = torch.sigmoid(outputs)

            # Move to CPU
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            B = probs_np.shape[0]

            for b in range(B):

                true_mask = masks_np[b]      # [2, H, W]
                pred_probs = probs_np[b]     # [2, H, W]

                for c in range(2):

                    all_true[c].extend(true_mask[c].flatten())
                    all_probs[c].extend(pred_probs[c].flatten())


    auprcs = []

    for c in range(2):

        y_true = np.array(all_true[c])
        y_prob = np.array(all_probs[c])

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall, precision)

        auprcs.append(auprc)


    # ✅ Now compute dataset average
    print("Mean IoU:", np.mean(iou_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.IOU"] = np.mean(iou_scores)
    print("Mean Dice:", np.mean(dice_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.Dice"] = np.mean(dice_scores)
    print("Mean Precision:", np.mean(precisions))
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.Mask_Precision"] = np.mean(precisions)
    print("Mean Recall:", np.mean(recalls))
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.Mask_Recall"] = np.mean(recalls)
    print("Mean F1:", np.mean(f1s))
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.F1-Score"] = np.mean(f1s)
    print("Mean AUPRC:", np.mean(auprcs))
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.AUPRC"] = np.mean(auprcs)


    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------

    # per class (except AUPRC above)
    # ---- U-Net inference loop ----

    import torch
    import torch.nn.functional as F

    model.eval()

    # IM storage
    iou_im_scores = []
    dice_im_scores = []
    prec_im_scores = []
    rec_im_scores = []
    f1_im_scores = []

    # OM storage
    iou_om_scores = []
    dice_om_scores = []
    prec_om_scores = []
    rec_om_scores = []
    f1_om_scores = []

    THRESHOLD = 0.5  # binarization threshold for sigmoid output

    with torch.no_grad():
        for images, masks in test_loader:

            images = images.to(device)
            masks = masks.to(device)       # [B, 2, H, W]
            outputs = model(images)        # [B, 2, H, W]

            probs = torch.sigmoid(outputs).cpu().numpy()   # [B, 2, H, W]
            masks_np = masks.cpu().numpy()                 # [B, 2, H, W]

            # Loop over each image in the batch
            for b in range(images.shape[0]):

                # --- Ground truth from loader (replaces COCO/YOLO block) ---
                gt_mask_im = masks_np[b, 0]   # [H, W]
                gt_mask_om = masks_np[b, 1]   # [H, W]

                # --- Prediction ---
                pred_mask_im = probs[b, 0] > THRESHOLD   # [H, W]
                pred_mask_om = probs[b, 1] > THRESHOLD   # [H, W]

                # --- Boolean conversion + OM cleaning ---
                gt_mask_im = gt_mask_im > 0
                gt_mask_om = gt_mask_om > 0

                # --- IM metrics ---
                true_im = np.stack([gt_mask_im])     # [1, H, W]
                pred_im = np.stack([pred_mask_im])

                iou_im_scores.append(calculate_iou(pred_im, true_im))
                dice_im_scores.append(calculate_dice(pred_im, true_im))
                p, r, f = calculate_precision_recall_f1(pred_im, true_im)
                prec_im_scores.append(p)
                rec_im_scores.append(r)
                f1_im_scores.append(f)

                # --- OM metrics ---
                true_om = np.stack([gt_mask_om])   # [1, H, W]
                pred_om = np.stack([pred_mask_om])

                iou_om_scores.append(calculate_iou(pred_om, true_om))
                dice_om_scores.append(calculate_dice(pred_om, true_om))
                p, r, f = calculate_precision_recall_f1(pred_om, true_om)
                prec_om_scores.append(p)
                rec_om_scores.append(r)
                f1_om_scores.append(f)



    print("\n=== Per-Class Pixel Segmentation Metrics ===\n")

    # ✅ Now compute dataset average

    print("IM IoU:", np.mean(iou_im_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[im_condition, "P.IOU"] = np.mean(iou_im_scores)
    print("IM Dice:", np.mean(dice_im_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[im_condition, "P.Dice"] = np.mean(dice_im_scores)
    print("IM Precision:", np.mean(prec_im_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[im_condition, "P.Mask_Precision"] = np.mean(prec_im_scores)
    print("IM Recall:", np.mean(rec_im_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[im_condition, "P.Mask_Recall"] = np.mean(rec_im_scores)
    print("IM F1:", np.mean(f1_im_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[im_condition, "P.F1-Score"] = np.mean(f1_im_scores)

    # ✅ Now compute dataset average
    print("\nOM IoU:", np.mean(iou_om_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[om_condition, "P.IOU"] = np.mean(iou_om_scores)
    print("OM Dice:", np.mean(dice_om_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[om_condition, "P.Dice"] = np.mean(dice_om_scores)
    print("OM Precision:", np.mean(prec_om_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[om_condition, "P.Mask_Precision"] = np.mean(prec_om_scores)
    print("OM Recall:", np.mean(rec_om_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[om_condition, "P.Mask_Recall"] = np.mean(rec_om_scores)
    print("OM F1:", np.mean(f1_om_scores))
    """ //#####|tree_root|#####\\ """
    df.loc[om_condition, "P.F1-Score"] = np.mean(f1_om_scores)

    auprcs = []

    for c in range(2):
        num_class_condition = (
            (df["Model"] == Model) &
            (df["Model_Electron_Dose"] == Model_Electron_Dose) &
            (df["Model_Image_Size"] == Model_Image_Size) &
            (df["Test_Electron_Dose"] == Test_Electron_Dose) &
            (df["Test_Image_Size"] == Test_Image_Size) &
            (df["Channel"].fillna(-1).astype(int) == c)
        )

        y_true = np.array(all_true[c])
        y_prob = np.array(all_probs[c])

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall, precision)

        auprcs.append(auprc)

        print(f"Class {c} AUPRC: {auprc:.4f}")
        """ //#####|tree_root|#####\\ """
        df.loc[num_class_condition, "P.AUPRC"] = auprc
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------


    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")



    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_u, test_loader, test_image_dir, test_annotation_file