# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def run_model_metrics_pipeline_s3(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_s3, test_img_folder, test_ann_path):
    # ── Standard library ──────────────────────────────────────────────────────
    import json
    import os
    import glob
    import scipy.spatial
    import matplotlib.pyplot as plt

    # ── Numeric / CV ──────────────────────────────────────────────────────────
    import numpy as np
    import cv2

    # ── PyTorch ───────────────────────────────────────────────────────────────
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF

    # ── PIL ───────────────────────────────────────────────────────────────────
    from PIL import Image

    # ── COCO / Metrics ────────────────────────────────────────────────────────
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    import pandas as pd


    # ── SAM3 ──────────────────────────────────────────────────────────────────
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.train.data.sam3_image_dataset import (
        InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
    )
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    )

    # Helper functions
    import torch
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    from sam3.eval.postprocessors import PostProcessImage


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
    # Main loop for folder
    # =========================



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_sam3_image_model(
        bpe_path="/content/CryoEM_membranes_top_model_decision_tree/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        enable_segmentation=True,
        eval_mode=False,
        load_from_HF=False,
    )
    ckpt = torch.load(MODEL_PATH_s3, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()



    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=False,
    )

    # Build datapoint
    def make_datapoint(pil_image, prompts):
        dp = Datapoint(find_queries=[], images=[])
        w, h = pil_image.size
        dp.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]
        ids = {}

        for i, prompt in enumerate(prompts):
            dp.find_queries.append(FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=i,
                    original_image_id=i,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            ))

            ids[prompt] = i

        return dp, ids

    def move_to_cuda(obj):
        """Recursively move all tensors in a dataclass/object to CUDA."""
        if isinstance(obj, torch.Tensor):
            return obj.cuda()
        elif hasattr(obj, '__dataclass_fields__'):
            for field_name in obj.__dataclass_fields__:
                val = getattr(obj, field_name)
                setattr(obj, field_name, move_to_cuda(val))
            return obj
        elif isinstance(obj, list):
            return [move_to_cuda(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: move_to_cuda(v) for k, v in obj.items()}
        return obj


    input_folder = test_img_folder


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

    #-------------------------------------------------------------------------------
    # METRICS
    print(f"Metrics for {Model}-{Model_Electron_Dose}:")

