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

    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------



    with open(test_ann_path) as f:
        test_ann = json.load(f)

    cat_name_to_id = {c["name"]: c["id"] for c in test_ann["categories"]}
    prompt_to_ch   = {"IM": 0, "OM": 1}
    coco_gt        = COCO(test_ann_path)

    # ── Collect predictions ───────────────────────────────────────────────────
    predictions  = []   # for COCO instance eval
    all_true     = {0: [], 1: []}  # for pixel eval
    all_probs    = {0: [], 1: []}

    for img_info in test_ann["images"]:
        img_path  = os.path.join(test_img_folder, img_info["file_name"])
        pil_image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = pil_image.size

        # GT masks
        gt = {0: np.zeros((orig_height, orig_width), dtype=np.uint8),
              1: np.zeros((orig_height, orig_width), dtype=np.uint8)}
        for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_info["id"])):
            cat_name = next(c["name"] for c in test_ann["categories"] if c["id"] == ann["category_id"])
            if cat_name not in prompt_to_ch:
                continue
            ch = prompt_to_ch[cat_name]
            m = coco_gt.annToMask(ann)
            if m.shape != (orig_height, orig_width):
                m = cv2.resize(m, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
            gt[ch] = np.maximum(gt[ch], m)

        # SAM3 inference
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                dp, ids = make_datapoint(pil_image, ["IM", "OM"])
                dp = transform(dp)
                batch = collate([dp], dict_key="dummy")["dummy"]
                batch = copy_data_to_device(batch, device, non_blocking=True)
                output = model(batch)
                pp = PostProcessImage(
                    max_dets_per_img=-1, iou_type="segm",
                    use_original_sizes_box=True, use_original_sizes_mask=True,
                    convert_mask_to_rle=False, detection_threshold=0.001, to_cpu=True,
                )
                results = pp.process_results(output, batch.find_metadatas)
        del batch, output
        torch.cuda.empty_cache()

        # Build prob maps + COCO predictions
        for prompt, ch in prompt_to_ch.items():
            cat_id = cat_name_to_id[prompt]
            prob_map = np.zeros((orig_height, orig_width), dtype=np.float32)

            for mask, score in zip(results[ids[prompt]]["masks"], results[ids[prompt]]["scores"]):
                m = mask.numpy().squeeze().astype(np.float32)
                if m.shape != (orig_height, orig_width):
                    m = cv2.resize(m, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
                prob_map = np.maximum(prob_map, m * float(score))

                # Add to COCO predictions
                binary = (m > 0.5).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(binary))
                rle["counts"] = rle["counts"].decode("utf-8")
                predictions.append({
                    "image_id":    img_info["id"],
                    "category_id": cat_id,
                    "segmentation": rle,
                    "score":       float(score),
                })

            all_true[ch].append(gt[ch].ravel())
            all_probs[ch].append(prob_map.ravel())

        print(f"  {img_info['file_name']} done")

    # Concatenate pixel arrays
    for ch in [0, 1]:
        all_true[ch]  = np.concatenate(all_true[ch])
        all_probs[ch] = np.concatenate(all_probs[ch])

    # ── COCO instance eval @IoU50 ─────────────────────────────────────────────
    coco_dt   = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract per-class AP@IoU50
    coco_ap50 = {}
    for cat_name, cat_id in cat_name_to_id.items():
        if cat_name not in prompt_to_ch:
            continue
        cat_idx = coco_eval.params.catIds.index(cat_id)
        coco_ap50[cat_name] = round(float(coco_eval.eval["precision"][0, :, cat_idx, 0, 2].mean()), 3)
    im_idx  = coco_eval.params.catIds.index(cat_name_to_id["IM"])
    om_idx  = coco_eval.params.catIds.index(cat_name_to_id["OM"])
    coco_ap50["all"] = round(float(
        np.mean([
            coco_eval.eval["precision"][0, :, im_idx, 0, 2].mean(),
            coco_eval.eval["precision"][0, :, om_idx, 0, 2].mean(),
        ])
    ), 3)

    # ── Pixel metrics @best threshold ────────────────────────────────────────
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
            auprc     = average_precision_score(y_true, y_prob)

            results[labels[ch]] = {
                "Mask Precision":    (precision),
                "Mask Recall":       (recall),
                "Pixel AUPRC":       (auprc),
                "Instance mAP50":    coco_ap50[labels[ch]],
                "F1 Score":          (f1),
                "Best Threshold":    (float(thresh)),
            }

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
        auprc     = average_precision_score(y_true_all, y_prob_all)

        results["all"] = {
            "Mask Precision":    (precision),
            "Mask Recall":       (recall),
            "Pixel AUPRC":       (auprc),
            "Instance mAP50":    coco_ap50["all"],
            "F1 Score":          (f1),
            "Best Threshold":    (float(thresh)),
        }

        return pd.DataFrame(results).T

    metrics_df = compute_metrics_at_best_f1(all_true, all_probs)
    print("\n── SAM3 Validation Metrics ──")
    print(metrics_df.to_string())



    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "O.Mask_Precision"] = metrics_df.loc["all", "Mask Precision"]
    df.loc[all_condition, "O.Mask_Recall"]    = metrics_df.loc["all", "Mask Recall"]
    df.loc[all_condition, "O.mAP50"]          = metrics_df.loc["all", "Instance mAP50"]
    df.loc[all_condition, "O.F1-Score"]       = metrics_df.loc["all", "F1 Score"]
    df.loc[im_condition, "O.Mask_Precision"] = metrics_df.loc["IM", "Mask Precision"]
    df.loc[im_condition, "O.Mask_Recall"]    = metrics_df.loc["IM", "Mask Recall"]
    df.loc[im_condition, "O.mAP50"]          = metrics_df.loc["IM", "Instance mAP50"]
    df.loc[im_condition, "O.F1-Score"]       = metrics_df.loc["IM", "F1 Score"]
    df.loc[om_condition, "O.Mask_Precision"] = metrics_df.loc["OM", "Mask Precision"]
    df.loc[om_condition, "O.Mask_Recall"]    = metrics_df.loc["OM", "Mask Recall"]
    df.loc[om_condition, "O.mAP50"]          = metrics_df.loc["OM", "Instance mAP50"]
    df.loc[om_condition, "O.F1-Score"]       = metrics_df.loc["OM", "F1 Score"]



    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------


    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")

    # METRICS - U-Net

    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------

    # overall pixel segmentation metrics


    with open(test_ann_path) as f:
        test_ann = json.load(f)

    coco_gt = COCO(test_ann_path)
    cat_name_to_id = {c["name"]: c["id"] for c in test_ann["categories"]}
    prompt_to_ch   = {"IM": 0, "OM": 1}

    # Per-image accumulators (mirrors U-Net loop)
    all_true  = {0: [], 1: []}
    all_probs = {0: [], 1: []}

    iou_scores, dice_scores, precisions, recalls, f1s = [], [], [], [], []
    iou_im_scores,  dice_im_scores,  prec_im_scores,  rec_im_scores,  f1_im_scores  = [], [], [], [], []
    iou_om_scores,  dice_om_scores,  prec_om_scores,  rec_om_scores,  f1_om_scores  = [], [], [], [], []

    THRESHOLD = 0.5

    for img_info in test_ann["images"]:
        img_path  = os.path.join(test_img_folder, img_info["file_name"])
        pil_image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = pil_image.size

        # ── Ground truth masks from COCO ──
        gt = {0: np.zeros((orig_height, orig_width), dtype=np.uint8),
              1: np.zeros((orig_height, orig_width), dtype=np.uint8)}
        for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_info["id"])):
            cat_name = next(c["name"] for c in test_ann["categories"] if c["id"] == ann["category_id"])
            if cat_name not in prompt_to_ch:
                continue
            ch = prompt_to_ch[cat_name]
            m = coco_gt.annToMask(ann)
            if m.shape != (orig_height, orig_width):
                m = cv2.resize(m, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
            gt[ch] = np.maximum(gt[ch], m)

        # ── SAM3 inference ──
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                dp, ids = make_datapoint(pil_image, ["IM", "OM"])
                dp = transform(dp)
                batch = collate([dp], dict_key="dummy")["dummy"]
                batch = copy_data_to_device(batch, device, non_blocking=True)
                output = model(batch)
                pp = PostProcessImage(
                    max_dets_per_img=-1, iou_type="segm",
                    use_original_sizes_box=True, use_original_sizes_mask=True,
                    convert_mask_to_rle=False, detection_threshold=0.001, to_cpu=True,
                )
                results = pp.process_results(output, batch.find_metadatas)

        # ── Build prob maps (mirrors probs[b, ch]) ──
        prob = {}
        for prompt, ch in prompt_to_ch.items():
            p = np.zeros((orig_height, orig_width), dtype=np.float32)
            for mask, score in zip(results[ids[prompt]]["masks"], results[ids[prompt]]["scores"]):
                m = mask.numpy().squeeze().astype(np.float32)
                if m.shape != (orig_height, orig_width):
                    m = cv2.resize(m, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
                p = np.maximum(p, m * float(score))
            prob[ch] = p
            all_true[ch].append(gt[ch].ravel())
            all_probs[ch].append(p.ravel())

        # ── Compute per-image metrics (mirrors U-Net batch loop) ──
        pred_mask = np.stack([prob[0] > THRESHOLD, prob[1] > THRESHOLD]).astype(np.float32)  # [2,H,W]
        true_mask = np.stack([gt[0] > 0,           gt[1] > 0          ]).astype(np.float32)  # [2,H,W]

        iou_scores.append(calculate_iou(pred_mask, true_mask))
        dice_scores.append(calculate_dice(pred_mask, true_mask))
        p, r, f = calculate_precision_recall_f1(pred_mask, true_mask)
        precisions.append(p); recalls.append(r); f1s.append(f)

        # Per-class
        for ch, (iou_list, dice_list, prec_list, rec_list, f1_list) in enumerate([
            (iou_im_scores, dice_im_scores, prec_im_scores, rec_im_scores, f1_im_scores),
            (iou_om_scores, dice_om_scores, prec_om_scores, rec_om_scores, f1_om_scores),
        ]):
            pm = np.stack([pred_mask[ch]])
            tm = np.stack([true_mask[ch]])
            iou_list.append(calculate_iou(pm, tm))
            dice_list.append(calculate_dice(pm, tm))
            p, r, f = calculate_precision_recall_f1(pm, tm)
            prec_list.append(p); rec_list.append(r); f1_list.append(f)

        print(f"  {img_info['file_name']} done")

    # Concatenate for PR/AUPRC
    for ch in [0, 1]:
        all_true[ch]  = np.concatenate(all_true[ch])
        all_probs[ch] = np.concatenate(all_probs[ch])

    # ── Print results (identical to U-Net output block) ────────────────────────
    print("\n=== Overall Pixel Segmentation Metrics ===\n")
    print("Mean IoU:",        round(np.mean(iou_scores),3))
    print("Mean Dice:",       round(np.mean(dice_scores),3))
    print("Mean Precision:",  round(np.mean(precisions),3))
    print("Mean Recall:",     round(np.mean(recalls),3))
    print("Mean F1:",         round(np.mean(f1s),3))

    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.IOU"] = np.mean(iou_scores)
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.Dice"] = np.mean(dice_scores)
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.Mask_Precision"] = np.mean(precisions)
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.Mask_Recall"] = np.mean(recalls)
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.F1-Score"] = np.mean(f1s)


    auprcs = []
    for c in range(2):
        y_true = all_true[c]
        y_prob = all_probs[c]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall, precision)
        auprcs.append(auprc)
    print("Mean AUPRC:", round(np.mean(auprcs),3))
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.AUPRC"] = np.mean(auprcs)


    print("\n=== Per-Class Pixel Segmentation Metrics ===\n")
    print("IM IoU:",       round(np.mean(iou_im_scores),3))
    """ //#####|tree_root|#####\\ """
    df.loc[im_condition, "P.IOU"] = np.mean(iou_im_scores)
    print("IM Dice:",      round(np.mean(dice_im_scores),3))
    """ //#####|tree_root|#####\\ """
    df.loc[im_condition, "P.Dice"] = np.mean(dice_im_scores)
    print("IM Precision:", round(np.mean(prec_im_scores),3))
    """ //#####|tree_root|#####\\ """
    df.loc[im_condition, "P.Mask_Precision"] = np.mean(prec_im_scores)
    print("IM Recall:",    round(np.mean(rec_im_scores),3))
    """ //#####|tree_root|#####\\ """
    df.loc[im_condition, "P.Mask_Recall"] = np.mean(rec_im_scores)
    print("IM F1:",        round(np.mean(f1_im_scores),3))
    """ //#####|tree_root|#####\\ """
    df.loc[im_condition, "P.F1-Score"] = np.mean(f1_im_scores)

    print("\nOM IoU:",       round(np.mean(iou_om_scores),3))
    """ //#####|tree_root|#####\\ """
    df.loc[om_condition, "P.IOU"] = np.mean(iou_om_scores)
    print("OM Dice:",      round(np.mean(dice_om_scores),3))
    """ //#####|tree_root|#####\\ """
    df.loc[om_condition, "P.Dice"] = np.mean(dice_om_scores)
    print("OM Precision:", round(np.mean(prec_om_scores),3))
    """ //#####|tree_root|#####\\ """
    df.loc[om_condition, "P.Mask_Precision"] = np.mean(prec_om_scores)
    print("OM Recall:",    round(np.mean(rec_om_scores),3))
    """ //#####|tree_root|#####\\ """
    df.loc[om_condition, "P.Mask_Recall"] = np.mean(rec_om_scores)
    print("OM F1:",        round(np.mean(f1_om_scores),3))
    """ //#####|tree_root|#####\\ """
    df.loc[om_condition, "P.F1-Score"] = np.mean(f1_om_scores)


    for c in range(2):
        num_class_condition = (
            (df["Model"] == Model) &
            (df["Model_Electron_Dose"] == Model_Electron_Dose) &
            (df["Model_Image_Size"] == Model_Image_Size) &
            (df["Test_Electron_Dose"] == Test_Electron_Dose) &
            (df["Test_Image_Size"] == Test_Image_Size) &
            (df["Channel"].fillna(-1).astype(int) == c)
        )
        y_true = all_true[c]
        y_prob = all_probs[c]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall, precision)
        print(f"Class {c} AUPRC: {auprc:.4f}")
        """ //#####|tree_root|#####\\ """
        df.loc[num_class_condition, "P.AUPRC"] = auprc

    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------


    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")

    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_s3, test_img_folder, test_ann_path
