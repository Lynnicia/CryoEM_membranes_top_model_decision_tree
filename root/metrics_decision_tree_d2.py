# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Detectron2
def run_model_metrics_pipeline_d2(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_d2, test_img_folder, test_loader):
    import numpy as np
    import glob
    from pathlib import Path
    import os
    import torch
    import matplotlib.pyplot as plt
    from detectron2.data import DatasetCatalog, build_detection_test_loader
    from detectron2.data import detection_utils as utils
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    import cv2
    from pycocotools import mask as mask_utils
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve, auc
    import torch
    import pandas as pd
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_coco_json
    import shutil
    import json


    ### mask count
 

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


    def reregister_coco(name, json_file, image_root):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

        register_coco_instances(name, {}, json_file, image_root)

    for t in ["test"]:
        reregister_coco(
            f"bacteria_{Test_Electron_Dose}_OMIM_{Test_Image_Size}_test",
            f"/content/CryoEM_membranes_top_model_decision_tree/Datasets/{Test_Electron_Dose}/COCO/test/{Test_Image_Size}/_filt_annotations.coco.json",
            f"/content/CryoEM_membranes_top_model_decision_tree/Datasets/{Test_Electron_Dose}/COCO/test/{Test_Image_Size}"
        )



    # Create predictor
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = MODEL_PATH_d2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # change if needed
    cfg.INPUT.MIN_SIZE_TEST = Test_Image_Size
    cfg.INPUT.MAX_SIZE_TEST = Test_Image_Size
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
    predictor = DefaultPredictor(cfg)
    MODEL_PATH = cfg.MODEL.WEIGHTS



    model_path = MODEL_PATH
    input_folder = test_img_folder
    output_folder = os.path.join(TARGET_FOLDER, f"Results_{Model}-{Model_Electron_Dose}")
    os.makedirs(output_folder, exist_ok=True)


    image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.png"))



    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset_name = f"bacteria_{Test_Electron_Dose}_OMIM_{Test_Image_Size}_test"

    bacteria_metadata = MetadataCatalog.get(dataset_name)
    bacteria_metadata.thing_classes = ["IM", "OM"]
    dataset = DatasetCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    classes = [0, 1]
    all_true     = {0: [], 1: []}
    all_probs    = {0: [], 1: []}


    def decode_coco_segmentation(ann, height, width):
        seg = ann["segmentation"]

        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, height, width)
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)

        elif isinstance(seg, dict):
            mask = mask_utils.decode(seg)

        else:
            raise ValueError(f"Unknown segmentation format: {type(seg)}")

        return (mask > 0).astype(np.uint8)

    with torch.no_grad():

        for d in dataset_dicts:

            image = cv2.imread(d["file_name"])
            H, W = image.shape[:2]

            # Forward pass
            outputs = model([{
                "image": torch.as_tensor(image.transpose(2,0,1)).float().to(device),
                "height": H,
                "width": W
            }])[0]

            instances = outputs["instances"].to("cpu")

            # -----------------------
            # Build GT masks
            # -----------------------
            gt_im = np.zeros((H, W), dtype=np.uint8)
            gt_om = np.zeros((H, W), dtype=np.uint8)

            for ann in d["annotations"]:
                mask = decode_coco_segmentation(ann, H, W)

                if ann["category_id"] == 0:
                    gt_im |= mask
                elif ann["category_id"] == 1:
                    gt_om |= mask

            # -----------------------
            # Build probability maps
            # -----------------------
            prob_im = np.zeros((H, W), dtype=np.float32)
            prob_om = np.zeros((H, W), dtype=np.float32)

            for i in range(len(instances)):

                cls = instances.pred_classes[i].item()
                score = instances.scores[i].item()
                mask = instances.pred_masks[i].numpy()

                if cls == 0:
                    prob_im[mask] = np.maximum(prob_im[mask], score)

                elif cls == 1:
                    prob_om[mask] = np.maximum(prob_om[mask], score)

            # -----------------------
            # Store flattened pixels
            # -----------------------
            all_true[0].append(gt_im.ravel())
            all_true[1].append(gt_om.ravel())
            all_probs[0].append(prob_im.ravel())
            all_probs[1].append(prob_om.ravel())

    for ch in [0,1]:
        all_true[ch] = np.concatenate(all_true[ch])
        all_probs[ch] = np.concatenate(all_probs[ch])

    #---------------------------------------------


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
    print(f"Metrics for {Model}-{Model_Electron_Dose}:")
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------

    def best_threshold(y_true, y_prob):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return thresholds[np.argmax(f1[:-1])]

    def compute_metrics_at_best_f1(all_true, all_probs):
        results = {}
        labels = {0: "IM", 1: "OM"}
        classes = [0, 1]

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
                "Mask Precision": round(precision, 3),
                "Mask Recall":    round(recall, 3),
                "mAP50":          round(ap50, 3),
                "F1 Score":       round(f1, 3),
                "Best Threshold": round(float(thresh), 3),
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
            "Mask Precision": round(precision, 3),
            "Mask Recall":    round(recall, 3),
            "mAP50":          round(ap50, 3),
            "F1 Score":       round(f1, 3),
            "Best Threshold": round(float(thresh), 3),
        }

        return pd.DataFrame(results).T

    metrics_df = compute_metrics_at_best_f1(all_true, all_probs)
    print("\n── Detectron2 Validation Metrics @Best F1 ──")
    print(metrics_df.to_string())

    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "O.Mask_Precision"] = metrics_df.loc["all", "Mask Precision"]
    df.loc[all_condition, "O.Mask_Recall"]    = metrics_df.loc["all", "Mask Recall"]
    df.loc[all_condition, "O.mAP50"]          = metrics_df.loc["all", "mAP50"]
    df.loc[all_condition, "O.F1-Score"]       = metrics_df.loc["all", "F1 Score"]
    df.loc[im_condition, "O.Mask_Precision"] = metrics_df.loc["IM", "Mask Precision"]
    df.loc[im_condition, "O.Mask_Recall"]    = metrics_df.loc["IM", "Mask Recall"]
    df.loc[im_condition, "O.mAP50"]          = metrics_df.loc["IM", "mAP50"]
    df.loc[im_condition, "O.F1-Score"]       = metrics_df.loc["IM", "F1 Score"]
    df.loc[om_condition, "O.Mask_Precision"] = metrics_df.loc["OM", "Mask Precision"]
    df.loc[om_condition, "O.Mask_Recall"]    = metrics_df.loc["OM", "Mask Recall"]
    df.loc[om_condition, "O.mAP50"]          = metrics_df.loc["OM", "mAP50"]
    df.loc[om_condition, "O.F1-Score"]       = metrics_df.loc["OM", "F1 Score"]


    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------

    iou_scores = []
    dice_scores = []
    precisions, recalls, f1s = [], [], []

    iou_im_scores = []
    dice_im_scores = []
    prec_im_scores = []
    rec_im_scores = []
    f1_im_scores = []

    iou_om_scores = []
    dice_om_scores = []
    prec_om_scores = []
    rec_om_scores = []
    f1_om_scores = []

    dataset_name = f"bacteria_{Test_Electron_Dose}_OMIM_{Test_Image_Size}_test"

    d2_test_loader = build_detection_test_loader(cfg, dataset_name)


    with torch.no_grad():
        for batch in d2_test_loader:

            outputs = model(batch)

            for i, output in enumerate(outputs):

                instances = output["instances"].to("cpu")

                file_name = batch[i]["file_name"]

                # 🔑 Match dataset entry
                d = next(x for x in dataset if x["file_name"] == file_name)

                image = cv2.imread(file_name)
                H, W = image.shape[:2]

                # ----------------------
                # Build prediction masks
                # ----------------------
                pred_mask_im = np.zeros((H, W), dtype=np.uint8)
                pred_mask_om = np.zeros((H, W), dtype=np.uint8)

                if instances.has("pred_masks"):
                    pred_masks = instances.pred_masks.numpy()
                    pred_classes = instances.pred_classes.numpy()

                    for mask, cls in zip(pred_masks, pred_classes):
                        if cls == 0:
                            pred_mask_im |= mask
                        elif cls == 1:
                            pred_mask_om |= mask

                # ----------------------
                # Build GT masks (COCO-safe)
                # ----------------------
                gt_mask_im = np.zeros((H, W), dtype=np.uint8)
                gt_mask_om = np.zeros((H, W), dtype=np.uint8)

                for ann in d["annotations"]:
                    mask = decode_coco_segmentation(ann, H, W)

                    if ann["category_id"] == 0:
                        gt_mask_im |= mask
                    elif ann["category_id"] == 1:
                        gt_mask_om |= mask

                # Convert to boolean
                gt_mask_im = gt_mask_im > 0
                gt_mask_om = gt_mask_om > 0
                pred_mask_im = pred_mask_im > 0
                pred_mask_om = pred_mask_om > 0

                # ----------------------
                # Overall Metrics
                # ----------------------
                pred_all = np.stack([pred_mask_im, pred_mask_om])
                gt_all = np.stack([gt_mask_im, gt_mask_om])

                iou_scores.append(calculate_iou(pred_all, gt_all))
                dice_scores.append(calculate_dice(pred_all, gt_all))
                p, r, f = calculate_precision_recall_f1(pred_all, gt_all)

                precisions.append(p)
                recalls.append(r)
                f1s.append(f)

                # ----------------------
                # IM Metrics
                # ----------------------
                pred_im = np.stack([pred_mask_im])
                gt_im = np.stack([gt_mask_im])

                iou_im_scores.append(calculate_iou(pred_im, gt_im))
                dice_im_scores.append(calculate_dice(pred_im, gt_im))
                p, r, f = calculate_precision_recall_f1(pred_im, gt_im)

                prec_im_scores.append(p)
                rec_im_scores.append(r)
                f1_im_scores.append(f)

                # ----------------------
                # OM Metrics
                # ----------------------
                pred_om = np.stack([pred_mask_om])
                gt_om = np.stack([gt_mask_om])

                iou_om_scores.append(calculate_iou(pred_om, gt_om))
                dice_om_scores.append(calculate_dice(pred_om, gt_om))
                p, r, f = calculate_precision_recall_f1(pred_om, gt_om)

                prec_om_scores.append(p)
                rec_om_scores.append(r)
                f1_om_scores.append(f)

    #AUPRC
    from sklearn.metrics import precision_recall_curve, auc
    import numpy as np
    import torch

    all_true     = {0: [], 1: []}
    all_probs    = {0: [], 1: []}

    dataset_name = f"bacteria_{Test_Electron_Dose}_OMIM_{Test_Image_Size}_test"

    evaluator = COCOEvaluator(
        dataset_name,
        tasks=("segm",),
        distributed=False,
        output_dir="./output"
    )

    d2_test_loader = build_detection_test_loader(cfg, dataset_name)

    results = inference_on_dataset(model, d2_test_loader, evaluator)

    with torch.no_grad():
        for d in dataset:

            image = cv2.imread(d["file_name"])
            H, W = image.shape[:2]

            inputs = [{
                "image": torch.as_tensor(image.transpose(2,0,1)).float().to(device),
                "height": H,
                "width": W
            }]

            outputs = model(inputs)[0]
            instances = outputs["instances"].to("cpu")

            # -----------------------
            # Build GT masks
            # -----------------------
            gt_im = np.zeros((H, W), dtype=np.uint8)
            gt_om = np.zeros((H, W), dtype=np.uint8)

            for ann in d["annotations"]:
                mask = decode_coco_segmentation(ann, H, W)
                if ann["category_id"] == 0:
                    gt_im |= mask
                elif ann["category_id"] == 1:
                    gt_om |= mask

            # -----------------------
            # Build probability maps
            # -----------------------
            prob_im = np.zeros((H, W), dtype=np.float32)
            prob_om = np.zeros((H, W), dtype=np.float32)

            for i in range(len(instances)):

                cls = instances.pred_classes[i].item()
                score = instances.scores[i].item()
                mask = instances.pred_masks[i].numpy()

                if cls == 0:
                    prob_im[mask] = np.maximum(prob_im[mask], score)
                elif cls == 1:
                    prob_om[mask] = np.maximum(prob_om[mask], score)

            # -----------------------
            # Store pixels
            # -----------------------
            all_true[0].extend(gt_im.flatten())
            all_true[1].extend(gt_om.flatten())

            all_probs[0].extend(prob_im.flatten())
            all_probs[1].extend(prob_om.flatten())

    auprcs = []

    for c in range(2):

        y_true = np.array(all_true[c])
        y_prob = np.array(all_probs[c])

        if np.unique(y_true).size > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            auprc = auc(recall, precision)
        else:
            auprc = np.nan

        auprcs.append(auprc)




    # ✅ Now compute dataset average
    df.loc[all_condition, "P.IOU"] = np.mean(iou_scores)
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.Dice"] = np.mean(dice_scores)
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.Mask_Precision"] = np.mean(precisions)
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.Mask_Recall"] = np.mean(recalls)
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.F1-Score"] = np.mean(f1s)
    print("Mean AUPRC:", np.mean(auprcs))
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "P.AUPRC"] = np.mean(auprcs)


    # ✅ Now compute dataset average
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
    print("IM AUPRC:", auprcs[0])
    """ //#####|tree_root|#####\\ """
    df.loc[im_condition, "P.AUPRC"] = auprcs[0]

    # ✅ Now compute dataset average
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
    print("OM AUPRC:", auprcs[1])
    """ //#####|tree_root|#####\\ """
    df.loc[om_condition, "P.AUPRC"] = auprcs[1]

    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")


    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_d2, test_img_folder, test_loader

