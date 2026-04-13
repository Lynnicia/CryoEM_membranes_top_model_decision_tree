# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
from ultralytics import YOLO
import numpy as np
import cv2
import glob
import os
from sklearn.metrics import precision_recall_curve, auc


# YOLOv11
def run_model_metrics_pipeline_yv11(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_yv11, test_images_yv11, test_images_orig_folder):
    MODEL_PATH = MODEL_PATH_yv11
    model = YOLO(MODEL_PATH)
    test_images = test_images_yv11


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
    model_path = MODEL_PATH_yv11
    input_folder = test_images_orig_folder


    model = YOLO(model_path)
    image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.png"))

    # Define YOLO txt to pixel masks
    def yolo_txt_to_mask(txt_path, image_shape, class_id=None):
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if not os.path.exists(txt_path):
            return None

        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            obj_class = int(parts[0])   # rename to avoid overwriting

            # 🔥 Filter by requested class
            if class_id is not None and obj_class != class_id:
                continue

            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)

            coords[:, 0] *= w
            coords[:, 1] *= h

            pts = coords.astype(np.int32)

            cv2.fillPoly(mask, [pts], 1)

        return mask

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

    # YOLO-specific object metrics


    metrics = model.val(
        data=f"/content/CryoEM_membranes_top_model_decision_tree/Datasets/{Test_Electron_Dose}/YOLO/test/{Test_Electron_Dose}_test.yaml",
        split="test",     # 👈 evaluate on test set
        imgsz=TARGET_SIZE
    )


    # overall object metrics (IOU not possible in object detection)
    print("\033[94m")
    print("\n=== Overall Object Segmentation Metrics ===\n")

    print("Overall Mask mAP50-95:", metrics.seg.map)
    print("Overall Mask mAP50:", metrics.seg.map50)
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "O.mAP50"] = metrics.seg.map50
    print("Overall Mask mAP75:", metrics.seg.map75)
    print("Overall Mask Precision:", metrics.seg.mp)
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "O.Mask_Precision"] = metrics.seg.mp
    print("Overall Mask Recall:", metrics.seg.mr)
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "O.Mask_Recall"] = metrics.seg.mr
    print("Overall Mask F1:", metrics.seg.f1.mean())
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "O.F1-Score"] = metrics.seg.f1.mean()
    print("\033[0m")


    # Per class pobject metrics

    class_names = ["IM", "OM"]

    ap_matrix = metrics.seg.all_ap  # (nc, 10)
    print("\033[94m")
    print("\n=== Per-Class Object Segmentation Metrics ===")
    print("\033[0m")
    for i, name in enumerate(class_names):

        """ //#####|tree_root|#####\\ """
        class_condition = (
            (df["Model"] == Model) &
            (df["Model_Electron_Dose"] == Model_Electron_Dose) &
            (df["Model_Image_Size"] == Model_Image_Size) &
            (df["Test_Electron_Dose"] == Test_Electron_Dose) &
            (df["Test_Image_Size"] == Test_Image_Size) &
            (df["Class"] == name)
        )

        print("\033[94m")
        print(f"\n{name}:")

        precision = metrics.seg.p[i]
        recall = metrics.seg.r[i]
        f1 = metrics.seg.f1[i]

        print("Mask mAP50-95:", metrics.seg.maps[i])
        print("Mask AP50:", ap_matrix[i, 0])
        """ //#####|tree_root|#####\\ """
        df.loc[class_condition, "O.mAP50"] = ap_matrix[i, 0]
        print("Mask AP75:", ap_matrix[i, 5])
        print("Mask Precision:", metrics.seg.p[i])
        """ //#####|tree_root|#####\\ """
        df.loc[class_condition, "O.Mask_Precision"] = metrics.seg.p[i]
        print("Mask Recall:", metrics.seg.r[i])
        """ //#####|tree_root|#####\\ """
        df.loc[class_condition, "O.Mask_Recall"] = metrics.seg.r[i]
        print("Mask F1:", metrics.seg.f1[i])
        """ //#####|tree_root|#####\\ """
        df.loc[class_condition, "O.F1-Score"] = metrics.seg.f1[i]
        print("\033[0m")


    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------

    # overall pixel segmentation metrics

    print("Total images before segmentation metrics:", len(test_images))

    print("\033[91m")
    print("\n=== Overall Pixel Segmentation Metrics (No OM Cleaning) ===\n")

    iou_scores = []
    dice_scores = []
    precisions, recalls, f1s = [], [], []


    for img_path in test_images:

        image = cv2.imread(img_path)
        if image is None:
            print("Bad image:", img_path)
            continue

        results = model.predict(img_path, imgsz=TARGET_SIZE, verbose=False)[0]

        txt_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

        # --- Ground truth ---
        gt_mask_im = yolo_txt_to_mask(txt_path, image.shape, class_id=0)
        if gt_mask_im is None:
            gt_mask_im = np.zeros(image.shape[:2], dtype=np.uint8)

        gt_mask_om = yolo_txt_to_mask(txt_path, image.shape, class_id=1)
        if gt_mask_om is None:
            gt_mask_om = np.zeros(image.shape[:2], dtype=np.uint8)


        # --- Prediction masks ---
        pred_mask_im = np.zeros_like(gt_mask_im, dtype=np.uint8)
        pred_mask_om = np.zeros_like(gt_mask_om, dtype=np.uint8)


        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for i, m in enumerate(masks):
                m_resized = cv2.resize(
                    m.astype(np.uint8),
                    (gt_mask_im.shape[1], gt_mask_im.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                if classes[i] == 0:
                    pred_mask_im = np.maximum(pred_mask_im, m_resized)

                elif classes[i] == 1:
                    pred_mask_om = np.maximum(pred_mask_om, m_resized)

        # Convert to boolean
        gt_mask_im = gt_mask_im > 0
        gt_mask_om = gt_mask_om > 0
        # gt_mask_om_clean = gt_mask_om & (~gt_mask_im)
        pred_mask_im = pred_mask_im > 0
        pred_mask_om = pred_mask_om > 0

        # Stack into channels
        true_mask = np.stack([gt_mask_im, gt_mask_om])
        pred_mask = np.stack([pred_mask_im, pred_mask_om])

        # ✅ Compute and STORE
        iou = calculate_iou(pred_mask, true_mask)
        dice = calculate_dice(pred_mask, true_mask)
        precision, recall, f1 = calculate_precision_recall_f1(pred_mask, true_mask)

        iou_scores.append(iou)
        dice_scores.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)





    all_true = [[], []]
    all_probs = [[], []]

    for img_path in test_images:

        image = cv2.imread(img_path)
        if image is None:
            continue

        results = model.predict(img_path, imgsz=TARGET_SIZE, verbose=False)[0]

        txt_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

        # --- Ground truth ---
        gt_mask_im = yolo_txt_to_mask(txt_path, image.shape, class_id=0)
        if gt_mask_im is None:
            gt_mask_im = np.zeros(image.shape[:2], dtype=np.uint8)

        gt_mask_om = yolo_txt_to_mask(txt_path, image.shape, class_id=1)
        if gt_mask_om is None:
            gt_mask_om = np.zeros(image.shape[:2], dtype=np.uint8)

        # --- Probability maps ---
        combined_prob_im = np.zeros_like(gt_mask_im, dtype=np.float32)
        combined_prob_om = np.zeros_like(gt_mask_om, dtype=np.float32)

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for i, m in enumerate(masks):

                m_resized = cv2.resize(
                    m,
                    (gt_mask_im.shape[1], gt_mask_im.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                if classes[i] == 0:
                    combined_prob_im = np.maximum(combined_prob_im, m_resized)

                elif classes[i] == 1:
                    combined_prob_om = np.maximum(combined_prob_om, m_resized)

        # Convert GT to boolean
        gt_mask_im = gt_mask_im > 0
        gt_mask_om = gt_mask_om > 0

        # Stack
        true_mask = np.stack([gt_mask_im, gt_mask_om])
        pred_probs = np.stack([combined_prob_im, combined_prob_om])

        # 🔥 Aggregate pixels
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
    print("Mean Dice:", np.mean(dice_scores))
    print("Mean Precision:", np.mean(precisions))
    print("Mean Recall:", np.mean(recalls))
    print("Mean F1:", np.mean(f1s))
    print("Mean AUPRC:", np.mean(auprcs))
    print("\033[0m")

    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------

    # per class (except AUPRC above)

    # IM storage
    iou_im_scores = []
    dice_im_scores = []
    prec_im_scores = []
    rec_im_scores = []
    f1_im_scores = []
    auprc_im_scores = []

    # OM storage
    iou_om_scores = []
    dice_om_scores = []
    prec_om_scores = []
    rec_om_scores = []
    f1_om_scores = []
    auprc_om_scores = []


    for img_path in test_images:

        image = cv2.imread(img_path)
        if image is None:
            continue

        results = model.predict(img_path, imgsz=TARGET_SIZE, verbose=False)[0]

        txt_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

        # --- Ground truth ---
        gt_mask_im = yolo_txt_to_mask(txt_path, image.shape, class_id=0)
        if gt_mask_im is None:
            gt_mask_im = np.zeros(image.shape[:2], dtype=np.uint8)

        gt_mask_om = yolo_txt_to_mask(txt_path, image.shape, class_id=1)
        if gt_mask_om is None:
            gt_mask_om = np.zeros(image.shape[:2], dtype=np.uint8)

        # --- Prediction ---
        pred_mask_im = np.zeros_like(gt_mask_im)
        pred_mask_om = np.zeros_like(gt_mask_om)

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for i, m in enumerate(masks):
                m_resized = cv2.resize(
                    m.astype(np.uint8),
                    (gt_mask_im.shape[1], gt_mask_im.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                if classes[i] == 0:
                    pred_mask_im = np.maximum(pred_mask_im, m_resized)
                elif classes[i] == 1:
                    pred_mask_om = np.maximum(pred_mask_om, m_resized)

        # Convert to boolean
        gt_mask_im = gt_mask_im > 0
        gt_mask_om = gt_mask_om > 0
        # gt_mask_om_clean = gt_mask_om & (~gt_mask_im)
        pred_mask_im = pred_mask_im > 0
        pred_mask_om = pred_mask_om > 0

        # --- IM metrics ---
        true_im = np.stack([gt_mask_im])
        pred_im = np.stack([pred_mask_im])

        iou_im_scores.append(calculate_iou(pred_im, true_im))
        dice_im_scores.append(calculate_dice(pred_im, true_im))
        p, r, f = calculate_precision_recall_f1(pred_im, true_im)
        prec_im_scores.append(p)
        rec_im_scores.append(r)
        f1_im_scores.append(f)

        # --- OM metrics ---
        true_om = np.stack([gt_mask_om])
        pred_om = np.stack([pred_mask_om])

        iou_om_scores.append(calculate_iou(pred_om, true_om))
        dice_om_scores.append(calculate_dice(pred_om, true_om))
        p, r, f = calculate_precision_recall_f1(pred_om, true_om)
        prec_om_scores.append(p)
        rec_om_scores.append(r)
        f1_om_scores.append(f)

    print("\033[91m")
    print("\n=== Per-Class Pixel Segmentation Metrics (No OM Cleaning) ===\n")

    # ✅ Now compute dataset average
    print("IM IoU:", np.mean(iou_im_scores))
    print("IM Dice:", np.mean(dice_im_scores))
    print("IM Precision:", np.mean(prec_im_scores))
    print("IM Recall:", np.mean(rec_im_scores))
    print("IM F1:", np.mean(f1_im_scores))

    # ✅ Now compute dataset average
    print("\nOM IoU:", np.mean(iou_om_scores))
    print("OM Dice:", np.mean(dice_om_scores))
    print("OM Precision:", np.mean(prec_om_scores))
    print("OM Recall:", np.mean(rec_om_scores))
    print("OM F1:", np.mean(f1_om_scores))


    auprcs = []

    for c in range(2):

        y_true = np.array(all_true[c])
        y_prob = np.array(all_probs[c])

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall, precision)

        auprcs.append(auprc)

        print(f"Class {c} AUPRC: {auprc:.4f}")

    print("\033[0m")

    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------




    # overall

    print("\n=== Overall Pixel Segmentation Metrics (OM Clean) ===\n")

    iou_scores = []
    dice_scores = []
    precisions, recalls, f1s = [], [], []


    for img_path in test_images:

        image = cv2.imread(img_path)
        if image is None:
            print("Bad image:", img_path)
            continue

        results = model.predict(img_path, imgsz=TARGET_SIZE, verbose=False)[0]

        txt_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

        # --- Ground truth ---
        gt_mask_im = yolo_txt_to_mask(txt_path, image.shape, class_id=0)
        if gt_mask_im is None:
            gt_mask_im = np.zeros(image.shape[:2], dtype=np.uint8)

        gt_mask_om = yolo_txt_to_mask(txt_path, image.shape, class_id=1)
        if gt_mask_om is None:
            gt_mask_om = np.zeros(image.shape[:2], dtype=np.uint8)


        # --- Prediction masks ---
        pred_mask_im = np.zeros_like(gt_mask_im, dtype=np.uint8)
        pred_mask_om = np.zeros_like(gt_mask_om, dtype=np.uint8)


        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for i, m in enumerate(masks):
                m_resized = cv2.resize(
                    m.astype(np.uint8),
                    (gt_mask_im.shape[1], gt_mask_im.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                if classes[i] == 0:
                    pred_mask_im = np.maximum(pred_mask_im, m_resized)

                elif classes[i] == 1:
                    pred_mask_om = np.maximum(pred_mask_om, m_resized)

        # Convert to boolean
        gt_mask_im = gt_mask_im > 0
        gt_mask_om = gt_mask_om > 0
        gt_mask_om_clean = gt_mask_om & (~gt_mask_im)
        pred_mask_im = pred_mask_im > 0
        pred_mask_om = pred_mask_om > 0

        # Stack into channels
        true_mask = np.stack([gt_mask_im, gt_mask_om_clean])
        pred_mask = np.stack([pred_mask_im, pred_mask_om])

        # ✅ Compute and STORE
        iou = calculate_iou(pred_mask, true_mask)
        dice = calculate_dice(pred_mask, true_mask)
        precision, recall, f1 = calculate_precision_recall_f1(pred_mask, true_mask)

        iou_scores.append(iou)
        dice_scores.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)



    all_true = [[], []]
    all_probs = [[], []]

    for img_path in test_images:

        image = cv2.imread(img_path)
        if image is None:
            continue

        results = model.predict(img_path, imgsz=TARGET_SIZE, verbose=False)[0]

        txt_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

        # --- Ground truth ---
        gt_mask_im = yolo_txt_to_mask(txt_path, image.shape, class_id=0)
        if gt_mask_im is None:
            gt_mask_im = np.zeros(image.shape[:2], dtype=np.uint8)

        gt_mask_om = yolo_txt_to_mask(txt_path, image.shape, class_id=1)
        if gt_mask_om is None:
            gt_mask_om = np.zeros(image.shape[:2], dtype=np.uint8)

        # --- Probability maps ---
        combined_prob_im = np.zeros_like(gt_mask_im, dtype=np.float32)
        combined_prob_om = np.zeros_like(gt_mask_om, dtype=np.float32)

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for i, m in enumerate(masks):

                m_resized = cv2.resize(
                    m,
                    (gt_mask_im.shape[1], gt_mask_im.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                if classes[i] == 0:
                    combined_prob_im = np.maximum(combined_prob_im, m_resized)

                elif classes[i] == 1:
                    combined_prob_om = np.maximum(combined_prob_om, m_resized)

        # Convert GT to boolean
        gt_mask_im = gt_mask_im > 0
        gt_mask_om = gt_mask_om > 0
        gt_mask_om_clean = gt_mask_om & (~gt_mask_im)

        # Stack
        true_mask = np.stack([gt_mask_im, gt_mask_om_clean])
        pred_probs = np.stack([combined_prob_im, combined_prob_om])

        # 🔥 Aggregate pixels
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


    # per class

    # IM storage
    iou_im_scores = []
    dice_im_scores = []
    prec_im_scores = []
    rec_im_scores = []
    f1_im_scores = []
    auprc_im_scores = []

    # OM storage
    iou_om_scores = []
    dice_om_scores = []
    prec_om_scores = []
    rec_om_scores = []
    f1_om_scores = []
    auprc_om_scores = []


    for img_path in test_images:

        image = cv2.imread(img_path)
        if image is None:
            continue

        results = model.predict(img_path, imgsz=TARGET_SIZE, verbose=False)[0]

        txt_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

        # --- Ground truth ---
        gt_mask_im = yolo_txt_to_mask(txt_path, image.shape, class_id=0)
        if gt_mask_im is None:
            gt_mask_im = np.zeros(image.shape[:2], dtype=np.uint8)

        gt_mask_om = yolo_txt_to_mask(txt_path, image.shape, class_id=1)
        if gt_mask_om is None:
            gt_mask_om = np.zeros(image.shape[:2], dtype=np.uint8)

        # --- Prediction ---
        pred_mask_im = np.zeros_like(gt_mask_im)
        pred_mask_om = np.zeros_like(gt_mask_om)

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for i, m in enumerate(masks):
                m_resized = cv2.resize(
                    m.astype(np.uint8),
                    (gt_mask_im.shape[1], gt_mask_im.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                if classes[i] == 0:
                    pred_mask_im = np.maximum(pred_mask_im, m_resized)
                elif classes[i] == 1:
                    pred_mask_om = np.maximum(pred_mask_om, m_resized)

        # Convert to boolean
        gt_mask_im = gt_mask_im > 0
        gt_mask_om = gt_mask_om > 0
        gt_mask_om_clean = gt_mask_om & (~gt_mask_im)
        pred_mask_im = pred_mask_im > 0
        pred_mask_om = pred_mask_om > 0

        # --- IM metrics ---
        true_im = np.stack([gt_mask_im])
        pred_im = np.stack([pred_mask_im])

        iou_im_scores.append(calculate_iou(pred_im, true_im))
        dice_im_scores.append(calculate_dice(pred_im, true_im))
        p, r, f = calculate_precision_recall_f1(pred_im, true_im)
        prec_im_scores.append(p)
        rec_im_scores.append(r)
        f1_im_scores.append(f)

        # --- OM metrics ---
        true_om = np.stack([gt_mask_om_clean])
        pred_om = np.stack([pred_mask_om])

        iou_om_scores.append(calculate_iou(pred_om, true_om))
        dice_om_scores.append(calculate_dice(pred_om, true_om))
        p, r, f = calculate_precision_recall_f1(pred_om, true_om)
        prec_om_scores.append(p)
        rec_om_scores.append(r)
        f1_om_scores.append(f)


    print("\n=== Per-Class Pixel Segmentation Metrics (OM Clean) ===\n")

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


    print("Total images after segmentation metrics:", len(test_images))

    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")






    

    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_yv11, test_images_yv11, test_images_orig_folder


