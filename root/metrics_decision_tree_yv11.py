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



























    total_bacteria = 0

    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping unreadable image {img_path}")
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]
        save_img = os.path.join(output_folder, f"{base}_thickness.png")
        save_csv = os.path.join(output_folder, f"{base}_thickness.csv")
        save_angles_csv = os.path.join(output_folder, f"{base}_angles.csv")

        # ✅ Skip already processed images
        #if os.path.exists(save_img) and os.path.exists(save_csv):
        #    print(f"Skipping {img_path} (already processed)")
        #    continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image_rgb.shape[:2]

        results = model.predict(image_rgb, imgsz = 1024, conf=0.5, verbose=False)[0]

        #  Skip images where YOLO found no masks
        if results.masks is None:
            print(f"Skipping {img_path} (no masks detected)")
            continue

        masks = results.masks.data.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        om_masks, im_masks = [], []
        for i, mask in enumerate(masks):
            resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            binary = (resized > 0.5).astype(np.uint8) * 255
            if class_ids[i] == 1:
                om_masks.append(binary)
            elif class_ids[i] == 0:
                im_masks.append(binary)

        # Skip if OM or IM missing
        if not om_masks or not im_masks:
            print(f"Skipping {img_path} (missing OM or IM contour)")
            continue

        im_combined = np.maximum.reduce(im_masks) if im_masks else np.zeros((img_height, img_width), dtype=np.uint8)
        om_combined = np.maximum.reduce(om_masks) if om_masks else np.zeros((img_height, img_width), dtype=np.uint8)

        cv2.imwrite(os.path.join(output_folder, f"{base}_IM_mask.png"), im_combined)
        cv2.imwrite(os.path.join(output_folder, f"{base}_OM_mask.png"), om_combined)
        
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        num_bacteria = np.sum(class_ids == 1)  # count OM only
        total_bacteria += num_bacteria
        print(f"{img_path}: {num_bacteria} bacteria")

    print(f"\nTotal predicted bacteria with YOLOv11 across all images: {total_bacteria}")
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "total_bacteria"] = total_bacteria

    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")
    

    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_yv11, test_images_yv11, test_images_orig_folder


