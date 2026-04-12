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
def run_model_pipeline_yv11(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_yv11, test_images_yv11, test_images_orig_folder):
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

    output_folder = os.path.join(TARGET_FOLDER, f"Results_{Model}")
    os.makedirs(output_folder, exist_ok=True)

    model = YOLO(model_path)
    image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.png"))

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

        results = model(image_rgb, verbose=False)[0]

        # Run YOLO on resized image
        results = model(image, verbose=False)[0]

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

        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        num_bacteria = np.sum(class_ids == 1)  # count OM only
        total_bacteria += num_bacteria
        print(f"{img_path}: {num_bacteria} bacteria")

    print(f"\nTotal bacteria across all images: {total_bacteria}")
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "total_bacteria"] = total_bacteria

    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")

    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_yv11, test_images_yv11, test_images_orig_folder


