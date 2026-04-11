# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

# Detectron2
def run_model_pipeline_d2(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_d2, test_img_folder):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import scipy.spatial
    import pandas as pd
    import os
    import glob


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

    # Create predictor
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = MODEL_PATH_d2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # change if needed
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    predictor = DefaultPredictor(cfg)
    MODEL_PATH = cfg.MODEL.WEIGHTS



    model_path = MODEL_PATH
    input_folder = test_img_folder
    output_folder = os.path.join(TARGET_FOLDER, f"Results_{Model}")
    os.makedirs(output_folder, exist_ok=True)


    image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.png"))

    total_bacteria = 0

    for img_path in image_files:
        used_ims = set()
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

        # Run on resized image
        outputs = predictor(image_rgb)
        instances = outputs["instances"].to("cpu")

        #  Skip images where YOLO found no masks
        if len(instances) == 0:
            print(f"Skipping {img_path} (no masks detected)")
            continue

        # Split OM & IM masks
        masks   = instances.pred_masks.numpy()     # [N, H, W] (bool)
        classes = instances.pred_classes.numpy()   # [N] (int)

        om_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        im_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        om_masks, im_masks = [], []

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

        
        for i, mask in enumerate(masks):
            # resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            binary = (mask > 0.5).astype(np.uint8) * 255
            if classes[i] == 1:
                om_masks.append(binary)
            elif classes[i] == 0:
                im_masks.append(binary)

        # Skip if OM or IM missing
        if not om_masks or not im_masks:
            print(f"Skipping {img_path} (missing OM or IM contour)")
            continue

    print(f"\nTotal bacteria across all images: {total_bacteria}")
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "total_bacteria"] = total_bacteria

    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")

    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_d2, test_img_folder




