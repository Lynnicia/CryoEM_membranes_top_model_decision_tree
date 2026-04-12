# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch

# Detectron2
def run_model_speed_pipeline_d2(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_d2, test_img_folder, test_loader):
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

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.WEIGHTS = MODEL_PATH_d2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.INPUT.MIN_SIZE_TEST = Test_Image_Size
    cfg.INPUT.MAX_SIZE_TEST = Test_Image_Size
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()


    # ===============================
    # COCO evaluation
    # ===============================

    for batch in test_loader:
        print(batch)
        break

    for images in test_loader:
        print(images.shape)
        break

    # -------------------------
    # WARMUP
    # -------------------------
    for i, images in enumerate(test_loader):

        batch = []

        for img in images:
            batch.append({
                "image": img.to(device),
                "height": img.shape[1],
                "width": img.shape[2]
            })

        with torch.no_grad():
            _ = model(batch)

        if i == 9:
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

        for images in test_loader:

            batch = []

            for img in images:
                batch.append({
                    "image": img.to(device),
                    "height": img.shape[1],
                    "width": img.shape[2]
                })

            starter.record()

            outputs = model(batch)

            ender.record()
            torch.cuda.synchronize()

            total_time += starter.elapsed_time(ender)
            total_images += len(batch)



    avg_time_per_image = total_time / total_images
    fps = 1000 / avg_time_per_image

    print(f"Speed test for {Model}-{Model_Electron_Dose}:")
    print(f"Detectron2 Avg time: {avg_time_per_image:.2f} ms")
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "Avg Time_Image"] = avg_time_per_image
    print(f"FPS: {fps:.2f}")
    df.loc[all_condition, "FPS"] = fps

        #-----------------------------------------------

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


    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_d2, test_img_folder, test_loader




