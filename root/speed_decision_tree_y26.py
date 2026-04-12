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
import torch

# YOLO26
def run_model_speed_pipeline_y26(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_y26, test_images_y26, test_images_orig_folder):

    test_images = test_images_y26
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
    model_path = MODEL_PATH_y26
    input_folder = test_images_orig_folder


    # YOLOv11 Timing
    model = YOLO(MODEL_PATH_y26)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(MODEL_PATH_y26)
    print(input_folder)

    images = []

    for img_path in test_images_y26:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
            img = img[:, :, ::-1]  # BGR → RGB
            images.append(img)


    # -------------------------
    # WARMUP
    # -------------------------
    for _ in range(10):
        _ = model.predict(images[0], imgsz=TARGET_SIZE, conf=0.5, verbose=False)[0]

    torch.cuda.synchronize()

    # -------------------------
    # TIMING
    # -------------------------
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    total_time = 0

    for img in images:
        starter.record()
        _ = model.predict(img, imgsz=TARGET_SIZE, conf=0.5, verbose=False)
        ender.record()

        torch.cuda.synchronize()
        total_time += starter.elapsed_time(ender)

    avg_time_per_image = total_time / len(images)
    fps = 1000 / avg_time_per_image

    print(f"Speed test for {Model}-{Model_Electron_Dose}:")
    print(f"Avg image time per image: {avg_time_per_image:.2f} ms")
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "Avg Time_Image"] = avg_time_per_image
    print(f"FPS: {fps:.2f}")
    """ //#####|tree_root|#####\\ """
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
        

    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_y26, test_images_y26, test_images_orig_folder

