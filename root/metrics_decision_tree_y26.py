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


# YOLO26
def run_model_pipeline_y26(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_y26, test_images_y26, test_images_orig_folder):
    MODEL_PATH = MODEL_PATH_y26
    model = YOLO(MODEL_PATH)
    test_images = test_images_y26


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