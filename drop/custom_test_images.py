#Do not touch below!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import glob
import cv2

def resize_images_bilinear(input_folder, output_folder, target_size=(1024,1024)):
    os.makedirs(output_folder, exist_ok=True)

    for img_path in glob.glob(os.path.join(input_folder, "*.jpg")):
        img = cv2.imread(img_path)

        resized = cv2.resize(
            img,
            target_size,
            interpolation=cv2.INTER_LINEAR   # Bilinear
        )

        filename = os.path.basename(img_path)

        cv2.imwrite(
            os.path.join(output_folder, filename),
            resized
        )
        
TARGET_FOLDER = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets"
os.makedirs(TARGET_FOLDER, exist_ok=True)


resize_images_bilinear(
    input_folder=custom_dataset_folder,
    output_folder=f"/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/{test_electron_dose_decision}/COCO/test/{test_image_size_decision}"
)
resize_images_bilinear(
    input_folder=custom_dataset_folder,
    output_folder=f"/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/{test_electron_dose_decision}/YOLO/test/{test_image_size_decision}"
)

import shutil

shutil.copy(
    custom_dataset_json,
    f"/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/{test_electron_dose_decision}/COCO/test/{test_image_size_decision}/_annotations.coco.json"
)