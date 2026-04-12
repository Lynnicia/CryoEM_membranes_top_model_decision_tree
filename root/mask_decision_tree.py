# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model_image_size_decision = 640     # (no quotes!)
MODEL_PATH_u, MODEL_PATH_yv11, MODEL_PATH_y26, MODEL_PATH_d2, MODEL_PATH_s3 = MODEL_MAP[(model_electron_dose_decision, model_image_size_decision)]()

if model_electron_dose_decision == "LD":
    results_folder = "/content/CryoEM_membranes_top_model_decision_tree/Masks_LD"
    os.makedirs(results_folder, exist_ok=True)

    Model_Image_Size = model_image_size_decision
    Model_Electron_Dose = model_electron_dose_decision
    Test_Image_Size = test_image_size_decision
    Test_Electron_Dose = test_electron_dose_decision
    TARGET_FOLDER = results_folder

    # YOLOv11
    input_folder = test_images_orig_folder
    from root.mask_decision_tree_yv11 import run_model_pipeline_yv11
    Model = f"{model_image_size_decision}-YOLOv11"
    run_model_pipeline_yv11(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_yv11, test_images_yv11, test_images_orig_folder)
    
    # YOLO26
    from root.mask_decision_tree_y26 import run_model_pipeline_y26
    Model = f"{model_image_size_decision}-YOLO26"
    run_model_pipeline_y26(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_y26, test_images_y26, test_images_orig_folder)
 
    # U-Net
    TARGET_SIZE = Test_Image_Size
    from root.mask_decision_tree_u import run_model_pipeline_u
    Model = f"{model_image_size_decision}-U-Net"
    run_model_pipeline_u(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_u, test_loader, test_image_dir)

    # Detectron2
    input_folder = test_img_folder
    from root.mask_decision_tree_d2 import run_model_pipeline_d2
    Model = f"{model_image_size_decision}-Detectron2"
    run_model_pipeline_d2(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_d2, test_img_folder)
    
    #SAM3
    COUNTER = 1
    input_folder = test_ann_path
    from root.mask_decision_tree_s3 import run_model_pipeline_s3
    Model = f"{model_image_size_decision}-SAM3"
    run_model_pipeline_s3(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_s3, test_img_folder, test_ann_path)
    
elif model_electron_dose_decision == "ULD":
    results_folder = "/content/CryoEM_membranes_top_model_decision_tree/Masks_ULD"
    os.makedirs(results_folder, exist_ok=True)

    Model_Image_Size = model_image_size_decision
    Model_Electron_Dose = model_electron_dose_decision
    Test_Image_Size = test_image_size_decision
    Test_Electron_Dose = test_electron_dose_decision
    TARGET_FOLDER = results_folder

    # YOLOv11
    input_folder = test_images_orig_folder
    from root.mask_decision_tree_yv11 import run_model_pipeline_yv11
    Model = f"{model_image_size_decision}-YOLOv11"
    run_model_pipeline_yv11(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_yv11, test_images_yv11, test_images_orig_folder)
    
    # YOLO26
    from root.mask_decision_tree_y26 import run_model_pipeline_y26
    Model = f"{model_image_size_decision}-YOLO26"
    run_model_pipeline_y26(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_y26, test_images_y26, test_images_orig_folder)
 
    # U-Net
    TARGET_SIZE = Test_Image_Size
    from root.mask_decision_tree_u import run_model_pipeline_u
    Model = f"{model_image_size_decision}-U-Net"
    run_model_pipeline_u(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_u, test_loader, test_image_dir)

    # Detectron2
    input_folder = test_img_folder
    from root.mask_decision_tree_d2 import run_model_pipeline_d2
    Model = f"{model_image_size_decision}-Detectron2"
    run_model_pipeline_d2(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_d2, test_img_folder)
    
    #SAM3
    COUNTER = 1
    input_folder = test_ann_path
    from root.mask_decision_tree_s3 import run_model_pipeline_s3
    Model = f"{model_image_size_decision}-SAM3"
    run_model_pipeline_s3(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_s3, test_img_folder, test_ann_path)

else:
    raise ValueError("Invalid option: choose 'LD' or 'ULD'")

model_image_size_decision = 1024     # (no quotes!)
MODEL_PATH_u, MODEL_PATH_yv11, MODEL_PATH_y26, MODEL_PATH_d2, MODEL_PATH_s3 = MODEL_MAP[(model_electron_dose_decision, model_image_size_decision)]()

if model_electron_dose_decision == "LD":
    results_folder = "/content/CryoEM_membranes_top_model_decision_tree/Masks_LD"
    os.makedirs(results_folder, exist_ok=True)
    Model_Image_Size = model_image_size_decision
    Model_Electron_Dose = model_electron_dose_decision
    Test_Image_Size = test_image_size_decision
    Test_Electron_Dose = test_electron_dose_decision
    TARGET_FOLDER = results_folder

    # YOLOv11
    input_folder = test_images_orig_folder
    from root.mask_decision_tree_yv11 import run_model_pipeline_yv11
    Model = f"{model_image_size_decision}-YOLOv11"
    run_model_pipeline_yv11(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_yv11, test_images_yv11, test_images_orig_folder)
    
    # YOLO26
    from root.mask_decision_tree_y26 import run_model_pipeline_y26
    Model = f"{model_image_size_decision}-YOLO26"
    run_model_pipeline_y26(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_y26, test_images_y26, test_images_orig_folder)
 
    # U-Net
    TARGET_SIZE = Test_Image_Size
    from root.mask_decision_tree_u import run_model_pipeline_u
    Model = f"{model_image_size_decision}-U-Net"
    run_model_pipeline_u(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_u, test_loader, test_image_dir)

    # Detectron2
    input_folder = test_img_folder
    from root.mask_decision_tree_d2 import run_model_pipeline_d2
    Model = f"{model_image_size_decision}-Detectron2"
    run_model_pipeline_d2(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_d2, test_img_folder)
    
    #SAM3
    COUNTER = 1
    input_folder = test_ann_path
    from root.mask_decision_tree_s3 import run_model_pipeline_s3
    Model = f"{model_image_size_decision}-SAM3"
    run_model_pipeline_s3(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_s3, test_img_folder, test_ann_path)


elif model_electron_dose_decision == "ULD":
    results_folder = "/content/CryoEM_membranes_top_model_decision_tree/Masks_ULD"
    os.makedirs(results_folder, exist_ok=True)
    Model_Image_Size = model_image_size_decision
    Model_Electron_Dose = model_electron_dose_decision
    Test_Image_Size = test_image_size_decision
    Test_Electron_Dose = test_electron_dose_decision
    TARGET_FOLDER = results_folder

    # YOLOv11
    input_folder = test_images_orig_folder
    from root.mask_decision_tree_yv11 import run_model_pipeline_yv11
    Model = f"{model_image_size_decision}-YOLOv11"
    run_model_pipeline_yv11(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_yv11, test_images_yv11, test_images_orig_folder)
    
    # YOLO26
    from root.mask_decision_tree_y26 import run_model_pipeline_y26
    Model = f"{model_image_size_decision}-YOLO26"
    run_model_pipeline_y26(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_y26, test_images_y26, test_images_orig_folder)
 
    # U-Net
    TARGET_SIZE = Test_Image_Size
    from root.mask_decision_tree_u import run_model_pipeline_u
    Model = f"{model_image_size_decision}-U-Net"
    run_model_pipeline_u(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_u, test_loader, test_image_dir)

    # Detectron2
    input_folder = test_img_folder
    from root.mask_decision_tree_d2 import run_model_pipeline_d2
    Model = f"{model_image_size_decision}-Detectron2"
    run_model_pipeline_d2(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_d2, test_img_folder)
    
    #SAM3
    COUNTER = 1
    input_folder = test_ann_path
    from root.mask_decision_tree_s3 import run_model_pipeline_s3
    Model = f"{model_image_size_decision}-SAM3"
    run_model_pipeline_s3(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_s3, test_img_folder, test_ann_path)


else:
    raise ValueError("Invalid option: choose 'LD' or 'ULD'")