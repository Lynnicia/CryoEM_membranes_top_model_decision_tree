# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def model_ULD_640():

    MODEL_PATH_u = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-13/ULD_UNET_640_02-10-2026_04-24-24/best_unet_dice.pt"
    MODEL_PATH_yv11 = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-13/ULD-OMIM-YOLOv11_02-08-2026_01-12-05/runs/segment/train/weights/best.pt"
    MODEL_PATH_y26 = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-13/ULD-OMIM-YOLO26_02-08-2026_02-50-28/runs/segment/train/weights/best.pt"
    MODEL_PATH_d2 = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-13/ULD-OMIM-Detectron2_03-03-2026_15-23-16/output/best_model.pth"
    MODEL_PATH_s3 = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-13/SAM3_OMIM_ULD_02-28-2026_20-35-52/runs/checkpoints/checkpoint.pt"

    return MODEL_PATH_u, MODEL_PATH_yv11, MODEL_PATH_y26, MODEL_PATH_d2, MODEL_PATH_s3

def model_ULD_1024():

    MODEL_PATH_u = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-12/ULD_UNET_1024_02-10-2026_05-28-20/best_unet_dice.pt"
    MODEL_PATH_yv11 = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-12/ULD-OMIM-YOLOv11_02-08-2026_01-51-07/runs/segment/train/weights/best.pt"
    MODEL_PATH_y26 = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-12/ULD-OMIM-YOLO26_02-08-2026_03-10-21/runs/segment/train/weights/best.pt"
    MODEL_PATH_d2 = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-12/ULD-OMIM-Detectron2_1024_03-03-2026_15-23-07/output/best_model.pth"
    MODEL_PATH_s3 = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-12/SAM3_OMIM_ULD_02-28-2026_22-37-25/runs/checkpoints/checkpoint.pt"

    return MODEL_PATH_u, MODEL_PATH_yv11, MODEL_PATH_y26, MODEL_PATH_d2, MODEL_PATH_s3

MODEL_MAP = {
    ("ULD", 640): model_ULD_640,
    ("ULD", 1024): model_ULD_1024,
}