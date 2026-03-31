# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def model_LD_640():

    MODEL_PATH_u = "/content/drive/MyDrive/content/bacteria-thickness_additional-4/SITA-MODEL-LD_UNET_640_02-10-2026_03-36-14/best_ld_omim_640_unet_dice-sita.pt"
    MODEL_PATH_yv11 = "/content/drive/MyDrive/content/bacteria-thickness_additional-4/SITA-MODEL-LD-OMIM-YOLOv11_02-09-2026_03-32-31/best-ld-omim-640x-yolov11-sita.pt"
    MODEL_PATH_y26 = "/content/drive/MyDrive/content/bacteria-thickness_additional-4/LD-OMIM-640-YOLO26_02-05-2026_00-53-17/runs/segment/train/weights/best.pt"
    MODEL_PATH_d2 = "/content/drive/MyDrive/content/bacteria-thickness_additional-4/LD-OMIM-Detectron2_02-24-2026_06-50-17/output/best_model.pth"
    MODEL_PATH_s3 = "/content/drive/MyDrive/content/bacteria-thickness_additional-4/SAM3_02-28-2026_06-56-20/runs/checkpoints/checkpoint.pt"

    return MODEL_PATH_u, MODEL_PATH_yv11, MODEL_PATH_y26, MODEL_PATH_d2, MODEL_PATH_s3

def model_LD_1024():

    MODEL_PATH_u = "/content/drive/MyDrive/content/bacteria-thickness_additional-6/LD_UNET_02-10-2026_02-43-47/best_unet_dice.pt"
    MODEL_PATH_yv11 = "/content/drive/MyDrive/content/bacteria-thickness_additional-6/LD-OMIM-YOLOv11_01-22-2026_19-22-08/runs/segment/train/weights/best.pt"
    MODEL_PATH_y26 = "/content/drive/MyDrive/content/bacteria-thickness_additional-6/LD-OMIM-YOLO26_01-23-2026_03-20-26/runs/segment/train/weights/best.pt"
    MODEL_PATH_d2 = "/content/drive/MyDrive/content/bacteria-thickness_additional-6/LD-OMIM-Detectron2_1024_03-02-2026_19-37-03/output/best_model.pth"
    MODEL_PATH_s3 = "/content/drive/MyDrive/content/bacteria-thickness_additional-6/SAM3_OMIM_LD_02-28-2026_18-43-34/runs/checkpoints/checkpoint.pt"

    return MODEL_PATH_u, MODEL_PATH_yv11, MODEL_PATH_y26, MODEL_PATH_d2, MODEL_PATH_s3

MODEL_MAP = {
    ("LD", 640): model_LD_640,
    ("LD", 1024): model_LD_1024,
}