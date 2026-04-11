# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def model_LD_640():

    MODEL_PATH_u = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/640-U-Net-LD.pt"
    MODEL_PATH_yv11 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/640-YOLOv11-LD.pt"
    MODEL_PATH_y26 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/640-YOLO26-LD.pt"
    MODEL_PATH_d2 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/640-Detectron2-LD.pth"
    MODEL_PATH_s3 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/640-SAM3-LD.pt"

    return MODEL_PATH_u, MODEL_PATH_yv11, MODEL_PATH_y26, MODEL_PATH_d2, MODEL_PATH_s3

def model_LD_1024():

    MODEL_PATH_u = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/1024-U-Net-LD.pt"
    MODEL_PATH_yv11 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/1024-YOLOv11-LD.pt"
    MODEL_PATH_y26 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/1024-YOLO26-LD.pt"
    MODEL_PATH_d2 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/1024-Detectron2-LD.pth"
    MODEL_PATH_s3 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/1024-SAM3-LD.pt"

    return MODEL_PATH_u, MODEL_PATH_yv11, MODEL_PATH_y26, MODEL_PATH_d2, MODEL_PATH_s3

MODEL_MAP = {
    ("LD", 640): model_LD_640,
    ("LD", 1024): model_LD_1024,
}
