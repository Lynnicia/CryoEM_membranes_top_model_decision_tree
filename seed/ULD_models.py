# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def model_ULD_640():

    MODEL_PATH_u = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/640-U-Net-ULD.pt"
    MODEL_PATH_yv11 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/640-YOLOv11-ULD.pt"
    MODEL_PATH_y26 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/640-YOLO26-ULD.pt"
    MODEL_PATH_d2 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/640-Detectron2-ULD.pt"
    MODEL_PATH_s3 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/640-SAM3-ULD.pt"

    return MODEL_PATH_u, MODEL_PATH_yv11, MODEL_PATH_y26, MODEL_PATH_d2, MODEL_PATH_s3

def model_ULD_1024():

    MODEL_PATH_u = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/1024-U-Net-ULD.pt"
    MODEL_PATH_yv11 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/1024-YOLOv11-ULD.pt"
    MODEL_PATH_y26 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/1024-YOLO26-ULD.pt"
    MODEL_PATH_d2 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/1024-Detectron2-ULD.pt"
    MODEL_PATH_s3 = "/content/CryoEM_membranes_top_model_decision_tree/seed/models/1024-SAM3-ULD.pt"

    return MODEL_PATH_u, MODEL_PATH_yv11, MODEL_PATH_y26, MODEL_PATH_d2, MODEL_PATH_s3

MODEL_MAP = {
    ("ULD", 640): model_ULD_640,
    ("ULD", 1024): model_ULD_1024,
}