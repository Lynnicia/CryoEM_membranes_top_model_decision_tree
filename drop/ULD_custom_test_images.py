# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from drop.coco_bacteria_dataset import CocoBacteriaDataset
from torch.utils.data import DataLoader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

import glob

# Safe copy json (_filt) and edit ONLY the json copy
from pathlib import Path
import shutil

src = Path(f"/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/{test_electron_dose_decision}/COCO/test/{test_image_size_decision}/_annotations.coco.json")
dst = Path(f"/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/{test_electron_dose_decision}/COCO/test/{test_image_size_decision}/_filt_annotations.coco.json")

# If destination file does NOT exist
if not dst.exists():

    # Ensure parent directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(src, dst)
    print("Copied to:", dst)

else:
    print("Destination already exists:", dst)




# Find original 3 classes: bacteria, IM and OM

with open(f"/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/{test_electron_dose_decision}/COCO/test/{test_image_size_decision}/_filt_annotations.coco.json") as f:
    coco = json.load(f)

for c in coco["categories"]:
    print(c["id"], c["name"])

cats = sorted(coco["categories"], key=lambda x: x["id"])


# Update annotations

BAD_ID = 0

coco["categories"] = [
    c for c in coco["categories"] if c["id"] != BAD_ID
]

coco["annotations"] = [
    ann for ann in coco["annotations"]
    if ann["category_id"] != BAD_ID
]

with open(f"/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/{test_electron_dose_decision}/COCO/test/{test_image_size_decision}/_filt_annotations.coco.json", "w") as f:
    json.dump(coco, f, indent=2)

for c in coco["categories"]:
    print(c["id"], c["name"])



#detectron registration needed here
def reregister_coco(name, json_file, image_root):
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)

    register_coco_instances(name, {}, json_file, image_root)


def test_image_ULD_640():

    #U-Net
    test_annotation_file = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/640/_annotations.coco.json"
    test_image_dir = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/640"
    test_dataset = CocoBacteriaDataset(test_annotation_file, test_image_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    #YOLOv11
    test_images_yv11 = glob.glob("/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/YOLO/test/images/*.jpg")
    test_images_orig_folder = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/YOLO/test/images"
    
    #YOLO26
    test_images_y26 = glob.glob("/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/YOLO/test/images/*.jpg")
    test_images_orig_folder = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/YOLO/test/images"

    #SAM3
    test_img_folder = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/640"
    test_ann_path = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/640/_annotations.coco.json"

    #Detectron2
    reregister_coco(
        "bacteria_ULD_OMIM_640_test",
        f"/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/640/_filt_annotations.coco.json",
        f"/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/640"
        )

    return test_loader, test_image_dir, test_images_yv11, test_images_y26, test_img_folder, test_ann_path, test_images_orig_folder


def test_image_ULD_1024():

    #U-Net
    test_annotation_file = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/1024/_annotations.coco.json"
    test_image_dir = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/1024"
    test_dataset = CocoBacteriaDataset(test_annotation_file, test_image_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    #YOLOv11
    test_images_yv11 = glob.glob("/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/YOLO/test/images/*.jpg")
    test_images_orig_folder = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/YOLO/test/images"
    
    #YOLO26
    test_images_y26 = glob.glob("/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/YOLO/test/images/*.jpg")
    test_images_orig_folder = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/YOLO/test/images"

    #SAM3
    test_img_folder = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/1024"
    test_ann_path = "/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/1024/_annotations.coco.json"

    #Detectron2
    reregister_coco(
        "bacteria_ULD_OMIM_1024_test",
        f"/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/1024/_filt_annotations.coco.json",
        f"/content/CryoEM_membranes_top_model_decision_tree/Custom_Datasets/ULD/COCO/test/1024"
        )

    return test_loader, test_image_dir, test_images_yv11, test_images_y26, test_img_folder, test_ann_path, test_images_orig_folder


TEST_MAP = {
    ("ULD", 640): test_image_ULD_640,
    ("ULD", 1024): test_image_ULD_1024,
}