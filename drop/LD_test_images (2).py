# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#detectron registration needed here
def reregister_coco(name, json_file, image_root):
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)

    register_coco_instances(name, {}, json_file, image_root)

def test_image_LD_640():
    #U-Net
    test_annotation_file = "/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/640/_annotations.coco.json"
    test_image_dir = "/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/640"
    test_dataset = CocoBacteriaDataset(test_annotation_file, test_image_dir, transform=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    #YOLOv11
    test_images_yv11 = glob.glob("/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/YOLO/test/images/*.jpg")
    
    #YOLO26
    test_images_y26 = glob.glob("/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/YOLO/test/images/*.jpg")

    #SAM3
    test_img_folder = "/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/640"
    test_ann_path = "/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/640/_annotations.coco.json"

    #Detectron2
    reregister_coco(
        "bacteria_LD_OMIM_640_test",
        f"/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/640/_filt_annotations.coco.json",
        f"/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/640"
        )

    return test_loader, test_image_dir, test_images_yv11, test_images_y26, test_img_folder, test_ann_path, register


def test_image_LD_1024():

    #U-Net
    test_annotation_file = "/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/1024/_annotations.coco.json"
    test_image_dir = "/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/1024"
    test_dataset = CocoBacteriaDataset(test_annotation_file, test_image_dir, transform=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    #YOLOv11
    test_images_yv11 = glob.glob("/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/YOLO/test/images/*.jpg")
    
    #YOLO26    
    test_images_y26 = glob.glob("/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/YOLO/test/images/*.jpg")
    
    #SAM3
    test_img_folder = "/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/1024"
    test_ann_path = "/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/1024/_annotations.coco.json"

    #Detectron2
    reregister_coco(
        "bacteria_LD_OMIM_1024_test",
        f"/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/1024/_filt_annotations.coco.json",
        f"/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/1024"
        )

    return test_loader, test_image_dir, test_images_yv11, test_images_y26, test_img_folder, test_ann_path


TEST_MAP = {
    ("LD", 640): test_image_LD_640,
    ("LD", 1024): test_image_LD_1024,
}