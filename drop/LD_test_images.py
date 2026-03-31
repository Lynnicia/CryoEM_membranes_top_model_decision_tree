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

    test_annotation_file = "/content/drive/MyDrive/content/bacteria-thickness_additional-4/SITA-MODEL-LD_UNET_640_02-10-2026_03-36-14/test/_annotations.coco.json"
    test_image_dir = "/content/drive/MyDrive/content/bacteria-thickness_additional-4/SITA-MODEL-LD_UNET_640_02-10-2026_03-36-14/test"
    test_dataset = CocoBacteriaDataset(test_annotation_file, test_image_dir, transform=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    test_images_yv11 = glob.glob("/content/drive/MyDrive/content/bacteria-thickness_additional-4/SITA-MODEL-LD-OMIM-YOLOv11_02-09-2026_03-32-31/test/images/*.jpg")
    test_images_y26 = glob.glob("/content/drive/MyDrive/content/bacteria-thickness_additional-4/LD-OMIM-640-YOLO26_02-05-2026_00-53-17/test/images/*.jpg")

    test_img_folder = "/content/drive/MyDrive/content/LD/COCO/test/640/"
    test_ann_path = "/content/drive/MyDrive/content/LD/COCO/test/640/_annotations.coco.json"

    return test_loader, test_image_dir, test_images_yv11, test_images_y26, test_img_folder, test_ann_path


def test_image_LD_1024():


    test_annotation_file = "/content/drive/MyDrive/content/bacteria-thickness_additional-6/LD_UNET_02-10-2026_02-43-47/test/_annotations.coco.json"
    test_image_dir = "/content/drive/MyDrive/content/bacteria-thickness_additional-6/LD_UNET_02-10-2026_02-43-47/test"
    test_dataset = CocoBacteriaDataset(test_annotation_file, test_image_dir, transform=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    test_images_yv11 = glob.glob("/content/drive/MyDrive/content/bacteria-thickness_additional-6/LD-OMIM-YOLOv11_01-22-2026_19-22-08/test/images/*.jpg")
    test_images_y26 = glob.glob("/content/drive/MyDrive/content/bacteria-thickness_additional-6/LD-OMIM-YOLO26_01-23-2026_03-20-26/test/images/*.jpg")

    test_img_folder = "/content/drive/MyDrive/content/LD/COCO/test/1024/"
    test_ann_path = "/content/drive/MyDrive/content/LD/COCO/test/1024/_annotations.coco.json"

    return test_loader, test_image_dir, test_images_yv11, test_images_y26, test_img_folder, test_ann_path


TEST_MAP = {
    ("LD", 640): test_image_LD_640,
    ("LD", 1024): test_image_LD_1024,
}