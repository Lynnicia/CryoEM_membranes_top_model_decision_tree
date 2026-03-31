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


def test_image_ULD_640():

    test_annotation_file = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-13/ULD_UNET_640_02-10-2026_04-24-24/test/_annotations.coco.json"
    test_image_dir = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-13/ULD_UNET_640_02-10-2026_04-24-24/test"
    test_dataset = CocoBacteriaDataset(test_annotation_file, test_image_dir, transform=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    test_images_yv11 = glob.glob("/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-13/ULD-OMIM-YOLOv11_02-08-2026_01-12-05/test/images/*.jpg")
    test_images_y26 = glob.glob("/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-13/ULD-OMIM-YOLO26_02-08-2026_02-50-28/test/images/*.jpg")

    test_img_folder = "/content/drive/MyDrive/content/ULD/COCO/test/640/"
    test_ann_path = "/content/drive/MyDrive/content/ULD/COCO/test/640/_annotations.coco.json"

    return test_loader, test_image_dir, test_images_yv11, test_images_y26, test_img_folder, test_ann_path


def test_image_ULD_1024():

    test_annotation_file = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-12/ULD_UNET_1024_02-10-2026_05-28-20/test/_annotations.coco.json"
    test_image_dir = "/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-12/ULD_UNET_1024_02-10-2026_05-28-20/test"
    test_dataset = CocoBacteriaDataset(test_annotation_file, test_image_dir, transform=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    test_images_yv11 = glob.glob("/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-12/ULD-OMIM-YOLOv11_02-08-2026_01-51-07/test/images/*.jpg")
    test_images_y26 = glob.glob("/content/drive/MyDrive/content/ultra_low_dose_bacteria_thick_v2-12/ULD-OMIM-YOLO26_02-08-2026_03-10-21/test/images/*.jpg")

    test_img_folder = "/content/drive/MyDrive/content/ULD/COCO/test/1024/"
    test_ann_path = "/content/drive/MyDrive/content/ULD/COCO/test/1024/_annotations.coco.json"

    return test_loader, test_image_dir, test_images_yv11, test_images_y26, test_img_folder, test_ann_path


TEST_MAP = {
    ("ULD", 640): test_image_ULD_640,
    ("ULD", 1024): test_image_ULD_1024,
}