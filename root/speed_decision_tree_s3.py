# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def run_model_speed_pipeline_s3(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_s3, test_img_folder, test_ann_path):
    # ── Standard library ──────────────────────────────────────────────────────
    import json
    import os
    import glob
    import scipy.spatial
    import matplotlib.pyplot as plt

    # ── Numeric / CV ──────────────────────────────────────────────────────────
    import numpy as np
    import cv2

    # ── PyTorch ───────────────────────────────────────────────────────────────
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF

    # ── PIL ───────────────────────────────────────────────────────────────────
    from PIL import Image

    # ── COCO / Metrics ────────────────────────────────────────────────────────
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    import pandas as pd


    # ── SAM3 ──────────────────────────────────────────────────────────────────
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.train.data.sam3_image_dataset import (
        InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
    )
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    )

    # Helper functions
    import torch
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    from sam3.eval.postprocessors import PostProcessImage


    TARGET_SIZE = Test_Image_Size

    csv_path = "/content/CryoEM_membranes_top_model_decision_tree/top_model_table.csv"
    df = pd.read_csv(csv_path)

    #convert Image size values to integers
    df["Model_Image_Size"] = pd.to_numeric(df["Model_Image_Size"], errors="coerce").astype("Int64")
    df["Test_Image_Size"] = pd.to_numeric(df["Test_Image_Size"], errors="coerce").astype("Int64")

    all_condition = (
        (df["Model"] == Model) &
        (df["Model_Electron_Dose"] == Model_Electron_Dose) &
        (df["Model_Image_Size"] == Model_Image_Size) &
        (df["Test_Electron_Dose"] == Test_Electron_Dose) &
        (df["Test_Image_Size"] == Test_Image_Size) &
        (df["Class"] == "All")
    )



    im_condition = (
        (df["Model"] == Model) &
        (df["Model_Electron_Dose"] == Model_Electron_Dose) &
        (df["Model_Image_Size"] == Model_Image_Size) &
        (df["Test_Electron_Dose"] == Test_Electron_Dose) &
        (df["Test_Image_Size"] == Test_Image_Size) &
        (df["Class"] == "IM")
    )

    om_condition = (
        (df["Model"] == Model) &
        (df["Model_Electron_Dose"] == Model_Electron_Dose) &
        (df["Model_Image_Size"] == Model_Image_Size) &
        (df["Test_Electron_Dose"] == Test_Electron_Dose) &
        (df["Test_Image_Size"] == Test_Image_Size) &
        (df["Class"] == "OM")
    )

    # =========================
    # Main loop for folder
    # =========================



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_sam3_image_model(
        bpe_path="/content/CryoEM_membranes_top_model_decision_tree/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        enable_segmentation=True,
        eval_mode=False,
        load_from_HF=False,
    )
    ckpt = torch.load(MODEL_PATH_s3, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()



    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=False,
    )

    # Build datapoint
    def make_datapoint(pil_image, prompts):
        dp = Datapoint(find_queries=[], images=[])
        w, h = pil_image.size
        dp.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]
        ids = {}

        for i, prompt in enumerate(prompts):
            dp.find_queries.append(FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=i,
                    original_image_id=i,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            ))

            ids[prompt] = i

        return dp, ids

    def move_to_cuda(obj):
        """Recursively move all tensors in a dataclass/object to CUDA."""
        if isinstance(obj, torch.Tensor):
            return obj.cuda()
        elif hasattr(obj, '__dataclass_fields__'):
            for field_name in obj.__dataclass_fields__:
                val = getattr(obj, field_name)
                setattr(obj, field_name, move_to_cuda(val))
            return obj
        elif isinstance(obj, list):
            return [move_to_cuda(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: move_to_cuda(v) for k, v in obj.items()}
        return obj


    input_folder = test_img_folder

    import os
    from PIL import Image
    from pycocotools.coco import COCO


    image_paths = [
        os.path.join(test_img_folder, f)
        for f in os.listdir(test_img_folder)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    pil_images = [Image.open(p).convert("RGB") for p in image_paths]

    print(f"Loaded {len(pil_images)} images")

    coco = COCO(test_ann_path)

    img_ids = coco.getImgIds()
    print("Number of images:", len(img_ids))


    image_paths = []

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        full_path = os.path.join(test_img_folder, file_name)
        image_paths.append(full_path)

    print(image_paths[:3])

    pil_images = [Image.open(p).convert("RGB") for p in image_paths]

    # -------------------------
    # WARMUP
    # -------------------------

    prompts = ["IM", "OM"]

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    total_time = 0

    for _ in range(10):
        dp, _ = make_datapoint(pil_images[0], prompts)
        dp = transform(dp)
        batched = collate_fn_api(batch=[dp], dict_key="find", with_seg_masks=False)["find"]
        batched = move_to_cuda(batched)

        h, w = dp.images[0].size
        num_queries = len(prompts)  # = 2
        target_sizes_boxes = torch.tensor([[h, w]] * num_queries, device='cuda')  # ← fixed
        target_sizes_masks = torch.tensor([[h, w]] * num_queries, device='cuda')  # ← fixed

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(batched)
            _ = postprocessor(outputs[-1], target_sizes_boxes, target_sizes_masks)

    torch.cuda.synchronize()


    # -------------------------
    # TIMING
    # -------------------------
    total_time = 0
    for img in pil_images:
        dp, _ = make_datapoint(img, prompts)
        dp = transform(dp)
        batched = collate_fn_api(batch=[dp], dict_key="find", with_seg_masks=False)["find"]
        batched = move_to_cuda(batched)

        h, w = dp.images[0].size
        num_queries = len(prompts)  # = 2
        target_sizes_boxes = torch.tensor([[h, w]] * num_queries, device='cuda')  # ← fixed
        target_sizes_masks = torch.tensor([[h, w]] * num_queries, device='cuda')  # ← fixed

        starter.record()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(batched)
            results = postprocessor(outputs[-1], target_sizes_boxes, target_sizes_masks)
        ender.record()

        torch.cuda.synchronize()
        total_time += starter.elapsed_time(ender)

    avg_time_per_image = total_time / len(pil_images)
    fps = 1000 / avg_time_per_image
    print(f"Speed test for {Model}-{Model_Electron_Dose}:")
    print(f"Avg time: {avg_time_per_image:.2f} ms")
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "Avg Time_Image"] = avg_time_per_image
    print(f"FPS: {fps:.2f}")
    df.loc[all_condition, "FPS"] = fps
    #-----------------------------------------------

    import torch

    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
        print("GPU Count:", torch.cuda.device_count())
        print("Current Device:", torch.cuda.current_device())
    else:
        print("No CUDA GPU detected.")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)

        print("Total Memory (GB):", props.total_memory / 1e9)
        print("Multiprocessors:", props.multi_processor_count)
        print("Compute Capability:", f"{props.major}.{props.minor}")


    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory (GB): {props.total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Compute Capability: {props.major}.{props.minor}")

    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")


    # CLEAR MEMORY
    import gc
    import torch
    del output, results, dp, pil_image, image
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_s3, test_img_folder, test_ann_path
