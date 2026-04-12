# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def run_model_pipeline_s3(Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_s3, test_img_folder, test_ann_path):
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

    output_folder = os.path.join(TARGET_FOLDER, f"Results_{Model}")
    os.makedirs(output_folder, exist_ok=True)


    image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.png"))

    total_bacteria = 0

    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping unreadable image {img_path}")
            continue


        # ✅ Skip already processed images
        #if os.path.exists(save_img) and os.path.exists(save_csv):
        #    print(f"Skipping {img_path} (already processed)")
        #    continue

        ibase = os.path.splitext(os.path.basename(img_path))[0]


        pil_image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = pil_image.size  # PIL is W, H

        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                dp, ids = make_datapoint(pil_image, ["IM", "OM"])
                dp = transform(dp)
                batch = collate([dp], dict_key="dummy")["dummy"]
                batch = copy_data_to_device(batch, device, non_blocking=True)
                output = model(batch)
                results = postprocessor.process_results(output, batch.find_metadatas)

        def masks_to_uint8(results, query_id, orig_height, orig_width):
            """Convert SAM3 mask tensors to list of uint8 numpy arrays (same as split_instances output)"""
            out = []
            for mask in results[query_id]["masks"]:
                m = mask.cpu().numpy().squeeze().astype(np.uint8) * 255
                if m.shape != (orig_height, orig_width):
                    m = cv2.resize(m, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
                out.append(m)
            return out

        im_masks = masks_to_uint8(results, ids["IM"], orig_height, orig_width)
        om_masks = masks_to_uint8(results, ids["OM"], orig_height, orig_width)


        # Skip if OM or IM missing
        if not om_masks or not im_masks:
            print(f"Skipping {img_path} (missing OM or IM contour)")
            continue

        # Save combined masks
        im_combined = np.maximum.reduce(im_masks)
        om_combined = np.maximum.reduce(om_masks)

        cv2.imwrite(os.path.join(output_folder, f"{ibase}_IM_mask.png"), im_combined)
        cv2.imwrite(os.path.join(output_folder, f"{ibase}_OM_mask.png"), om_combined)

        num_bacteria = len(om_masks)  # count OM only
        total_bacteria += num_bacteria
        print(f"{img_path}: {num_bacteria} bacteria")




    print(f"\nTotal predicted bacteria with SAM3 across all images: {total_bacteria}")
    """ //#####|tree_root|#####\\ """
    df.loc[all_condition, "total_bacteria"] = total_bacteria

    df.to_csv(csv_path, index=False)
    print("CSV updated successfully to Tree ✅")

    # CLEAR MEMORY
    import gc
    import torch
    del batch, output, results, dp, pil_image, image
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return Model, Model_Image_Size, Model_Electron_Dose, Test_Image_Size, Test_Electron_Dose, input_folder, TARGET_FOLDER, MODEL_PATH_s3, test_img_folder, test_ann_path
