# Top Model Decision Tree: CryoEM Bacterial Membranes
### Background
Multiple deep learning model architectures can be used to segment bacterial membranes in cryoEM images. However, an AI-based tool advancement is often presented with only a single segmentation model for broad use, and this single model may show inconsistent results across datasets from different users. Here, we present the Top Model Decision Tree, a model screening framwework to screen for the best model to generate bacterial inner and outer membrane masks based on user priorities. We use pre-trained segmentation models from YOLOv11, YOLO26, U-Net, Detectron2 and SAM3 fine-tuned on bacterial inner and outer membranes imaged with cryoEM. 
### Overview
This repository hosts a model screening framework for *Pantoea* sp. YR343 low dose and ultralow dose cryo-electron microscopy (cryoEM) datasets. This framework can be used as a plug and play to select the top models that output segmentation masks used in AI-based tool pipelines. We have chosen the Bacterial Cell Envelope Tool hosted at Constellation (DOI: 10.13139/ORNLNCCS/2997581) and GitHub (https://github.com/Sireesiru/Cryo-TEM-Ultrastructures in Membrane_Thickness_Tool.ipynb) as a representative AI-based tool. This workflow streamlines for model selection process to target tool compatibility and tool scalability across cryoEM imaging conditions. 

<div align="center">
<img width="569" height="408" alt="image" src="https://github.com/user-attachments/assets/e64ff81f-c265-487c-ad3b-95ea3f7cdd05" />
</div>

### Run the Framework
1. This notebook must be opened in Google Colab.  Mount Google Drive and run with a GPU-based runtime. ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)
(https://colab.research.google.com/github/Lynnicia/CryoEM_membranes_top_model_decision_tree/blob/main/top_model_decision_tree.ipynb)
2. Open the notebook and follow steps to git clone in folders and files within this repository. There will be a repeating top_model_decision_tree.ipynb (notebook clone) that will not be used.
3. Save your csv table outputs within your Google Drive or download before closing the notebook. 

### Framework Workflow
#### 1. Select mode for electron dose and test images
Load in your top model checkpoints and architecture. Next, load in test images. 
#### 2. Select mask count
Prepare a manual count of bacteria in your test images. We used 15 images to manually count a total of 20 bacteria. Around 10-20 images can be used for a manual count with a custom dataset. This will output predicted masks and mask counts for the test images. Refer to https://github.com/Sireesiru/Cryo-TEM-Ultrastructures or Constellation (DOI: 10.13139/ORNLNCCS/2997581) for the notebook and instructions to use the Bacterial Cell Envelope Thickness Tool.
#### 3. Select evaluation critera
This will output metrics for the class of interest. This has been hard-coded to output All classes (averaged OM and IM metrics), OM class and IM class. 

### Repository Contents
#### Datasets 
The low dose and ultralow dose test images of *Pantoea* sp. YR343 and annotations are located within this folder in both COCO and YOLO formats (Roboflow). The images have also been resized to 640 x 640 and 1024 x 1024. 

For custom datasets, use the Datasets folder for correct data processing. Upload images (mask only) and annotations (metrics) from your local computer to the Google Colab Datasets folder. Please use the exact folder names, folder levels, `_annotations.coco.json` text and annotation class order of 0 = inner membrane (IM) and 1 = outer membrane (OM). In addition, please resize your images to either 640 x 640 or 1024 x 1024 before proceeding. 
#### Masks (generated in Notebook)
Output IM and OM binary masks with all models based on user preference for test image size and test electron dose. 
#### Drop
Test image python code is located in this folder. If using a custom dataset, be sure to resize your images and refactor your dataset path to exactly match this Dataset folder setup (i.e. resize your images to 1024 x 1024, then change your path from [your_folder/test] and [your_folder/test/_annotations.coco.json] to [/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/1024] and [/content/CryoEM_membranes_top_model_decision_tree/Datasets/LD/COCO/test/1024/_annotations.coco.json]). 
#### Misc
Example Detectron2 and SAM3 segmentation model training steps. Please use your own data and Hugging Face token where applicable as these training steps are for demonstrative purposes only. For U-Net, Refer to https://github.com/Sireesiru/Semantic-Segmentation-of-bacterial-cell-envelope-using-U-Nets to be forked to a more in-depth example of U-Net segmentation model training steps. Refer to https://github.com/Sireesiru/Cryo-TEM-Ultrastructures for steps on how to train on a custom dataset, check README.md. Demonstrations for YOLOv11 and YOLO26 segmentation model training can be found at https://github.com/ultralytics/ultralytics.
#### Root
Mask and metrics screening python code. Tools to generate masks, metrics and speed tests for test images with YOLOv11, YOLO26, U-Net, Detectron2 and SAM3 fine-tuned pre-trained models. 
#### Seed
Python code for top models and model architectures. 
#### |  Models
All best model checkpoints are located in this subfolder. Placeholder python code to load in all model checkpoints. A placeholder has been added for models too large to add to this repository. Please run the placeholder routine in the misc folder to load in all best model chekpoints. All models will be either loaded from Constellation (DOI: 10.13139/ORNLNCCS/3025228), GitHub releases (YOLOv11, YOLO26, U-Net and Detectron2) or from Hugging Face (SAM3). 
#### Top (generated in Notebook)
Python code to sort top_model_table.csv and output a top 5 (or top #) subset table based on metrics of interest. 
#### top_model_decision_tree.ipynb
Main Notebook to run the Top Model Decision Tree. This notebook is only compatible with Google Colab. Open this notebook in Google Colab and git clone the remaining folder components to use the framework.
#### top_model_table.csv
Blank .csv file that will populate with metrics based on test image inferences. 
#### example_top_model_table.csv
Example .csv output from the Top Model Decision Tree based on example LD and ULD test images.

### Future Outlook
This framework presented in Constellation and GitHub is not limited to only bacterial membrane segmentation. Please feel free to restructure the model architectures to adapt to your fine-tuned pre-trained model checkpoints to custom datasets. 

__________________________________________________________________________________________



This readme file repository update was generated on 2025-05-21 by Lynnicia Massenburg


GENERAL INFORMATION

1. Title of Dataset:  CryoEM_membranes_top_model_decision_tree

2. Author Information
	A. Principal Investigator Contact Information
		Name: Alexis Williams
		ORCID: 0000-0002-5283-5822
		Institution: Oak Ridge National Laboratory
		Email: williamsan@ornl.gov

	B. Alternate Contact Information
		Name: Lynnicia Massenburg
		ORCID: 0000-0002-6590-273X
		Institution: Oak Ridge National Laboratory
		Email: massenb2@hotmail.com 

3. Date of data collection: 2025-06-26

4. Geographic location of data collection: Oak Ridge, TN

5. Information about funding sources that supported the collection of the data: 

This work is supported by the U.S. Department of Energy, Office of Science FWP ERKCZ64, Structure Guided Design of Materials to Optimize the Abiotic-Biotic Material Interface, as part of the Biopreparedness Research Virtual Environment (BRaVE) initiative. Sample preparation, imaging and image analysis were conducted as part of a user project at the Center for Nanophase Materials Sciences (CNMS), which is a US Department of Energy, Office of Science User Facility at Oak Ridge National Laboratory. Electron microscopy data was collected using instrumentation within ORNL's Materials Characterization Core provided by UT-Battelle, LLC, under Contract No. DE-AC05- 00OR22725 with the DOE and sponsored by the Laboratory Directed Research and Development Program of Oak Ridge National Laboratory, managed by UT-Battelle, LLC, for the U.S. Department of Energy.

SHARING/ACCESS INFORMATION

1. Reuse restrictions placed on the data:  MIT license

2. Links to publications that cite or use the data:  (paper publication in progress)

3. Links to other publicly accessible locations of the data:  N/A

4. Links/relationships to ancillary data sets: 

GitHub (same dataset): https://github.com/Lynnicia/CryoEM_membranes_top_model_decision_tree

HuggingFace (same dataset): 
https://huggingface.co/LynnMass/640-SAM3-ULD
https://huggingface.co/LynnMass/640-SAM3-LD
https://huggingface.co/LynnMass/1024-SAM3-LD
https://huggingface.co/LynnMass/1024-SAM3-ULD

5. Was data derived from another source? If yes, list source(s):  No

6. Recommended citation for this dataset: 

Massenburg, L. N., Madugula, S. S., Brown, S. R., Bible, A. N., Zhang, L., Parker, K., Retterer, S.T., Morrell-Falvey, J.L., Vasudevan, R. K. and Williams, A. (2026). Dataset for Top Model Decision Tree: Selecting Segmentation Models for Reliable Quantitative Analysis in Low- and Ultralow-Dose CryoEM. Constellation. DOI: 10.13139/ORNLNCCS/3025229


DATA & FILE OVERVIEW

1. File List: 

```
FOLDER: Datasets
	SUBFOLDER: LD
		SUBFOLDER: COCO
			SUBFOLDER: test
				SUBFOLDER: 640
					FILES: [images].jpg (15)
					FILE: _annotations.coco.json
					FILE: _filt_annotations.coco.json
				SUBFOLDER: 1024
					FILES: [images].jpg (15)
					FILE: _annotations.coco.json
					FILE: _filt_annotations.coco.json
				SUBFOLDER: 2048				
					FILES: [images].jpg (15)
					FILE: _annotations.coco.json
				SUBFOLDER: 4096
					FILES: [images].jpg (15)
					FILE: _annotations.coco.json
		SUBFOLDER: YOLO
			SUBFOLDER: test
				SUBFOLDER: images
					FILES: [images].jpg (15)
				SUBFOLDER: labels
					FILES: [images].txt (15)
				FILE: labels.cache
			FILE: LD_test.yaml
	SUBFOLDER: ULD
		SUBFOLDER: COCO
			SUBFOLDER: test
				SUBFOLDER: 640
					FILES: [images].jpg (18)
					FILE: _annotations.coco.json
					FILE: _filt_annotations.coco.json
				SUBFOLDER: 1024
					FILES: [images].jpg (18)
					FILE: _annotations.coco.json
					FILE: _filt_annotations.coco.json
				SUBFOLDER: 2048				
					FILES: [images].jpg (18)
					FILE: _annotations.coco.json
				SUBFOLDER: 4096
					FILES: [images].jpg (18)
					FILE: _annotations.coco.json
		SUBFOLDER: YOLO
			SUBFOLDER: test
				SUBFOLDER: images
					FILES: [images].jpg (18)
				SUBFOLDER: labels
					FILES: [images].txt (18)
				FILE: labels.cache
			FILE: LD_test.yaml

FOLDER: drop
	SUBFOLDER: __pycache__
	FILE: __init__.py
	FILE: coco_bacteria_dataset.py
	FILE: custom_test_images.py
	FILE: drop_test_images.py
	FILE: LD_custom_test_images.py
	FILE: LD_test_images.py
	FILE: ULD_custom_test_images.py
	FILE: ULD_test_images.py

FOLDER: misc
	FILE: Detectron2_example_segmentation_model_training.ipynb
	FILE: SAM3_example_segmentation_model_training (1).ipynb

FOLDER: root
	SUBFOLDER: __pycache__
	FILE: mask_decision_tree_d2.py
	FILE: mask_decision_tree_s3.py
	FILE: mask_decision_tree_u.py
	FILE: mask_decision_tree_y26.py
	FILE: mask_decision_tree_yv11.py
	FILE: metrics_decision_tree_d2.py
	FILE: metrics_decision_tree_s3.py
	FILE: metrics_decision_tree_u.py
	FILE: metrics_decision_tree_y26.py
	FILE: metrics_decision_tree_yv11.py
	FILE: speed_decision_tree_d2.py
	FILE: speed_decision_tree_s3.py
	FILE: speed_decision_tree_u.py
	FILE: speed_decision_tree_y26.py
	FILE: speed_decision_tree_yv11.py
	
FOLDER: seed
	SUBFOLDER: __pycache__
	SUBFOLDER: models
		FILE: model_repo.ipynb
		FILE: 640-YOLOv11-LD
		FILE: 640-YOLOv11-ULD
		FILE: 1024-YOLOv11-LD
		FILE: 1024-YOLOv11-ULD
		FILE: 640-YOLO26-LD
		FILE: 640-YOLO26-ULD
		FILE: 1024-YOLO26-LD
		FILE: 1024-YOLO26-ULD
		FILE: 640-U-Net-LD
		FILE: 640-U-Net-ULD
		FILE: 1024-U-Net-LD
		FILE: 1024-U-Net-ULD		
		FILE: 640-Detectron2-LD
		FILE: 640-Detectron2-ULD
		FILE: 1024-Detectron2-LD
		FILE: 1024-Detectron2-ULD
		FILE: 640-SAM3-LD
		FILE: 640-SAM3-ULD
		FILE: 1024-SAM3-LD
		FILE: 1024-SAM3-ULD
	FILE: __init__.py
	FILE: LD_models.py
	FILE: model_arch.ipynb
	FILE: ULD_models.py
	
FILE: example_top_model_table.csv
	
FILE: top_model_decision_tree.ipynb
	
FILE: top_model_table.csv
```

2. Relationship between files: 
Cryogenic electron miroscopy images of Pantoea sp. YR343 bacteria in [images]. Raw images (.jpeg files) and annotated (YOLO text and COCO json files) images. Instances listed in this file are model segmentation predictions per class. Folders named "drop", "root" and "seed" build the Top Model Decision Tree framework. The "misc" folder provides example model training routines for Detectron2 and SAM3 for custom datasets. 

3. Additional related data collected that was not included in the current data package: 
No

4. Are there multiple versions of this dataset? If yes, what files were updated and why?
Only one version of the dataset is available. 

METHODOLOGICAL INFORMATION

1. Description of methods used for collection/generation of data: 
Madugula, S. S., Massenburg, L. N., Brown, S. R., Bible, A. N., Harris, C. R., Zhang, L. X., Parker, K., Retterer, S. T., Morrell-Falvey, J. L., Vasudevan, R. K., & Williams, A. N. (2026). Automated Bacterial Identification and Morphological Feature Analysis in Low-Dose Cryo-EM Using YOLOv11. Advanced Intelligent Discovery, n/a(n/a), e202500241. https://doi.org/https://doi.org/10.1002/aidi.202500241 

2. Methods for processing the data: 
Load images into Google Colab and run the Top Model Decision Tree framework to segment bacterial membranes. Walk through the steps outlined in top_model_decision_tree.ipynb to obtain masks and/or metrics. 

3. Instrument- or software-specific information needed to interpret the data: 
YOLOv11, YOLO26, U-Net, Detectron2, SAM3

4. Standards and calibration information, if appropriate: 
N/A

5. Environmental/experimental conditions: 
Manual cryoEM image collection was described in Madugula et al. for low-dose images collected at 40 e⁻/Å2 at –5 μm defocus using the Falcon 3EC direct electron detector (Thermo Fisher Scientific, counting mode) on the Thermo Fisher Scientific Krios G4 operated in nanoprobe TEM mode (Madugula et al., 2026). 

Madugula, S. S., Massenburg, L. N., Brown, S. R., Bible, A. N., Harris, C. R., Zhang, L. X., Parker, K., Retterer, S. T., Morrell-Falvey, J. L., Vasudevan, R. K., & Williams, A. N. (2026). Automated Bacterial Identification and Morphological Feature Analysis in Low-Dose Cryo-EM Using YOLOv11. Advanced Intelligent Discovery, n/a(n/a), e202500241. https://doi.org/https://doi.org/10.1002/aidi.202500241 

6. Describe any quality-assurance procedures performed on the data: 
visual check

7. People involved with sample collection, processing, analysis and/or submission: 
Massenburg, L. N., Madugula, S. S., Brown, S. R., Bible, A. N., Zhang, L., Parker, K., Retterer, S.T., Morrell-Falvey, J.L., Vasudevan, R. K. and Williams, A.

DATA-SPECIFIC INFORMATION FOR: All [images].jpg in LD subfolder

1. Number of variables: 
2 classes (class 0: inner membrane (IM), class 1: outer membrane (OM))

2. Number of cases/rows: 
15 images

3. Variable List: 
class 0: 20 instances
class 1: 20 instances

4. Codes used for missing data: 
N/A

5. Specialized formats or other abbreviations used: 
N/A

DATA-SPECIFIC INFORMATION FOR: All [images].jpg in ULD subfolder

1. Number of variables: 
2 classes (class 0: inner membrane (IM), class 1: outer membrane (OM))

2. Number of cases/rows: 
18 images
 
3. Variable List: 
class 0: 55 instances
class 1: 55 instances

4. Codes used for missing data: 
N/A

5. Specialized formats or other abbreviations used: 
N/A