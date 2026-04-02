# Top Model Decision Tree: CryoEM Bacterial Membranes
### Overview
This repository hosts a model screening framework for *Pantoea* sp. YR343 low dose and ultralow dose cryo-electron microscopy (cryoEM) datasets. This framework can be used to screen for the top models that output segmentation masks used in AI-based tool pipelines. We have chosen the Bacterial Cell Envelope Tool hosted at https://github.com/Sireesiru/Cryo-TEM-Ultrastructures in Membrane_Thickness_Tool.ipynb as a representative AI-based tool. This workflow streamlines for model selection process to target tool compatibility and tool scalability across cryoEM imaging conditions. 

<div align="center">
<img width="569" height="408" alt="image" src="https://github.com/user-attachments/assets/e64ff81f-c265-487c-ad3b-95ea3f7cdd05" />
</div>

### Motivation
Multiple deep learning model architectures can be used to segment bacterial membranes in cryoEM images. However, an AI-based tool advancement is often presented with only a single segmentation model for broad use, and this single model may show inconsistent results across datasets from different users. Here, we present the Top Model Decision Tree, a model screening framwework to screen for the best model to generate bacterial inner and outer membrane masks based on user priorities. We use pre-trained segmentation models from YOLOv11, YOLO26, U-Net, Detectron2 and SAM3 fine-tuned on bacterial inner and outer membranes imaged with cryoEM. 

### Framework Workflow
#### 1. What model and test image is needed?
Load in your top model checkpoints and architecture. Next, load in test images. 
#### 2. What mask count is needed?
Prepare a manual count of bacteria in your test images. We used 15 images to manually count a total of 20 bacteria. Around 10-20 images can be used for a manual count with a custom dataset. This will output predicted masks and mask counts for the test images. Refer to https://github.com/Sireesiru/Cryo-TEM-Ultrastructures for the notebook and instructions to use the Bacterial Cell Envelope Thickness Tool.
#### 3. What classes are needed? 
This will output metrics for the class of interest. This has been hard-coded to output All classes (averaged OM and IM metrics), OM class and IM class. 

### Repository Contents
#### Datasets 
The low dose and ultralow dose test images of *Pantoea* sp. YR343 and annotations are located within this folder in both COCO and YOLO formats (Roboflow). The images have also been resized to 640 x 640 and 1024 x 1024. When loading in custom datasets, please use the exact folder format, `_annotations.coco.json` text and annotation class order of 0 = inner membrane (IM) and 1 = outer membrane (OM). In addition, please resize your images to either 640 x 640 or 1024 x 1024 before proceeding. 
#### Drop
Test image python code is located in this folder. Be sure to update the dataset links for a custom dataset. 
#### Misc
Example YOLOv11, YOLO26, Detectron2 and SAM3 segmentation model training steps. Please use your own data and Hugging Face token where applicable as these training steps are for demonstrative purposes only. For U-Net, Refer to https://github.com/Sireesiru/Semantic-Segmentation-of-bacterial-cell-envelope-using-U-Nets to be forked to a more in-depth example of U-Net segmentation model training steps. Refer to https://github.com/Sireesiru/Cryo-TEM-Ultrastructures for steps on how to train on a custom dataset, check README.md.
#### Seed
Python code for top models and model architectures. 
#### |  Models
All best model checkpoints are located in this subfolder. Placeholder python code to load in all model checkpoints. A placeholder has been added for models too large to add to this repository. Please run the placeholder routine in the misc folder to load in all best model chekpoints. All models will be either loaded from the GitHub releases (YOLOv11, YOLO26, U-Net and Detectron2) or from Hugging Face (SAM3). 
#### Root
Mask and metrics screening python code. 
#### Top
Python code to sort top_model_table.csv and output a subset table. 
#### top_model_decision_tree.ipynb
Main Notebook to run the Top Model Decision Tree. This notebook is only compatible with Google Colab. Open this notebook in Google Colab and git clone the remaining folder components to use the framework.
#### top_model_table.csv
Example .csv output from the Top Model Decision Tree.  

### Run the Framework
1. This notebook must be opened in Google Colab.  Mount Google Drive and run with a GPU-based runtime. 
2. Open the notebook and follow steps to git clone in folders and files within this repository. There will be a repeating top_model_decision_tree.ipynb (notebook clone) that will not be used.
3. Save your csv table outputs within your Google Drive or download before closing the notebook. 

## Future Outlook
This framework presented in this GitHub is not limited to only bacterial membrane segmentation. Please feel free to restructure the model architectures to adapt to your fine-tuned pre-trained model checkpoints to custom datasets. 
