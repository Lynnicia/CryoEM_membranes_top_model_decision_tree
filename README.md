# CryoEM_membranes_top_model_decision_tree
### Overview
This repository hosts a model screening framework for *Pantoea* sp. YR343 low dose and ultralow dose cryo-electron microscopy (cryoEM) datasets. This framework can be used to screen for the top models that output segmentation masks used in AI-based tool pipelines. We have chosen the Bacterial Cell Envelope Tool hosted at https://github.com/Sireesiru/Cryo-TEM-Ultrastructures in Membrane_Thickness_Tool.ipynb as a representative AI-based tool. This workflow streamlines for model selection process to target tool compatibility and tool scalability across cryoEM imaging conditions. 

<div align="center">
<img width="569" height="408" alt="image" src="https://github.com/user-attachments/assets/e64ff81f-c265-487c-ad3b-95ea3f7cdd05" />
</div>

### Motivation
Multiple deep learning model architectures can be used to segment bacterial membranes in cryoEM images. However, an AI-based tool advancement is often presented with only a single segmentation model for broad use, and this single model may show inconsistent results across datasets from different users. Here, we present the Top Model Decision Tree, a model screening framwework to screen for the best model to generate bacterial inner and outer membrane masks based on user priorities. We use pre-trained segmentation models from YOLOv11, YOLO26, U-Net, Detectron2 and SAM3 fine-tuned on bacterial inner and outer membranes imaged with cryoEM. 

### Framework Workflow
##### 1. 
##### 2. 
##### 3. 

### Repository Contents
##### Datasets 
The low dose and ultralow dose test images of *Pantoea* sp. YR343 and annotations are located within this folder in both COCO and YOLO formats (Roboflow). The images have also be resized to 640 x 640 and 1024 x 1024. Please use the exact folder format, `_annotations.coco.json` text and annotation class order of 0 = inner membrane (IM) and 1 = outer membrane (OM) for replacing these datasets with your own dataset.  
##### Drop
Test image python code is located in this folder. Be sure to update the dataset links for your own dataset. 
##### Misc
Placeholder python code to load in all model checkpoints. Example YOLOv11, YOLO26, Detectron2 and SAM3 segmentation model training steps. Please use your own data and Hugging Face token where applicable as these training steps are for demonstrative purposes only. For U-Net, Refer to https://github.com/Sireesiru/Semantic-Segmentation-of-bacterial-cell-envelope-using-U-Nets to be forked to a more in-depth example of U-Net segmentation model training steps.
##### Seed
Python code for top models and model architectures. 
##### |  Models
All best model checkpoints are located in this subfolder. A placeholder has been added for models too large to add to this repository. Please run the placeholder routine in the misc folder to load in all best model chekpoints. 
##### Root
Mask and metrics screening python code. 
##### top_model_decision_tree.ipynb
Main Notebook to run the Top Model Decision Tree. This notebook is only copatible with Google Colab. Open this notebook in Google Colab and git clone the remaining folder components to use the framework.
##### top_model_table.csv
Example .csv output from the Top Model Decision Tree.  

### Run the Framework
1. 
