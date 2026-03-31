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
The low dose and ultralow dose test images of *Pantoea* sp. YR343 and annotations are located within this folder in both COCO and YOLO formats. The images have also be resized to 640 x 640 and 1024 x 1024. Please use the exact folder format, `_annotations.json` text and annotation class order of 0 = inner membrane (IM) and 1 = outer membrane (OM) for replacing these datasets with your own dataset.  
##### Drop
##### Misc
##### Seed
##### Root
##### top_model_decision_tree.ipynb
##### top_model_table.csv


### Run the Framework
1. 
