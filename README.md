# AI-Pipeline-for-Image-Segmentation-and-Object-Analysis

## Overview

This project provides a pipeline for image segmentation and object analysis using advanced models. It includes a Streamlit application for a user-friendly interface where users can upload images or directories of images to be processed. The pipeline uses the SAM model for segmentation, ResNet for object identification, EasyOCR for text extraction, Summarize Object Attributes using predefined dictionaries, and Data Mapping output in a single json file. The results include segmented images, segmented objects, and metadata.

## Setup Instructions
python 3.10 or higher
# Download the default model from here: 
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 
and edit the path of model in the ocr_seg_iden_model.py file 

# Install Required Packages
pip install -r requirements.txt

# Usage Guidelines
streamlit run app.py

## system requirements
tested on GP100 and RTX4060 mobile GPUs

