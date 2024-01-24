
# Introduction
 The source code and models for our paper Semi-supervised Semantic Segmentation with Multi-Level Consistency Learning

# Framework
<img width="745" alt="image" src="https://github.com/MKSAQW/MLCL/assets/155893338/7982cdc1-ac88-480e-a876-5af7f28f617f">


# Installation
  After creating a virtual environment of python 3.7, run pip install -r requirements.txt to install all dependencies

# Usage
## MLCL
### Step 1:  Prepare the required dataset (Pascal VOC2012 and Cityscapes) and set the image path in the configuration file.
### Step 2:  Run the following instruction
     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup  python -m torch.distributed.launch --nproc_per_node=8 --master_port=6719  MLCL.py >out.log &

  
