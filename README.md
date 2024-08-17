
# Semi-supervised Semantic Segmentation with Multi-Level Consistency Learning
 Abstract:—Consistency regularization has prevailed in semi-supervised semantic segmentation and achieved promising performance. However, existing methods typically concentrate on enhancing the Image-augmentation based Prediction consistency and optimizing the segmentation network as a whole, resulting in insufficient utilization of potential supervisory information. In this paper, we propose a Multi-Level Consistency Learning (MLCL) approach to facilitate the staged enhancement of the encoder and decoder. Specifically, we first design a feature knowledge alignment (FKA) strategy to promote the feature consistency learning of the encoder from image-augmentation. Our FKA encourages the encoder to derive consistent features for strongly and weakly augmented views from the perspectives of point-to-point alignment and prototype-based intra-class compactness. Moreover, we propose a self-adaptive intervention (SAI) module to increase the discrepancy of aligned intermediate feature representations, promoting Feature-perturbation based Prediction consistency learning. Self-adaptive feature masking and noise injection are designed in an instance-specific manner to perturb the features for robust learning of the decoder.  Experimental results on Pascal VOC2012 and Cityscapes datasets demonstrate that our proposed MLCL achieves new state-of-the-art performance. 

 # Pipeline
 ![Network](https://github.com/MKSAQW/MLCL/blob/main/Network.png)

# Installation
> pip install -r requirements.txt

# Datasets
We have demonstrated state-of-the-art experimental performance of our method on Pascal VOC2012 and Cityscapes datasets.
You can download the Pascal VOC2012 on [this](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html).

You can download the Cityscapes on [this](https://www.cityscapes-dataset.com/).

# Training 
## How to train on Pascal VOC2012
### If training is performed on the 1/2 setting, set the configuration file for the VOC dataset, set the path  for labeled data and the path  for unlabeled data, as well as the corresponding training model parameter storage path. Here is an example shell script to run UCCL on Pascal VOC2012 :

     CUDA_VISIBLE_DEVICES=0,1 nohup  python -m torch.distributed.launch --nproc_per_node=2 --master_port=1001   MLCL.py >VOC_1_2.log &

## How to train on Cityscapes
### If training is performed on the 1/2 setting, set the configuration file for the Cityscapes dataset, set the path  for labeled data and the path  for unlabeled data, as well as the corresponding training model parameter storage path. Here is an example shell script to run UCCL on Pascal VOC2012 :

     CUDA_VISIBLE_DEVICES=0,1,2,3 nohup  python -m torch.distributed.launch --nproc_per_node=4 --master_port=6719   MLCL.py >Cityscapes_1_2.log &

#  Results on original Pascal VOC2012.
<img src="https://github.com/YUKEKEJAN/UCCL/blob/main/Table1.png" width="600" alt="Results on original Pascal VOC2012">

#  Results on blended Pascal VOC2012.
<img src="https://github.com/YUKEKEJAN/UCCL/blob/main/Table2.png" width="600" alt="Results on blended Pascal VOC2012">

#  Results on Cityscapes.
<img src="https://github.com/YUKEKEJAN/UCCL/blob/main/Table3.png" width="600" alt="Results on Cityscapes">

#  Comparison of visualization results on Pascal VOC2012 and Cityscapes.
<img src="https://github.com/YUKEKEJAN/UCCL/blob/main/Visual.png" width="600" alt="Results on Pascal VOC2012">

  
