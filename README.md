# Order-ViT

<div align="center">

# Order-ViT: Order Learning Vision Transformer for Cancer Classification in Pathology Images

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/html/Lee_Order-ViT_Order_Learning_Vision_Transformer_for_Cancer_Classification_in_Pathology_ICCVW_2023_paper.html)
[![Conference](http://img.shields.io/badge/ICCV(Workshoh)_CVAMD_Conference-2023-4b44ce.svg)](https://cvamd2023.github.io/)
</div>

## Description

![Order-ViT](/new_model.png)

The overall model architecture is as follows. Categorical classification and sequential relationship classification problems are performed simultaneously.


## Datasets

All the models in this project were evaluated on the following datasets:

- [Colon_KBSMC](https://github.com/QuIIL/KBSMC_colon_cancer_grading_dataset) (Colon TMA from Kangbuk Samsung Hospital)
- [Colon_KBSMC](https://github.com/QuIIL/KBSMC_colon_cancer_grading_dataset) (Colon WSI from Kangbuk Samsung Hospital)
- [Gastric_KBSMC](-) (Gastric from Kangbuk Samsung Hospital)



## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/JcLee96/Order-ViT.git


# [OPTIONAL] create conda environment
conda env create -f order_vit_environment.yml
conda activate order_vit_environment

#We install Pytorch version 1.10.0 with CUDA 11.4
```

## Repository
```
/config: data and model parameter setting
/scripts: .sh file
/src: data load and augmentation, model code
```
 
## How to training for Only Categorical classification and Order-learning
```
## Only Categorical classification
# model.name = timm model name & ../train_test: Code for validating different datasets using the best model
Using /scripts/classification.sh

## Order-learning
# ../train_test: Code for validating different datasets using the best model
Using /scripts/order_learning.sh

## feature extracture for voting (Using mamory bank)
# Feature vectors for voting through the following paths are selected in advance.: ../src/models/save_features_module.py
Using /scripts/features.sh

## voting
# sub_prob: First prob - Second prob
# trust: Meaning validated datasets described in the paper
(How to correctly predict among feature vectors extracted through features.sh and select a picture vector with a probability of 0.9 or higher at the time)
Using /scripts/voting.sh
```


