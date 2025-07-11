# PID-Inspired Continuous Correction for Visual Reprogramming

This repository is the official PyTorch implementation of the paper:[PID-Inspired Continuous Correction for Visual Reprogramming].

**Key Words:**
Visual Reprogramming, Closed-loop Correction Reprogramming (CCR), PID, Proportional Adjustment Controller (PAC), Fine-grained, Machine Learnin

**Abstract:**
Visual Reprogramming (VR) adapts pre-trained models to new tasks through pixel-level attention modulation without parameter modification. While existing methods achieve competent performance on basic classification, they dispersed attention in critical discriminative regions for fine-grained tasks. Inspired by PID control theory, we propose Closed-loop Correction Reprogramming (CCR) that integrates proportional feedback. Concretely, the framework comprises dual streams: a Foundation Flow for initial attention patterns and a Correction Flow that iteratively refines them with residual feedback, alternating between both. A Proportional Adjustment Controller (PAC) dynamically calibrates perturbation intensity via learnable error mapping--enhancing the correction flow's contribution in response to increased foundational stream errors, otherwise maintaining the foundation's dependable attributes. Experiments on 11 benchmarks demonstrate CCR achieves up to 10.8% accuracy gain with only 0.64% parameter increase, attaining 8.62% average improvement on five challenging fine-grained datasets (GTSRB, FLOWERS102, DTD, UCF101, FOOD101). The framework offers enhanced visual cues that improve discrimination in fine-grained classification.

**Method:**
The **Closed-loop Correction Reprogramming (CCR)** model enhances pre-trained models for fine-grained classification tasks using **Visual Reprogramming (VR)**. It employs a **dual-stream architecture** consisting of **Foundation Flow** and **Correction Flow** to dynamically refine attention maps.

- **Foundation Flow** generates initial perturbation patterns for coarse adaptation.
- **Correction Flow** iteratively refines these patterns based on residual feedback from the Foundation Flow, improving the attention distribution for fine-grained tasks.

A key component, the **Proportional Adjustment Controller (PAC)**, estimates the perturbation magnitude during testing without needing ground-truth labels. PAC dynamically adjusts the influence of the two flows: it strengthens the Correction Flow when the Foundation Flow’s errors exceed a threshold, and preserves reliable features when errors are small.

<p align="center">
  <img src="https://anonymous.4open.science/r/CCR_VR-84D0/pic/model.png?raw=true" width=100%/>
</p>


## Dataset
- For CIFAR10, CIFAR100, GTSRB, SVHN, simply run our code and use [TorchVision](https://pytorch.org/vision/0.15/datasets.html) to download them automatically.
- For other datasets, follow [this paper](https://github.com/OPTML-Group/ILM-VP) to prepare them.

Then put all the download datasets in `/dataset/`

## Environment

- Python (3.10.0)
- PyTorch (2.5.1)+cu124
- TorchVision (0.20.1)+cu124

        pip install -r requirements.txt
  
## Training

To train the model, simply run the **CCR.ipynb** notebook. Modify the following section in the notebook to adjust the model and dataset settings:

### Model and Dataset Configuration
```python
class Args:
    network = "resnet18"  # Choose from ["resnet18", "resnet50", "ViT_B32"]
    dataset = "flowers102"   # Choose from ["cifar10", "cifar100", "gtsrb", "svhn", "food101", "eurosat", "sun397", "UCF101", "flowers102", "DTD", "oxfordpets"]
```
### Pre-trained Models
The **model_pth** directory should contain the following pre-trained models:
- **resnet18**Link：https://pan.baidu.com/s/1pOGIZkY9ndo6nkPRgrU2kQ 
Passwords：36cs 
- **resnet50**Link：https://pan.baidu.com/s/1pOGIZkY9ndo6nkPRgrU2kQ 
Passwords：36cs 
- **ViT_B32**Link：https://pan.baidu.com/s/1pOGIZkY9ndo6nkPRgrU2kQ 
Passwords：36cs 

### Training Results
The final training results will be saved in the **results** folder, allowing easy access to evaluate and analyze the outcomes of your training.

## Visual result
<p align="center">
  <img src="https://anonymous.4open.science/r/CCR_VR-84D0/pic/visual result.png?raw=true" width=100%/>
</p>

Visual results of trained VR on the Flowers102 dataset.ResNet-18 is used as the pre-trained model as an example.

## Acknowledgements

This repo is built upon these previous works:

- [lukemelas/PyTorch-Pretrained-ViT](https://github.com/lukemelas/PyTorch-Pretrained-ViT)
- [OPTML-Group/ILM-VP](https://github.com/OPTML-Group/ILM-VP)
- [tmlr-group/SMM](https://github.com/tmlr-group/SMM/tree/main)
