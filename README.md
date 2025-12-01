# ViTX-Demo: Visualizing Vision Transformer Attention for Medical Image Localization
This repository contains a demonstration (ViTX_Demo.ipynb) of how to leverage Vision Transformers (ViT) for Explainable AI (XAI) in medical imaging. Specifically, it shows how to extract self-attention maps from a pre-trained ViT model to localize pathologies in Chest X-rays without explicit object detection training.
# 1. Project Overview
Vision Transformers process images as sequences of patches. By analyzing the self-attention weights—specifically how the Class Token ([CLS]) attends to spatial patches in the final layers—we can generate heatmaps indicating regions of high importance.
## This project implements a pipeline to:
* Load a Pretrained ViT: Uses timm to load models like vit_base_patch16_224.
* Extract Attention: Registers forward hooks to capture attention weights during inference.
* Visualize: Overlays attention heatmaps onto original X-ray images.
* Evaluate: Measures localization performance using Intersection over Union (IoU) against ground truth masks.
* ## 2. Technical Methodology
### The ViTExplainer Class
The core logic resides in the ViTExplainer class. Unlike CNNs which use Class Activation Maps (CAM), this implementation directly accesses the multi-head self-attention mechanism.
* Preprocessing: Images are resized and normalized (default ImageNet stats) to match the model's expected input (e.g., 224x224).
* Attention Hook: We register a hook on the attention modules to intercept the query-key product
* Map Generation: The code extracts the attention weights of the [CLS] token with respect to all other patch tokens from the last block. This $1 \times N_{patches}$ vector is reshaped into a 2D grid (e.g., 14x14) and bicubicly upsampled to the original image size to form a smooth heatmap.
* 
