# ViTX-Demo: Visualizing Vision Transformer Attention for Medical Image Localization
This repository contains a demonstration (ViTX_Demo.ipynb) of how to leverage Vision Transformers (ViT) for Explainable AI (XAI) in medical imaging. Specifically, it shows how to extract self-attention maps from a pre-trained ViT model to localize pathologies in Chest X-rays without explicit object detection training.
# 1. Project Overview
Vision Transformers process images as sequences of patches. By analyzing the self-attention weights—specifically how the Class Token ([CLS]) attends to spatial patches in the final layers—we can generate heatmaps indicating regions of high importance.
## This project implements a pipeline to:
* Load a Pretrained ViT: Uses timm to load models like vit_base_patch16_224.
* Extract Attention: Registers forward hooks to capture attention weights during inference.
* Visualize: Overlays attention heatmaps onto original X-ray images.
* Evaluate: Measures localization performance using Intersection over Union (IoU) against ground truth masks.
## 2. Technical Methodology

![VIT Image](https://github.com/Arjun1443/Vision-transformers-from-scratch/blob/main/VIT%20DataFlow.png)


### The ViTExplainer Class
The core logic resides in the ViTExplainer class. Unlike CNNs which use Class Activation Maps (CAM), this implementation directly accesses the multi-head self-attention mechanism.
* Preprocessing: Images are resized and normalized (default ImageNet stats) to match the model's expected input (e.g., 224x224).
* Attention Hook: We register a hook on the attention modules to intercept the query-key product
* Map Generation: The code extracts the attention weights of the [CLS] token with respect to all other patch tokens from the last block. This $1 \times N_{patches}$ vector is reshaped into a 2D grid (e.g., 14x14) and bicubicly upsampled to the original image size to form a smooth heatmap.
  
## Data Handling
The notebook supports multiple dataset formats, configured via the CFG dictionary:
* CheXlocalize: Handles hierarchical folder structures and JSON-encoded RLE masks.
* CSV Bounding Boxes: Reads standard bounding box annotations.
* Mask Directories: Loads binary segmentation masks directly.

## Evaluation Metrics
To quantify how well the attention map locates the pathology, the notebook calculates Intersection over Union (IoU) at specific thresholds (0.3, 0.5, 0.7).
* Heatmap Thresholding: The continuous attention map is binarized. Pixels with attention values above threshold $\tau$ are set to 1, others to 0.
* Metric Calculation: The code computes this overlap between the binarized attention map and the Ground Truth (GT) pathology mask.
## Getting Started
### Prerequisites

* Ensure you have the following libraries installed (as seen in Cell 1):
```python
!pip install torch timm numpy pandas pillow matplotlib

CFG = {
    "dataset.type": "chexlocalize",
    "paths.images": Path("../data/images"),
    # ...
    "model.name": "vit_base_patch16_224"
}
```
### Running the Benchmark
Execute the run_benchmark() function. It will process images up to run.max_images, calculate IoU scores, and save visualization overlays to ../results/attn_overlays.

```python
# Run benchmark and get results dataframe
df = run_benchmark()
print(df.head())
```
## Sample Results
The notebook generates overlay images where red/jet areas indicate high model attention. These visualizations verify if the model is focusing on the lungs/pathology rather than artifacts or background tags.

![VIT Image](https://github.com/Arjun1443/Vision-transformers-from-scratch/blob/main/Results.png)


