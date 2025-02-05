# UNet_MangaText_Segmentation

This repository contains an implementation of a **U-Net-based model** for text segmentation in manga images. The model is trained using the Hugging Face `Trainer` API and achieves high accuracy in detecting and segmenting text regions. The output masks are used to extract bounding boxes for downstream tasks like OCR (Optical Character Recognition).

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Training](#training)
4. [Inference](#inference)
5. [Sample Results](#sample-results)
6. [Requirements](#requirements)
7. [Usage](#usage)
8. [Acknowledgments](#acknowledgments)

---

## Overview

The goal of this project is to segment text regions in manga images using a **U-Net architecture**. The model predicts binary masks for text regions, which are then converted into bounding boxes. This pipeline can be used as a preprocessing step for OCR systems or other text-related tasks.

---

## Model Architecture

The model is based on the **U-Net architecture**, which is widely used for image segmentation tasks. Key features include:
- **Encoder**: A series of convolutional blocks with max-pooling layers to downsample the input image.
- **Bottleneck**: A central block that captures high-level features.
- **Decoder**: A series of upsampling blocks with skip connections to reconstruct the segmentation mask.
- **Output Layer**: Produces a single-channel binary mask for text regions.

The model has been optimized to reduce the number of parameters while maintaining high performance.

---

## Training

The model was trained using the Hugging Face `Trainer` API, which simplifies the training process by handling batching, logging, and evaluation.

### Training Details
- **Dataset**: Manga images with corresponding binary masks for text regions(Fixed image size [3,1024,1024] and mask size [1,1024,1024]).
- **Loss Function**: BCE.
- **Optimizer**: AdamW with a learning rate scheduler.
- **Batch Size**: 4 (adjustable based on GPU memory).
- **Epochs**: 10 (adjustable based on convergence).

---

## Inference

Once trained, the model can be used to predict text masks for new manga images. The predicted masks are post-processed to extract bounding boxes around text regions.

### Steps for Inference
1. Load the trained model weights.
2. Pass image path to inference function and image size as 1024 to get the predicted mask.
3. Use bounding box extraction function to identify text regions.
4. Plot the bounding boxes on the original image.

---

## Sample Results

Below are three sample images with bounding boxes plotted using the trained model:

### Sample 1
![Sample 1](samples/Figure_1.png)
*Description: Text regions in a manga panel are accurately detected and bounded.*

### Sample 2
![Sample 2](images/Figure_2.png)
*Description: Multiple text regions are segmented and boxed correctly.*

### Sample 3
![Sample 3](images/Figure_3.png)
*Description: Fine-grained text detection in complex backgrounds.*

---
