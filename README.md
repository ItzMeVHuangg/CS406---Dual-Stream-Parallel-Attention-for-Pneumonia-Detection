# CS406--Dual-Stream PAFM for Pneumonia Detection on Chest X-Ray

## Overview

This project focuses on automatic pneumonia detection from chest X-ray (CXR) images using deep learning-based object detection models.

The study evaluates multiple detection architectures:

* Faster R-CNN with ResNet50-FPN
* RetinaNet with ResNet50-FPN
* Anchor-Free One-Stage Detector
* Proposed Dual-Stream PAFM (Parallel Attention Fusion Module)

The proposed Dual-Stream architecture improves feature extraction by combining:

* Original chest X-ray images
* Contrast-enhanced chest X-ray images

through a parallel attention fusion mechanism.

---

## Problem Statement

Pneumonia detection from chest X-ray images is a challenging medical imaging task due to:

* Low contrast lung opacity regions
* Blurry lesion boundaries
* Overlapping anatomical structures
* Severe class imbalance

The objective of this project is:

* Detect pneumonia regions automatically
* Predict bounding boxes for lesion localization
* Improve detection accuracy using dual-stream attention-based fusion

---

## Proposed Architecture

### 1. Faster R-CNN Baseline

Two-stage object detector using:

* ResNet50 backbone
* Feature Pyramid Network (FPN)
* Region Proposal Network (RPN)

Strengths:

* High localization accuracy

Limitations:

* Slow inference speed

---

### 2. RetinaNet Baseline

One-stage detector optimized for speed.

Key components:

* ResNet50 backbone
* Feature Pyramid Network (FPN)
* Focal Loss for class imbalance

Strengths:

* Faster inference
* Better handling of hard samples

---

### 3. Anchor-Free Detector

Anchor-free one-stage detector that predicts:

* Object centers
* Bounding box offsets directly

Advantages:

* No anchor box tuning
* Simpler detection pipeline
* Better flexibility for irregular lesion shapes

---

### 4. Proposed Dual-Stream PAFM

Main contribution of this project.

#### Dual-Stream Input

The model processes two parallel image streams:

| Stream   | Description                     |
| -------- | ------------------------------- |
| Stream 1 | Original normalized X-ray image |
| Stream 2 | Histogram-equalized image       |

#### Parallel Attention Fusion Module (PAFM)

The fusion module applies:

* Cross-attention
* Residual fusion
* Feature refinement

This allows the model to:

* Enhance subtle pneumonia regions
* Capture complementary features
* Improve localization robustness

#### Backbone

* ResNet50
* Feature Pyramid Network (FPN)

#### Optimizer

* AdamW

#### Scheduler

* Cosine Annealing Learning Rate

---

## Model Pipeline

```text
Chest X-Ray Image
        │
        ▼
Dual-Stream Processing
 ├── Original Image
 └── Contrast Enhanced Image
        │
        ▼
ResNet50 Backbones
        │
        ▼
Parallel Attention Fusion Module
        │
        ▼
Feature Pyramid Network (FPN)
        │
        ▼
Region Proposal Network (RPN)
        │
        ▼
Detection Head
        │
        ▼
Bounding Box Prediction
```

---

## Dataset

### RSNA Pneumonia Detection Challenge

Dataset source:

* RSNA Pneumonia Detection Challenge (2018)

Dataset characteristics:

* Chest X-ray grayscale images
* Bounding box annotations
* Pneumonia localization labels

### Dataset Statistics

| Split    | Samples |
| -------- | ------- |
| Training | 30,227  |
| Testing  | 3,000   |

---

## Data Augmentation

### Geometric Transformations

* Horizontal Flip
* Shift
* Scale
* Rotation

### Pixel-Level Transformations

#### Stream 1

* ColorJitter

#### Stream 2

* Histogram Equalization

### Preprocessing

* Resize to 512×512
* Normalize using ImageNet statistics
* Tensor conversion

---

## Evaluation Metrics

### 1. IoU (Intersection over Union)

Measures overlap between:

* Predicted bounding box
* Ground truth bounding box

---

### 2. mAP (Mean Average Precision)

Evaluated across multiple IoU thresholds:

```text
0.40 → 0.75 (step 0.05)
```

---

### 3. F1-Score

Measures balance between:

* Precision
* Recall

Important for medical diagnosis tasks.

---

## Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Input Size    | 512 × 512        |
| Optimizer     | AdamW            |
| Learning Rate | 2e-5             |
| Weight Decay  | 0.01             |
| Scheduler     | Cosine Annealing |
| Batch Size    | 4                |
| Epochs        | 15               |

Additional techniques:

* Mixed Precision Training (AMP)
* Gradient Clipping
* Warmup Strategy
* Transfer Learning from ImageNet

---

## Technologies Used

### Frameworks

* PyTorch
* Torchvision
* Scikit-learn
* Albumentations

### Hardware

* NVIDIA CUDA GPU

---

## Installation

Clone repository:

```bash
git clone <your-repository-url>
cd <repository-folder>
```

Install dependencies:

```bash
pip install torch torchvision
pip install albumentations scikit-learn
pip install matplotlib numpy pandas tqdm
```

---

## Project Structure

```text
.
├── ipynb/
│   ├── DualStream-PAFM.ipynb
│   ├── RetinaNet.ipynb
│   ├── anchor-free.ipynb
│   └── faster-rcnn-resnet50fpn.ipynb
│
├── CS406_Report.pdf
├── README.md
└── outputs/
```

### File Descriptions

| File                          | Description                                   |
| ----------------------------- | --------------------------------------------- |
| DualStream-PAFM.ipynb         | Proposed Dual-Stream PAFM implementation      |
| RetinaNet.ipynb               | RetinaNet baseline experiment                 |
| anchor-free.ipynb             | Anchor-free detector implementation           |
| faster-rcnn-resnet50fpn.ipynb | Faster R-CNN baseline with ResNet50-FPN       |
| CS406_Report.pdf              | Full project report and experimental analysis |

---

## Running Training

Train baseline model:

```bash
python train.py --model faster_rcnn
```

Train RetinaNet:

```bash
python train.py --model retinanet
```

Train Dual-Stream PAFM:

```bash
python train.py --model dual_stream_pafm
```

---

## Inference

Run prediction on test images:

```bash
python inference.py --weights checkpoints/best_model.pth
```

Outputs:

* Predicted bounding boxes
* Confidence scores
* Visualization images

---

## Results

The proposed Dual-Stream PAFM architecture improves:

* Feature representation quality
* Detection robustness
* Localization accuracy

Compared to standard single-stream baselines.

Key improvements:

* Better detection of low-contrast lesions
* Reduced missed pneumonia regions
* More stable training with attention fusion

---

## Future Work

Potential future improvements:

* Vision Transformers (ViT)
* Swin Transformer backbones
* Mamba-based medical vision models
* Segmentation-based lesion localization
* Self-supervised pretraining
* Multi-modal clinical integration
* Lightweight deployment optimization

---

## Academic Context

This project was developed for:

* Course: Image Processing and Applications (CS406)
* University of Information Technology (UIT)
* Vietnam National University Ho Chi Minh City

---

## Authors

* Mai Thai Binh
* Dang Van Duy
* Vu Viet Hoang
* Le Ngoc Thanh

---

## References

Main references include:

* Faster R-CNN
* RetinaNet
* RSNA Pneumonia Detection Challenge
* Feature Pyramid Networks (FPN)
* Attention-based object detection methods
