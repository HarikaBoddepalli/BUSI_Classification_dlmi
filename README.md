# BUSI Breast Ultrasound Classification

A deep learning project for classifying breast ultrasound images into **Normal**, **Benign**, and **Malignant** categories using ResNet50 transfer learning.

---

## Project Overview

This project was built as part of the **Deep Learning for Medical Imaging (DLMI)** course. The goal is to train a model that takes a breast ultrasound image as input and predicts whether it is Normal, Benign, or Malignant — without data leakage between patients.

---

## Dataset

**BUSI — Breast Ultrasound Images Dataset**

| Class | Images |
|---|---|
| Benign | 437 |
| Malignant | 210 |
| Normal | 133 |
| **Total** | **780** |

> Mask files (`_mask.png`) are excluded. Dataset available on [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

---

## Model

- **Architecture:** ResNet50 pretrained on ImageNet
- **Custom Head:** GlobalAvgPool → FC(512) → FC(128) → FC(3) + Softmax
- **Framework:** TensorFlow / Keras
- **Training:** Two-phase fine-tuning
  - Phase 1: Backbone frozen, head only (LR = 1e-3)
  - Phase 2: conv5_block unfrozen (LR = 1e-5)

---

## Key Features

- **Patient Hold-Out Split** — no data leakage between train/val/test
- **CLAHE preprocessing** — enhances ultrasound contrast
- **Class weights** — handles Benign/Malignant/Normal imbalance
- **Data augmentation** — flips, rotation, zoom, brightness, noise (training only)

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 75.83% |
| Macro F1 Score | 74.04% |
| Test Images | 120 |
| Train / Val / Test Split | 545 / 115 / 120 |

---


## Requirements

```
tensorflow >= 2.19
opencv-python
scikit-learn
matplotlib
seaborn
kaggle
```
