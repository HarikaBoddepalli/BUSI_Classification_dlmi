# BUSI Breast Ultrasound Classification — Assignment 3
## Depthwise Separable Convolutions & MobileNetV1

Built on top of **Assignment 2** (ResNet50 transfer learning). This assignment focuses on **proper loss function design** and computationally efficient architectures using Depthwise Separable Convolutions (DSC).

---

## Assignment Focus
- Satisfy all **3 desirable properties of a loss function** (monotonically increasing, differentiable, faster convergence)
- Replace standard convolutions with **DSC** and observe the loss
- Train **MobileNetV1** and report accuracy + trainable parameters

---

## Dataset
**BUSI — Breast Ultrasound Images Dataset** (780 images, masks excluded)

| Class | Images | Class Weight |
|---|---|---|
| Benign | 437 (56%) | 0.5956 |
| Malignant | 210 (27%) | 1.2358 |
| Normal | 133 (17%) | 1.9534 |

**Split:** 545 train / 115 val / 120 test — patient hold-out, zero overlap verified

---

## Parts

### Part A — ResNet50 + DSC Head + ELU
- Replaced the Dense classification head with **2 Depthwise Separable Convolution blocks**
- Used **ELU activation** instead of ReLU (no dead neurons, centred output)
- DSC applied on the **(7×7×2048) feature map** before pooling
- Two-phase training: head-only (LR=2e-3) → conv5_block fine-tune (LR=1e-5)

| Metric | Value |
|---|---|
| Test Accuracy | 67.50% |
| Macro F1 | 55.15% |
| Trainable Params (Phase 1) | 1,143,939 |
| DSC Saving (K=3, Cout=512) | 88.7% fewer computations |

### Part B — MobileNetV1 + ELU
- Full DSC backbone — 28 DSC layers throughout the entire network
- ELU head: Dense(256) + BN + ELU + Dense(64) + BN + ELU + Dense(3, softmax)
- Phase 1 best model used (Phase 2 did not improve — only 545 training images)

| Metric | Value |
|---|---|
| Test Accuracy | 75.00% |
| Macro F1 | 73.33% |
| Total Params | 3,509,187 |
| Trainable Params (Phase 1) | 279,683 |
| vs ResNet50 | **85.8% fewer parameters** |

---

## Key Design Decisions

**Loss Function — Categorical Cross-Entropy** satisfies all 3 properties:
1. Monotonically increasing: `L = −Σ y_true · log(y_pred)` — loss always increases when prediction worsens
2. Differentiable: gradient = `y_pred − y_true` — smooth everywhere, no halting problem
3. Faster convergence: gradient ∝ prediction error — large errors → fast updates

**ELU Activation** — no dead neurons proof:
- For x > 0: `f'(x) = 1` → gradient constant, no decay
- For x < 0: `f'(x) = α·e^x > 0` → always positive, neuron never dies

**DSC Ratio** (from class formula):
```
C(DSC) / C(RC) = (1/Cout) + (1/K²)
For K=3, Cout=512 → saving = 88.7%
```

---

## How to Run

1. Open `BUSI_DSC_MobileNet.ipynb` in **Google Colab**
2. Set Runtime → **GPU (T4)**
3. Run all cells top to bottom
4. Upload `kaggle.json` when prompted in Step 1

---

## Final Comparison

| Model | Accuracy | Macro F1 | Trainable Params |
|---|---|---|---|
| Assignment 2 — ResNet50 + ReLU | 75.83% | 74.04% | 1,116,419 |
| Part A — ResNet50 + DSC + ELU | 67.50% | 55.15% | 1,143,939 |
| Part B — MobileNetV1 + ELU | 75.00% | 73.33% | 279,683 |

**Takeaway:** MobileNetV1 achieves the same accuracy as ResNet50 with **85.8% fewer parameters** — proving DSC is computationally efficient without sacrificing performance when used as a complete architecture.

---

## 🛠️ Requirements
```
tensorflow >= 2.19
opencv-python
scikit-learn
matplotlib
seaborn
kaggle
```
