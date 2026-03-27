**English** | [日本語](README.md)

# **[Columnar ED Method]** — Extension of the Original ED Method with Cortical Column Structure

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-orange.svg)](https://numpy.org/)

## Table of Contents

- [Overview](#overview)
- [Target Reader and Fast Path](#target-reader-and-fast-path)
- [Features](#features)
- [Quick Start](#quick-start)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Usage Examples](#usage-examples)
- [Claims and Verifiability (FAQ)](#claims-and-verifiability-faq)
- [How It Works](#how-it-works)
  - [What Is the Columnar ED Method](#what-is-the-columnar-ed-method)
  - [What Is the Original ED Method](#what-is-the-original-ed-method)
  - [Weight Update via Amine Diffusion](#weight-update-via-amine-diffusion)
  - [Column Structure](#column-structure)
  - [Gabor Feature Extraction](#gabor-feature-extraction)
  - [Reservoir Computing Characteristics](#reservoir-computing-characteristics)
- [Achieved Accuracy](#achieved-accuracy)
- [Directory Structure](#directory-structure)
- [Automatic Parameter Configuration](#automatic-parameter-configuration)
- [Compliance with the Original ED Method](#compliance-with-the-original-ed-method)
- [References](#references)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

**The Columnar ED Method** is a neural network implementation that extends Isamu Kaneko's Error Diffusion learning algorithm (ED method, hereinafter "the original ED method") by introducing cortical column structure from the cerebral cortex.

The Columnar ED Method **does not use backpropagation based on the chain rule of derivatives at all**, and instead learns through biologically plausible amine diffusion mechanisms. Despite this, it achieves **97.36%** test accuracy on MNIST handwritten digit recognition (6-layer configuration, 50,000 training samples).

This repository provides two implementations:

| Implementation | File | Purpose |
|----------------|------|---------|
| **Main Version** | `columnar_ed_ann.py` | High accuracy with minimal arguments. This document covers this version. |
| Experiment Version | `columnar_ed_ann_experiment.py` | More parameters can be specified than the main version. See [README_experiment_en.md](README_experiment_en.md) for details. |

The **Main Version** is a **self-contained implementation that uses only the `modules/` directory**. It operates with just `columnar_ed_ann.py` and `modules/`, allowing you to understand the ED method and column structure without referencing the experiment version code.

## Target Reader and Fast Path

This README targets readers who already have **basic Python and machine learning knowledge** but are new to the original ED method / Columnar ED method.

Recommended reading order:

1. Run the minimum Quick Start command once
2. Compare your output with the Achieved Accuracy table at a high level
3. Read Claims and Verifiability (FAQ) for the definition and verification points of "no Backpropagation"
4. Move on to How It Works

---

## Features

### 1. No Backpropagation Based on the Chain Rule

Conventional neural networks update weights using "backpropagation based on the chain rule of derivatives," but this implementation does not use it at all. Instead, learning is performed through a mechanism that models the diffusion of neurotransmitters (amines) in the brain.

### 2. Cortical Column Structure

The column structure found in the visual cortex of the cerebral cortex is introduced, assigning a subset of neurons to specific classes. This enables multi-class classification within a single network (weight space), which was difficult for the original ED method.

### 3. Gabor Feature Extraction

Gabor filter-based feature extraction modeling the simple cells of the primary visual cortex (V1) is built in (ON by default). By improving input quality, accuracy exceeding 95% can be achieved even with a single-layer configuration under appropriate parameter settings.

### 4. Learning with Only Biologically Plausible Functions

Learning relies solely on biologically plausible mechanisms (amine diffusion, column structure, Gabor filters) without depending on mathematical optimization theories such as error function minimization or the chain rule of derivatives.

### 5. Fast Learning

When the training data is sufficiently large, the model reaches over 90% of the final test accuracy in the first epoch. High accuracy is achieved with a small number of epochs without requiring many repetitions.

Furthermore, per-sample training is fast. Under identical conditions (NumPy, CPU, per-sample learning), it operates approximately **3.7–4.4× faster** than backpropagation (BP). Since no backward-pass computation is required and fewer neurons are updated, the speed advantage increases with deeper networks.

### 6. Easy Parameter Tuning

The network responds to parameter changes in a stable and monotonic manner, without the sudden learning collapse caused by vanishing or exploding gradients as seen with backpropagation. Accuracy does not drop drastically when parameters deviate slightly from their optimal values, making hyperparameter tuning straightforward.

### 7. Reservoir Computing Characteristics

When the number of neurons per column is set to 1 (`column_neurons=1`), the implementation operates on the same principle as reservoir computing. Non-column neurons in the hidden layer are maintained as fixed random weights (reservoir), and only a small number of column neurons are trained using the original ED method. With `column_neurons=10` (default for 2–3 layer configurations), 10 column neurons are assigned per class, increasing the number of learning neurons so that each class can be represented by more diverse features. The reservoir computing-like structure (majority of weights remain fixed) is maintained while classification performance improves with the increase in column neurons.

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yoiwa0714/columnar_ed_ann.git
cd columnar_ed_ann

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run

```bash
# Default configuration (2-layer + Gabor features, ~10 minutes)
python columnar_ed_ann.py --train 10000 --test 10000

# 1-layer configuration (~3 minutes)
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000

# With visualization (learning curve, confusion matrix, activation heatmap)
python columnar_ed_ann.py --train 10000 --test 10000 --viz --heatmap
```

With seed=42 (default), approximately 96–97% test accuracy is obtained.

> Runtime and accuracy are environment-dependent. Values in this README are representative benchmark values.

## Reproducibility Checklist

- Record OS / Python version / CPU-GPU environment
- Fix `--seed` (default: 42)
- Explicitly specify `--train`, `--test`, and `--epochs`
- Record whether `config/hyperparameters.yaml` was modified
- Record whether visualization options (`--viz`, `--heatmap`) were enabled

Keeping these fixed makes it easier to explain differences from README values.

### 3. Checking Results

When the run completes, results similar to the following are displayed:

```
======================================================================
Training Complete
======================================================================
Final Accuracy: Train=0.9714, Test=0.9685
Best Accuracy:  Test=0.9685 (Epoch 10)
```

---

## Usage Examples

### Basic Execution Patterns

```bash
# 1-layer + Gabor features (~3 minutes)
python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000
# → Best ≈ 96.13% / Final ≈ 96.13%

# 2-layer + Gabor features (default configuration, ~10 minutes)
python columnar_ed_ann.py --train 10000 --test 10000
# → Best ≈ 96.85% / Final ≈ 96.84%

# 3-layer + Gabor features (~30 minutes)
python columnar_ed_ann.py --hidden 2048,1024,1024 --train 10000 --test 10000
# → Best ≈ 97.11% / Final ≈ 96.78%

# Without Gabor (to verify the pure learning capability of the original ED method)
python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000 --no_gabor
# → Best ≈ 90.37% / Final ≈ 90.37%
```

### Visualization

```bash
# Display real-time learning curve
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --viz

# Learning curve + hidden layer and output layer heatmaps
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --viz --heatmap

# Save visualization results to a directory
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --viz --heatmap --save_viz results/
```

### Other Options

```bash
# Fashion-MNIST
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --dataset fashion
```

### All Command-Line Arguments

**Network Configuration:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden` | `2048,1024` | Hidden layer neuron count (comma-separated, e.g., `2048`=1 layer, `2048,1024`=2 layers) |
| `--train` | `10000` | Number of training samples |
| `--test` | `10000` | Number of test samples |
| `--epochs` | Auto (YAML) | Number of epochs |
| `--seed` | `42` | Random seed |
| `--dataset` | `mnist` | Dataset name (`mnist`, `fashion`, `cifar10`) or custom data path (see [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md)) |

**Gabor Feature Extraction:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--no_gabor` | OFF | Disable Gabor feature extraction (ON by default) |

**Visualization:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--viz` | OFF | Display real-time learning curve |
| `--heatmap` | OFF | Display heatmap (used together with `--viz`) |
| `--save_viz` | None | Directory or file path to save visualization results |

## Claims and Verifiability (FAQ)

### Q1. What exactly does "no Backpropagation" mean here?

In this project, it means **weight updates do not use error backpropagation based on the chain rule of derivatives**.
Error signals are handled via amine diffusion and column-structured local updates.

### Q2. Where can I verify this in code/docs?

- Learning core: `modules/ed_network.py`
- Activations and related helpers: `modules/activation_functions.py`
- Operational explanation: `docs/en/Columnar_ED_Method_Flow.md`

### Q3. Why might my reproduced accuracy differ?

Typical causes are environment differences (CPU/GPU/library versions), sample counts, epoch counts, visualization settings, and YAML config differences.
Start by checking the Reproducibility Checklist above.

---

## How It Works

> 📖 For detailed explanations of the internal operation flow and core algorithms, see **[Columnar ED Method — How It Works](docs/en/Columnar_ED_Method_Flow.md)**.
>
> 🔗 For Mermaid-based code-anchored flow diagrams, see **[ED Learning Mechanism (Mermaid Anchors, EN)](docs/en/ed_learning_mechanism_anchors_en.md)**.
>
> 📘 For equation-level formalization aligned with the implementation, see **[Columnar ED Method: Detailed Principles](docs/en/Columnar_ED_Method_Detailed_Principles.md)**.

### What Is the Columnar ED Method

**The Columnar ED Method (Columnar Error Diffusion Method)** is an extended implementation of Isamu Kaneko's original ED method neural network, with the cortical column structure of the cerebral cortex introduced.

The original ED method is a biologically plausible learning algorithm that models the diffusion of neurotransmitters in the brain. However, it was designed for binary classification and struggled with multi-class classification involving 10 or more classes. The root cause was that hidden layer neurons in the network had no information about which output class they contributed to.

The Columnar ED Method introduces the column structure found in the cerebral cortex, explicitly mapping **column neurons** in the hidden layer to **output class neurons**. This enables each column neuron to receive learning signals only from its designated class, making multi-class classification possible within a single network (weight space).

Furthermore, by combining Gabor filter-based feature extraction that models simple cells in the primary visual cortex (V1), test accuracy improves by approximately 6% without modifying the learning mechanism itself (MNIST 1-layer: 90.37% → 96.13%).

### What Is the Original ED Method

The **Error Diffusion Learning Algorithm (original ED method)** is a neural network learning algorithm conceived by Isamu Kaneko in 1999.

The mainstream approach for training neural networks — "backpropagation based on the chain rule of derivatives (BP method)" — computes the error between the network's output and ground-truth labels, then propagates it backwards from the output layer to the input layer using the chain rule to efficiently update weights and biases in each layer.

However, Isamu Kaneko, the developer of the original ED method, was unwilling to accept this biologically implausible learning approach. He modeled the phenomenon of amine-type neurotransmitters (noradrenaline, dopamine, etc.) diffusing through space to develop the original ED method. In the original ED method, each layer **independently** updates its weights based on diffused amine concentration information.

**Essential Differences from BP:**

| | BP (Backpropagation) | Original ED Method (Error Diffusion) |
|---|---|---|
| Error propagation | Flows backward along axons via chain rule | Diffuses through space as amine concentration |
| Layer-wise learning | Depends on downstream gradients | **Each layer learns independently** |
| Biological plausibility | Low | High |

> For details, see [docs/en/ED_Method_Explanation.md](docs/en/ED_Method_Explanation.md).

### Weight Update via Amine Diffusion

The original ED method performs learning in the following steps:

```
1. Forward pass: Input → Hidden layer (tanh) → Output layer (SoftMax) → Prediction
2. Error computation: Probability error for correct class → converted to amine concentration
3. Amine diffusion: Output layer → Hidden layer (selectively diffused along column structure)
4. Weight update: Each layer independently updates using amine concentration × saturation suppression term × input
```

**Saturation Suppression Term** (the core of the original ED method):

```
Saturation suppression = |z| × (1 - |z|)
```

- This is neither the sigmoid derivative `z(1-z)` nor the tanh derivative `1-z²`
- As the neuron activation value `z` saturates (approaches ±1), the update magnitude decreases
- This allows each layer to independently determine the appropriate update magnitude without using the chain rule

### Column Structure

The primary visual cortex and association areas of the cerebral cortex exhibit a structure in which neurons with similar properties cluster into columns[*1]. This implementation incorporates this structure to extend the multi-class classification capability of the original ED method.

**How the Column Structure Works:**

1. Hidden layer neurons are placed in a 2D space using a hexagonal (honeycomb 2-3-3-2) arrangement
2. Column centers for each class are positioned in this space (10 centers for 10-class classification)
3. The neuron closest to each column center is assigned as the **column neuron** for that class
4. During training, only column neurons of the correct class receive amine signals and update their weights
5. Neurons not belonging to any column retain fixed random weights (the number of learning neurons changes based on the `column_neurons` setting)

![Hexagonal column structure spatial arrangement](images/hexagonal_column_structure_no_origin.png)

*Figure: Neuron arrangement in hexagonal structure (2048 neurons, column_neurons=1). Gray dots = non-column neurons (2038), colored dots = column neurons (10), ★ = class centers (0–9)*

```
[*1] "NeUro+" — Neuroscience start-up combining Tohoku University knowledge and Hitachi technology
     (https://neu-brains.co.jp/neuro-plus/glossary/ka/140/)
```

### Gabor Feature Extraction

Simple cells in the primary visual cortex (V1) respond selectively to edges of specific orientations and frequencies. This implementation models them with Gabor filters to extract features from input images (ON by default).

**Filter Configuration:**
- 8 orientations × 2 frequencies = 16 Gabor filters + 2 Sobel edge filters = **18 filters total**
- Kernel size: 11×11, Pooling: 4×4 average pooling
- Output dimensions: 784 → 882 (for MNIST 28×28)

**Effect:** With Gabor features, the MNIST 1-layer configuration improves from 90.37% → 96.13% (+5.76%). This is a biologically plausible approach that improves accuracy through input quality enhancement without modifying the learning mechanism itself.

### Reservoir Computing Characteristics

With `column_neurons=1` (default for 1-layer configurations), this implementation functions on the same operating principle as reservoir computing.

**Why cn=1 Becomes Reservoir Computing:**

`column_neurons=1` assigns one column neuron per class. For 10-class classification with a 2048-neuron hidden layer, only 10 neurons (0.5% of the total) are training targets; the remaining 2038 neurons retain their initialization-time random weights unchanged.

This configuration exactly matches the fundamental principles of reservoir computing:

| Reservoir Computing | Columnar ED Method (cn=1) |
|---|---|
| **Reservoir**: Fixed random connections project input into a high-dimensional space | **Non-column neurons (2038)**: Fixed random weights project input into a high-dimensional space |
| **Readout layer**: Learns classification boundary from reservoir output | **Output layer**: Learns classification from the activation patterns of the entire hidden layer |

Neurons with fixed random weights serve the role of randomly projecting input data into a high-dimensional nonlinear space. Because each neuron with different random weights responds to different aspects of the input, patterns that are difficult to separate linearly in the original input space become more separable after high-dimensional projection. The output layer only needs to learn the classification boundary from this projected representation, allowing high classification accuracy to be achieved by training only a small number of neurons.

This configuration allows the Columnar ED Method to reconcile biological plausibility (only a small number of neurons learn) with high generalization performance (overfitting suppression through fixed weights).

**When cn>1:**

For example, with `column_neurons=10` (default for 2–3 layer configurations), 10 column neurons are assigned per class. In a 2048-neuron hidden layer, 100 neurons (about 4.9% of total) become training targets, while the remaining 1948 retain fixed random weights.

Compared to cn=1, the increased number of learning neurons allows each class to be represented by more diverse features. The reservoir computing-like structure (majority of weights remain fixed) is maintained while classification performance improves with the increase in column neurons.

---

## Achieved Accuracy

Experimental results on MNIST handwritten digit recognition (seed=42, reproducible):

### With Gabor Features (default) — 10k samples

| Configuration | Hidden Layers | Test Accuracy | Runtime (*) |
|---------------|---------------|---------------|-------------|
| 1-layer | [2048] | Best 96.13% / Final 96.13% | ~3 min |
| 2-layer | [2048, 1024] | Best 96.85% / Final 96.84% | ~10 min |
| 3-layer | [2048, 1024, 1024] | Best 97.11% / Final 96.78% | ~30 min |

### With Gabor Features — 50k samples

| Configuration | Hidden Layers | Test Accuracy | Runtime (*) |
|---------------|---------------|---------------|-------------|
| 2-layer | [2048, 1024] | **Best 97.15%** | ~24 min |
| 3-layer | [2048, 1024, 1024] | **Best 97.23%** | ~30 min |
| 4-layer | [1024, 1024, 1024, 1024] | Best 97.12% | ~26 min |
| 5-layer | [1024, 1024, 1024, 1024, 1024] | Best 97.25% | ~33 min |
| 6-layer | [1024, 1024, 1024, 1024, 1024, 1024] | **Best 97.36%** | ~37 min |

\* Runtimes measured on an Intel Core i5-11th gen / RTX 3060 system and will vary depending on your environment.

### Without Gabor Features (`--no_gabor`) — 10k samples

| Configuration | Hidden Layers | Test Accuracy |
|---------------|---------------|---------------|
| 1-layer | [2048] | Best 90.37% / Final 90.37% |
| 2-layer | [2048, 1024] | 89.38% |
| 3-layer | [2048, 1024, 1024] | 89.41% |

> **Experimental conditions:** seed=42 (all under identical conditions, fully reproducible). Epoch counts are automatically set from `config/hyperparameters.yaml`. Higher accuracy can be achieved by increasing training data and epochs.

---

## Directory Structure

The following is a **summary of key files** for understanding and reproducing the method.
If you are new to this repository, start with these files first.

```
columnar_ed_ann/
├── columnar_ed_ann.py              # ★ Main script (recommended)
├── columnar_ed_ann_experiment.py   # Experiment version (all parameters configurable)
├── README.md                       # ★ This document (Japanese)
├── README_en.md                    # ★ This document (English)
├── README_experiment.md            # Experiment version documentation (Japanese)
├── README_experiment_en.md         # Experiment version documentation (English)
├── LICENSE                         # License
├── requirements.txt                # Dependencies
├── CUSTOM_DATASET_GUIDE.md         # Custom dataset usage guide
│
├── modules/                        # ★ Main version modules
├── modules_experiment/             # Experiment version modules
├── config/                         # Parameter configuration files
│   ├── hyperparameters.yaml        #   Per-layer optimal parameters (editable)
│   └── hyperparameters_initial.yaml#   Initial state (for restoration)
├── docs/                           # Related documentation
│   ├── ja/ED法_解説資料.md          #   Detailed explanation of original ED method (Japanese)
│   ├── ja/EDLA_金子勇氏.md          #   Academic background & Isamu Kaneko's achievements (Japanese)
│   ├── ja/コラムED法_動作の流れ.md  #   Core function details of Columnar ED Method (Japanese)
│   ├── en/ED_Method_Explanation.md  #   Detailed explanation of original ED method (English)
│   ├── en/EDLA_Isamu_Kaneko.md      #   Academic background & Isamu Kaneko's achievements (English)
│   └── en/Columnar_ED_Method_Flow.md#   Core function details of Columnar ED Method (English)
├── images/                         # Column structure diagrams
└── original-c-source-code/         # Isamu Kaneko's original C source code
```

---

## Automatic Parameter Configuration

In the main version, optimal parameters are automatically loaded from `config/hyperparameters.yaml` based on the number of hidden layers. Users only need to specify minimal arguments: hidden layer configuration, data size, and epoch count.

### Key Automatically Configured Parameters

| Parameter | 1-layer | 2-layer | 3-layer | Description |
|-----------|---------|---------|---------|-------------|
| column_neurons | 1 | 10 | 10 | Number of column neurons |
| init_scales | [0.4, 1.0] | [0.7, 1.8, 0.8] | [0.7, 1.8, 1.8, 0.8] | Per-layer initialization scales |
| output_lr | 0.15 | 0.15 | 0.15 | Output layer learning rate |
| non_column_lr | [0.15] | [0.15, 0.15] | [0.15, 0.15, 0.15] | Hidden layer base learning rate (per layer) * |
| column_lr_factors | [0.01] | [0.005, 0.003] | [0.005, 0.004, 0.002] | Column neuron learning rate multiplier (per layer) |
| hidden_sparsity | 0.4 | [0.4, 0.4] | [0.4, 0.4, 0.4] | Hidden layer sparsity |
| gradient_clip | 0.05 | 0.03 | 0.06 | Gradient clipping |

> \* `non_column_lr` is used as the base learning rate for the entire hidden layer. Non-column neurons do not actually learn because they receive no amine signals; only column neurons (updated with `column_lr_factors` multiplier) and the output layer (updated with `output_lr`) train.
>
> Parameters for 4+ layers are also defined in `config/hyperparameters.yaml`.

### Customization

1. **Direct YAML editing**: Edit `config/hyperparameters.yaml` in a text editor
2. **View settings**: Refer to the YAML file directly

> If the YAML file is accidentally corrupted, restore it by copying from `config/hyperparameters_initial.yaml`.

---

## Compliance with the Original ED Method

The implementation has been verified against Isamu Kaneko's C source code as a reference to confirm compliance with the original ED method.

| Item | Implementation | Compliant |
|------|---------------|-----------|
| Output layer saturation term | `\|z\| × (1 - \|z\|)` | ✓ |
| Hidden layer saturation term | `\|z\| × (1 - \|z\|)` | ✓ |
| Amine diffusion | Selective diffusion along column structure | ✓ |
| Weight update | Amine concentration-based, no chain rule | ✓ |
| Dale's Principle | Excitatory/inhibitory neuron pairs | ✓ |
| SoftMax | Used only for probability normalization (forward pass) | ✓ |

---

## References

- [Original ED Method Explanation (Japanese)](docs/ja/ED法_解説資料.md) — Detailed explanation of the theory and operation of the original ED method
- [EDLA — Isamu Kaneko's Error Diffusion Learning Algorithm (Japanese)](docs/ja/EDLA_金子勇氏.md) — Academic background of the ED method and Isamu Kaneko's contributions
- [ED Learning Mechanism (Mermaid Anchors, EN)](docs/en/ed_learning_mechanism_anchors_en.md) — Code-anchored execution and feature flow diagrams
- [ED Learning Mechanism (Mermaid Anchors, Japanese)](docs/ja/ed_learning_mechanism_anchors.md) — Code-anchored execution and feature flow diagrams (Japanese)
- [Columnar ED Method: Detailed Principles](docs/en/Columnar_ED_Method_Detailed_Principles.md) — Implementation-aligned mathematical formalization (English)
- [Columnar ED Method: Detailed Principles (Japanese)](docs/ja/コラムED法の動作原理詳細.md) — Implementation-aligned mathematical formalization (Japanese)
- [Isamu Kaneko (1999) Original ED Method C Source Code](original-c-source-code/main.c) — The original implementation on which this work is based
- [Cortical Column Structure](https://neu-brains.co.jp/neuro-plus/glossary/ka/140/) — Biological background of column structure

---

## License

See the [LICENSE](LICENSE) file.

- **Non-commercial use**: MIT License (personal use, academic research, educational purposes)
- **Commercial use**: A separate commercial license is required (contact via GitHub Issues)

---

## Acknowledgements

This implementation is based on the Error Diffusion (ED) learning algorithm conceived by Isamu Kaneko in 1999. We express our deepest respect and gratitude for his pioneering work in biologically plausible neural network learning algorithms.

---

## Author

yoiwa0714
