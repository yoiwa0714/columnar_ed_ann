[日本語](README.md) | **English**

# **[Columnar ED Method] Full Version — Extension of the Original ED Method with Cortical Column Structure**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-orange.svg)](https://numpy.org/)

## Table of Contents

- [Overview](#overview)
- [Target Reader and Fast Path](#target-reader-and-fast-path)
- [Features](#features)
- [Quick Start](#quick-start)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Usage Examples](#usage-examples)
- [Command-Line Arguments](#command-line-arguments)
- [Claims and Verifiability (FAQ)](#claims-and-verifiability-faq)
- [How It Works](#how-it-works)
  - [What Is the Original ED Method](#what-is-the-original-ed-method)
  - [Column Structure](#column-structure)
  - [Weight Update via Amine Diffusion](#weight-update-via-amine-diffusion)
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

**The Columnar ED Method** is a neural network implementation that extends the Error Diffusion learning algorithm (ED method, hereinafter referred to as the "original ED method") conceived by Isamu Kaneko, by introducing cortical column structure from the cerebral cortex.

The Columnar ED Method **does not use backpropagation based on the chain rule of derivatives at all**, and instead learns through biologically plausible amine diffusion mechanisms. Despite this, it achieves **97.16%** test accuracy on MNIST handwritten digit recognition (4-layer configuration, 10,000 training samples).

This repository provides two implementations:

| Implementation | File | Purpose |
|----------------|------|---------|
| **Full Version** | `columnar_ed_ann.py` | All parameters configurable. This document covers this version. |
| Simple Version | `columnar_ed_ann_simple.py` | High accuracy with minimal arguments. See [README_simple_en.md](README_simple_en.md). |

> **Key differences from the Simple Version:**
> - Gabor feature extraction is OFF by default; enable explicitly with `--gabor_features` (Simple Version has it ON by default)
> - Learning rates, amine diffusion coefficients, dynamic synaptic pruning, gradient clipping, early stopping, etc. can be specified from the command line — enabling parameter exploration and detailed experimentation
> - GPU (CuPy) support
> - Detailed Gabor filter parameters (number of orientations, frequencies, kernel size, etc.) are configurable

## Target Reader and Fast Path

This README targets readers who already have **basic Python and machine learning knowledge** but are new to the original ED method / Columnar ED method.

Recommended reading order:

1. Run the minimum Quick Start command once
2. Compare your output with the Achieved Accuracy section at a high level
3. Read Claims and Verifiability (FAQ) for the definition of "no backpropagation"
4. Move on to How It Works

---

## Features

### 1. No Backpropagation Based on the Chain Rule

Conventional neural networks update weights using "backpropagation based on the chain rule of derivatives," but this implementation does not use it at all. Instead, learning is performed through a mechanism that models the diffusion of neurotransmitters (amines) in the brain.

### 2. Cortical Column Structure

The column structure found in the visual cortex of the cerebral cortex is introduced, assigning a subset of neurons to specific classes. This enables multi-class classification within a single network (weight space), which was difficult for the original ED method.

### 3. Gabor Feature Extraction

Gabor filter-based feature extraction is built in, modeling the simple cells of the primary visual cortex (V1) (enabled via `--gabor_features`). By improving input quality, accuracy exceeding 95% can be achieved even with a single-layer configuration under appropriate parameter settings.

### 4. Learning with Only Biologically Plausible Functions

Learning relies solely on biologically plausible mechanisms (amine diffusion, column structure, Gabor filters) without depending on mathematical optimization theories such as error function minimization or the chain rule of derivatives.

### 5. Fast Learning

When the training data is sufficiently large, accuracy exceeding 90% of the final test accuracy is reached in the first epoch. High accuracy can be achieved with a small number of epochs without needing many repetitions.

### 6. Easy Parameter Tuning

The network responds to parameter changes in a stable and monotonic manner, without the sudden learning collapse caused by vanishing or exploding gradients as seen with backpropagation. Accuracy does not drop drastically when parameters deviate slightly from their optimal values, making hyperparameter tuning straightforward.

### 7. Reservoir Computing Characteristics

When the number of neurons per column is set to 1 (`--column_neurons 1`), this implementation operates on the same principle as reservoir computing. Non-column neurons in the hidden layer are maintained as fixed random weights (reservoir), and only a small number of column neurons are trained using the original ED method.

### 8. Dynamic Synaptic Pruning

A dynamic synaptic pruning feature models the developmental pruning observed in the brain (enabled via `--dynamic_pruning_fs`). Unnecessary synaptic connections are progressively removed during training, improving the computational efficiency of the network.

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
# 1-layer + Gabor features (~3 minutes)
python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000 --gabor_features

# 1-layer + Gabor features + visualization (learning curve, confusion matrix, activation heatmap)
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --gabor_features --viz 2 --heatmap
```

With seed=42 (default), approximately 96% test accuracy is obtained.

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
Final Accuracy: Train=0.9722, Test=0.9613
Best Accuracy:  Test=0.9613 (Epoch 10)
```

---

## Usage Examples

### Basic Execution Patterns

```bash
# 1-layer + Gabor features (~3 minutes)
python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000 --gabor_features
# → Best ≈ 96.13% / Final ≈ 96.13%

# 2-layer + Gabor features (~10 minutes)
python columnar_ed_ann.py --hidden 2048,1024 --train 10000 --test 10000 --gabor_features
# → Best ≈ 96.85% / Final ≈ 96.84%

# 3-layer + Gabor features (~30 minutes)
python columnar_ed_ann.py --hidden 2048,1024,1024 --train 10000 --test 10000 --gabor_features
# → Best ≈ 97.11% / Final ≈ 96.78%

# 4-layer + Gabor features (MNIST, cn=20 latest)
python columnar_ed_ann.py --dataset mnist --hidden 1024[4] --train 10000 --test 10000 --epochs 10 --seed 42 --column_neurons 20 --init_method he --gabor_features --lr 0.04 --column_lr_factors 0.005,0.004,0.003,0.002 --gradient_clip 0.03 --init_scales 0.9,1.6,1.8,1.6,0.8 --viz 2 --heatmap
# → Best = 97.16% (Epoch 10), Final = 97.16%

# 5-layer + Gabor features (MNIST, T3M adopted)
python columnar_ed_ann.py --dataset mnist --hidden 1024[5] --train 10000 --test 10000 --epochs 10 --seed 42 --column_neurons 20 --init_method he --gabor_features --lr 0.04 --column_lr_factors 0.005,0.004,0.003,0.002,0.0015 --gradient_clip 0.03 --init_scales 0.9,1.6,1.8,1.2,1.4,0.8 --viz 2 --heatmap
# → Best = 96.78% (Epoch 10), Final = 96.78%

# Without Gabor (to verify the pure learning capability of the original ED method)
python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000
# → Best ≈ 90.37% / Final ≈ 90.37%
```

### Visualization

```bash
# Display real-time learning curve (size levels: 1=base, 2=1.3x, 3=1.6x, 4=2x; window size)
# Omitting SIZE is equivalent to --viz 1
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --gabor_features --viz 2

# Learning curve + hidden layer and output layer heatmaps
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --gabor_features --viz 2 --heatmap

# Save visualization to a directory (auto-named with timestamp)
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --gabor_features --viz 2 --heatmap --save_viz results/

# Save with a specified filename
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --gabor_features --viz 2 --heatmap --save_viz results/my_experiment.png

# Display misclassified training data (scrollable window after final epoch)
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --gabor_features --show_train_errors

# Specify maximum number of errors displayed per class (default: 20)
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --gabor_features --show_train_errors --max_errors_per_class 50
```

### Specifying Training Parameters

```bash
# Manually specify learning rates
python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000 --gabor_features \
    --output_lr 0.15 --column_lr 0.0015

# Use repeat notation for concise per-layer learning-rate specification (5 layers)
python columnar_ed_ann.py --hidden 1024[5] --train 10000 --test 10000 --gabor_features \
    --output_lr 0.04 --non_column_lr 0.04[5] --column_lr 0.0002,0.00016,0.00012,8e-05,6e-05

# Adjust amine diffusion coefficients
python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000 --gabor_features \
    --u1 0.5 --u2 0.8

# Specify gradient clipping for 2 layers
python columnar_ed_ann.py --hidden 2048,1024 --train 10000 --test 10000 --gabor_features \
    --gradient_clip 0.03
```

### Dynamic Synaptic Pruning

```bash
# Prune 40% of weights (equivalent to developmental pruning in the brain)
python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000 --gabor_features \
    --dynamic_pruning_fs 0.4

# Display detailed pruning log
python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000 --gabor_features \
    --dynamic_pruning_fs 0.4 --pruning_verbose

# Specify pruning start and end epochs
python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000 --epochs 20 --gabor_features \
    --dynamic_pruning_fs 0.4 --pruning_start_epoch 3 --pruning_end_epoch 15
```

### Early Stopping

```bash
# Early stopping for grid search (stop if Test accuracy ≤ 15% at Epoch 3)
python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000 --gabor_features \
    --early_stop_epoch 3 --early_stop_threshold 0.15
```

### Other Options

```bash
# Fashion-MNIST
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --gabor_features --dataset fashion

# Manually specify column neuron count and initialization scales (overrides YAML defaults)
python columnar_ed_ann.py --hidden 2048,1024 --gabor_features --column_neurons 10 --init_scales 0.7,1.8,0.8

# Change weight initialization method
python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000 --gabor_features --init_method xavier

# Display YAML configuration list
python columnar_ed_ann.py --list_hyperparams

# Diagnose column structure
python columnar_ed_ann.py --hidden 2048 --diagnose_column
```

---

## Command-Line Arguments

### Network Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden` | `2048` | Hidden layer neuron count (e.g., `2048`=1 layer, `2048,1024`=2 layers, `1024[5]`=five identical layers) |
| `--train` | `3000` | Number of training samples |
| `--test` | `1000` | Number of test samples |
| `--epochs` | Auto (YAML) | Number of epochs |
| `--seed` | `42` | Random seed |
| `--dataset` | `mnist` | Dataset name (`mnist`, `fashion`, `cifar10`) or custom data path (see [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md)) |
| `--batch_size` | None | Mini-batch size (online learning when not specified) |
| `--use_cupy` | OFF | Enable CuPy (GPU) acceleration |

### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_lr` | Auto (YAML) | [Recommended] Output layer learning rate |
| `--non_column_lr` | Auto (YAML) | [Recommended] Non-column neuron learning rate per layer (comma-separated, repeat notation supported: `0.04[5]`) |
| `--column_lr` | Auto (YAML) | [Recommended] Column neuron learning rate per layer (comma-separated, repeat notation supported: `0.0002[3],0.0001[2]`) |
| `--lr` | `0.15` | [Compatibility] Learning rate (used when 3-system learning rates are not specified) |
| `--column_lr_factors` | Auto (YAML) | [Compatibility] Per-layer suppression factors for column rows (effective `column_lr = lr × factor`, comma-separated) |
| `--u1` | Auto (YAML) | Amine diffusion coefficient u1 |
| `--u2` | Auto (YAML) | Amine diffusion coefficient u2 |
| `--gradient_clip` | `0.03` | Gradient clipping value |

One-line compatibility mode example:
```bash
python columnar_ed_ann.py --dataset mnist --hidden 2048,1024 --train 10000 --test 10000 --epochs 20 --lr 0.15 --column_lr_factors 0.005,0.003
```

### Column Structure

| Argument | Default | Description |
|----------|---------|-------------|
| `--column_neurons` | Auto (YAML) | Number of column neurons per class |
| `--base_column_radius` | `0.4` | Base column radius |
| `--column_radius` | Auto | Column influence radius |
| `--participation_rate` | `0.1` | Column participation rate |
| `--diagnose_column` | OFF | Run detailed column structure diagnostics |
| `--diagnose_hidden_weights` | OFF | Run detailed diagnostics on hidden layer weight state |

### Initialization

| Argument | Default | Description |
|----------|---------|-------------|
| `--init_method` | `he` | Weight initialization method (`uniform`, `xavier`, `he`) |
| `--hidden_sparsity` | Auto (YAML) | Hidden layer sparsity (supports per-layer comma-separated values) |
| `--init_scales` | Auto (YAML) | Per-layer initialization scales (e.g., `0.7,1.8,0.8`) |

### Visualization

| Argument | Default | Description |
|----------|---------|-------------|
| `--viz [SIZE]` | OFF | Display real-time learning curve (`1=base`, `2=1.3x`, `3=1.6x`, `4=2x` window size; omitted SIZE defaults to `1`) |
| `--heatmap` | OFF | Display heatmap (used together with `--viz`) |
| `--save_viz` | None | Directory to save visualization results |
| `--save_weights` | OFF | Save weight statistics per epoch |
| `--show_train_errors` | OFF | Display misclassified training data after final epoch |
| `--max_errors_per_class` | `20` | Maximum number of errors displayed per class |

### Early Stopping

| Argument | Default | Description |
|----------|---------|-------------|
| `--early_stop_epoch` | None | Epoch at which early stopping is evaluated (disabled if not specified) |
| `--early_stop_threshold` | `0.15` | Stop if Test accuracy falls below this value |

### Dynamic Synaptic Pruning

Models developmental pruning in the brain. Unnecessary synaptic connections are progressively removed during training.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dynamic_pruning_fs` | None | Target sparsity ratio (e.g., `0.4` = 40% pruning; enabled when specified) |
| `--pruning_start_epoch` | Auto-detect | Epoch to start pruning |
| `--pruning_end_epoch` | Final epoch | Epoch to end pruning |
| `--stability_threshold` | `0.01` | Threshold for stable-phase detection (Test accuracy change rate) |
| `--pruning_verbose` | OFF | Output detailed pruning log |

### Gabor Feature Extraction

Extracts input features using fixed filters that model V1 simple cells.

| Argument | Default | Description |
|----------|---------|-------------|
| `--gabor_features` | OFF | Enable Gabor feature extraction |
| `--gabor_orientations` | `8` | Number of filter orientations |
| `--gabor_frequencies` | `2` | Number of spatial frequencies |
| `--gabor_kernel_size` | `7` | Filter kernel size |
| `--gabor_pool_size` | `4` | Average pooling window size |
| `--gabor_pool_stride` | `4` | Pooling stride |
| `--gabor_no_edge` | OFF | Exclude Sobel edge filters |

### Utilities

| Argument | Default | Description |
|----------|---------|-------------|
| `--list_hyperparams` | — | Display YAML configuration list (layer count can be specified: `--list_hyperparams 2`) |
| `--verbose [LEVEL]` | OFF | Show detailed logs. `--verbose` or `--verbose 0`: without activation statistics; `--verbose 1`: with activation statistics |
| `--activation-stats` | OFF | Show activation statistics (compatible with `--verbose 1`) |

## Claims and Verifiability (FAQ)

### Q1. What exactly does "no backpropagation" mean here?

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

### What Is the Columnar ED Method?

**The Columnar ED Method** is an extended implementation of Isamu Kaneko's original ED method, with the cortical column structure of the cerebral cortex introduced into the neural network.

The original ED method is a biologically plausible learning algorithm that models the diffusion of neurotransmitters in the brain. However, it was designed for binary classification and struggled with multi-class classification of 10 or more classes. The root cause was that hidden layer neurons in the network had no information about which output class they contributed to.

The Columnar ED Method introduces the column structure found in the cerebral cortex, explicitly mapping **column neurons** in the hidden layer to **output class neurons**. This enables each column neuron to receive learning signals only from its designated class, making multi-class classification possible within a single network (weight space).

Furthermore, by combining Gabor filter-based feature extraction that models simple cells in the primary visual cortex (V1), test accuracy improves by approximately 6% without modifying the learning mechanism itself (MNIST 1-layer: 90.37% → 96.13%).

### What Is the Original ED Method?

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
5. Neurons not belonging to any column retain fixed random weights (the number of learning neurons (column neurons) changes based on the `column_neurons` setting)

![Hexagonal column structure spatial arrangement](images/hexagonal_column_structure_no_origin.png)

*Figure: Neuron arrangement in hexagonal structure (2048 neurons, column_neurons=1). Gray dots = non-column neurons (2038), colored dots = column neurons (10), ★ = class centers (0–9)*

```
[*1] "NeUro+" — Neuroscience start-up combining Tohoku University knowledge and Hitachi technology
     (https://neu-brains.co.jp/neuro-plus/glossary/ka/140/)
```

### Gabor Feature Extraction

Simple cells in the primary visual cortex (V1) respond selectively to edges of specific orientations and frequencies. This implementation models them with Gabor filters to extract features from input images.

**Filter Configuration:**
- 8 orientations × 2 frequencies = 16 Gabor filters + 2 Sobel edge filters = **18 filters total**
- Kernel size: 7×7, Pooling: 4×4 average pooling
- Output dimensions: 784 → 882 (for MNIST 28×28)

**Effect:** With Gabor features, the MNIST 1-layer configuration improves from 90.37% → 96.13% (+5.76%). This is a biologically plausible approach that improves accuracy through input quality enhancement without modifying the learning mechanism itself.

> In the Full Version, detailed filter parameters can be adjusted via `--gabor_orientations`, `--gabor_frequencies`, `--gabor_kernel_size`, `--gabor_pool_size`, `--gabor_pool_stride`, and `--gabor_no_edge`.

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

For example, with `column_neurons=10` (default for 2+ layer configurations), 10 column neurons are assigned per class. In a 2048-neuron hidden layer, 100 neurons (about 4.9% of total) become training targets, while the remaining 1948 retain fixed random weights.

Compared to cn=1, the increased number of learning neurons allows each class to be represented by more diverse features. While the reservoir computing-like structure (majority of weights remain fixed) is maintained, the increased column neurons improve classification performance. For 2-3 layer configurations, cn=10 is the default; for 4-5 layer configurations, cn=20 is the default. This achieves Best 97.11% (Final 96.78%) for 3 layers, Best 97.16% (Final 97.16%) for 4 layers, and Best 96.78% (Final 96.78%) for 5 layers.

---

## Achieved Accuracy

Experimental results on MNIST handwritten digit recognition (seed=42, reproducible):

### With Gabor Features (`--gabor_features`)

| Configuration | Hidden Layers | Test Accuracy | Runtime (*) |
|------|--------|-----------|----------------|
| 1-layer | [2048] | Best 96.13% / Final 96.13% | ~3 min |
| 2-layer | [2048, 1024] | Best 96.85% / Final 96.84% | ~10 min |
| 3-layer | [2048, 1024, 1024] | **Best 97.11% / Final 96.78%** | ~30 min |

### With Gabor Features (Additional 4/5-layer results)

| Configuration | Hidden Layers | Test Accuracy | Runtime (*) |
|------|--------|-----------|----------------|
| 4-layer (MNIST, cn=20) | [1024, 1024, 1024, 1024] | **Best 97.16% / Final 97.16%** | ~20 min |
| 5-layer (MNIST, T3M adopted) | [1024, 1024, 1024, 1024, 1024] | Best 96.78% / Final 96.78% | ~20 min |

\* Runtimes measured on an Intel Core i5-11th gen / RTX 3060 system and will vary depending on your environment.

### Without Gabor Features

| Configuration | Hidden Layers | Test Accuracy |
|------|--------|-----------|
| 1-layer | [2048] | Best 90.37% / Final 90.37% |
| 2-layer | [2048, 1024] | 89.38% |
| 3-layer | [2048, 1024, 1024] | 89.41% |

> **Experimental conditions:** 10,000 training samples, 10,000 test samples, seed=42 (all under identical conditions, fully reproducible). Epoch counts vary by each experiment (automatically set from `config/hyperparameters.yaml` or explicitly specified via CLI).

> **Note:** For all layer configurations, column neuron counts and initialization scales are automatically set to optimal values from `config/hyperparameters.yaml` (6+ layers fall back to 5-layer parameters). Higher accuracy can be achieved by increasing training data and epochs (e.g., 2-layer + Gabor with 20k samples achieves 97.43%).

---

## Directory Structure

The following is a **summary of key files** for understanding and reproducing the method.
If you are new to this repository, start with these files first.

```
columnar_ed_ann/
├── columnar_ed_ann.py              # ★ Full Version main script (this document)
├── columnar_ed_ann_simple.py       # Simple Version main script
├── README.md                       # ★ This document (Full Version, Japanese)
├── README_en.md                    # This document (Full Version, English)
├── README_simple.md                # Simple Version documentation (Japanese)
├── README_simple_en.md             # Simple Version documentation (English)
├── LICENSE                         # License
├── requirements.txt                # Dependencies
├── CUSTOM_DATASET_GUIDE.md         # Custom dataset usage guide
│
├── modules/                        # ★ Full Version modules
│   ├── ed_network.py               #   ED network core (training and evaluation)
│   ├── column_structure.py         #   Column structure generation (hexagonal layout)
│   ├── gabor_features.py           #   Gabor feature extraction (V1 simple cell model)
│   ├── activation_functions.py     #   Activation functions (tanh, softmax)
│   ├── neuron_structure.py         #   E/I pair structure (Dale's Principle)
│   ├── data_augmentation.py        #   Data augmentation (referenced when using `--augment`)
│   ├── hyperparameters.py          #   YAML parameter loading
│   ├── data_loader.py              #   Dataset loading
│   └── visualization_manager.py    #   Visualization (learning curves, heatmaps, training error analysis)
│
├── modules_simple/                 # Simple Version modules
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

Even in the Full Version, optimal parameters are automatically loaded from `config/hyperparameters.yaml` based on the number of hidden layers. Values explicitly specified via command-line arguments take precedence over automatic settings.

### Key Automatically Configured Parameters

| Parameter | 1-layer | 2-layer | 3-layer | 4-layer | 5-layer (T3M adopted) | Description |
|-----------|---------|---------|---------|---------|---------|-------------|
| output_lr | 0.15 | 0.15 | 0.15 | 0.04 | 0.04 | Output layer learning rate |
| non_column_lr | [0.15] | [0.15, 0.15] | [0.15, 0.15, 0.15] | [0.04, 0.04, 0.04, 0.04] | [0.04, 0.04, 0.04, 0.04, 0.04] | Hidden layer base learning rate (per layer) *1 |
| column_lr | [0.0015] | [0.00075, 0.00045] | [0.00075, 0.0006, 0.0003] | [0.0002, 0.00016, 0.00012, 0.00008] | [0.0002, 0.00016, 0.00012, 0.00008, 0.00006] | Column neuron learning rate (per layer) |
| lr | 0.15 | 0.15 | 0.15 | 0.04 | 0.04 | [Compatibility] Base learning rate (used when 3-system learning rates are not specified) |
| column_lr_factors (clf) | [0.01] | [0.005, 0.003] | [0.005, 0.004, 0.002] | [0.005, 0.004, 0.003, 0.002] | [0.005, 0.004, 0.003, 0.002, 0.0015] | Per-layer suppression factors for column rows |
| column_neurons | 1 | 10 | 10 | 20 | 20 | Number of column neurons |
| init_scales | [0.4, 1.0] | [0.7, 1.8, 0.8] | [0.7, 1.8, 1.8, 0.8] | [0.9, 0.9, 1.8, 1.6, 0.8] | [0.9, 1.6, 1.8, 1.2, 1.4, 0.8] | Per-layer initialization scales |
| hidden_sparsity | 0.4 | [0.4, 0.4] | [0.4, 0.4, 0.4] | [0.4, 0.4, 0.4, 0.4] | [0.4, 0.4, 0.4, 0.4, 0.4] | Hidden layer sparsity |
| gradient_clip | 0.05 | 0.03 | 0.06 | 0.03 | 0.03 | Gradient clipping |

> **3-system learning rates**: Learning rates are independently controlled via three systems: `output_lr` (output layer), `non_column_lr` (hidden layer base, per layer), and `column_lr` (column neurons, per layer). `column_lr_factors` is a per-layer suppression factor applied only to column rows.
>
> *1 `non_column_lr` is used as the base learning rate for the entire hidden layer. Non-column neurons do not actually learn because they receive no amine signals; only column neurons (updated with `column_lr`) and the output layer (updated with `output_lr`) train.

### Customization

1. **Command-line arguments**: Override with `--output_lr`, `--column_lr`, `--column_lr_factors`, `--column_neurons`, `--init_scales`, etc.
2. **Direct YAML editing**: Edit `config/hyperparameters.yaml` in a text editor
3. **View settings**: `python columnar_ed_ann.py --list_hyperparams`

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
- [ED学習メカニズム（Mermaidアンカー・日本語）](docs/ja/ed_learning_mechanism_anchors.md) — 日本語版のコード行番号アンカー付きフロー図
- [Columnar ED Method: Detailed Principles](docs/en/Columnar_ED_Method_Detailed_Principles.md) — Implementation-aligned mathematical formalization
- [コラムED法の動作原理詳細](docs/ja/コラムED法の動作原理詳細.md) — 数式ベースの実装整合型定式化（日本語）
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

---

**Note:** This implementation is for research and educational purposes. For commercial use, please review the [LICENSE](LICENSE).
