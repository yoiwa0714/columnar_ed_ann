# ED-Multi SNN Project Overview

## Project Purpose

Apply Isamu Kaneko's (1999) innovative Error-Diffusion (ED) method to Spiking Neural Networks (SNN) to build an image classification system using biologically plausible learning algorithms.

## Core Technologies

### 1. ED Method (Error-Diffusion Method)

- **Developer**: Isamu Kaneko (1999)
- **Feature**: Does not use "error backpropagation with chain rule of differentiation"
- **Biological Plausibility**: Close to actual brain learning mechanisms
- **Amine Diffusion**: Error signal propagation through neurotransmitter diffusion

### 2. Spiking Neural Network (SNN)

- **Full LIF-ization**: All 1706 neurons are Leaky Integrate-and-Fire (LIF) models
- **Spike Encoding**: Poisson encoding (150Hz, 50ms)
- **Temporal Dynamics**: Temporal integration of membrane potential and spike firing

### 3. Adherence to Biological Constraints

- **E/I Pair Structure**: Excitatory/Inhibitory neuron pairs
- **Dale's Principle**: Excitatory weights ≥0, Inhibitory weights ≤0
- **Independent Output Neurons**: Independent networks for each class

## Achievements

### Accuracy Performance

- **MNIST**: Test accuracy 85.0% (Training accuracy 85.9%)
- **Fashion-MNIST**: Test accuracy 82.0%
- **Generalization Gap**: 0.9% (No overfitting)

### Technical Advantages

1. **Parallel Computation Capable**: Each layer independently updates weights
2. **Avoids Vanishing Gradient Problem**: Local learning without chain rule
3. **Biological Plausibility**: Based on actual brain learning principles
4. **Energy Efficiency**: Spike-based computation

## Implementation Variations

### 1. Complete LIF Version (ed_multi_lif_snn.py)

- All neurons are LIF type
- Most biologically plausible
- Optimal for research purposes

### 2. Simple Version (ed_multi_lif_snn_simple.py)

- Easy-to-understand basic implementation
- For learning and educational purposes
- Optimal for algorithm understanding

### 3. FReLU Version (ed_multi_frelu_snn.py)

- Experimental implementation of FReLU activation function
- Hybrid configuration
- Validation of new activation functions

## Extended Features

### Extensions from Original Theory

1. **Multi-layer Network Support**: Arbitrary hidden layer structures
2. **Mini-batch Learning**: Efficient batch processing
3. **NumPy Matrix Operations**: 1,899x acceleration
4. **Dynamic Memory Management**: Automatic adjustment based on data volume
5. **Real-time Visualization**: Dynamic display of learning progress

### Modern Integration

6. **Modern Data Loader**: TensorFlow integration
7. **GPU Computing Support**: Automatic GPU computation via CuPy integration
8. **Detailed Profiling**: Performance bottleneck identification
9. **Heatmap Visualization**: Neuron firing pattern display

### SNN-Specific Features

10. **Multiple Spike Encoding**: Poisson/Rate/Temporal/Population
11. **LIF Neuron Integration**: Biologically plausible spike generation
12. **Spike-ED Conversion**: Bidirectional conversion between spike activity and ED method input
13. **High-speed SNN Implementation**: Optimization for large-scale SNN computation

## Research Significance

### Contribution to Neuroscience

- Learning algorithms with high biological plausibility
- Practical application of spike-based computation
- Promoting understanding of brain learning mechanisms

### Engineering Value

- Energy-efficient learning for edge devices
- Acceleration through parallel processing
- Fundamental solution to vanishing gradient problem

### Future Prospects

- Application to neuromorphic hardware
- Realization of large-scale spiking networks
- Engineering implementation of biological intelligence

## Technical Specifications

### System Requirements

- Python 3.7+
- NumPy, TensorFlow, matplotlib
- Optional: CuPy (GPU acceleration)

### Performance

- MNIST 1000 samples × 10 epochs: approximately 21 minutes (CPU)
- Memory usage: Automatic adjustment within 16GB RAM
- GPU acceleration: Transparent switching

### Supported Datasets

- MNIST (handwritten digits)
- Fashion-MNIST (clothing classification)
- CIFAR-10/100 (color images)

## Project Structure

```
ed_multi_snn/
├── src/
│   ├── ja/                    # Japanese commented version
│   └── en/                    # English commented version
├── docs/
│   ├── ja/                    # Japanese documentation
│   └── en/                    # English documentation
├── modules/                   # Common module suite
├── test_results/              # Learning result records
└── viz_results/               # Visualization results
```

## Related Literature and Materials

### Technical Specifications

- [ed_multi_snn.prompt_EN.md](ed_multi_snn.prompt_EN.md) - Complete technical specification

### Theoretical Background

- [ED_Method_Overview.md](ED_Method_Overview.md) - Detailed ED method explanation
- [SNN_Fundamentals.md](SNN_Fundamentals.md) - SNN fundamental principles

### Experimental Results

- [test_results/](../../test_results/) - Learning trial records
- [viz_results/](../../viz_results/) - Visualization results

---

**This project aims to provide foundational technology for next-generation AI by engineering the principles of biological intelligence.**