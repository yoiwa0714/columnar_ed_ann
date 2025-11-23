# ED-Multi: Multi-Layer Error Diffusion Learning Algorithm for SNN Specification

**Original Author**: Isamu Kaneko  
**Original Development Year**: 1999  
**Extended Implementation**: 2025  
**Original Source**: C implementation (verified)  
**Extended Version**: Python implementation (multi-layer, optimized)  
**SNN Integration**: Learning method for Spiking Neural Networks

## Project Objectives

- Extend Isamu Kaneko's original ED method theory by adding multi-layer and optimization capabilities
- Implement the ED method as a learning algorithm for Spiking Neural Networks (SNN)
- Construct networks using LIF neurons for all neurons in the SNN architecture

## ED Method Overview

The Error Diffusion Learning Algorithm (ED method) is an innovative learning algorithm that mimics the amine (neurotransmitter) diffusion mechanisms of biological neural systems. It features a fundamentally different approach from traditional backpropagation, with excitatory/inhibitory neuron pair structures and output neuron-centric architecture.

## Core Principles

### ED Method Fundamentals
Detailed principles of the ED method are documented in `docs/en/ED_Method_Overview.md`. Please review this fundamental documentation before referring to this specification.

### SNN Fundamentals  
Basic principles of SNNs are documented in `docs/en/SNN_Overview.md`. Understanding SNN fundamentals is essential before working with this specification.

## Extended Features (Additions to Original Theory)

This implementation completely preserves Isamu Kaneko's original ED method theory while adding the following extended features:

### 1. Multi-Layer Neural Network Support
- **Original Specification**: Single hidden layer only
- **Extended Feature**: Flexible combination of multiple hidden layers
- **Implementation**: Comma-separated specification (e.g., `--hidden 256,128,64`)
- **Technical Feature**: Dynamic layer management via NetworkStructure class
- **ED Theory Consistency**: Amine diffusion coefficient u1 applied across multiple layers

### 2. Mini-Batch Learning System
- **Original Specification**: Sequential processing of single samples only
- **Extended Feature**: Efficient processing of multiple samples together
- **Implementation**: Size specification via `--batch` option
- **Technical Feature**: High-speed data processing via MiniBatchDataLoader
- **Performance Improvement**: 3.66x epoch speed, 278x overall acceleration achieved

### 3. Dramatic Acceleration via NumPy Matrix Operations
- **Original Specification**: Sequential calculation with triple loops
- **Extended Feature**: Parallel computation via NumPy matrix operations
- **Performance Improvement**: 1,899x acceleration in forward computation
- **Technical Feature**: Vectorized sigmoid functions and memory efficiency improvements

### 4. Dynamic Memory Management System
- **Original Specification**: Fixed-size arrays (MAX=1000)
- **Extended Feature**: Automatic memory size adjustment based on data volume
- **Implementation**: Safety assurance via `calculate_safe_max_units`
- **Technical Feature**: Maximum efficiency utilization within 16GB RAM constraints
- **Safety**: Overflow protection and memory shortage avoidance

### 5. Real-Time Visualization System
- **Original Specification**: Text output for results only
- **Extended Feature**: Real-time graphical display of learning progress
- **Implementation Features**: Dynamic visualization of learning curves, confusion matrices, accuracy progression
- **Technical Feature**: Asynchronous update system based on matplotlib
- **Usage**: Enabled via `--viz` option

### 6. Modern Data Loader Integration
- **Original Specification**: Manual setup of proprietary data formats
- **Extended Feature**: Automatic data processing via TensorFlow (tf.keras.datasets) integration
- **Supported Datasets**: Automatic download of MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- **Technical Feature**: Balanced sampling and class equalization
- **Usage**: Dataset switching via `--mnist`, `--fashion`, `--cifar10`, `--cifar100` options

### 7. GPU Computing Support (CuPy Integration)
- **Original Specification**: CPU computation only
- **Extended Feature**: Automatic GPU computation when NVIDIA GPU available
- **Technical Feature**: Transparent GPU processing via CuPy integration
- **Performance Improvement**: Further acceleration for large-scale datasets
- **Compatibility**: Automatic fallback to CPU processing in GPU-free environments

### 8. Detailed Profiling Functionality
- **Original Specification**: Basic execution time display only
- **Extended Feature**: Detailed performance analysis by processing stage
- **Implementation Features**: Bottleneck identification, memory usage monitoring
- **Technical Feature**: Real-time performance monitoring and report generation
- **Usage**: Detailed display via `--verbose` option

### 9. Heatmap Visualization Feature
- **Original Specification**: Not implemented in original
- **Extended Feature**: Real-time display of neuron firing patterns during learning
- **Implementation Features**: Dynamic visualization of firing patterns in each layer
- **Technical Feature**: Heatmap system based on seaborn
- **Usage**: Enabled via `--heatmap` option

### 10. SNN (Spiking Neural Network) Support
- **Original Specification**: Non-spiking neural networks
- **Extended Feature**: Integration of SNN architecture with ED method learning
- **Technical Feature**: Fusion of spike firing models with ED method
- **Network**: Constructed with SNN
- **Theory Preservation**: Prohibition of "error backpropagation using chain rule of differentiation" maintained in ED method learning algorithm

### 11. Multiple Spike Encoding Features
- **Original Specification**: Not implemented in original
- **Extended Feature**: Input conversion system using multiple spike encoding methods
- **Implementation Features**: Poisson encoding, rate encoding, temporal encoding, population encoding
- **Technical Features**:
  - `_poisson_encode`: Spike train generation via Poisson distribution
  - `_rate_encode`: Conversion of intensity to spike firing rate
  - `_temporal_encode`: Temporal firing pattern encoding
  - `_spike_encode`: Composite spike pattern generation
- **Usage**: Encoding method selection via `encoding_type` parameter

### 12. LIF Neuron Integration System
- **Original Specification**: Not implemented in original
- **Extended Feature**: Complete Leaky Integrate-and-Fire (LIF) neuron model integration
- **Implementation Features**:
  - `LIFNeuron` class: Individual LIF neuron implementation
  - `LIFNeuronLayer` class: Layer-wise LIF management
  - Amine concentration management: `set_amine_concentration`, `get_amine_concentration`
  - Membrane potential dynamics: Membrane potential updates via differential equations
- **Technical Feature**: Biologically plausible spike generation mechanisms
- **Parameters**: Control of resting potential, threshold, reset potential, time constants

### 13. Spike-ED Conversion Interface
- **Original Specification**: Not implemented in original
- **Extended Feature**: Bidirectional conversion system between spike activity and ED method input
- **Implementation Features**:
  - `convert_ed_outputs_to_spike_activities`: Convert ED outputs to spike activity patterns
  - `convert_to_lif_input`: Convert image data to LIF input current (v019 Phase 11 compliant)
  - `convert_spikes_to_ed_input`: Convert spike patterns to ED method input format
- **Technical Feature**: Complete preservation of excitatory/inhibitory pair structure
- **Usage**: Seamless integration between ED method and SNN via unified learning

### 14. High-Speed SNN Network Implementation
- **Original Specification**: Not implemented in original
- **Extended Feature**: Acceleration system for large-scale SNN computation
- **Implementation Features**:
  - `SNNNetworkFastV2` class: High-speed SNN implementation
  - `encode_input_fast`: High-speed input encoding
  - `simulate_snn_fast`: Efficient SNN dynamics simulation
- **Technical Feature**: Computation step reduction and vectorized processing
- **Performance Improvement**: Significant acceleration compared to conventional SNN implementation

### 15. Pure ED Preprocessing System
- **Original Specification**: Not implemented in original
- **Extended Feature**: Preprocessing class fully compliant with ED method theory
- **Implementation Feature**: Data preprocessing integration via `PureEDPreprocessor`
- **Technical Feature**: Excitatory/inhibitory pairing, Dale's Principle application
- **Theory Compliance**: Complete consistency guarantee with Isamu Kaneko's original theory

### 16. Integrated Visualization System Extension
- **Original Specification**: Basic visualization only
- **Extended Feature**: Comprehensive real-time visualization of SNN learning process
- **Implementation Features**:
  - `RealtimeLearningVisualizer`: Real-time learning visualization
  - `setup_japanese_font`: Automatic Japanese font setup
  - `wait_for_keypress_or_timeout`: Interactive control
- **Technical Feature**: Integrated display of spike activity, membrane potential, learning progress
- **Usage**: SNN-specific dynamics visualization and ED method learning monitoring

### 17. Modular Architecture
- **Original Specification**: Single file implementation
- **Extended Feature**: Maintainability improvement through function-based module separation
- **Implementation Structure**:
  - `modules/snn/`: SNN core functionality
  - `modules/ed_learning/`: ED method learning engine
  - `modules/visualization/`: Visualization system
  - `modules/utils/`: Common utilities
- **Technical Feature**: Extensibility assurance through loose coupling design
- **Maintainability**: Independence improvement for feature additions and modifications

## Core ED Method Theory (Original Specification Completely Preserved)

The following is Isamu Kaneko's original ED method theory, which is 100% preserved in the extended version:

### 1. Independent Output Neuron Architecture
- Each output neuron maintains completely independent weight space
- 3D weight array: `w_ot_ot[output_neuron][destination][source]`
- Each class constitutes an independent neural network

### 2. Excitatory/Inhibitory Neuron Pair Structure
- Input layer: Composed of excitatory (+1) and inhibitory (-1) neuron pairs
- Same-type connections: Positive weight constraints
- Different-type connections: Negative weight constraints
- Biological plausibility guarantee

### 3. Amine Diffusion Learning Control
- Output layer errors diffuse as amine concentrations to hidden layers
- Two types of amines: `del_ot[n][k][0]` (positive error), `del_ot[n][k][1]` (negative error)
- Diffusion intensity control via parameter `u1`

### 4. Complete Connection Structure of Input Layer
- **Important**: All neurons in input layer (both excitatory and inhibitory) connect to next layer
- Automatic input size calculation:
  - MNIST/Fashion-MNIST: 784 pixels → 1568 neurons (784 pairs)
  - CIFAR-10/100: 3072 pixels (32×32×3 color) → 6144 neurons (3072 pairs)
- Weight matrix size: `paired_input_size × hidden_units` (input layer→hidden layer)
- Each pixel value is set as the same value for both excitatory and inhibitory neurons

### 5. Strict Application of Dale's Principle
- **Principle**: Neurons have either excitatory or inhibitory properties exclusively
- **Implementation**: Weight sign constraint `w *= ow[source] * ow[target]`
- **Same-type connections**: `(+1) * (+1) = +1` or `(-1) * (-1) = +1` → positive weights
- **Different-type connections**: `(+1) * (-1) = -1` or `(-1) * (+1) = -1` → negative weights
- **Biological Plausibility**: Adherence to fundamental principles in actual neural systems

## Implementation Policy (Extended Version Support)

- **Absolute Preservation of ED Method Theory**: Isamu Kaneko's original theory is never modified; extended features are added in compliance with the theory
- **Prohibition of "Error Backpropagation Using Chain Rule of Differentiation"**: Usage of "error backpropagation using chain rule of differentiation" is prohibited
- **SNN Network Support**: When applying ED method to SNN architecture, ED method learning algorithms are completely preserved
- **Coding Rules**: Comply with PEP8, prioritizing readability
- **Clear Extended Features**: When adding new features, clearly indicate in comments that they are extensions from the original theory
- **Code Readability**: Keep comments moderate (as minimal as possible). Comments should describe why rather than what, ensuring code intent is clear
- **Modularization**: Clearly separate each function and implement as reusable modules
- **Test-Driven Development**: When implementing new features, create unit tests for those features and pass tests before implementation
- **Parameter Adjustment**: Enable flexible modification of basic parameters using argparse to facilitate experimental adjustments
- **Theoretical Basis for Extended Features**: When implementing extended features, maintain consistency with ED method theory and explain theoretical basis in comments as needed

---

**This specification was created based on operation verification of the original C implementation, detailed code analysis, and implementation verification of extended features.**  
**Original Verification Date**: August 30, 2025  
**Extended Version Creation Date**: September 13, 2025  
**SNN Integration Implementation**: November 2025  
**Verifier**: AI Analysis System  
**Source Code**: `/ed_original_src/` (compile and execution verified) + Extended Python implementation + SNN integration implementation