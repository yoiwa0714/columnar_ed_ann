# Columnar ED Method — Operational Flow and Code Walkthrough

This document illustrates the operational flow of the Columnar ED method using block diagrams and provides annotated explanations of the core ED method functions.

> **Target files:** `columnar_ed_ann.py` (main script), `modules/ed_network.py` (ED network), `modules/column_structure.py` (column structure)

---

## Table of Contents

- [1. Overall Flow](#1-overall-flow)
- [2. Single Epoch Training Flow](#2-single-epoch-training-flow)
- [3. Single Sample Learning Flow (Core of ED Method)](#3-single-sample-learning-flow-core-of-ed-method)
- [4. Core Function Details](#4-core-function-details)
  - [4.1 forward() — Forward Propagation](#41-forward--forward-propagation)
  - [4.2 Output Layer Gradient — Saturation Suppression Term](#42-output-layer-gradient--saturation-suppression-term)
  - [4.3 Amine Concentration and Diffusion](#43-amine-concentration-and-diffusion)
  - [4.4 Hidden Layer Weight Update](#44-hidden-layer-weight-update)
  - [4.5 create_column_membership() — Column Structure](#45-create_column_membership--column-structure)

---

## 1. Overall Flow

The processing flow of the main script `columnar_ed_ann.py`.

```
┌─────────────────────────────────────────────────┐
│         Parse Command-Line Arguments              │
│  (hidden layers, data size, epochs, etc.)         │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│       Load Auto Parameters from YAML              │
│  (learning rates, init scales per layer count)    │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│              Load Dataset                         │
│  (MNIST / Fashion-MNIST / Custom)                 │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│     Gabor Feature Extraction (ON by default)      │
│  (V1 simple cell model, input quality boost)      │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│       ED Network Construction & Initialization    │
│  ┌──────────────────────────────────────┐        │
│  │ • Column structure (honeycomb layout) │        │
│  │ • He initialization × init_scales     │        │
│  │ • Dale's Principle sign matrix        │        │
│  │ • Non-column neuron sparsification    │        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│        Epoch Loop (→ details in next section)     │
│  ┌──────────────────────────────────────┐        │
│  │  Each epoch:                          │        │
│  │   1. train_epoch() → weight updates   │        │
│  │   2. evaluate_parallel() → accuracy   │        │
│  │   3. Record best accuracy             │        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│              Display Result Summary               │
│  (final accuracy, best accuracy, per-class)       │
└─────────────────────────────────────────────────┘
```

---

## 2. Single Epoch Training Flow

The processing flow of the `train_epoch()` method. The ED method performs online learning (weights are updated immediately after each sample).

```
┌─────────────────────────────────────────────────┐
│    Loop over n_samples samples                    │
│                                                   │
│    ┌───────────────────────────────────────┐      │
│    │  train_one_sample(x, y_true)          │      │
│    │                                       │      │
│    │  1. forward(x)         → prediction   │      │
│    │  2. cross_entropy_loss → loss calc    │      │
│    │  3. update_weights()   → weight update│      │
│    │     (→ details in next section)       │      │
│    └───────────────────────────────────────┘      │
│                                                   │
│    * Weights updated immediately per sample       │
│      (online learning)                            │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  evaluate_parallel() — re-evaluate with final     │
│  weights (for fair comparison with test accuracy) │
└─────────────────────────────────────────────────┘
```

---

## 3. Single Sample Learning Flow (Core of ED Method)

The processing flow of `update_weights()` → `_compute_gradients()`. This is the complete picture of the ED method's learning mechanism.

```
Input x
  │
  ▼
┌─────────────────────────────────────────────────┐
│  E/I Pairing: x → [x, x]                        │
│  (Foundation for Dale's Principle)                │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  Forward Propagation forward()                    │
│  ┌──────────────────────────────────────┐        │
│  │ Hidden: dot(W, z) → tanh(a) → z     │        │
│  │ Output: dot(W, z) → softmax(a) → p  │        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  Output Layer Weight Update                       │
│  ┌──────────────────────────────────────┐        │
│  │  error     = target_onehot − pred    │        │
│  │  sat_term  = |z| × (1 − |z|)        │ ★core  │
│  │  ΔW = lr × (error × sat_term) ⊗ z_in│        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  Amine Concentration Calculation                  │
│  ┌──────────────────────────────────────┐        │
│  │ Inject amine for correct class ONLY  │ ★key   │
│  │ amine = (1 − pred_prob_correct)      │        │
│  │         × initial_amine_conc         │        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  Hidden Layer Weight Update (last → first layer)  │
│                                                   │
│  For each hidden layer:                           │
│  ┌──────────────────────────────────────┐        │
│  │ 1. Amine diffusion: amine × coef     │        │
│  │    (u1/u2)                            │        │
│  │ 2. Column membership identifies       │        │
│  │    target neurons; rank by activation │ ★core  │
│  │ 3. Saturation term: |z| × (1 − |z|)  │        │
│  │ 4. ΔW = lr × amine × sat_term ⊗ z_in │        │
│  │ 5. Gradient clipping                  │        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  Apply weights + Dale's Principle sign            │
│  enforcement (layer 0 only)                       │
│  Mild regularization on output weights            │
└─────────────────────────────────────────────────┘
```

**Fundamental differences from BP (Backpropagation):**

| | BP | ED Method (this implementation) |
|---|---|---|
| Error signal propagation | Chain rule of derivatives | Amine concentration spatial diffusion |
| Layer updates | Depends on downstream gradients | **Each layer updates independently** |
| Update magnitude | Derivative of activation function | **Saturation term** `\|z\| × (1 − \|z\|)` |
| Learning targets | All neurons | **Column neurons only** |

---

## 4. Core Function Details

### 4.1 forward() — Forward Propagation

**File:** `modules/ed_network.py`

Forward propagation is relatively straightforward. After converting the input to E/I pairs, each hidden layer applies tanh activation, and the output layer converts to a probability distribution via softmax.

```python
def forward(self, x):
    # ★ Dale's Principle: duplicate input as [x, x]
    # Combined with sign constraints on weight matrix to realize
    # excitatory/inhibitory biological constraints
    x_paired = create_ei_pairs(x)

    z_hiddens = []
    z_current = x_paired

    for layer_idx in range(self.n_layers):
        # ★ Each layer is a simple matrix product → tanh (no derivatives used)
        a_hidden = np.dot(self.w_hidden[layer_idx], z_current)
        z_hidden = tanh_activation(a_hidden)         # output range: -1 to +1
        z_hiddens.append(z_hidden)
        z_current = z_hidden

    # ★ Output layer: softmax for probabilities (used only in forward pass,
    #   NOT used for learning signal computation)
    a_output = np.dot(self.w_output, z_hiddens[-1])
    z_output = softmax(a_output)                     # sum = 1.0

    return z_hiddens, z_output, x_paired
```

**Key points:**
- `create_ei_pairs(x)` simply concatenates the input as `[x, x]`, but when combined with the sign constraints (Dale's Principle) applied to layer 0's weight matrix, the first half functions as excitatory input and the second half as inhibitory input
- The tanh in each hidden layer has saturation characteristics (becomes less responsive near ±1), which works in concert with the ED method's saturation suppression term
- Softmax is used to interpret outputs as probabilities, but unlike BP, the derivative of softmax is NOT used for backpropagation

<details>
<summary>📄 Full code of forward()</summary>

```python
def forward(self, x):
    """
    Forward propagation (multi-class classification)

    Hidden layers: tanh activation (bidirectional, saturation characteristics)
    Output layer: SoftMax (probability distribution)

    Args:
        x: Input data (shape: [n_input])

    Returns:
        z_hiddens: List of outputs from each hidden layer
        z_output: Output layer probability distribution (SoftMax, sum=1.0)
        x_paired: Input pair (excitatory + inhibitory)
    """
    # Input pair structure: x → [x, x] (Dale's Principle sign matrix
    # realizes excitatory/inhibitory behavior)
    x_paired = create_ei_pairs(x)

    z_hiddens = []
    z_current = x_paired

    for layer_idx in range(self.n_layers):
        a_hidden = np.dot(self.w_hidden[layer_idx], z_current)
        z_hidden = tanh_activation(a_hidden)
        z_hiddens.append(z_hidden)
        z_current = z_hidden

    # Output layer: SoftMax activation
    a_output = np.dot(self.w_output, z_hiddens[-1])
    z_output = softmax(a_output)

    return z_hiddens, z_output, x_paired
```

</details>

---

### 4.2 Output Layer Gradient — Saturation Suppression Term

**File:** `modules/ed_network.py` — first part of `_compute_gradients()`

The output layer weight update uses the ED method's saturation suppression term. This is fundamentally different from BP's softmax derivative (cross-entropy derivative).

```python
# --- Output layer gradient computation ---

# Error is the difference between the one-hot target and predicted probabilities
target_probs = np.zeros(self.n_output)
target_probs[y_true] = 1.0
error_output = target_probs - z_output

# ★★★ Core of the ED method: Saturation Suppression Term ★★★
# This is NOT sigmoid derivative z(1-z) nor tanh derivative (1-z²)!
# It uses the absolute value: |z| × (1 - |z|)
saturation_output = np.abs(z_output) * (1.0 - np.abs(z_output))

# Weight update = learning_rate × outer_product(error × saturation_term, input)
output_lr = self.layer_lrs[-1]
gradients['w_output'] = output_lr * np.outer(
    error_output * saturation_output,
    z_hiddens[-1]
)
```

**Key points:**
- The **saturation suppression term `|z| × (1 - |z|)`** is the core of the ED method. The closer a neuron's activation value `z` is to saturation (0 or ±1), the smaller its update magnitude becomes
- In BP, `∂loss/∂z_output` is computed via the chain rule of derivatives, but in the ED method, the saturation suppression term replaces the derivative
- This term reaches its maximum value of `0.25` when `z=0.5` and becomes `0` when `z=0` or `z=1`. This means "neurons with moderate confidence are updated the most" — an intuitively reasonable behavior

---

### 4.3 Amine Concentration and Diffusion

**File:** `modules/ed_network.py` — middle part of `_compute_gradients()`

Amine concentration is generated from the output layer error and diffused to hidden layers along the column structure. This is the mechanism that replaces BP's "chain rule backpropagation."

```python
# --- Amine concentration calculation ---

# ★★★ Pure ED method: inject amine for correct class ONLY ★★★
# No negative learning signal for incorrect classes
# (no "this is NOT class X" teaching)
amine_concentration = np.zeros(self.n_output)
error_correct = 1.0 - z_output[y_true]   # shortfall in correct class probability
if error_correct > 0:
    # Amine is released only when the correct class output is insufficient
    amine_concentration[y_true] = error_correct * self.initial_amine
```

```python
# --- Amine diffusion to hidden layers (reverse order loop) ---

for layer_idx in range(self.n_layers - 1, -1, -1):
    # ★ Diffusion coefficient: different for last hidden layer (u1) vs others (u2)
    # u1 > u2 makes amine signal stronger for layers closer to output
    if layer_idx == self.n_layers - 1:
        diffusion_coef = self.u1
    else:
        diffusion_coef = self.u2

    # Amine diffuses spatially (concentration × diffusion coefficient)
    amine_diffused = amine_concentration * diffusion_coef

    # ★★★ Selective diffusion via column membership ★★★
    # Amine reaches only the column neurons assigned to the relevant class
    membership = self.column_membership_all_layers[layer_idx]
    active_classes = np.where(amine_diffused >= 1e-8)[0]

    # Rank column members by activation value
    active_membership = membership[active_classes]
    masked_activations = np.where(active_membership, z_current, -np.inf)
    sorted_indices = np.argsort(-masked_activations, axis=1)
    ranks = np.argsort(sorted_indices, axis=1)

    # ★ Rank-dependent learning rate: more active neurons learn more strongly
    learning_weights = self._learning_weight_lut[np.minimum(ranks, len(self._learning_weight_lut) - 1)]

    # ★ Non-column neurons do not learn (maintain fixed random weights)
    learning_weights = np.where(active_membership, learning_weights, 0.0)

    # Amine amount per neuron = diffusion value × rank-dependent learning rate
    amine_hidden[active_classes] = amine_diffused[active_classes, np.newaxis] * learning_weights
```

**Key points:**
- **Correct class only learning** (pure ED method): Rather than penalizing incorrect classes, learning only strengthens the correct class output. This corresponds to how reward-system neurotransmitters work in the brain
- **Amine diffusion** models the spatial diffusion of neurotransmitters such as noradrenaline and dopamine in the brain. Unlike BP's mechanism of "backpropagating derivatives across layers," each layer receives a "spatial signal" in the form of amine concentration and learns independently
- **Column membership** ensures that amine signals reach only the neurons assigned to the target class. This is the key to enabling multi-class classification
- **Rank-dependent learning rate** ensures that more activated neurons within a column learn more strongly (a winner-learns-more mechanism)

---

### 4.4 Hidden Layer Weight Update

**File:** `modules/ed_network.py` — latter part of `_compute_gradients()`

Using amine diffusion amounts and the saturation suppression term, each hidden layer updates its weights **independently**.

```python
# --- Hidden layer weight update (for each layer) ---

# ★★★ Saturation suppression term (same formula as output layer) ★★★
# Neurons with saturated activations (close to ±1) receive smaller updates
z_active = z_hiddens[layer_idx][active_neurons]
saturation_term = np.abs(z_active) * (1.0 - np.abs(z_active))
saturation_term = np.maximum(saturation_term, 1e-3)    # minimum floor

# ★★★ Learning signal = learning_rate × amine_diffusion × saturation_term ★★★
# NOT the "chain rule" of BP — three independent factors multiplied together
layer_lr = self.layer_lrs[layer_idx]
learning_signals = layer_lr * amine_hidden[:, active_neurons] * saturation_term[np.newaxis, :]

# Sum signals from all classes → outer product with input for weight deltas
signal_sum = learning_signals.sum(axis=0)
delta_w = signal_sum[:, np.newaxis] * z_input[np.newaxis, :]

# Layers 1+: maintain weight sign constraint
if layer_idx > 0:
    w_sign = np.sign(self.w_hidden[layer_idx][active_neurons, :])
    w_sign[w_sign == 0] = 1
    delta_w *= w_sign

# Gradient clipping (prevent divergence)
if self.gradient_clip > 0:
    delta_w_norms = np.linalg.norm(delta_w, axis=1, keepdims=True)
    clip_mask = delta_w_norms > self.gradient_clip
    delta_w = np.where(clip_mask, delta_w * (self.gradient_clip / delta_w_norms), delta_w)
```

**Key points:**
- **Each layer learns independently**: The learning signal is the product of `learning_rate × amine_diffusion × saturation_term`. Unlike BP, which multiplies and backpropagates downstream gradients, each layer determines its update amount using only the amine concentration and its own activation values
- Note that the **saturation suppression term** uses the same formula as the output layer. This demonstrates the uniformity of the ED method: "all layers operate on the same principle"
- **Sign constraint for layers 1+**: Prevents weight sign flips (generalization of Dale's Principle)

<details>
<summary>📄 Full code of _compute_gradients()</summary>

```python
def _compute_gradients(self, x_paired, z_hiddens, z_output, y_true):
    """
    Gradient computation using the ED method

    ★Core★ Does NOT use the chain rule of derivatives. Instead:
    1. Generate amine concentration from output layer error
    2. Amine diffuses to hidden layers along column structure
    3. Each layer independently learns using saturation term abs(z)*(1-abs(z))

    Args:
        x_paired: Input pair
        z_hiddens: Outputs from each hidden layer
        z_output: Output layer probability distribution
        y_true: True class label

    Returns:
        gradients: Dictionary of gradients for each layer
    """
    gradients = {
        'w_output': None,
        'w_hidden': [None] * self.n_layers,
    }

    # ============================================
    # 1. Output layer gradient computation
    # ============================================
    target_probs = np.zeros(self.n_output)
    target_probs[y_true] = 1.0
    error_output = target_probs - z_output

    # ★Saturation suppression term★ Core of ED method — NOT sigmoid derivative
    saturation_output = np.abs(z_output) * (1.0 - np.abs(z_output))

    output_lr = self.layer_lrs[-1]
    gradients['w_output'] = output_lr * np.outer(
        error_output * saturation_output,
        z_hiddens[-1]
    )

    # ============================================
    # 2. Output layer amine concentration
    # ============================================
    # Pure ED method: correct class only learning
    amine_concentration = np.zeros(self.n_output)
    error_correct = 1.0 - z_output[y_true]
    if error_correct > 0:
        amine_concentration[y_true] = error_correct * self.initial_amine

    # ============================================
    # 3. Multi-layer amine diffusion and gradient computation
    #    (reverse order, NO chain rule of derivatives)
    # ============================================
    for layer_idx in range(self.n_layers - 1, -1, -1):
        if layer_idx == 0:
            z_input = x_paired
        else:
            z_input = z_hiddens[layer_idx - 1]

        # Diffusion coefficient selection (last hidden=u1, others=u2)
        if layer_idx == self.n_layers - 1:
            diffusion_coef = self.u1
        else:
            diffusion_coef = self.u2

        # Amine diffusion (selective along column structure)
        amine_mask = amine_concentration >= 1e-8
        amine_diffused = amine_concentration * diffusion_coef

        # Membership-based: activation rank Top-K learning
        membership = self.column_membership_all_layers[layer_idx]
        z_current = z_hiddens[layer_idx]
        n_neurons = self.n_hidden[layer_idx]

        active_classes = np.where(amine_diffused >= 1e-8)[0]
        n_active = len(active_classes)

        if n_active == 0:
            amine_hidden = np.zeros((self.n_output, n_neurons))
        else:
            # Rank column members by activation value
            active_membership = membership[active_classes]
            masked_activations = np.where(active_membership, z_current, -np.inf)
            sorted_indices = np.argsort(-masked_activations, axis=1)
            ranks = np.argsort(sorted_indices, axis=1)

            # Look up learning rates from rank via LUT
            clamped_ranks = np.minimum(ranks, len(self._learning_weight_lut) - 1)
            learning_weights = self._learning_weight_lut[clamped_ranks]

            # Non-column neurons get amine=0 (do not learn)
            learning_weights = np.where(active_membership, learning_weights, 0.0)

            # Apply learning rates to diffused amine values
            amine_hidden = np.zeros((self.n_output, n_neurons))
            amine_hidden[active_classes] = (
                amine_diffused[active_classes, np.newaxis] *
                learning_weights
            )

        amine_hidden = amine_hidden * amine_mask[:, np.newaxis]

        # Identify active neurons (only rows with non-zero amine)
        neuron_mask = np.any(amine_hidden >= 1e-8, axis=0)
        active_neurons = np.where(neuron_mask)[0]

        if len(active_neurons) == 0:
            gradients['w_hidden'][layer_idx] = None
            continue

        # ★Saturation suppression term★ abs(z)*(1-abs(z)) — NOT chain rule
        z_active = z_hiddens[layer_idx][active_neurons]
        saturation_term_raw = np.abs(z_active) * (1.0 - np.abs(z_active))
        saturation_term = np.maximum(saturation_term_raw, 1e-3)

        # Learning signal = learning_rate × amine_diffusion × saturation_term
        layer_lr = self.layer_lrs[layer_idx]
        learning_signals = (
            layer_lr *
            amine_hidden[:, active_neurons] *
            saturation_term[np.newaxis, :]
        )

        # Gradient computation (sum signals across classes × input outer product)
        signal_sum = learning_signals.sum(axis=0)
        delta_w_batch = signal_sum[:, np.newaxis] * z_input[np.newaxis, :]

        # Sign constraint for layers 1+
        if layer_idx > 0:
            w_sign = np.sign(self.w_hidden[layer_idx][active_neurons, :])
            w_sign[w_sign == 0] = 1
            delta_w_batch *= w_sign

        # Gradient clipping
        if self.gradient_clip > 0:
            delta_w_norms = np.linalg.norm(delta_w_batch, axis=1, keepdims=True)
            clip_mask = delta_w_norms > self.gradient_clip
            delta_w_batch = np.where(
                clip_mask,
                delta_w_batch * (self.gradient_clip / delta_w_norms),
                delta_w_batch
            )

        # Column neuron learning rate suppression (prevent weight saturation)
        layer_lr_factor = self.column_lr_factors[layer_idx]
        if layer_lr_factor < 1.0 and layer_idx < len(self.column_membership_all_layers):
            membership = self.column_membership_all_layers[layer_idx]
            is_column_neuron = np.any(membership, axis=0)
            active_is_column = is_column_neuron[active_neurons]
            if np.any(active_is_column):
                delta_w_batch[active_is_column, :] *= layer_lr_factor

        # Save in sparse format (update only active rows)
        gradients['w_hidden'][layer_idx] = (active_neurons, delta_w_batch)

    return gradients
```

</details>

<details>
<summary>📄 Full code of update_weights()</summary>

```python
def update_weights(self, x_paired, z_hiddens, z_output, y_true):
    """
    Weight update (ED method compliant, no chain rule of derivatives)

    Each layer independently updates weights based on amine diffusion signals.
    Dale's Principle sign constraint applied to layer 0 only.
    """
    gradients = self._compute_gradients(x_paired, z_hiddens, z_output, y_true)

    # Apply gradients
    self.w_output += gradients['w_output']

    for layer_idx in range(self.n_layers):
        sparse_grad = gradients['w_hidden'][layer_idx]
        if sparse_grad is not None:
            active_neurons, delta_w_batch = sparse_grad
            self.w_hidden[layer_idx][active_neurons] += delta_w_batch

            # Dale's Principle enforcement for layer 0 (active rows only)
            if layer_idx == 0:
                self.w_hidden[0][active_neurons] = (
                    np.abs(self.w_hidden[0][active_neurons]) *
                    self._sign_matrix_layer0[active_neurons]
                )

    # Mild regularization on output weights
    self.w_output *= (1.0 - 0.00001)
```

</details>

---

### 4.5 create_column_membership() — Column Structure

**File:** `modules/column_structure.py`

The column structure places hidden layer neurons in a 2D space and assigns the nearest neurons to each class.

```python
def create_column_membership(n_hidden, n_classes, ..., column_neurons=None):
    membership = np.zeros((n_classes, n_hidden), dtype=bool)

    # ★ Number of neurons assigned to each class
    # column_neurons=1: reservoir computing (only 1 neuron per class learns)
    # column_neurons=10: more neurons represent each class
    if column_neurons is not None:
        neurons_per_class = column_neurons

    # ★ Honeycomb layout (2-3-3-2 pattern) determines 10 class centers
    # Mimics the column structure found in the visual cortex of the cerebral cortex
    class_coords = {
        0: (center + scale*(-1), center + scale*(-1)),  # top-left
        1: (center + scale*(+1), center + scale*(-1)),  # top-right
        2: (center + scale*(-2), center + scale*(0)),   # mid-left
        3: (center + scale*(0),  center + scale*(0)),   # mid-center
        4: (center + scale*(+2), center + scale*(0)),   # mid-right
        5: (center + scale*(-2), center + scale*(+1)),  # lower-mid-left
        6: (center + scale*(0),  center + scale*(+1)),  # lower-mid-center
        7: (center + scale*(+2), center + scale*(+1)),  # lower-mid-right
        8: (center + scale*(-1), center + scale*(+2)),  # bottom-left
        9: (center + scale*(+1), center + scale*(+2)),  # bottom-right
    }

    # ★ Assign the nearest neurons_per_class neurons to each class center
    for class_idx in range(n_classes):
        center_row, center_col = class_coords[class_idx]
        distances = np.sqrt(
            (neuron_positions[:, 0] - center_row)**2 +
            (neuron_positions[:, 1] - center_col)**2
        )
        closest_indices = np.argsort(distances)[:neurons_per_class]
        membership[class_idx, closest_indices] = True

    return membership, neuron_positions, class_coords
```

**Key points:**
- **Honeycomb layout (2-3-3-2 pattern)**: Places 10 class center coordinates in a hexagonal-grid-like arrangement. This mimics the structure in the visual cortex where neurons with similar properties cluster in columnar formations
- **Nearest-neighbor assignment**: The `neurons_per_class` neurons closest to each class center become that class's "column neurons"
- With **`column_neurons=1`**, only 1 neuron per class is a learning target, making this equivalent to reservoir computing (only 10 out of 2048 learn, 99.5% are fixed weights)
- `membership` is a boolean array of shape `[n_classes, n_hidden]` that determines "which neurons receive amine signals" during learning

<details>
<summary>📄 Full code of create_column_membership()</summary>

```python
def create_column_membership(n_hidden, n_classes, participation_rate=1.0,
                             use_hexagonal=True, column_radius=0.4, column_neurons=None):
    """
    Create column membership flags

    Manage which class column each neuron belongs to using boolean flags.
    Maintains column structure while weights are acquired through learning.

    Args:
        n_hidden: Total number of hidden layer neurons
        n_classes: Number of output classes
        participation_rate: Proportion of neurons assigned to each class (0.0-1.0)
        use_hexagonal: True for honeycomb layout, False for sequential assignment
        column_radius: Column radius (reference value for honeycomb layout)
        column_neurons: Number of neurons per class (explicit, highest priority)

    Returns:
        membership: Boolean array of shape [n_classes, n_hidden]
        neuron_positions: 2D coordinate array of shape [n_hidden, 2]
        class_coords: Dictionary of column center coordinates per class
    """
    membership = np.zeros((n_classes, n_hidden), dtype=bool)
    neuron_positions = None
    class_coords = None

    # Neurons per class (priority: column_neurons > participation_rate)
    if column_neurons is not None:
        neurons_per_class = column_neurons
    else:
        neurons_per_class = int(n_hidden * participation_rate / n_classes)

    if neurons_per_class == 0:
        neurons_per_class = 1

    if use_hexagonal:
        # Honeycomb layout (2-3-3-2 pattern for 10 classes, centered)
        grid_size = int(np.ceil(np.sqrt(n_hidden)))
        grid_center = grid_size / 2.0

        # Scale factor: spread columns over ~4/5 of the hidden layer area
        scale_factor = (grid_size * 0.8) / 4.0

        # 2-3-3-2 layout with 10 class coordinates
        class_coords = {
            0: (grid_center + scale_factor * (-1), grid_center + scale_factor * (-1)),
            1: (grid_center + scale_factor * (+1), grid_center + scale_factor * (-1)),
            2: (grid_center + scale_factor * (-2), grid_center + scale_factor * (0)),
            3: (grid_center + scale_factor * (0),  grid_center + scale_factor * (0)),
            4: (grid_center + scale_factor * (+2), grid_center + scale_factor * (0)),
            5: (grid_center + scale_factor * (-2), grid_center + scale_factor * (+1)),
            6: (grid_center + scale_factor * (0),  grid_center + scale_factor * (+1)),
            7: (grid_center + scale_factor * (+2), grid_center + scale_factor * (+1)),
            8: (grid_center + scale_factor * (-1), grid_center + scale_factor * (+2)),
            9: (grid_center + scale_factor * (+1), grid_center + scale_factor * (+2))
        }

        neuron_positions = np.array([
            [i // grid_size, i % grid_size] for i in range(n_hidden)
        ])

        for class_idx in range(min(n_classes, len(class_coords))):
            center_row, center_col = class_coords[class_idx]

            distances = np.sqrt(
                (neuron_positions[:, 0] - center_row) ** 2 +
                (neuron_positions[:, 1] - center_col) ** 2
            )

            closest_indices = np.argsort(distances)[:neurons_per_class]
            membership[class_idx, closest_indices] = True
    else:
        # Sequential assignment
        grid_size = int(np.ceil(np.sqrt(n_hidden)))
        neuron_positions = np.array([
            [i // grid_size, i % grid_size] for i in range(n_hidden)
        ])
        class_coords = None

        for class_idx in range(n_classes):
            start_idx = class_idx * neurons_per_class
            end_idx = min(start_idx + neurons_per_class, n_hidden)
            membership[class_idx, start_idx:end_idx] = True

    return membership, neuron_positions, class_coords
```

</details>

---

## Summary

The learning in the Columnar ED method consists of three key elements:

1. **Saturation suppression term** `|z| × (1 - |z|)` — A mechanism that allows each layer to independently determine appropriate update magnitudes without using the chain rule of derivatives
2. **Amine diffusion** — A mechanism that converts output layer errors into amine concentrations and diffuses them to hidden layers along the column structure
3. **Column structure** — A mechanism that assigns hidden layer neurons to classes, enabling selective transmission of amine signals

All of these are biologically plausible mechanisms, and **no backpropagation based on the chain rule of derivatives** is used whatsoever.
