# Columnar ED Method: Detailed Principles

## Overview
This document is a draft that formalizes the implemented Columnar ED method as a set of logical equations, based on the original ED method description.

The goals are:

1. Describe learning as a local plasticity rule without the chain rule.
2. Explicitly show how column structure (membership) and rank weighting affect learning.
3. Provide implementation-aligned update equations in a paper-ready form.

---

## 1. Forward Pass (Implementation-Aligned)

Define the excitatory/inhibitory paired input as

$$
\tilde{\mathbf{x}} \in \mathbb{R}^{2d}, \qquad \mathbf{z}^{(0)} = \tilde{\mathbf{x}}
$$

For hidden layers $\ell=1,\dots,L$:

$$
\mathbf{a}^{(\ell)} = W^{(\ell)}\mathbf{z}^{(\ell-1)}, \qquad
\mathbf{z}^{(\ell)} = \phi\!\left(\mathbf{a}^{(\ell)}\right)
$$

(Main setting: $\phi=\tanh$.)

Output layer:

$$
\mathbf{a}^{(o)} = W^{(o)}\mathbf{z}^{(L)}, \qquad
\mathbf{p} = \mathrm{softmax}(\mathbf{a}^{(o)})
$$

---

## 2. ED-Type Output Layer Update

With one-hot target $\mathbf{t}$, define output error:

$$
\mathbf{e}^{(o)} = \mathbf{t} - \mathbf{p}
$$

Output saturation suppression term:

$$
\mathbf{s}^{(o)} = |\mathbf{p}| \odot (1 - |\mathbf{p}|)
$$

Output weight update:

$$
\Delta W^{(o)} = \eta_o\left(\mathbf{e}^{(o)} \odot \mathbf{s}^{(o)}\right)\mathbf{z}^{(L)\top}
$$

This equation uses only output-local quantities and does not use chain-rule backpropagation.

---

## 3. Amine Generation (Pure ED Setting)

Let the correct class be $y$. Generate amine only for the correct class:

$$
A_{c,+}^{(o)} = \alpha_0\,(1-p_y)\,\mathbf{1}[c=y], \qquad
A_{c,-}^{(o)} = 0
$$

where $\alpha_0$ is initial amine strength and $\mathbf{1}[\cdot]$ is the indicator function.

---

## 4. Intra-Layer Diffusion and Column Weighting

Diffusion coefficient at layer $\ell$:

$$
d_\ell =
\begin{cases}
u_1, & \ell=L \\
u_2, & \ell<L
\end{cases}
$$

(With `uniform_amine`, all layers use $d_\ell=u_1$.)

Column membership for class $c$:

$$
M_{c,j}^{(\ell)} \in \{0,1\}
$$

Let rank within class be $r_{c,j}^{(\ell)}$ and rank LUT be $g(\cdot)$:

$$
\lambda_{c,j}^{(\ell)}=
\begin{cases}
 g\!\left(r_{c,j}^{(\ell)}\right), & M_{c,j}^{(\ell)}=1 \\
 \lambda^{\mathrm{NC}}_{c,j,\ell}, & M_{c,j}^{(\ell)}=0
\end{cases}
$$

$\lambda^{\mathrm{NC}}_{c,j,\ell}$ is one of:

- 0 (no non-column learning)
- nearest-class assigned NC strength
- spatially decayed diffusion
- uniform micro value

Hence, hidden-layer amine signal:

$$
H_{c,\sigma,j}^{(\ell)} = d_\ell\,A_{c,\sigma}^{(o)}\,\lambda_{c,j}^{(\ell)}
$$

with $\sigma\in\{+, -\}$.

---

## 5. Hidden Layer Update (Core of Columnar ED)

Hidden saturation suppression:

$$
q_j^{(\ell)} = \max\!\left(|z_j^{(\ell)}|(1-|z_j^{(\ell)}|),\,\varepsilon\right)
$$

Local signal summed over class/sign:

$$
u_j^{(\ell)} = \eta_\ell\sum_{c,\sigma} H_{c,\sigma,j}^{(\ell)}\,q_j^{(\ell)}
$$

With input-side activation $\mathbf{z}^{(\ell-1)}$, base update:

$$
\Delta W_{j,:}^{(\ell)} = u_j^{(\ell)}\,\mathbf{z}^{(\ell-1)\top}
$$

Implementation additionally applies:

1. sign-alignment mask (intermediate layers)
2. norm clipping
3. per-layer column LR factor $\beta_\ell\le 1$ on column rows

$$
\Delta W_{j,:}^{(\ell)} \leftarrow
\mathrm{Clip}\!\left(\Gamma_j^{(\ell)}\odot\Delta W_{j,:}^{(\ell)}\right)
$$

$$
\Delta W_{j,:}^{(\ell)} \leftarrow
\begin{cases}
\beta_\ell\,\Delta W_{j,:}^{(\ell)}, & j\in\mathcal{C}_\ell \\
\Delta W_{j,:}^{(\ell)}, & j\notin\mathcal{C}_\ell
\end{cases}
$$

Final update:

$$
W^{(\ell)} \leftarrow W^{(\ell)} + \Delta W^{(\ell)}
$$

---

## 6. Why Columnar ED Learns (Propositions)

### Proposition 1: Correct-Class Selectivity

$$
A_{c,+}^{(o)} \propto \mathbf{1}[c=y]
$$

Concentrates reinforcement on correct-class column system and reduces multi-class interference.

### Proposition 2: Closure of Local Plasticity

Each layer update is expressed as:

$$
\Delta W \propto
(\text{local amine})\times(\text{local saturation})\times(\text{previous-layer activity})
$$

So each layer is independently defined without chain rule.

### Proposition 3: Structured Regularization

Membership, rank LUT, and $\beta_\ell$ control:

- which neurons learn
- how strongly they learn

This geometric and hierarchical control balances representation specialization and stability, enabling practical accuracy.

---

## Implementation Mapping (Equations <-> Functions/Lines)

This section maps the equations above to concrete implementation locations.

| Equation / Concept | Math (summary) | Implementation | Key implementation point | Observable metric in logs |
|---|---|---|---|---|
| Hidden forward pass | $\mathbf{a}^{(\ell)}=W^{(\ell)}\mathbf{z}^{(\ell-1)},\ \mathbf{z}^{(\ell)}=\phi(\mathbf{a}^{(\ell)})$ | [modules/ed_network.py](../../modules/ed_network.py#L348), [modules/ed_network.py](../../modules/ed_network.py#L349) | `forward()` uses `np.dot` then `tanh_activation` | Layer activation heatmaps, activation range |
| Output forward pass | $\mathbf{p}=\mathrm{softmax}(W^{(o)}\mathbf{z}^{(L)})$ | [modules/ed_network.py](../../modules/ed_network.py#L399), [modules/activation_functions.py](../../modules/activation_functions.py#L24) | Linear map then softmax | Class-wise test accuracy, winner frequency |
| Output error | $\mathbf{e}^{(o)}=\mathbf{t}-\mathbf{p}$ | [modules/ed_network.py](../../modules/ed_network.py#L579) | difference between one-hot and predicted prob | early misclassification trend |
| Output saturation | $\mathbf{s}^{(o)}=|\mathbf{p}|\odot(1-|\mathbf{p}|)$ | [modules/ed_network.py](../../modules/ed_network.py#L582) | ED saturation term at output | probability saturation ratio |
| Output update | $\Delta W^{(o)}=\eta_o(\mathbf{e}^{(o)}\odot\mathbf{s}^{(o)})\mathbf{z}^{(L)\top}$ | [modules/ed_network.py](../../modules/ed_network.py#L585) | direct outer-product update | convergence speed |
| Correct-only amine | $A_{c,+}^{(o)}=\alpha_0(1-p_y)\mathbf{1}[c=y]$ | [modules/ed_network.py](../../modules/ed_network.py#L595), [modules/ed_network.py](../../modules/ed_network.py#L605) | amine only for correct class positive channel | pure-ED vs control comparison |
| Layer diffusion coeff | $d_\ell\in\{u_1,u_2\}$ | [modules/ed_network.py](../../modules/ed_network.py#L615) | switch by last vs earlier layers | layer-wise heatmap differences |
| Column membership | $M_{c,j}^{(\ell)}\in\{0,1\}$ | [modules/column_structure.py](../../modules/column_structure.py#L16), [modules/ed_network.py](../../modules/ed_network.py#L645) | mask for class-specific learning | class assignment diagnostics |
| Rank LUT weight | $\lambda_{c,j}^{(\ell)}=g(r_{c,j}^{(\ell)})$ | [modules/ed_network.py](../../modules/ed_network.py#L655) | rank and `_learning_weight_lut` weighting | best accuracy under LUT settings |
| NC branch | $\lambda^{NC}_{c,j,\ell} = 0$ | [modules/ed_network.py](../../modules/ed_network.py#L658) | non-column neurons do not learn (frozen random weights) | — |
| Hidden saturation | $q_j^{(\ell)}=\max(|z_j^{(\ell)}|(1-|z_j^{(\ell)}|),\varepsilon)$ | [modules/ed_network.py](../../modules/ed_network.py#L679) | minimum floor `1e-3` for stability | update stagnation / saturation |
| Hidden local signal | $u_j^{(\ell)}=\eta_\ell\sum H\,q$ | [modules/ed_network.py](../../modules/ed_network.py#L679), [modules/ed_network.py](../../modules/ed_network.py#L693) | class/sign summed signal, outer-product update | per-epoch gain, update statistics |
| Gradient clipping | $\Delta W\leftarrow\mathrm{Clip}(\Delta W)$ | [modules/ed_network.py](../../modules/ed_network.py#L703) | row-norm clip by `gradient_clip` | stability vs gap |
| Column LR factor | $\Delta W_{j,:}\leftarrow\beta_\ell\Delta W_{j,:}$ | [modules/ed_network.py](../../modules/ed_network.py#L716) | suppress only column rows using `column_lr_factors` | overfitting suppression / class balance |
| Weight application | $W^{(\ell)}\leftarrow W^{(\ell)}+\Delta W^{(\ell)}$ | [modules/ed_network.py](../../modules/ed_network.py#L744), [modules/ed_network.py](../../modules/ed_network.py#L755) | sequential additive updates | train/test trajectory |

Notes:

- Line numbers may shift as code changes.
- The table emphasizes local-update structure without chain rule.

---

## Minimal Validation Protocol (Three Experiments)

This section defines a minimum set of experiments to validate Propositions 1-3.

### Common Fixed Setup

- Dataset: MNIST (Fashion-MNIST as optional follow-up)
- Seed: 42 (primary)
- Train/Test: 10000/10000
- Hidden structure: use current high-accuracy setup first
- Epochs: 10 (extend to 20 only if needed)
- Metrics: best_test, final_test, best_epoch, class-wise accuracy, winner frequency

Note:

- Columnar ED accuracy is sensitive to training set size.
- Comparisons must keep data size fixed.

---

### Experiment 1: Correct-Class Selectivity (Proposition 1)

Goal:
Validate whether generating amine only for the correct class suppresses multi-class interference and improves/maintains accuracy.

Conditions:

1. Pure ED (current): correct-class-only amine generation
2. Control: allow learning signal on wrong-class side (reproduce prior implementation where possible)

Fixed:

- Same architecture, lr, init_scales, column_lr_factors, gradient_clip
- Fixed seed (42 first, optional multi-seed follow-up)

Metrics:

- Global: best_test, final_test, best-final gap
- Local: class-wise accuracy (especially historically fragile classes)
- Auxiliary: winner selection bias

Decision criterion:

- Pure ED shows best_test not lower than control and reduces class-wise collapse.

Command templates:

```bash
# Exp1-A: Pure ED (current baseline)
python columnar_ed_ann_v048.py \
	--dataset mnist --train 5000 --test 5000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	--output_lr 0.15 --non_column_lr 0.15,0.15 --column_lr 0.00075,0.00045 \
	> logs/exp1_pure_ed.log 2>&1

# Exp1-B: Control (branch-dependent option)
python columnar_ed_ann_v048.py \
	--dataset mnist --train 5000 --test 5000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	--output_lr 0.15 --non_column_lr 0.15,0.15 --column_lr 0.00075,0.00045 \
	<control-option> \
	> logs/exp1_control.log 2>&1
```

---

### Experiment 2: Closure of Local Plasticity (Proposition 2)

Goal:
Verify stable convergence under local updates without chain rule.

Conditions:

1. Baseline: current setup (with saturation term, with clipping)
2. Variant A: weaken/disable gradient clipping
3. Variant B: change saturation epsilon floor

Fixed:

- Keep all other hyperparameters identical
- Same seed

Metrics:

- Convergence speed: best_epoch
- Stability: best-final gap, late-epoch oscillation amplitude
- Failures: abrupt drops, NaN/Inf, severe class bias

Decision criterion:

- Baseline is most stable (smallest gap) and achieves best_test not lower than variants.

---

### Experiment 3: Structured Regularization (Proposition 3)

Goal:
Verify whether membership + rank LUT + column LR factors improve interference control and generalization.

Ablation sequence:

1. Full: current (membership + LUT + column_lr_factors)
2. Ablation-1: weaken/disable column_lr_factors
3. Ablation-2: reduce LUT contrast (e.g., equal mode if available)
4. Ablation-3: weaken column structure effect (`column_neurons` / participation)

Fixed:

- lr, init_scales, gradient_clip, epochs, seed

Metrics:

- Generalization: best_test, final_test, best-final gap
- Balance: standard deviation of class-wise accuracy
- Dynamics: winner frequency bias, column/non-column contribution skew

Decision criterion:

- Full condition simultaneously satisfies:
  - best_test not lower than ablations
  - smaller best-final gap
  - smaller class variance

---

## Post-Run Aggregation Templates

### 1) Summary CSV (best/final/gap)

```bash
cd /home/yoichi/develop/ai/column_ed_snn
mkdir -p tmp
out="tmp/protocol_summary_$(date +%Y%m%d_%H%M%S).csv"
echo "label,best_test,best_epoch,final_test,final_epoch,gap,log" > "$out"
# ... (same parsing flow as Japanese document)
```

### 2) Class-wise CSV (best epoch vs final epoch)

```bash
cd /home/yoichi/develop/ai/column_ed_snn
mkdir -p tmp
class_out="tmp/protocol_class_compare_$(date +%Y%m%d_%H%M%S).csv"
echo "label,best_epoch,best_C2,best_C6,best_C7,best_C9,best_avg,final_epoch,final_C2,final_C6,final_C7,final_C9,final_avg,log" > "$class_out"
# ... (same parsing flow as Japanese document)
```

Notes:

- The class-column extraction assumes the current log format.
- If log columns change, update `awk` column indices accordingly.

---

## Wrapper Script Template (Run 3 experiments + aggregate)

Purpose:
Automate Exp1-3 execution and produce summary/class CSV outputs in one script.

Usage:

1. Save as `scripts/run_protocol_3experiments.sh`
2. `chmod +x scripts/run_protocol_3experiments.sh`
3. `bash scripts/run_protocol_3experiments.sh`

(Use the same script body as in the Japanese version, replacing labels/comments with English if needed.)

---

## Recommended Execution Order

1. Experiment 1 (signal direction validity)
2. Experiment 2 (update-rule stability)
3. Experiment 3 (structured regularization contribution)

This order provides the clearest causal separation:
signal design -> update stability -> structural effect.

---

## Remarks

This formalization is an implementation-consistent model that prioritizes operation-level correspondence with code.

To improve paper readiness:

1. Add lemmas for each proposition (monotonicity, boundedness, near-fixed-point behavior)
2. Unify notation for NC branches (nearest assignment, spatial diffusion)
3. Tighten mapping between equations and convergence observables (best/final gap, class-wise contribution)
