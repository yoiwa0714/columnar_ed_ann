# ED Learning Mechanism (Mermaid, 1:1 Code Anchors)

This document maps implementation paths to exact symbol names and file/line anchors in:
- `columnar_ed_ann.py` (published version v1.2.0)
- `modules/ed_network.py`
- `modules/column_structure.py`
- `modules/activation_functions.py`
- `modules/gabor_features.py`

Note:
- Line anchors are based on the current published version (v1.2.0).
- Line numbers may shift as code changes.

## 1. End-to-end execution path

```mermaid
flowchart TD
    A["Entry point<br/>columnar_ed_ann.py:664<br/>def main()"] --> C["CLI parse<br/>columnar_ed_ann.py:166<br/>def parse_args()"]

    C --> C1["arg --hidden<br/>columnar_ed_ann.py:180"]
    C --> C2["arg --epochs<br/>columnar_ed_ann.py:188"]
    C --> C3["arg --no_gabor<br/>columnar_ed_ann.py:194"]

    C --> D["HyperParams load<br/>columnar_ed_ann.py:705<br/>hp = HyperParams(); get_config(n_layers)"]
    D --> E["Build network<br/>columnar_ed_ann.py:882<br/>SimpleColumnEDNetwork(...)"]

    E --> F["Train loop<br/>columnar_ed_ann.py:1083<br/>network.train_epoch(...)"]
    E --> G["Eval loop<br/>columnar_ed_ann.py:1088<br/>network.evaluate_parallel(...)"]

    F --> H["train_epoch<br/>modules/ed_network.py:991"]
    H --> I["train_one_sample<br/>modules/ed_network.py:974"]
    I --> J["forward(x)<br/>modules/ed_network.py:319"]
    I --> K["update_weights(...)<br/>modules/ed_network.py:732"]
    K --> L["_compute_gradients(...)<br/>modules/ed_network.py:537"]

    J --> J1["x_paired = np.concatenate([x, x])<br/>modules/ed_network.py:334"]
    J --> J2["z_hidden = tanh_activation(a_hidden)<br/>modules/ed_network.py:349<br/>modules/activation_functions.py:14"]
    J --> J3["z_output = softmax(a_output)<br/>modules/ed_network.py:400<br/>modules/activation_functions.py:24"]

    L --> L1["saturation_output<br/>modules/ed_network.py:582"]
    L --> L2["error_correct = 1.0 - z_output[y_true]<br/>modules/ed_network.py:595"]
    L --> L3["amine_diffused = amine_concentration * diffusion_coef<br/>modules/ed_network.py:615"]
    L --> L4["learning_weights (rank + LUT + membership mask)<br/>modules/ed_network.py:645-658"]
    L --> L5["layer_lr = self.layer_lrs[layer_idx]<br/>modules/ed_network.py:92"]
    L --> L6["gradient clipping<br/>modules/ed_network.py:703"]
    L --> L7["column gradient suppression<br/>modules/ed_network.py:716"]

    G --> G1["evaluate_parallel<br/>modules/ed_network.py:1041"]
```

## 2. Feature section: Forward and activation flow

```mermaid
flowchart LR
    A["forward(x)<br/>modules/ed_network.py:319"] --> B["x_paired = np.concatenate([x, x])<br/>modules/ed_network.py:334"]
    B --> C["a_hidden = W*z_current<br/>modules/ed_network.py:348"]
    C --> D["tanh_activation(a_hidden)<br/>modules/ed_network.py:349<br/>modules/activation_functions.py:14"]
    D --> E["a_output = W_out*z_hidden<br/>modules/ed_network.py:399"]
    E --> F["softmax(a_output)<br/>modules/ed_network.py:400<br/>modules/activation_functions.py:24"]
```

## 3. Feature section: ED gradient core (no backprop chain rule)

```mermaid
flowchart LR
    A["_compute_gradients(...)<br/>modules/ed_network.py:537"] --> B["saturation_output = abs(z)*(1-abs(z))<br/>modules/ed_network.py:582"]
    A --> C["error_correct = 1 - z_output[y_true]<br/>modules/ed_network.py:595"]
    C --> D["amine concentration setup<br/>modules/ed_network.py:595-605"]
    D --> E["amine_diffused = amine * diffusion_coef<br/>modules/ed_network.py:615"]
    E --> F["rank of active neurons<br/>modules/ed_network.py:645-655"]
    F --> G["LUT learning_weights<br/>modules/ed_network.py:655"]
    G --> H["membership mask via np.where<br/>modules/ed_network.py:658"]
    H --> I["learning_signals_3d<br/>modules/ed_network.py:684"]
    I --> J["delta_w layer update matrix<br/>modules/ed_network.py:694"]
    J --> K["gradient clip<br/>modules/ed_network.py:703-709"]
    K --> L["column_lr_factors suppression<br/>modules/ed_network.py:716"]
```

## 4. Feature section: Column structure and class-specific suppression

```mermaid
flowchart LR
    A["create_column_membership(...)<br/>modules/column_structure.py:16"] --> B["membership bool map [class, neuron]<br/>modules/column_structure.py:16-123"]
    B --> C["network stores membership per layer<br/>modules/ed_network.py:197"]
    C --> D["active_membership for active classes<br/>modules/ed_network.py:645"]
    D --> E["rank/LUT masked by membership<br/>modules/ed_network.py:655-658"]
    E --> F["column_lr_factors suppress gradients<br/>modules/ed_network.py:716"]
```

## 5. Feature section: Gabor preprocessing path

```mermaid
flowchart LR
    A["--no_gabor flag (default: Gabor ON)<br/>columnar_ed_ann.py:194"] --> B["GaborFeatureExtractor init<br/>columnar_ed_ann.py:776,827<br/>modules/gabor_features.py:19"]
    B --> C["x_train = extractor.transform(x_train)<br/>modules/gabor_features.py:138"]
    B --> D["x_test = extractor.transform(x_test)<br/>modules/gabor_features.py:138"]
    C --> E["network input dim switches 784 -> gabor_dim<br/>columnar_ed_ann.py:882"]
```

## 6. Feature section: Visualization

```mermaid
flowchart LR
    A["--viz option (1-4)<br/>columnar_ed_ann.py:241"] --> B["VisualizationManager<br/>columnar_ed_ann.py:951<br/>(from modules.visualization_manager)"]
    B --> C["Heatmap callback in train loop<br/>columnar_ed_ann.py:1083"]
```
