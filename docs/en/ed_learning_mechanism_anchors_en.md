# ED Learning Mechanism (Mermaid, 1:1 Code Anchors)

This document maps implementation paths to exact symbol names and file/line anchors in:
- `columnar_ed_ann.py` (published/remote main implementation)
- `modules/ed_network.py`
- `modules/column_structure.py`
- `modules/neuron_structure.py`
- `modules/activation_functions.py`
- `modules/gabor_features.py`
- `modules/visualization_manager.py`

Note:
- Line anchors for `columnar_ed_ann.py` are based on the current published/remote repository version.
- If `columnar_ed_ann.py` changes later, anchors should be re-synced to the final confirmed version.

## 1. End-to-end execution path

```mermaid
flowchart TD
    A["Entry point<br/>columnar_ed_ann.py:446<br/>def main()"] --> C["CLI parse<br/>columnar_ed_ann.py:29<br/>def parse_args()"]

    C --> C1["arg --lr<br/>columnar_ed_ann.py:72"]
    C --> C2["arg --column_neurons<br/>columnar_ed_ann.py:113"]
    C --> C3["arg --column_lr_factors<br/>columnar_ed_ann.py:129"]
    C --> C4["arg --init_scales<br/>columnar_ed_ann.py:225"]
    C --> C5["arg --gabor_features<br/>columnar_ed_ann.py:305"]

    C --> D["HyperParams load<br/>columnar_ed_ann.py:508,523<br/>hp = HyperParams(); get_config(n_layers)"]
    D --> E["Build network<br/>columnar_ed_ann.py:1009<br/>RefinedDistributionEDNetwork(...)"]

    E --> F["Train loop<br/>columnar_ed_ann.py:1281-1285<br/>network.train_epoch(...)"]
    E --> G["Eval loop<br/>columnar_ed_ann.py:1291<br/>network.evaluate_parallel(...)"]

    F --> H["train_epoch<br/>modules/ed_network.py:1925"]
    H --> I["train_one_sample<br/>modules/ed_network.py:1740"]
    I --> J["forward(x)<br/>modules/ed_network.py:967"]
    I --> K["update_weights(...)<br/>modules/ed_network.py:1570"]
    K --> L["_compute_gradients(...)<br/>modules/ed_network.py:1296"]

    J --> J1["x_paired = create_ei_pairs(x)<br/>modules/ed_network.py:984<br/>modules/neuron_structure.py:32"]
    J --> J2["z_hidden = tanh_activation(a_hidden)<br/>modules/ed_network.py:998<br/>modules/activation_functions.py:46"]
    J --> J3["z_output = softmax(a_output)<br/>modules/ed_network.py:1097<br/>modules/activation_functions.py:63"]

    L --> L1["saturation_output<br/>modules/ed_network.py:1328"]
    L --> L2["error_correct = 1.0 - z_output[y_true]<br/>modules/ed_network.py:1349"]
    L --> L3["amine_diffused = amine_concentration_output * diffusion_coef<br/>modules/ed_network.py:1377"]
    L --> L4["learning_weights (rank + LUT + membership mask)<br/>modules/ed_network.py:1413,1433-1478"]
    L --> L5["layer_lr = self.layer_lrs[layer_idx]<br/>modules/ed_network.py:1511"]
    L --> L6["gradient clipping<br/>modules/ed_network.py:1541"]
    L --> L7["column gradient suppression<br/>modules/ed_network.py:1554-1564"]

    K --> K1["output_lr_factor = self.column_lr_factors[-1]<br/>modules/ed_network.py:1590-1602"]

    G --> G1["evaluate_parallel<br/>modules/ed_network.py:2345"]
```

## 2. Feature section: Forward and activation flow

```mermaid
flowchart LR
    A["forward(x)<br/>modules/ed_network.py:967"] --> B["create_ei_pairs(x)<br/>modules/ed_network.py:984<br/>modules/neuron_structure.py:32"]
    B --> C["a_hidden = W*x + b<br/>modules/ed_network.py:994-997"]
    C --> D["tanh_activation(a_hidden)<br/>modules/ed_network.py:998<br/>modules/activation_functions.py:46"]
    D --> E["a_output = W_out*z + b_out<br/>modules/ed_network.py:1093-1096"]
    E --> F["softmax(a_output)<br/>modules/ed_network.py:1097<br/>modules/activation_functions.py:63"]
```

## 3. Feature section: ED gradient core (no backprop chain rule)

```mermaid
flowchart LR
    A["_compute_gradients(...)<br/>modules/ed_network.py:1296"] --> B["saturation_output = abs(z)*(1-abs(z))<br/>modules/ed_network.py:1328"]
    A --> C["error_correct = 1 - z_output[y_true]<br/>modules/ed_network.py:1349"]
    C --> D["amine concentration setup<br/>modules/ed_network.py:1356-1369"]
    D --> E["amine_diffused = amine * u_coef<br/>modules/ed_network.py:1377"]
    E --> F["rank of active neurons<br/>modules/ed_network.py:1393-1410"]
    F --> G["LUT learning_weights<br/>modules/ed_network.py:1413"]
    G --> H["membership/NC masks via np.where<br/>modules/ed_network.py:1433-1478"]
    H --> I["learning_signals_3d<br/>modules/ed_network.py:1486-1490"]
    I --> J["delta_w layer update matrix<br/>modules/ed_network.py:1497-1506"]
    J --> K["layer_lr scale<br/>modules/ed_network.py:1511"]
    K --> L["gradient clip<br/>modules/ed_network.py:1541-1546"]
    L --> M["column_lr_factors suppression<br/>modules/ed_network.py:1554-1564"]
```

## 4. Feature section: Column structure and class-specific suppression

```mermaid
flowchart LR
    A["create_column_membership(...)<br/>modules/column_structure.py:44"] --> B["membership bool map [class, neuron]<br/>modules/column_structure.py:73-142"]
    B --> C["network stores membership per layer<br/>modules/ed_network.py:380,421,429"]
    C --> D["active_membership for active classes<br/>modules/ed_network.py:1401-1405"]
    D --> E["rank/LUT masked by membership<br/>modules/ed_network.py:1433-1478"]
    E --> F["column_lr_factors suppress gradients<br/>modules/ed_network.py:1554-1564"]
    F --> G["output side factor suppression<br/>modules/ed_network.py:1590-1602"]
```

## 5. Feature section: Gabor preprocessing path

```mermaid
flowchart LR
    A["--gabor_features flag<br/>columnar_ed_ann.py:305"] --> B["GaborFeatureExtractor init<br/>columnar_ed_ann.py:921,943<br/>modules/gabor_features.py:24"]
    B --> C["x_train = extractor.transform(x_train)<br/>columnar_ed_ann.py:963<br/>modules/gabor_features.py:188"]
    B --> D["x_test = extractor.transform_test(x_test)<br/>columnar_ed_ann.py:965<br/>modules/gabor_features.py:213"]
    C --> E["network input dim switches 784 -> 882<br/>columnar_ed_ann.py:967"]
```

## 6. Feature section: Visualization and heatmap window

```mermaid
flowchart LR
    A["VisualizationManager(..., window_scale)<br/>columnar_ed_ann.py:1099-1105<br/>modules/visualization_manager.py:169"] --> B["update_heatmap callback in train loop<br/>columnar_ed_ann.py:1183,1468<br/>modules/visualization_manager.py:378"]
    B --> C["Top row: info + raw image + Gabor tiles<br/>modules/visualization_manager.py:491-571"]
    B --> D["Bottom row: input/hidden/output layers<br/>modules/visualization_manager.py:606-685"]
    D --> E["Input layer uses raw image when available<br/>modules/visualization_manager.py:609-614,623-652"]
    E --> F["Input layer label uses raw dimension (e.g., 784)<br/>modules/visualization_manager.py:609-614"]
```

## 7. Feature section: Misclassified training samples viewer

```mermaid
flowchart LR
    A["--show_train_errors option<br/>columnar_ed_ann.py:192-198"] --> B["collect_errors in final epoch train call<br/>columnar_ed_ann.py:1273-1285<br/>modules/ed_network.py:1925"]
    B --> C["show_train_errors(...) call<br/>columnar_ed_ann.py:1639"]
    C --> D["scrollable error viewer<br/>modules/visualization_manager.py:743"]
    D --> E["max_errors_per_class default=20<br/>columnar_ed_ann.py:195<br/>modules/visualization_manager.py:743"]
```
