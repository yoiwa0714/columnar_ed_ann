# ED学習メカニズム（Mermaid, 1:1コードアンカー）— 実験版

このドキュメントは、実装パスを実際のシンボル名とファイル/行番号アンカーに対応付けたものです。
対象ファイル:
- `columnar_ed_ann_experiment.py`（実験版）
- `modules_experiment/ed_network.py`
- `modules_experiment/column_structure.py`
- `modules_experiment/neuron_structure.py`
- `modules_experiment/activation_functions.py`
- `modules_experiment/gabor_features.py`
- `modules_experiment/visualization_manager.py`

> **注記**: メイン版（`columnar_ed_ann.py` + `modules/`）のアンカーは、[ed_learning_mechanism_anchors.md](ed_learning_mechanism_anchors.md) を参照してください。

注記:
- 行番号アンカーは、現在の実験版実装を基準にしています。
- コード更新により行番号が変動する可能性があります。

## 1. End-to-end実行パス

```mermaid
flowchart TD
    A["Entry point<br/>columnar_ed_ann_experiment.py:446<br/>def main()"] --> C["CLI parse<br/>columnar_ed_ann_experiment.py:29<br/>def parse_args()"]

    C --> C1["arg --lr<br/>columnar_ed_ann_experiment.py:72"]
    C --> C2["arg --column_neurons<br/>columnar_ed_ann_experiment.py:113"]
    C --> C3["arg --column_lr_factors<br/>columnar_ed_ann_experiment.py:129"]
    C --> C4["arg --init_scales<br/>columnar_ed_ann_experiment.py:225"]
    C --> C5["arg --gabor_features<br/>columnar_ed_ann_experiment.py:305"]

    C --> D["HyperParams load<br/>columnar_ed_ann_experiment.py:508,523<br/>hp = HyperParams(); get_config(n_layers)"]
    D --> E["Build network<br/>columnar_ed_ann_experiment.py:1009<br/>RefinedDistributionEDNetwork(...)"]

    E --> F["Train loop<br/>columnar_ed_ann_experiment.py:1281-1285<br/>network.train_epoch(...)"]
    E --> G["Eval loop<br/>columnar_ed_ann_experiment.py:1291<br/>network.evaluate_parallel(...)"]

    F --> H["train_epoch<br/>modules_experiment/ed_network.py:1885"]
    H --> I["train_one_sample<br/>modules_experiment/ed_network.py:1700"]
    I --> J["forward(x)<br/>modules_experiment/ed_network.py:927"]
    I --> K["update_weights(...)<br/>modules_experiment/ed_network.py:1530"]
    K --> L["_compute_gradients(...)<br/>modules_experiment/ed_network.py:1256"]

    J --> J1["x_paired = create_ei_pairs(x)<br/>modules_experiment/ed_network.py:944<br/>modules_experiment/neuron_structure.py:32"]
    J --> J2["z_hidden = tanh_activation(a_hidden)<br/>modules_experiment/ed_network.py:948<br/>modules_experiment/activation_functions.py:46"]
    J --> J3["z_output = softmax(a_output)<br/>modules_experiment/ed_network.py:954<br/>modules_experiment/activation_functions.py:63"]

    L --> L1["saturation_output<br/>modules_experiment/ed_network.py:1288"]
    L --> L2["error_correct = 1.0 - z_output[y_true]<br/>modules_experiment/ed_network.py:1309"]
    L --> L3["amine_diffused = amine_concentration_output * diffusion_coef<br/>modules_experiment/ed_network.py:1337"]
    L --> L4["learning_weights (rank + LUT + membership mask)<br/>modules_experiment/ed_network.py:1373,1383-1437"]
    L --> L5["layer_lr = self.layer_lrs[layer_idx]<br/>modules_experiment/ed_network.py:1514"]
    L --> L6["gradient clipping<br/>modules_experiment/ed_network.py:1502-1507"]
    L --> L7["column gradient suppression<br/>modules_experiment/ed_network.py:1514"]

    K --> K1["output_lr_factor = self.column_lr_factors[-1]<br/>modules_experiment/ed_network.py:1572"]

    G --> G1["evaluate_parallel<br/>modules_experiment/ed_network.py:2305"]
```

## 2. 機能別セクション: 順伝播と活性化フロー

```mermaid
flowchart LR
    A["forward(x)<br/>modules_experiment/ed_network.py:927"] --> B["create_ei_pairs(x)<br/>modules_experiment/ed_network.py:944<br/>modules_experiment/neuron_structure.py:32"]
    B --> C["a_hidden = W*x + b<br/>modules_experiment/ed_network.py:946-947"]
    C --> D["tanh_activation(a_hidden)<br/>modules_experiment/ed_network.py:948<br/>modules_experiment/activation_functions.py:46"]
    D --> E["a_output = W_out*z + b_out<br/>modules_experiment/ed_network.py:952-953"]
    E --> F["softmax(a_output)<br/>modules_experiment/ed_network.py:954<br/>modules_experiment/activation_functions.py:63"]
```

## 3. 機能別セクション: ED勾配コア（連鎖律ベース逆伝播なし）

```mermaid
flowchart LR
    A["_compute_gradients(...)<br/>modules_experiment/ed_network.py:1256"] --> B["saturation_output = abs(z)*(1-abs(z))<br/>modules_experiment/ed_network.py:1288"]
    A --> C["error_correct = 1 - z_output[y_true]<br/>modules_experiment/ed_network.py:1309"]
    C --> D["amine concentration setup<br/>modules_experiment/ed_network.py:1305-1311"]
    D --> E["amine_diffused = amine * diffusion_coef<br/>modules_experiment/ed_network.py:1337"]
    E --> F["rank of active neurons<br/>modules_experiment/ed_network.py:1348-1373"]
    F --> G["LUT learning_weights<br/>modules_experiment/ed_network.py:1373"]
    G --> H["membership/NC masks via np.where<br/>modules_experiment/ed_network.py:1383-1437"]
    H --> I["learning_signals_3d<br/>modules_experiment/ed_network.py:1464-1467"]
    I --> J["delta_w layer update matrix<br/>modules_experiment/ed_network.py:1492-1493"]
    J --> K["gradient clip<br/>modules_experiment/ed_network.py:1502-1507"]
    K --> L["column_lr_factors suppression<br/>modules_experiment/ed_network.py:1514"]
```

## 4. 機能別セクション: コラム構造とクラス特異的抑制

```mermaid
flowchart LR
    A["create_column_membership(...)<br/>modules_experiment/column_structure.py:44"] --> B["membership bool map [class, neuron]<br/>modules_experiment/column_structure.py:73-142"]
    B --> C["network stores membership per layer<br/>modules_experiment/ed_network.py:63"]
    C --> D["active_membership for active classes<br/>modules_experiment/ed_network.py:1362"]
    D --> E["rank/LUT masked by membership<br/>modules_experiment/ed_network.py:1383-1437"]
    E --> F["column_lr_factors suppress gradients<br/>modules_experiment/ed_network.py:1514"]
    F --> G["output side factor suppression<br/>modules_experiment/ed_network.py:1572"]
```

## 5. 機能別セクション: Gabor前処理パス

```mermaid
flowchart LR
    A["--gabor_features flag<br/>columnar_ed_ann_experiment.py:305"] --> B["GaborFeatureExtractor init<br/>columnar_ed_ann_experiment.py:921,943<br/>modules_experiment/gabor_features.py:24"]
    B --> C["x_train = extractor.transform(x_train)<br/>modules_experiment/gabor_features.py:188"]
    B --> D["x_test = extractor.transform_test(x_test)<br/>modules_experiment/gabor_features.py:213"]
    C --> E["network input dim switches 784 -> gabor_dim<br/>columnar_ed_ann_experiment.py:967"]
```

## 6. 機能別セクション: 可視化とヒートマップウィンドウ

```mermaid
flowchart LR
    A["VisualizationManager(..., window_scale)<br/>columnar_ed_ann_experiment.py:1099-1105<br/>modules_experiment/visualization_manager.py:158"] --> B["update_heatmap callback in train loop<br/>columnar_ed_ann_experiment.py:1183,1468<br/>modules_experiment/visualization_manager.py:476"]
    B --> C["Top row: info + raw image + Gabor tiles<br/>modules_experiment/visualization_manager.py:491-571"]
    B --> D["Bottom row: input/hidden/output layers<br/>modules_experiment/visualization_manager.py:606-685"]
    D --> E["Input layer uses raw image when available<br/>modules_experiment/visualization_manager.py:609-614,623-652"]
    E --> F["Input layer label uses raw dimension (e.g., 784)<br/>modules_experiment/visualization_manager.py:609-614"]
```

## 7. 機能別セクション: 不正解学習サンプルビューア

```mermaid
flowchart LR
    A["--show_train_errors option<br/>columnar_ed_ann_experiment.py:192-198"] --> B["collect_errors in final epoch train call<br/>columnar_ed_ann_experiment.py:1273-1285<br/>modules_experiment/ed_network.py:1885"]
    B --> C["show_train_errors(...) call<br/>columnar_ed_ann_experiment.py:1639"]
    C --> D["scrollable error viewer<br/>modules_experiment/visualization_manager.py:851"]
    D --> E["max_errors_per_class default=20<br/>columnar_ed_ann_experiment.py:195<br/>modules_experiment/visualization_manager.py:851"]
```
