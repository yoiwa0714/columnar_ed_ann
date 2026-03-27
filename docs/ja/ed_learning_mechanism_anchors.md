# ED学習メカニズム（Mermaid, 1:1コードアンカー）— メイン版

このドキュメントは、実装パスを実際のシンボル名とファイル/行番号アンカーに対応付けたものです。
対象ファイル:
- `columnar_ed_ann.py`（メイン版）
- `modules/ed_network.py`
- `modules/column_structure.py`
- `modules/activation_functions.py`
- `modules/gabor_features.py`

> **注記**: 実験版（`columnar_ed_ann_experiment.py` + `modules_experiment/`）のアンカーは、[ed_learning_mechanism_anchors_experiment.md](ed_learning_mechanism_anchors_experiment.md) を参照してください。

注記:
- 行番号アンカーは、現在のメイン版実装を基準にしています。
- コード更新により行番号が変動する可能性があります。

## 1. End-to-end実行パス

```mermaid
flowchart TD
    A["Entry point<br/>columnar_ed_ann.py:91<br/>def main()"] --> C["CLI parse<br/>columnar_ed_ann.py:47<br/>def parse_args()"]

    C --> C1["arg --hidden<br/>columnar_ed_ann.py:58"]
    C --> C2["arg --epochs<br/>columnar_ed_ann.py:66"]
    C --> C3["arg --no_gabor<br/>columnar_ed_ann.py:72"]

    C --> D["HyperParams load<br/>columnar_ed_ann.py:113<br/>hp = HyperParams(); get_config(n_layers)"]
    D --> E["Build network<br/>columnar_ed_ann.py:204<br/>SimpleColumnEDNetwork(...)"]

    E --> F["Train loop<br/>columnar_ed_ann.py:318<br/>network.train_epoch(...)"]
    E --> G["Eval loop<br/>columnar_ed_ann.py:323<br/>network.evaluate_parallel(...)"]

    F --> H["train_epoch<br/>modules/ed_network.py:506"]
    H --> I["train_one_sample<br/>modules/ed_network.py:489"]
    I --> J["forward(x)<br/>modules/ed_network.py:226"]
    I --> K["update_weights(...)<br/>modules/ed_network.py:459"]
    K --> L["_compute_gradients(...)<br/>modules/ed_network.py:289"]

    J --> J1["x_paired = np.concatenate([x, x])<br/>modules/ed_network.py:241"]
    J --> J2["z_hidden = tanh_activation(a_hidden)<br/>modules/ed_network.py:248<br/>modules/activation_functions.py:14"]
    J --> J3["z_output = softmax(a_output)<br/>modules/ed_network.py:254<br/>modules/activation_functions.py:24"]

    L --> L1["saturation_output<br/>modules/ed_network.py:334"]
    L --> L2["error_correct = 1.0 - z_output[y_true]<br/>modules/ed_network.py:347"]
    L --> L3["amine_diffused = amine_concentration * diffusion_coef<br/>modules/ed_network.py:363"]
    L --> L4["learning_weights (rank + LUT + membership mask)<br/>modules/ed_network.py:377-389"]
    L --> L5["layer_lr = self.layer_lrs[layer_idx]<br/>modules/ed_network.py:92"]
    L --> L6["gradient clipping<br/>modules/ed_network.py:433"]
    L --> L7["column gradient suppression<br/>modules/ed_network.py:443"]

    G --> G1["evaluate_parallel<br/>modules/ed_network.py:549"]
```

## 2. 機能別セクション: 順伝播と活性化フロー

```mermaid
flowchart LR
    A["forward(x)<br/>modules/ed_network.py:226"] --> B["x_paired = np.concatenate([x, x])<br/>modules/ed_network.py:241"]
    B --> C["a_hidden = W*z_current<br/>modules/ed_network.py:246"]
    C --> D["tanh_activation(a_hidden)<br/>modules/ed_network.py:248<br/>modules/activation_functions.py:14"]
    D --> E["a_output = W_out*z_hidden<br/>modules/ed_network.py:252"]
    E --> F["softmax(a_output)<br/>modules/ed_network.py:254<br/>modules/activation_functions.py:24"]
```

## 3. 機能別セクション: ED勾配コア（連鎖律ベース逆伝播なし）

```mermaid
flowchart LR
    A["_compute_gradients(...)<br/>modules/ed_network.py:289"] --> B["saturation_output = abs(z)*(1-abs(z))<br/>modules/ed_network.py:334"]
    A --> C["error_correct = 1 - z_output[y_true]<br/>modules/ed_network.py:347"]
    C --> D["amine concentration setup<br/>modules/ed_network.py:346-349"]
    D --> E["amine_diffused = amine * diffusion_coef<br/>modules/ed_network.py:363"]
    E --> F["rank of active neurons<br/>modules/ed_network.py:377-386"]
    F --> G["LUT learning_weights<br/>modules/ed_network.py:386"]
    G --> H["membership mask via np.where<br/>modules/ed_network.py:389"]
    H --> I["learning_signals_3d<br/>modules/ed_network.py:414"]
    I --> J["delta_w layer update matrix<br/>modules/ed_network.py:424"]
    J --> K["gradient clip<br/>modules/ed_network.py:433-439"]
    K --> L["column_lr_factors suppression<br/>modules/ed_network.py:443"]
```

## 4. 機能別セクション: コラム構造とクラス特異的抑制

```mermaid
flowchart LR
    A["create_column_membership(...)<br/>modules/column_structure.py:16"] --> B["membership bool map [class, neuron]<br/>modules/column_structure.py:16-123"]
    B --> C["network stores membership per layer<br/>modules/ed_network.py:128"]
    C --> D["active_membership for active classes<br/>modules/ed_network.py:377"]
    D --> E["rank/LUT masked by membership<br/>modules/ed_network.py:386-389"]
    E --> F["column_lr_factors suppress gradients<br/>modules/ed_network.py:443"]
```

## 5. 機能別セクション: Gabor前処理パス

```mermaid
flowchart LR
    A["--no_gabor flag (default: Gabor ON)<br/>columnar_ed_ann.py:72"] --> B["GaborFeatureExtractor init<br/>columnar_ed_ann.py:159,177<br/>modules/gabor_features.py:19"]
    B --> C["x_train = extractor.transform(x_train)<br/>modules/gabor_features.py:138"]
    B --> D["x_test = extractor.transform(x_test)<br/>modules/gabor_features.py:138"]
    C --> E["network input dim switches 784 -> gabor_dim<br/>columnar_ed_ann.py:204"]
```

## 6. 機能別セクション: 可視化

```mermaid
flowchart LR
    A["--viz option (1-4)<br/>columnar_ed_ann.py:77"] --> B["VisualizationManager<br/>columnar_ed_ann.py:229<br/>(from modules_experiment.visualization_manager)"]
    B --> C["Heatmap callback in train loop<br/>columnar_ed_ann.py:318"]
```

注記: メイン版の可視化は `modules_experiment/visualization_manager.py` を利用します。
`--show_train_errors` 機能は実験版のみで利用可能です。
