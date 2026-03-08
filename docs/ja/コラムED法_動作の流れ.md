# コラムED法 — 動作の流れとコード解説

本ドキュメントでは、コラムED法の動作の流れをブロックダイアグラムで示し、ED法の核となる関数のコードを注釈付きで解説します。

> **対象ファイル:** `columnar_ed_ann_simple.py`（メインスクリプト）、`modules_simple/ed_network.py`（EDネットワーク）、`modules_simple/column_structure.py`（コラム構造）

---

## 目次

- [1. 全体フロー](#1-全体フロー)
- [2. 1エポックの学習フロー](#2-1エポックの学習フロー)
- [3. 1サンプルの学習フロー（ED法の核心）](#3-1サンプルの学習フローed法の核心)
- [4. 核関数の詳細解説](#4-核関数の詳細解説)
  - [4.1 forward() — 順伝播](#41-forward--順伝播)
  - [4.2 出力層の勾配計算 — 飽和抑制項](#42-出力層の勾配計算--飽和抑制項)
  - [4.3 アミン濃度計算と拡散](#43-アミン濃度計算と拡散)
  - [4.4 隠れ層の重み更新](#44-隠れ層の重み更新)
  - [4.5 create_column_membership() — コラム構造](#45-create_column_membership--コラム構造)
- [5. 実装アンカー付き学習メカニズム（フル版）](#5-実装アンカー付き学習メカニズムフル版)

---

## 1. 全体フロー

メインスクリプト `columnar_ed_ann_simple.py` の処理の流れです。

```
┌─────────────────────────────────────────────────┐
│              コマンドライン引数解析                │
│  (隠れ層構成、データ数、エポック数 等)              │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│         YAML自動パラメータ読み込み                 │
│  (層数に応じた学習率、初期化スケール 等)            │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│              データセット読み込み                   │
│  (MNIST/Fashion-MNIST/カスタム)                    │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│       Gabor特徴抽出（デフォルトON、省略可能）       │
│  (V1野単純型細胞モデル、入力品質向上)               │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│          EDネットワーク構築・初期化                 │
│  ┌──────────────────────────────────────┐        │
│  │ ・コラム構造生成(ハニカム配置)         │        │
│  │ ・He初期化 × init_scales              │        │
│  │ ・Dale's Principle符号行列            │        │
│  │ ・非コラムニューロンのスパース化       │        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│          エポックループ（→ 詳細は次節）            │
│  ┌──────────────────────────────────────┐        │
│  │  各エポック:                          │        │
│  │   1. train_epoch() → 重み更新         │        │
│  │   2. evaluate_parallel() → 精度評価   │        │
│  │   3. ベスト精度の記録                  │        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│              結果サマリー表示                      │
│  (最終精度、ベスト精度、クラス別精度)               │
└─────────────────────────────────────────────────┘
```

---

## 2. 1エポックの学習フロー

`train_epoch()` メソッドの処理の流れです。ED法はオンライン学習（1サンプルずつ即座に重み更新）を行います。

```
┌─────────────────────────────────────────────────┐
│    n_samples個のサンプルに対してループ              │
│                                                   │
│    ┌───────────────────────────────────────┐      │
│    │  train_one_sample(x, y_true)          │      │
│    │                                       │      │
│    │  1. forward(x)         → 予測          │      │
│    │  2. cross_entropy_loss → 損失計算      │      │
│    │  3. update_weights()   → 重み更新      │      │
│    │     （→ 詳細は次節）                    │      │
│    └───────────────────────────────────────┘      │
│                                                   │
│    ※ 1サンプルごとに即座に重み更新（オンライン学習）│
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  evaluate_parallel() — 最終重みで訓練精度を再評価  │
│  （テスト精度と公平に比較できるようにするため）     │
└─────────────────────────────────────────────────┘
```

---

## 3. 1サンプルの学習フロー（ED法の核心）

`update_weights()` → `_compute_gradients()` の処理の流れです。これがED法の学習メカニズムの全体像です。

```
入力 x
  │
  ▼
┌─────────────────────────────────────────────────┐
│  E/Iペア化: x → [x, x]                          │
│  (Dale's Principleの基盤)                         │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  順伝播 forward()                                 │
│  ┌──────────────────────────────────────┐        │
│  │ 隠れ層: dot(W, z) → tanh(a) → z     │        │
│  │ 出力層: dot(W, z) → softmax(a) → p  │        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  出力層の重み更新                                  │
│  ┌──────────────────────────────────────┐        │
│  │  誤差      = 正解one-hot − 予測確率   │        │
│  │  飽和抑制項 = |z| × (1 − |z|)        │ ★核心  │
│  │  ΔW = lr × (誤差 × 飽和抑制項) ⊗ z_in │        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  アミン濃度計算                                    │
│  ┌──────────────────────────────────────┐        │
│  │ 正解クラスのみアミンを注入             │ ★重要  │
│  │ amine = (1 − 正解クラスの予測確率)     │        │
│  │         × 初期アミン濃度               │        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  隠れ層の重み更新（最終層→第1層の逆順ループ）     │
│                                                   │
│  各隠れ層について:                                │
│  ┌──────────────────────────────────────┐        │
│  │ 1. アミン拡散: amine × 拡散係数(u1/u2) │        │
│  │ 2. コラムメンバーシップで対象ニューロン │        │
│  │    を特定、活性値ランクで学習率決定    │ ★核心  │
│  │ 3. 飽和抑制項: |z| × (1 − |z|)       │        │
│  │ 4. ΔW = lr × amine × 飽和抑制項 ⊗ z_in│        │
│  │ 5. 勾配クリッピング                    │        │
│  └──────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  重み適用 + Dale's Principle符号強制（第1層のみ）  │
│  出力層の微弱正則化                                │
└─────────────────────────────────────────────────┘
```

**BP法（誤差逆伝播法）との決定的な違い:**

| | BP法 | ED法（本実装） |
|---|---|---|
| 誤差信号の伝達 | 微分の連鎖律で逆伝播 | アミン濃度として空間拡散 |
| 各層の更新 | 後段の勾配に依存 | **各層が独立に更新** |
| 更新量の決定 | 活性化関数の微分 | **飽和抑制項** `\|z\| × (1 − \|z\|)` |
| 学習対象 | 全ニューロン | **コラムニューロンのみ** |

---

## 4. 核関数の詳細解説

### 4.1 forward() — 順伝播

**ファイル:** `modules_simple/ed_network.py`

順伝播は比較的シンプルです。入力をE/Iペアに変換した後、各隠れ層ではtanh活性化、出力層ではsoftmaxで確率分布に変換します。

```python
def forward(self, x):
    # ★ Dale's Principle: 入力を [x, x] に複製
    # 重み行列の符号制約（興奮性/抑制性）と組み合わせて生物学的制約を実現
    x_paired = create_ei_pairs(x)

    z_hiddens = []
    z_current = x_paired

    for layer_idx in range(self.n_layers):
        # ★ 各層は単純な行列積 → tanh（微分は使わない）
        a_hidden = np.dot(self.w_hidden[layer_idx], z_current)
        z_hidden = tanh_activation(a_hidden)         # 出力範囲: -1 〜 +1
        z_hiddens.append(z_hidden)
        z_current = z_hidden

    # ★ 出力層: softmaxで確率化（順伝播のみに使用、学習信号には使わない）
    a_output = np.dot(self.w_output, z_hiddens[-1])
    z_output = softmax(a_output)                     # 合計 = 1.0

    return z_hiddens, z_output, x_paired
```

**ポイント:**
- `create_ei_pairs(x)` は入力を `[x, x]` に連結するだけの単純な処理ですが、第1層の重み行列に適用される符号制約（Dale's Principle）と組み合わさることで、前半が興奮性入力、後半が抑制性入力として機能します
- 各隠れ層のtanhは飽和特性（±1に近づくと変化しにくくなる）を持ち、これがED法の飽和抑制項と連携します
- softmaxは出力を確率として解釈するために使用されますが、BP法のようにsoftmaxの微分を逆伝播に使用することはありません

<details>
<summary>📄 forward() の全コード</summary>

```python
def forward(self, x):
    """
    順伝播（多クラス分類版）

    隠れ層: tanh活性化（双方向性、飽和特性）
    出力層: SoftMax（確率分布化）

    Args:
        x: 入力データ（shape: [n_input]）

    Returns:
        z_hiddens: 各隠れ層の出力のリスト
        z_output: 出力層の確率分布（SoftMax、合計=1.0）
        x_paired: 入力ペア（興奮性+抑制性）
    """
    # 入力ペア構造: x → [x, x]（Dale's Principleの符号行列で興奮性・抑制性を実現）
    x_paired = create_ei_pairs(x)

    z_hiddens = []
    z_current = x_paired

    for layer_idx in range(self.n_layers):
        a_hidden = np.dot(self.w_hidden[layer_idx], z_current)
        z_hidden = tanh_activation(a_hidden)
        z_hiddens.append(z_hidden)
        z_current = z_hidden

    # 出力層: SoftMax活性化
    a_output = np.dot(self.w_output, z_hiddens[-1])
    z_output = softmax(a_output)

    return z_hiddens, z_output, x_paired
```

</details>

---

### 4.2 出力層の勾配計算 — 飽和抑制項

**ファイル:** `modules_simple/ed_network.py` — `_compute_gradients()` の前半部分

出力層の重み更新は、ED法の飽和抑制項を使います。これはBP法のsoftmax微分（クロスエントロピー微分）とは本質的に異なります。

```python
# --- 出力層の勾配計算 ---

# 正解のone-hotベクトルと予測確率の差が誤差
target_probs = np.zeros(self.n_output)
target_probs[y_true] = 1.0
error_output = target_probs - z_output

# ★★★ ED法の核心: 飽和抑制項 ★★★
# これはシグモイド微分 z(1-z) でも tanh微分 (1-z²) でもない！
# 活性値の絶対値を使う: |z| × (1 - |z|)
saturation_output = np.abs(z_output) * (1.0 - np.abs(z_output))

# 重み更新量 = 学習率 × (誤差 × 飽和抑制項) の外積
output_lr = self.layer_lrs[-1]
gradients['w_output'] = output_lr * np.outer(
    error_output * saturation_output,
    z_hiddens[-1]
)
```

**ポイント:**
- **飽和抑制項 `|z| × (1 - |z|)`** がED法の核心です。ニューロンの活性値 `z` が飽和（0 または ±1 に近い）しているほど更新量が小さくなります
- BP法では `∂loss/∂z_output` を微分の連鎖律で計算しますが、ED法では飽和抑制項が微分を代替します
- この飽和抑制項は `z=0.5` のとき最大値 `0.25` を取り、`z=0` または `z=1` で `0` になります。つまり「確信度が中程度のニューロンほど大きく更新される」という直感的に合理的な動作をします

---

### 4.3 アミン濃度計算と拡散

**ファイル:** `modules_simple/ed_network.py` — `_compute_gradients()` の中間部分

出力層の誤差からアミン濃度を生成し、コラム構造に沿って隠れ層に拡散させます。これがBP法の「微分の連鎖律による逆伝播」を代替する機構です。

```python
# --- アミン濃度計算 ---

# ★★★ 純粋ED法: 正解クラスのみにアミンを注入 ★★★
# 不正解クラスへの負の学習信号（「〜ではない」と学ばせる）は使わない
amine_concentration = np.zeros(self.n_output)
error_correct = 1.0 - z_output[y_true]   # 正解クラスの予測確率の不足分
if error_correct > 0:
    # 正解クラスの出力が不十分な場合のみ、アミンが放出される
    amine_concentration[y_true] = error_correct * self.initial_amine
```

```python
# --- 隠れ層へのアミン拡散（逆順ループ） ---

for layer_idx in range(self.n_layers - 1, -1, -1):
    # ★ 拡散係数: 最終隠れ層(u1)とそれ以外(u2)で異なる
    # u1 > u2 とすることで、出力に近い層ほどアミン信号が強くなる
    if layer_idx == self.n_layers - 1:
        diffusion_coef = self.u1
    else:
        diffusion_coef = self.u2

    # アミンが空間的に拡散（濃度 × 拡散係数）
    amine_diffused = amine_concentration * diffusion_coef

    # ★★★ コラムメンバーシップによる選択的拡散 ★★★
    # アミンはコラム構造に沿って、該当クラスのコラムニューロンにのみ届く
    membership = self.column_membership_all_layers[layer_idx]
    active_classes = np.where(amine_diffused >= 1e-8)[0]

    # コラムメンバーの活性値でランク計算
    active_membership = membership[active_classes]
    masked_activations = np.where(active_membership, z_current, -np.inf)
    sorted_indices = np.argsort(-masked_activations, axis=1)
    ranks = np.argsort(sorted_indices, axis=1)

    # ★ ランク依存学習率: 活性値の高いニューロンほど強く学習
    learning_weights = self._learning_weight_lut[np.minimum(ranks, len(self._learning_weight_lut) - 1)]

    # ★ 非コラムニューロンは学習しない（固定ランダム重みを維持）
    learning_weights = np.where(active_membership, learning_weights, 0.0)

    # 各ニューロンのアミン量 = 拡散値 × ランク依存学習率
    amine_hidden[active_classes] = amine_diffused[active_classes, np.newaxis] * learning_weights
```

**ポイント:**
- **正解クラスのみ学習**（純粋ED法）: 不正解クラスにペナルティを与えるのではなく、正解クラスの出力を強化する方向にのみ学習します。これは「報酬系」の神経伝達物質の働きに対応します
- **アミン拡散**は、脳のノルアドレナリンやドーパミンなどの神経伝達物質が空間に拡散する現象をモデル化したものです。BP法の「微分を層間で逆伝播する」機構とは全く異なり、各層がアミン濃度という「空間的な信号」を受け取って独立に学習します
- **コラムメンバーシップ**により、アミン信号は対象クラスに割り当てられたニューロンにのみ届きます。これが多クラス分類を可能にする鍵です
- **ランク依存学習率**により、コラム内でより活性化しているニューロンほど強く学習します（勝者がより多く学習する仕組み）

---

### 4.4 隠れ層の重み更新

**ファイル:** `modules_simple/ed_network.py` — `_compute_gradients()` の後半部分

アミン拡散量と飽和抑制項を使って、各隠れ層が**独立に**重みを更新します。

```python
# --- 隠れ層の重み更新（各層について） ---

# ★★★ 飽和抑制項（出力層と同じ式） ★★★
# 活性値が飽和（±1に近い）しているニューロンは更新量が小さくなる
z_active = z_hiddens[layer_idx][active_neurons]
saturation_term = np.abs(z_active) * (1.0 - np.abs(z_active))
saturation_term = np.maximum(saturation_term, 1e-3)    # 最小値保証

# ★★★ 学習信号 = 学習率 × アミン拡散量 × 飽和抑制項 ★★★
# BP法の「微分の連鎖律」ではなく、3つの独立した要素の積
layer_lr = self.layer_lrs[layer_idx]
learning_signals = layer_lr * amine_hidden[:, active_neurons] * saturation_term[np.newaxis, :]

# 全クラスからの信号を合計 → 入力との外積で重み更新量を計算
signal_sum = learning_signals.sum(axis=0)
delta_w = signal_sum[:, np.newaxis] * z_input[np.newaxis, :]

# 第2層以降: 重みの符号を維持する制約
if layer_idx > 0:
    w_sign = np.sign(self.w_hidden[layer_idx][active_neurons, :])
    w_sign[w_sign == 0] = 1
    delta_w *= w_sign

# 勾配クリッピング（発散防止）
if self.gradient_clip > 0:
    delta_w_norms = np.linalg.norm(delta_w, axis=1, keepdims=True)
    clip_mask = delta_w_norms > self.gradient_clip
    delta_w = np.where(clip_mask, delta_w * (self.gradient_clip / delta_w_norms), delta_w)
```

**ポイント:**
- **各層が独立に学習**: 学習信号は `学習率 × アミン拡散量 × 飽和抑制項` の積です。BP法のように後段の勾配を乗算しながら逆伝播するのではなく、各層がアミン濃度と自身の活性値だけで更新量を決定します
- **飽和抑制項**が出力層と同じ式であることに注目してください。これは「すべての層が同じ原理で動作する」というED法の統一性を示しています
- **第2層以降の符号制約**: 重みの正負が反転しないように制約をかけます（Dale's Principleの一般化）

<details>
<summary>📄 _compute_gradients() の全コード</summary>

```python
def _compute_gradients(self, x_paired, z_hiddens, z_output, y_true):
    """
    ED法による勾配計算

    ★核心★ 微分の連鎖律を使わない。代わりに:
    1. 出力層の誤差からアミン濃度を生成
    2. アミンがコラム構造に沿って隠れ層に拡散
    3. 各層は独立に飽和抑制項 abs(z)*(1-abs(z)) で学習

    Args:
        x_paired: 入力ペア
        z_hiddens: 各隠れ層の出力
        z_output: 出力層の確率分布
        y_true: 正解クラス

    Returns:
        gradients: 各層の勾配辞書
    """
    gradients = {
        'w_output': None,
        'w_hidden': [None] * self.n_layers,
    }

    # ============================================
    # 1. 出力層の勾配計算
    # ============================================
    target_probs = np.zeros(self.n_output)
    target_probs[y_true] = 1.0
    error_output = target_probs - z_output

    # ★飽和抑制項★ ED法の核心 ─ シグモイド微分ではない
    saturation_output = np.abs(z_output) * (1.0 - np.abs(z_output))

    output_lr = self.layer_lrs[-1]
    gradients['w_output'] = output_lr * np.outer(
        error_output * saturation_output,
        z_hiddens[-1]
    )

    # ============================================
    # 2. 出力層のアミン濃度計算
    # ============================================
    # 純粋ED法: 正解クラスのみ学習
    amine_concentration = np.zeros(self.n_output)
    error_correct = 1.0 - z_output[y_true]
    if error_correct > 0:
        amine_concentration[y_true] = error_correct * self.initial_amine

    # ============================================
    # 3. 多層アミン拡散と勾配計算（逆順、微分の連鎖律不使用）
    # ============================================
    for layer_idx in range(self.n_layers - 1, -1, -1):
        if layer_idx == 0:
            z_input = x_paired
        else:
            z_input = z_hiddens[layer_idx - 1]

        # 拡散係数の選択（最終隠れ層=u1、それ以外=u2）
        if layer_idx == self.n_layers - 1:
            diffusion_coef = self.u1
        else:
            diffusion_coef = self.u2

        # アミン拡散（コラム構造に沿って選択的に拡散）
        amine_mask = amine_concentration >= 1e-8
        amine_diffused = amine_concentration * diffusion_coef

        # Membership方式: 活性値ランクベースTop-K学習
        membership = self.column_membership_all_layers[layer_idx]
        z_current = z_hiddens[layer_idx]
        n_neurons = self.n_hidden[layer_idx]

        active_classes = np.where(amine_diffused >= 1e-8)[0]
        n_active = len(active_classes)

        if n_active == 0:
            amine_hidden = np.zeros((self.n_output, n_neurons))
        else:
            # コラムメンバーの活性値でランク計算
            active_membership = membership[active_classes]
            masked_activations = np.where(active_membership, z_current, -np.inf)
            sorted_indices = np.argsort(-masked_activations, axis=1)
            ranks = np.argsort(sorted_indices, axis=1)

            # ランクからLUT参照で学習率を取得
            clamped_ranks = np.minimum(ranks, len(self._learning_weight_lut) - 1)
            learning_weights = self._learning_weight_lut[clamped_ranks]

            # 非コラムニューロンはamine=0（学習しない）
            learning_weights = np.where(active_membership, learning_weights, 0.0)

            # アミン拡散値に学習率を適用
            amine_hidden = np.zeros((self.n_output, n_neurons))
            amine_hidden[active_classes] = (
                amine_diffused[active_classes, np.newaxis] *
                learning_weights
            )

        amine_hidden = amine_hidden * amine_mask[:, np.newaxis]

        # 活性ニューロンの特定（アミン非ゼロの行のみ更新）
        neuron_mask = np.any(amine_hidden >= 1e-8, axis=0)
        active_neurons = np.where(neuron_mask)[0]

        if len(active_neurons) == 0:
            gradients['w_hidden'][layer_idx] = None
            continue

        # ★飽和抑制項★ abs(z)*(1-abs(z)) ─ 微分の連鎖律ではない
        z_active = z_hiddens[layer_idx][active_neurons]
        saturation_term_raw = np.abs(z_active) * (1.0 - np.abs(z_active))
        saturation_term = np.maximum(saturation_term_raw, 1e-3)

        # 学習信号 = 学習率 × アミン拡散量 × 飽和抑制項
        layer_lr = self.layer_lrs[layer_idx]
        learning_signals = (
            layer_lr *
            amine_hidden[:, active_neurons] *
            saturation_term[np.newaxis, :]
        )

        # 勾配計算（全クラスの信号を合計 × 入力でouter product）
        signal_sum = learning_signals.sum(axis=0)
        delta_w_batch = signal_sum[:, np.newaxis] * z_input[np.newaxis, :]

        # 第2層以降の符号制約
        if layer_idx > 0:
            w_sign = np.sign(self.w_hidden[layer_idx][active_neurons, :])
            w_sign[w_sign == 0] = 1
            delta_w_batch *= w_sign

        # 勾配クリッピング
        if self.gradient_clip > 0:
            delta_w_norms = np.linalg.norm(delta_w_batch, axis=1, keepdims=True)
            clip_mask = delta_w_norms > self.gradient_clip
            delta_w_batch = np.where(
                clip_mask,
                delta_w_batch * (self.gradient_clip / delta_w_norms),
                delta_w_batch
            )

        # コラムニューロンの学習率抑制（重み飽和防止）
        layer_lr_factor = self.column_lr_factors[layer_idx]
        if layer_lr_factor < 1.0 and layer_idx < len(self.column_membership_all_layers):
            membership = self.column_membership_all_layers[layer_idx]
            is_column_neuron = np.any(membership, axis=0)
            active_is_column = is_column_neuron[active_neurons]
            if np.any(active_is_column):
                delta_w_batch[active_is_column, :] *= layer_lr_factor

        # スパース形式で保存（active行のみ更新）
        gradients['w_hidden'][layer_idx] = (active_neurons, delta_w_batch)

    return gradients
```

</details>

<details>
<summary>📄 update_weights() の全コード</summary>

```python
def update_weights(self, x_paired, z_hiddens, z_output, y_true):
    """
    重みの更新（ED法準拠、微分の連鎖律不使用）

    各層が独立にアミン拡散信号に基づいて重み更新を行う。
    第1層にはDale's Principleの符号制約を適用。
    """
    gradients = self._compute_gradients(x_paired, z_hiddens, z_output, y_true)

    # 勾配適用
    self.w_output += gradients['w_output']

    for layer_idx in range(self.n_layers):
        sparse_grad = gradients['w_hidden'][layer_idx]
        if sparse_grad is not None:
            active_neurons, delta_w_batch = sparse_grad
            self.w_hidden[layer_idx][active_neurons] += delta_w_batch

            # 第1層のDale's Principle強制（active行のみ）
            if layer_idx == 0:
                self.w_hidden[0][active_neurons] = (
                    np.abs(self.w_hidden[0][active_neurons]) *
                    self._sign_matrix_layer0[active_neurons]
                )

    # 出力重みの微弱正則化
    self.w_output *= (1.0 - 0.00001)
```

</details>

---

### 4.5 create_column_membership() — コラム構造

**ファイル:** `modules_simple/column_structure.py`

コラム構造は、隠れ層のニューロンを2次元空間に配置し、各クラスに最も近いニューロンを割り当てる仕組みです。

```python
def create_column_membership(n_hidden, n_classes, ..., column_neurons=None):
    membership = np.zeros((n_classes, n_hidden), dtype=bool)

    # ★ 各クラスに割り当てるニューロン数
    # column_neurons=1: リザバーコンピューティング（各クラス1個のみ学習）
    # column_neurons=10: より多くのニューロンで各クラスを表現
    if column_neurons is not None:
        neurons_per_class = column_neurons

    # ★ ハニカム配置（2-3-3-2パターン）で10クラスの中心を決定
    # 大脳皮質の視覚野で見られるコラム構造の配置を模倣
    class_coords = {
        0: (center + scale*(-1), center + scale*(-1)),  # 上段左
        1: (center + scale*(+1), center + scale*(-1)),  # 上段右
        2: (center + scale*(-2), center + scale*(0)),   # 中段左
        3: (center + scale*(0),  center + scale*(0)),   # 中段中
        4: (center + scale*(+2), center + scale*(0)),   # 中段右
        5: (center + scale*(-2), center + scale*(+1)),  # 中下段左
        6: (center + scale*(0),  center + scale*(+1)),  # 中下段中
        7: (center + scale*(+2), center + scale*(+1)),  # 中下段右
        8: (center + scale*(-1), center + scale*(+2)),  # 下段左
        9: (center + scale*(+1), center + scale*(+2)),  # 下段右
    }

    # ★ 各クラスの中心に最も近い neurons_per_class 個のニューロンを割り当て
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

**ポイント:**
- **ハニカム配置（2-3-3-2パターン）**: 10クラスの中心座標を六角格子風に配置します。これは大脳皮質の視覚野で類似した特性を持つニューロンが柱状に集まる構造を模倣しています
- **最近傍割り当て**: 各クラスの中心に最も近い `neurons_per_class` 個のニューロンがそのクラスの「コラムニューロン」になります
- **`column_neurons=1`** の場合、各クラスたった1個のニューロンだけが学習対象になり、リザバーコンピューティングと同等の動作になります（2048個中10個のみ学習、99.5%は固定重み）
- `membership` は `[n_classes, n_hidden]` のブール配列で、これが学習時の「アミン信号がどのニューロンに届くか」を決定します

<details>
<summary>📄 create_column_membership() の全コード</summary>

```python
def create_column_membership(n_hidden, n_classes, participation_rate=1.0,
                             use_hexagonal=True, column_radius=0.4, column_neurons=None):
    """
    コラムメンバーシップフラグを作成

    各ニューロンがどのクラスのコラムに所属するかをブールフラグで管理。
    コラム構造を維持しつつ、重みは学習で獲得される。

    Args:
        n_hidden: 隠れ層のニューロン総数
        n_classes: 出力クラス数
        participation_rate: 各クラスに割り当てるニューロンの割合（0.0-1.0）
        use_hexagonal: Trueならハニカム配置、Falseなら順次割り当て
        column_radius: コラム半径（ハニカム配置時の参考値）
        column_neurons: 各クラスに割り当てるニューロン数（明示指定、優先度最高）

    Returns:
        membership: shape [n_classes, n_hidden] のブール配列
        neuron_positions: shape [n_hidden, 2] の2D座標配列
        class_coords: 各クラスのコラム中心座標辞書
    """
    membership = np.zeros((n_classes, n_hidden), dtype=bool)
    neuron_positions = None
    class_coords = None

    # 各クラスに割り当てるニューロン数（優先順位: column_neurons > participation_rate）
    if column_neurons is not None:
        neurons_per_class = column_neurons
    else:
        neurons_per_class = int(n_hidden * participation_rate / n_classes)

    if neurons_per_class == 0:
        neurons_per_class = 1

    if use_hexagonal:
        # ハニカム配置（2-3-3-2パターンで10クラスを中心化配置）
        grid_size = int(np.ceil(np.sqrt(n_hidden)))
        grid_center = grid_size / 2.0

        # スケール係数: コラムを隠れ層全体の約4/5の範囲に分散
        scale_factor = (grid_size * 0.8) / 4.0

        # 2-3-3-2配置の10クラス座標
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
        # 順次割り当て
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

## まとめ

コラムED法の学習は、以下の3つの要素で構成されています:

1. **飽和抑制項** `|z| × (1 - |z|)` — 微分の連鎖律を使わずに、各層が独立に適切な更新量を決定する仕組み
2. **アミン拡散** — 出力層の誤差をアミン濃度に変換し、コラム構造に沿って隠れ層に拡散させる仕組み
3. **コラム構造** — 隠れ層のニューロンをクラスに割り当て、アミン信号の選択的な伝達を実現する仕組み

これらはすべて生物学的に妥当なメカニズムであり、微分の連鎖律を用いた誤差逆伝播法（BP法）を一切使用していません。

---

## 5. 実装アンカー付き学習メカニズム（フル版）

本章は、上記の概念説明を**現行フル版実装**に1対1対応させるためのアンカー集です。

- 対象ファイル: `columnar_ed_ann.py`, `modules/ed_network.py`
- 目的: 理論要素を「どの行で実装しているか」を即確認できるようにする

### 5.1 実行フロー（mainから学習まで）

1. CLI引数の受け取り（学習率、コラム関連、初期化、Gabor）
`columnar_ed_ann.py:72`
`columnar_ed_ann.py:113`
`columnar_ed_ann.py:129`
`columnar_ed_ann.py:218`
`columnar_ed_ann.py:298`

2. 層数に応じたYAML自動パラメータ適用
`columnar_ed_ann.py:501`
`columnar_ed_ann.py:516`

3. ネットワーク生成（ED学習則パラメータ注入）
`columnar_ed_ann.py:987`

4. 学習実行（オンラインED）
`columnar_ed_ann.py:1245`
`modules/ed_network.py:1930`

5. 評価実行（学習後の性能計測）
`columnar_ed_ann.py:1251`
`modules/ed_network.py:2291`

### 5.2 理論要素とコード対応（1:1）

1. 連鎖律を使わないED勾配計算
`modules/ed_network.py:1301`
`_compute_gradients()` で各層を独立更新し、BPの連鎖微分を使わない。

2. 出力誤差の正解クラス基準化
`modules/ed_network.py:1354`
`error_correct = 1.0 - z_output[y_true]` により正解クラス由来の学習信号を生成。

3. 飽和抑制項 `abs(z) * (1 - abs(z))`
`modules/ed_network.py:1333`
`modules/ed_network.py:1516`
出力層と隠れ層で同一原理を適用。

4. アミン拡散（u1/u2）
`modules/ed_network.py:1382`
`amine_diffused = amine_concentration_output * diffusion_coef`。

5. コラムmembershipに基づく選択的学習
`modules/ed_network.py:1438`
`modules/ed_network.py:1481`
`learning_weights = np.where(...)` でコラム/非コラムの学習信号を分岐。

6. 層別学習率
`modules/ed_network.py:204`
`modules/ed_network.py:1516`
`self.layer_lrs` を各層更新信号に直接適用。

7. コラム学習率抑制（層別）
`modules/ed_network.py:250`
`modules/ed_network.py:1559`
`self.column_lr_factors[layer_idx]` でコラム行の勾配だけ抑制。

8. 勾配クリッピング
`modules/ed_network.py:279`
`modules/ed_network.py:1547`
過大更新を抑制して学習安定化。

9. 順伝播（tanh + softmax）
`modules/ed_network.py:972`
`modules/ed_network.py:1102`

10. オンライン逐次更新（EDの時間順更新）
`modules/ed_network.py:1745`
`modules/ed_network.py:1930`
`train_epoch()` が各サンプルで `train_one_sample()` を呼び、即時更新。

### 5.3 実行モードごとの注意

1. 通常実行（`--batch_size` なし）
`columnar_ed_ann.py:1245`
NumPyオンライン経路（本章のアンカー対象）が使われる。

2. `--batch_size` または `--use_cupy` 指定時
`columnar_ed_ann.py:1226`
ミニバッチ/高速化経路に分岐するため、内部実装アンカーは別系統も確認が必要。
