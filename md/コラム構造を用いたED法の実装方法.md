# コラム構造を用いたED法の実装方法

## 概要

本ドキュメントでは、オリジナルのED法（Error Diffusion Learning Algorithm）にコラム構造を導入し、多クラス分類を実現するための実装方法と注意点をまとめます。

最終的な成果：
- **訓練精度**: 79.0%（3000サンプル、10エポック）
- **テスト精度**: 73.2%（1000サンプル）
- **第1層の学習**: 正常に機能（相殺問題を解決）

---

## 1. コラム構造の作成

### 1.1 基本概念

コラム構造は、脳の大脳皮質の柱状構造を模倣したもので、各出力クラスに対応する専用のニューロン群を作成します。

**重要なパラメータ**:
- `column_neurons`: 各クラスに割り当てるニューロン数（推奨: 25-30）
- `column_overlap`: コラム間の重複係数（推奨: 0.3-0.5）

### 1.2 実装コード

```python
def create_column_affinity(n_neurons, n_classes, column_neurons=25, overlap=0.3):
    """
    コラム帰属度マップの作成
    
    Args:
        n_neurons: 隠れ層のニューロン総数
        n_classes: 出力クラス数
        column_neurons: 各クラスに割り当てるニューロン数
        overlap: コラム間の重複係数（0.0-1.0）
    
    Returns:
        column_affinity: shape [n_classes, n_neurons]
                        各ニューロンの各クラスへの帰属度（0.0-1.0）
    """
    column_affinity = np.zeros((n_classes, n_neurons))
    
    # 各クラスに対してコラムを作成
    for class_idx in range(n_classes):
        # コラムの中心ニューロンをランダムに選択
        center = np.random.randint(0, n_neurons)
        
        # 各ニューロンとの距離を計算（円環トポロジー）
        distances = np.minimum(
            np.abs(np.arange(n_neurons) - center),
            n_neurons - np.abs(np.arange(n_neurons) - center)
        )
        
        # 距離に基づいて帰属度を計算（ガウス型）
        sigma = column_neurons / 3.0  # 標準偏差
        affinity = np.exp(-0.5 * (distances / sigma) ** 2)
        
        # 閾値以上の帰属度を持つニューロンを選択
        threshold = np.exp(-0.5 * 9)  # 3σ点
        affinity[affinity < threshold] = 0
        
        # 重複を考慮（他のクラスのコラムと重複する場合）
        affinity *= (1.0 - overlap * np.sum(column_affinity, axis=0))
        affinity = np.clip(affinity, 0, 1)
        
        column_affinity[class_idx, :] = affinity
    
    return column_affinity
```

### 1.3 実装時の注意点

**重要**: コラム構造は各隠れ層ごとに独立して作成する必要があります。

```python
# 初期化時に各層のコラム帰属度を作成
self.column_affinity_all_layers = []
for layer_idx, n_hidden in enumerate(n_hidden):
    column_affinity = create_column_affinity(
        n_neurons=n_hidden,
        n_classes=n_output,
        column_neurons=column_neurons,
        overlap=column_overlap
    )
    self.column_affinity_all_layers.append(column_affinity)
```

**確認ポイント**:
1. 各クラスが十分なニューロン数を持つか（`column_neurons`）
2. コラム間の重複が適切か（過度な重複は性能低下の原因）
3. 全ニューロンが少なくとも1つのクラスに帰属しているか

---

## 2. コラム構造へのED法の実装

### 2.1 アミン濃度の管理

オリジナルED法のアミン拡散メカニズムを、コラムごとに適用します。

```python
# アミン濃度の初期化
# shape: [n_output, n_hidden, 2]
# [:, :, 0]: 興奮性アミン濃度
# [:, :, 1]: 抑制性アミン濃度
self.amine_concentrations = np.zeros((n_output, n_hidden, 2))
```

### 2.2 コラム単位の重み更新

**重要**: 各クラスのコラムに属するニューロンのみを選択的に更新します。

```python
# Winner-Takes-All方式の学習対象決定
if winner_class == y_true:
    # 正解時: 正解クラスのみ学習
    learning_configs = [(y_true, 0, 1.0)]  # (クラス, アミンタイプ, 係数)
else:
    # 誤答時: 勝者クラスを抑制、正解クラスを強化
    learning_configs = [
        (winner_class, 1, 1.0),  # 勝者を抑制（抑制性アミン）
        (y_true, 0, 1.0)         # 正解を強化（興奮性アミン）
    ]

# 各対象クラスのコラムを更新
for target_class, amine_type, learning_coef in learning_configs:
    # コラムに属するニューロンを選択
    column_affinity = self.column_affinity_all_layers[layer]
    column_scale = column_affinity[target_class]
    column_neurons_mask = column_scale > 0.5  # 帰属度閾値
    
    if not np.any(column_neurons_mask):
        continue
    
    column_indices = np.where(column_neurons_mask)[0]
    
    # アミン濃度を取得
    d_column = self.amine_concentrations[target_class, column_indices, amine_type]
    
    # 活性化関数の微分
    f_prime_column = self.tanh_scaled_derivative(
        a_hiddens[layer][column_indices], 
        z_hiddens[layer][column_indices]
    )
    
    # 重み更新量を計算
    amine_f_column = (self.alpha * d_column * f_prime_column * learning_coef).reshape(-1, 1)
    delta_w_column = amine_f_column * z_input.reshape(1, -1)
    
    # 重みを更新
    self.w_hidden[layer][column_indices, :] += delta_w_column
```

### 2.3 実装時の注意点

**致命的なバグ防止**:
1. **コラム選択の閾値**: `column_scale > 0.5`を使用（0.0にしない）
2. **インデックスの確認**: `column_indices`が空でないことを確認
3. **アミン濃度の次元**: `[n_output, n_hidden, 2]`の順序を守る

---

## 3. Dale's Principleの実装（重要）

### 3.1 問題の背景

初期実装では、第1層が全く学習しない問題が発生しました。原因は**興奮性と抑制性の重み変化の完全な相殺**でした。

**問題のメカニズム**:
```python
# NG例: 重み更新時にsign_constraintを適用
sign_constraint = np.outer(ei_output, ei_input)  # [+1, -1, +1, -1, ...]
delta_w = amine_f * z_input.reshape(1, -1) * sign_constraint

# 結果: 興奮性への変化 = +Δw、抑制性への変化 = -Δw
# ペア入力（同じ値）× 逆符号 → 出力への影響が完全に相殺
```

### 3.2 正しい実装方法

**解決策**: 重み更新時は符号制約を適用せず、**学習後に符号を強制**します。

```python
def columnar_ed_update(self, x, y_true, a_hiddens, z_hiddens, z_output):
    """コラムED法による重み更新"""
    
    # ... （省略）
    
    for layer in range(self.n_layers - 1, -1, -1):
        # ... （省略）
        
        if layer == 0:  # 第1層の場合
            z_input = create_excitatory_inhibitory_pairs(x)
            ei_input = self.excitatory_inhibitory[:len(z_input)]
        else:
            z_input = z_hiddens[layer-1]
            ei_input = None
        
        # 各コラムの重み更新
        for target_class, amine_type, learning_coef in learning_configs:
            # ... （コラム選択）
            
            # ★重要★ 第1層: sign_constraintを適用しない
            if ei_input is not None:
                # 符号制約なしで自由に学習
                amine_f_column = (self.alpha * d_column * f_prime_column * learning_coef).reshape(-1, 1)
                delta_w_column = amine_f_column * z_input.reshape(1, -1)  # sign_constraint削除
            else:
                # 第2層以降: 重みの符号を保持
                w_sign_column = np.sign(self.w_hidden[layer][column_indices, :])
                w_sign_column[w_sign_column == 0] = 1
                
                amine_f_column = (self.alpha * d_column * f_prime_column * learning_coef).reshape(-1, 1)
                delta_w_column = amine_f_column * z_input.reshape(1, -1) * w_sign_column
            
            # 重みを更新
            self.w_hidden[layer][column_indices, :] += delta_w_column
        
        # ★最も重要★ 第1層: 学習後に符号を強制
        if layer == 0:
            w1 = self.w_hidden[0]
            ei_input_full = self.excitatory_inhibitory[:w1.shape[1]]
            ei_hidden = np.ones(w1.shape[0])
            sign_matrix = np.outer(ei_hidden, ei_input_full)
            
            # 重みの符号を強制: w = |w| × sign
            self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
```

### 3.3 初期化時の実装

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    # 第1層の初期化時にDale's Principleを適用
    if len(self.w_hidden) > 0:
        w1 = self.w_hidden[0]
        
        # 入力側のE/Iフラグ
        ei_input = self.excitatory_inhibitory[:w1.shape[1]]
        
        # 出力側（隠れ層）は全て興奮性
        ei_hidden = np.ones(w1.shape[0])
        
        # ★重要★ 絶対値を取ってから符号制約を適用
        sign_matrix = np.outer(ei_hidden, ei_input)
        self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
```

### 3.4 実装時の重大な注意点

**絶対に守るべきルール**:

1. **重み更新時**: 第1層では`sign_constraint`を適用しない
2. **学習後**: 毎サンプルごとに`w = |w| × sign_matrix`で符号を強制
3. **初期化時**: `w *= sign_matrix`ではなく`w = |w| × sign_matrix`を使用

**理由**: ペア構造で同じ値を複製するため、重み更新時に符号制約を適用すると相殺が発生します。

---

## 4. Winner-Takes-All方式の実装

### 4.1 基本概念

各サンプルについて、最も出力が大きいクラス（勝者クラス）と正解クラスの関係に基づいて学習を制御します。

### 4.2 実装コード

```python
# 勝者クラスの決定
winner_class = np.argmax(z_output)

# 学習対象の決定
if winner_class == y_true:
    # 正解時: 正解クラスのコラムのみ学習
    learning_configs = [
        (y_true, 0, 1.0)  # (クラス, アミンタイプ, 学習係数)
    ]
else:
    # 誤答時: 勝者を抑制、正解を強化
    learning_configs = [
        (winner_class, 1, 1.0),  # 勝者クラスを抑制（抑制性アミン）
        (y_true, 0, 1.0)         # 正解クラスを強化（興奮性アミン）
    ]
```

### 4.3 アミン濃度の計算

```python
# アミン濃度の計算（オリジナルED法の関数を使用）
if winner_class == y_true:
    # 正解時: 興奮性アミンのみ
    target = 1.0
    error_correct = target - z_output[y_true]
    calculate_amine_separation(
        np.array([error_correct]), 
        self.amine_concentrations, 
        y_true, 
        initial_amine=0.7
    )
else:
    # 誤答時: 勝者クラスに抑制性アミン
    target_winner = 0.0
    error_winner = target_winner - z_output[winner_class]
    calculate_amine_separation(
        np.array([error_winner]), 
        self.amine_concentrations, 
        winner_class, 
        initial_amine=0.7
    )
    
    # 正解クラスに興奮性アミン
    target_correct = 1.0
    error_correct = target_correct - z_output[y_true]
    calculate_amine_separation(
        np.array([error_correct]), 
        self.amine_concentrations, 
        y_true, 
        initial_amine=0.7
    )
```

### 4.4 実装時の注意点

1. **アミン濃度の初期値**: `initial_amine=0.7`が推奨値
2. **誤差の符号**: 必ず`target - output`の順序で計算
3. **アミンタイプ**: 0=興奮性、1=抑制性

---

## 5. 側方抑制の実装

### 5.1 基本概念

クラス間の競合関係をモデル化し、誤答時の学習を促進します。

### 5.2 側方抑制重みの作成

```python
def create_lateral_inhibition_weights(n_classes, w1=0.1):
    """
    側方抑制重みマトリクスの作成
    
    Args:
        n_classes: クラス数
        w1: 抑制強度（推奨: 0.05-0.15）
    
    Returns:
        lateral_weights: shape [n_classes, n_classes]
                        対角成分は0、非対角成分は-w1
    """
    lateral_weights = -w1 * np.ones((n_classes, n_classes))
    np.fill_diagonal(lateral_weights, 0)
    return lateral_weights
```

### 5.3 学習への適用

```python
# 側方抑制の学習への適用
lateral_effect_to_true = self.lateral_weights[winner_class, y_true]

if winner_class != y_true:
    # 誤答時: 側方抑制を考慮してアミン濃度を増強
    target_correct = 1.0
    error_correct = target_correct - z_output[y_true]
    
    if lateral_effect_to_true < 0:
        # 抑制が強い場合、アミン濃度を増強
        enhanced_amine = 0.7 * (1.0 - lateral_effect_to_true)
    else:
        enhanced_amine = 0.7
    
    calculate_amine_separation(
        np.array([error_correct]), 
        self.amine_concentrations, 
        y_true, 
        initial_amine=enhanced_amine
    )
```

### 5.4 実装時の注意点

1. **抑制強度**: `w1=0.1`が推奨値（0.05-0.15の範囲で調整）
2. **増強係数**: `1.0 - lateral_effect_to_true`で抑制を相殺
3. **適用条件**: 誤答時のみ適用

---

## 6. コラム間結合の実装

### 6.1 コラム係数による誤差配分

コラムに属するニューロンと属さないニューロンで、異なる学習係数を適用します。

```python
# コラム帰属度マップの取得
column_affinity = self.column_affinity_all_layers[layer]
column_scale = column_affinity[y_true]  # 正解クラスへの帰属度

# コラムED法の係数
column_coef = 0.5      # コラム内の係数（小さめ）
non_column_coef = 1.5  # コラム外の係数（大きめ）

# 誤差を配分
error_column = error_current * column_scale * column_coef
error_non_column = error_current * (1.0 - np.clip(column_scale, 0, 1)) * non_column_coef
error_hidden = (error_column + error_non_column) * self.tanh_scaled_derivative(a_hiddens[layer], z_hiddens[layer])
```

### 6.2 実装時の注意点

**重要なバランス**:
- `column_coef < 1.0`: コラム内は控えめに学習（専門性を保持）
- `non_column_coef > 1.0`: コラム外は積極的に学習（汎化性能向上）

**推奨値**:
- `column_coef = 0.5`
- `non_column_coef = 1.5`

---

## 7. 活性化関数の実装

### 7.1 Tanhスケール活性化関数

```python
def tanh_scaled(self, u):
    """
    スケールされたtanh活性化関数
    
    Args:
        u: 線形出力
    
    Returns:
        z: 活性化後の出力（範囲: 0-1）
    """
    sig = 2.0  # スケール係数
    return 0.5 * (1.0 + np.tanh(u / sig))

def tanh_scaled_derivative(self, u, z):
    """
    スケールされたtanhの微分
    
    Args:
        u: 線形出力
        z: 活性化後の出力
    
    Returns:
        微分値
    """
    sig = 2.0
    return (1.0 - (2 * z - 1) ** 2) / sig
```

### 7.2 実装時の注意点

1. **出力範囲**: 0-1（シグモイドと同等）
2. **スケール係数**: `sig=2.0`が推奨値
3. **微分計算**: `z`を使って効率的に計算

---

## 8. 学習率とハイパーパラメータ

### 8.1 推奨値

| パラメータ | 推奨値 | 範囲 | 説明 |
|-----------|--------|------|------|
| `alpha` | 0.05 | 0.03-0.1 | 学習率 |
| `column_neurons` | 25-30 | 20-40 | 各クラスのニューロン数 |
| `column_overlap` | 0.3 | 0.2-0.5 | コラム間の重複係数 |
| `w1` | 0.1 | 0.05-0.15 | 側方抑制強度 |
| `column_coef` | 0.5 | 0.3-0.7 | コラム内学習係数 |
| `non_column_coef` | 1.5 | 1.2-2.0 | コラム外学習係数 |
| `initial_amine` | 0.7 | 0.5-0.9 | 初期アミン濃度 |

### 8.2 パラメータチューニングの指針

**学習率（alpha）**:
- 小さすぎる → 収束が遅い
- 大きすぎる → 発散または振動

**コラムニューロン数（column_neurons）**:
- 小さすぎる → 表現力不足
- 大きすぎる → 過学習、コラム間の干渉

**側方抑制強度（w1）**:
- 小さすぎる → クラス間競合が弱い
- 大きすぎる → 学習が不安定

---

## 9. デバッグとトラブルシューティング

### 9.1 第1層が学習しない場合

**確認項目**:

```python
# 学習前後の重みをコピー
w1_before = net.w_hidden[0].copy()  # ★.copy()を忘れずに

# 1エポック学習
net.train_epoch(train_images, train_labels)

# 学習後の重み
w1_after = net.w_hidden[0].copy()

# 変化を確認
w1_change = w1_after - w1_before
print(f"変化した重み数: {np.sum(np.abs(w1_change) > 1e-10)}")

# 出力の変化を確認
for i in range(5):
    a_h_before, z_h_before, _ = net.forward(train_images[i])
    # ... （学習）
    a_h_after, z_h_after, _ = net.forward(train_images[i])
    diff = z_h_after[0] - z_h_before[0]
    print(f"サンプル{i} RMS変化: {np.sqrt(np.mean(diff**2))}")
```

**期待される結果**:
- 重み変化: 数千〜数万個（総重みの1-10%）
- 出力RMS変化: 0.1以上

### 9.2 精度が上がらない場合

**チェックリスト**:

1. **コラム構造**:
   ```python
   # 各クラスが十分なニューロンを持つか
   for c in range(n_output):
       n_neurons_in_column = np.sum(column_affinity[c] > 0.5)
       print(f"クラス{c}: {n_neurons_in_column}個のニューロン")
   ```

2. **アミン濃度**:
   ```python
   # アミン濃度が正しく計算されているか
   print(f"アミン濃度: {net.amine_concentrations[y_true, column_indices, 0]}")
   ```

3. **Dale's Principle**:
   ```python
   # 第1層の重みの符号が正しいか
   ei_input = net.excitatory_inhibitory[:net.w_hidden[0].shape[1]]
   for i in range(10):
       if ei_input[i] > 0:
           assert np.all(net.w_hidden[0][:, i] >= 0), f"興奮性入力{i}への重みに負値"
       else:
           assert np.all(net.w_hidden[0][:, i] <= 0), f"抑制性入力{i}への重みに正値"
   ```

### 9.3 数値的不安定性の回避

```python
# オーバーフロー防止
u = np.clip(u, -50, 50)  # tanh計算前

# ゼロ除算防止
f_prime = np.maximum(f_prime, 1e-10)

# NaN/Inf検出
assert not np.any(np.isnan(delta_w)), "NaN detected in delta_w"
assert not np.any(np.isinf(delta_w)), "Inf detected in delta_w"
```

---

## 10. 実装の全体フロー

### 10.1 初期化フェーズ

```python
class ColumnarEDNetwork:
    def __init__(self, n_input, n_hidden, n_output, alpha=0.05, 
                 column_neurons=25, w1=0.1, column_overlap=0.3):
        # 1. 基本パラメータ
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.alpha = alpha
        
        # 2. 重みの初期化
        self.w_hidden = []
        for layer in range(len(n_hidden)):
            if layer == 0:
                w = np.random.randn(n_hidden[layer], n_input * 2) * 0.1
            else:
                w = np.random.randn(n_hidden[layer], n_hidden[layer-1]) * 0.1
            self.w_hidden.append(w)
        
        self.w_output = np.random.randn(n_output, n_hidden[-1]) * 0.1
        
        # 3. コラム構造の作成
        self.column_affinity_all_layers = []
        for layer_idx, n_h in enumerate(n_hidden):
            affinity = create_column_affinity(n_h, n_output, column_neurons, column_overlap)
            self.column_affinity_all_layers.append(affinity)
        
        # 4. 側方抑制重みの作成
        self.lateral_weights = create_lateral_inhibition_weights(n_output, w1)
        
        # 5. アミン濃度の初期化
        self.amine_concentrations = np.zeros((n_output, n_hidden[0], 2))
        
        # 6. E/Iフラグの作成
        self.excitatory_inhibitory = np.array([1 if i % 2 == 0 else -1 
                                                for i in range(n_input * 2)])
        
        # 7. Dale's Principleの適用（第1層）
        if len(self.w_hidden) > 0:
            ei_input = self.excitatory_inhibitory[:self.w_hidden[0].shape[1]]
            ei_hidden = np.ones(self.w_hidden[0].shape[0])
            sign_matrix = np.outer(ei_hidden, ei_input)
            self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
```

### 10.2 順伝播フェーズ

```python
def forward(self, x):
    """順伝播"""
    a_hiddens = []
    z_hiddens = []
    
    # 第1層: ペア構造入力
    z_input = create_excitatory_inhibitory_pairs(x)
    
    # 各隠れ層
    for layer in range(len(self.w_hidden)):
        a = np.dot(self.w_hidden[layer], z_input)
        z = self.tanh_scaled(a)
        
        a_hiddens.append(a)
        z_hiddens.append(z)
        z_input = z
    
    # 出力層
    a_output = np.dot(self.w_output, z_hiddens[-1])
    z_output = 1.0 / (1.0 + np.exp(-a_output / self.sig))
    
    return a_hiddens, z_hiddens, z_output
```

### 10.3 学習フェーズ

```python
def columnar_ed_update(self, x, y_true, a_hiddens, z_hiddens, z_output):
    """重み更新"""
    
    # 1. 出力層の誤差
    y_target = np.zeros(self.n_output)
    y_target[y_true] = 1.0
    error_output = y_target - z_output
    
    # 2. Winner-Takes-All
    winner_class = np.argmax(z_output)
    
    # 3. 側方抑制の効果
    lateral_effect = self.lateral_weights[winner_class, y_true]
    
    # 4. アミン濃度の計算
    # ... （セクション4参照）
    
    # 5. 出力層の重み更新
    # ... （省略）
    
    # 6. 隠れ層の重み更新（逆順）
    for layer in range(self.n_layers - 1, -1, -1):
        # 6.1 コラム係数による誤差配分
        # ... （セクション6参照）
        
        # 6.2 入力の取得
        if layer == 0:
            z_input = create_excitatory_inhibitory_pairs(x)
            ei_input = self.excitatory_inhibitory[:len(z_input)]
        else:
            z_input = z_hiddens[layer-1]
            ei_input = None
        
        # 6.3 学習対象の決定
        # ... （セクション4参照）
        
        # 6.4 各コラムの重み更新
        for target_class, amine_type, learning_coef in learning_configs:
            # コラム選択
            # ... （セクション2参照）
            
            # 重み更新（Dale's Principle考慮）
            # ... （セクション3参照）
        
        # 6.5 Dale's Principleの強制（第1層のみ）
        if layer == 0:
            # ... （セクション3参照）
```

---

## 11. 性能評価

### 11.1 達成した性能

**大規模テスト**（3000訓練/1000テスト、10エポック）:
- 訓練精度: 79.0%
- テスト精度: 73.2%（最高）、72.6%（最終）
- 第1層の学習: 正常に機能

**学習曲線**:
| Epoch | 訓練精度 | テスト精度 |
|-------|---------|-----------|
| 1 | 21.4% | 25.9% |
| 3 | 58.4% | 56.6% |
| 5 | 70.6% | 64.3% |
| 7 | 75.3% | 71.5% |
| 9 | 78.2% | **73.2%** |
| 10 | 79.0% | 72.6% |

### 11.2 バージョン別比較

| バージョン | 訓練精度 | テスト精度 | 第1層学習 | 主な特徴 |
|-----------|---------|-----------|----------|---------|
| v0.1.3 | 28.4% | - | ❌ | 第2層以降のみ学習 |
| v0.1.5 | 63.5% | 46.5% | ❌ | v0.1.3と同等 |
| v0.1.6 | 37.4% | 46.5% | ❌ | 初期化のみ修正（不十分） |
| **v0.1.7** | **79.0%** | **73.2%** | ✅ | Dale's Principle完全修正 |

---

## 12. まとめ

### 12.1 重要なポイント

1. **コラム構造の設計**: 適切な`column_neurons`と`overlap`の設定
2. **Dale's Principleの正しい実装**: 学習後の符号強制が必須
3. **Winner-Takes-All方式**: 正解/誤答で異なる学習戦略
4. **側方抑制の活用**: クラス間競合のモデル化
5. **コラム係数による誤差配分**: コラム内外で異なる学習率

### 12.2 実装の落とし穴

**絶対に避けるべきミス**:

1. ❌ 重み更新時に`sign_constraint`を適用（相殺が発生）
2. ❌ `.copy()`を忘れて参照で比較
3. ❌ コラム選択で閾値0.0を使用（全ニューロンが選択される）
4. ❌ アミン濃度の次元順序を間違える
5. ❌ 初期化時に`w *= sign_matrix`を使用（逆符号の対称値になる）

### 12.3 今後の展開

**性能向上の可能性**:
- パラメータチューニング（グリッドサーチ）
- 多層化（3層以上の隠れ層）
- バッチ正規化の導入
- 学習率スケジューリング

**応用可能性**:
- Fashion-MNISTへの適用
- より複雑なデータセットへの拡張
- SNNとの統合

---

## 参考文献

- Kaneko, I. (1999). "Error Diffusion Learning Algorithm"
- Dale's Principle in Neural Networks
- Columnar Organization of the Cerebral Cortex

---

**作成日**: 2025年11月30日  
**最終更新**: 2025年11月30日  
**バージョン**: v0.1.7準拠
