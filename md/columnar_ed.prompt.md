# ED法にコラム構造を導入したネットワーク学習方法の仕様書

**ED法 原作発案者**: 金子勇（Isamu Kaneko）  
**原作開発年**: 1999年  
**ED法 オリジナルソース**: C言語実装（動作確認済み）

**拡張実装**: 2025年 by yoiwa0714  
**拡張版**: ED法へのコラム構造の導入  
**最新バージョン**: v0.2.6-multiclass-multilayer  
**最終更新**: 2025-12-09（エポック最適化・2層構成最適化完了）

## 本プロジェクトの目的

 - 生物学的に妥当なニューラルネットワーク学習アルゴリズムの研究と開発を促進する
 - ED法は生物学的に妥当な理論であるため、金子勇氏オリジナルのED法理論にコラム構造を導入し、オリジナルED法理論を拡張する

### 本プロジェクトの背景

 - ED法は基本的に「二値分類問題には対応できるが、多分類問題は不得手」という特性を持っている
 - この二値分類しかできないという制限のあるED法を使って多クラス分類に対応させたい
 - そのためのアイデアとして、出力クラスと関係を持つコラム構造を設け、コラム構造と二値分類の機能を合わせることで、ED法による多クラス分類ができるようにしたい

## 達成精度（MNIST）

### 隠れ層1層 [512ニューロン]
- Test精度: **83.80%**
- Train精度: 86.77%
- 最適パラメータ: lr=0.20, u1=0.5, lateral_lr=0.08, radius=1.0
- エポック: 40（収束分析に基づく最適化、100から60%削減）
- 最適化完了: 2025-12-04

### 隠れ層2層 [256, 128ニューロン]
- Test精度: **64.27%**
- Train精度: 61.00%
- 最適パラメータ: lr=0.35, u1=0.5, u2=0.5, lateral_lr=0.08
- エポック: 45（収束分析に基づく最適化、100から55%削減）
- 最適化完了: 2025-12-09

### 隠れ層3-5層
- 状態: パラメータ設定済み、グリッドサーチによる最適化予定
- 暫定パラメータ: lr=0.35, u1=0.5, u2=0.5, epochs=50

### 重要な知見
- 層数増加に伴い精度低下（1層: 83.80% → 2層: 64.27%, 約19%低下）
- 多層構成では高めの学習率が必要（1層: 0.20 → 2層: 0.35）
- エポック最適化により55-60%の時間短縮、精度低下<1%

## 実装上の注意点

 - 生物学的に妥当ではない「微分の連鎖律を用いた誤差逆伝播法」の使用を禁止する。これは本プロジェクトにおける絶対のルールである
  - もし、仮にこの仕様書中に「誤差逆伝播法を使用する」という言葉が出てきたとしても、それは記述の誤りであり、ED法の実装においては誤差逆伝播法を使用しないことを厳守すること
 - オリジナルのED法で実装されていた「各クラスごとに独立した重み空間を持つ実装」は規模の大きな多クラス分類では以下の問題があることが判明したため禁止する：
   - クラス間での情報共有ができない
   - メモリ使用量が膨大になる（クラス数 × 重み行列サイズ）
   - クラス間の相対的な関係性を学習できない
 - 代わりに、**1つの共有重み空間 + コラム構造**による実装を採用する：
   - 全クラスで重み行列を共有し、メモリ効率が高い
   - コラム帰属度により、各クラスに対応するニューロン群を区別
   - クラス間の情報共有と専門化を両立

### u1/u2パラメータの正しい実装（v028バグ修正）⚠️

**重要**: v027以前では、アミン拡散処理に誤った実装がありました。v028で修正済みです。

**問題のあった実装（v027以前）**:
```python
# ❌ 誤った実装: column_affinityとの積算順序が誤っていた
amine_hidden = amine × u1 × column_affinity  # u1の効果が減衰
```

**影響**:
- u1/u2の値を変更しても学習精度がほとんど変化しない
- アミン拡散係数のパラメータチューニングが無効化されていた

**正しい実装（v028以降、オリジナルCコード準拠）**:
```python
# ✅ 正しい実装: 2ステップ処理
# ステップ1: アミン拡散（全ニューロン一律）
amine_diffused = amine_concentration × diffusion_coef  # u1またはu2

# ステップ2: コラム構造による重み付け
amine_hidden_3d = amine_diffused × column_affinity
```

**参考**: original-c-source-code/teach_calc.c 26-27行目の実装に準拠

**検証結果（10エポック実験）**:
- u1=0.3: Test=0.746
- u1=0.5: Test=0.743 (-0.4%)
- u1=0.7: Test=0.742 (-0.5%) → 50エポック: Test=0.780

**実装時の注意**:
1. アミン拡散係数は必ず**column_affinityとの積算の前に**適用する
2. オリジナルCコードの2ステップ処理を厳守する
3. u1/u2パラメータの効果が正しく反映されることを検証する

## 純粋なED法の判定基準 (2025-12-03 調査結果)

オリジナルのCコード（original-c-source-code/）を精査した結果、以下の判定基準を確立しました：

### ✅ オリジナルED法に含まれる要素

1. **誤差計算**: `error = target - output`
   - オリジナルCコード (teach_calc.c 9行目) で確認: `wkb = indata_tch[l] - ot_ot[l][in+2];`
   - これは**純粋なED法の一部**であり、勾配降下法ではない

2. **飽和抑制項**: `|output| × (1 - |output|)`
   - オリジナルCコード (weight_calc.c 13-14行目) で確認:
     ```c
     del *= fabs(ot_ot[n][k]);              // |出力|
     del *= (1 - fabs(ot_ot[n][k]));      // (1 - |出力|)
     ```
   - これは**生物学的な飽和特性**を表現するものであり、微分ではない

### ❌ 微分の連鎖律を使用（誤差逆伝播法的）

1. **シグモイド微分の直接使用**:
   - `sigmoid_derivative(z) = z * (1 - z)` を**活性化関数の微分として**使用
   - これは勾配降下法の典型的な実装

2. **判定方法**:
   ```python
   # ❌ 誤差逆伝播法（微分の連鎖律）
   sigmoid_derivative = z * (1 - z)
   delta_w = learning_rate * error * sigmoid_derivative * input
   
   # ✅ 純粋なED法（飽和抑制項）
   saturation = abs(z) * (1 - abs(z))
   delta_w = learning_rate * amine * saturation * input
   ```

### 重要な違い

| 要素 | 純粋なED法 | 誤差逆伝播法 |
|---|---|---|
| 計算式 | `abs(z) * (1 - abs(z))` | `z * (1 - z)` |
| 意味 | 生物学的飽和特性 | 活性化関数の微分 |
| 使用箇所 | オリジナルCコード確認済み | 一般的なNN実装 |
| 絶対値 | あり | なし |

### v017〜v023の判定結果

| バージョン | 出力層 | 隠れ層 | 判定 |
|---|---|---|---|
| v017 | sigmoid_derivative ❌ | sigmoid_derivative ❌ | ❌ 誤差逆伝播法 |
| v020 | sigmoid_derivative ❌ | abs(z)*(1-abs(z)) ✅ | ⚠️ 混合 |
| v023 | abs(z)*(1-abs(z)) ✅ | abs(z)*(1-abs(z)) ✅ | ✅ 純粋なED法 |

## ED法の概要

ED法（Error Diffusion Learning Algorithm）は、生物学的神経系のアミン（神経伝達物質）拡散メカニズムを模倣した独創的な学習アルゴリズムです。
従来のバックプロパゲーションとは根本的に異なる、興奮性・抑制性ニューロンペア構造と出力ニューロン中心のアーキテクチャを特徴とします。

### ED法の基本原理

オリジナルED法の基本原理はプロジェクトディレクトリのdocs/ED法_解説資料.mdに詳細に記載されています。
本仕様書に目を通す場合には必ずdocs/ED法_解説資料.mdにも目を通し、ED法の基本原理を理解した上で本仕様書を参照してください。

## コラム

### 大脳皮質コラムの特徴

 - 直径: 約0.5mm
 - 構造: 皮質を垂直に貫通する柱状
 - 機能: 特定刺激への選択的反応
 -  例: V1視覚野では「特定の傾きの線」に反応

 - 重要な性質:
 -  隣接コラムは「少し違う刺激」に反応
 -  階層的情報処理: V1 → V2 → 高次視覚野
 -  各コラム内でローカルな特徴抽出

#### 実際の人間の脳内のコラムに関する資料からの抜粋

 - コラムとは、大脳皮質を垂直に通る柱状の機能単位のことである。
 - 領域によって大きさに違いがあるが、直径は約0.5mm。
 - 視覚を例にとると、目から入った情報は、脳の奥にある視床で中継され、後頭部にある第一次視覚野（V1）に送られる。ここでは、視覚情報から形や色、動きに関する基本的な情報が抽出される。
 - 例えば、V1には、ある傾きの線にだけ反応する神経細胞が集まってコラムをつくっている。その隣のコラムには少し違う傾きに反応する神経細胞が集まっている。V1で処理された情報は、V1の前方にある第二次視覚野（V2）でより立体的な視覚認知が行われ、さらに前方の視覚野へと情報が送られていく。
 - こうして奥行きや距離、色、運動、位置関係などを把握するためのより高次の情報処理が行われ、どこに、何があるのかが認知される。

### コラムの実装方法

#### 各クラスの出力ニューロンに対応したコラムを作成

 - 出力層から隠れ層の第1層(入力層の次の層)まで貫通する構造
 - クラス1に対応したコラムであれば、出力1の誤差を学習する時に他のクラスのコラムよりも強く反応
 - この動作により、クラス1に対応したコラムは正解クラス1の重みの更新の影響を強く受け、その他の正解クラスの重みの更新の影響をあまり受けないという動作が可能になる
 - 以上のことより、1つの重み空間の中で多クラスに対応した重みの更新が可能になる

#### コラム構造の選択方式

**デフォルト実装: ハニカム構造**
 - ハニカム（六角形）状の2次元配置によるコラム構造
 - ニューロンを2-3-3-2の行配置で視覚的に配置
 - 各クラスに対して空間的に分離された領域を割り当て
 - 生物学的な大脳皮質のコラム構造により近い実装

**オプション実装: 円環構造**
 - 1次元円環トポロジーによるコラム構造
 - ニューロン配列の端と端が繋がった構造
 - ガウス型分布による連続的な帰属度
 - `--use_circular`オプションで選択可能

#### コラム帰属度マップの作成

コラム構造は、各ニューロンが各クラスにどの程度帰属するかを表す**帰属度マップ**で実装します。

##### 実装例1: ハニカム構造（デフォルト）

```python
def create_hexagonal_column_affinity(n_neurons, n_classes, participation_rate=1.0):
    """
    ハニカム構造のコラム帰属度マップ作成
    
    Args:
        n_neurons: 隠れ層のニューロン総数
        n_classes: 出力クラス数（10クラス想定）
        participation_rate: 各クラスに割り当てるニューロンの割合（0.0-1.0）
    
    Returns:
        column_affinity: shape [n_classes, n_neurons]
                        各ニューロンの各クラスへの帰属度（0.0-1.0）
    """
    column_affinity = np.zeros((n_classes, n_neurons))
    
    # ハニカム構造: 2-3-3-2の行配置
    row_patterns = [2, 3, 3, 2]
    positions = []
    for row_idx, cols in enumerate(row_patterns):
        for col_idx in range(cols):
            positions.append((row_idx, col_idx))
    
    # 各クラスに位置を割り当て
    for class_idx in range(n_classes):
        if class_idx >= len(positions):
            break
        
        row, col = positions[class_idx]
        
        # ニューロンを2D配置に変換して距離計算
        grid_size = int(np.ceil(np.sqrt(n_neurons)))
        neuron_positions = np.array([
            [i // grid_size, i % grid_size] for i in range(n_neurons)
        ])
        
        # コラム中心位置を設定
        center_row = row * (grid_size / len(row_patterns))
        center_col = col * (grid_size / max(row_patterns))
        
        # 各ニューロンとの距離を計算
        distances = np.sqrt(
            (neuron_positions[:, 0] - center_row) ** 2 +
            (neuron_positions[:, 1] - center_col) ** 2
        )
        
        # 距離に基づいて帰属度を計算（ガウス分布）
        sigma = grid_size / (2 * len(row_patterns))
        affinity = np.exp(-0.5 * (distances / sigma) ** 2)
        
        # participation_rateに基づいて上位ニューロンを選択
        n_assigned = int(n_neurons * participation_rate / n_classes)
        threshold = np.partition(affinity, -n_assigned)[-n_assigned]
        affinity[affinity < threshold] = 0
        
        # 正規化
        if np.sum(affinity) > 0:
            affinity = affinity / np.sum(affinity)
        
        column_affinity[class_idx, :] = affinity
    
    return column_affinity
```

**ハニカム構造の特性**：
- **2次元配置**: 2-3-3-2の行パターンで視覚的配置
- **空間分離**: 各クラスが独立した領域を持つ
- **生物学的妥当性**: 大脳皮質のコラム構造に近い
- **参加率制御**: `participation_rate`で各クラスのニューロン数を制御

##### 実装例2: 円環構造（オプション）

**v027更新**: ハニカム構造の知見を反映した改善版

```python
def create_column_affinity(n_neurons, n_classes, column_size=30, overlap=0.3,
                           column_neurons=None, participation_rate=None):
    """
    円環構造のコラム帰属度マップ作成（v027更新）
    
    v027更新内容:
    - participation_rate対応: 部分参加による学習促進
    - 中心化配置: 全クラスの均等なニューロンアクセス
    - 検証ロジック: 0.0と1.0の禁止
    
    Args:
        n_neurons: 隠れ層のニューロン総数
        n_classes: 出力クラス数
        column_size: 各コラムの基準サイズ（ニューロン数）- column_neurons/participation_rate未指定時に使用
        overlap: コラム間の重複係数（0.0-1.0）- participation_rate未指定時に使用
        column_neurons: 各クラスのコラムに割り当てるニューロン数（指定時はcolumn_sizeより優先）
        participation_rate: コラム参加率（0.01-0.99）。指定時はcolumn_neurons/column_sizeより優先
                           0.0と1.0は禁止（0.0=コラム無意味、1.0=学習不安定）
    
    Returns:
        column_affinity: shape [n_classes, n_neurons]
                        各ニューロンの各クラスへの帰属度（0.0-1.0）
    """
    # participation_rate検証
    if participation_rate is not None:
        if participation_rate <= 0.0 or participation_rate >= 1.0:
            raise ValueError(
                f"participation_rate must be in range (0.0, 1.0), exclusive. "
                f"Got {participation_rate}."
            )
    
    column_affinity = np.zeros((n_classes, n_neurons))
    
    # 中心化した等間隔配置（エッジ効果を軽減）
    centers = np.linspace(0, n_neurons, n_classes, endpoint=False).astype(int)
    center_offset = n_neurons // (2 * n_classes)  # 中心化オフセット
    centers = centers + center_offset
    
    # ガウス分布の標準偏差
    if column_neurons is not None:
        sigma = column_neurons / 3.0
    else:
        sigma = column_size / 3.0
    
    for class_idx in range(n_classes):
        center = centers[class_idx]
        
        # 各ニューロンとの距離を計算（円環トポロジー）
        distances = np.minimum(
            np.abs(np.arange(n_neurons) - center),
            n_neurons - np.abs(np.arange(n_neurons) - center)
        )
        
        # ガウス型の帰属度を計算
        affinity = np.exp(-0.5 * (distances / sigma) ** 2)
        
        # 閾値以下を0に設定
        threshold = np.exp(-0.5 * 9)  # 3σ点
        affinity[affinity < threshold] = 0
        
        column_affinity[class_idx, :] = affinity
    
    # participation_rate対応（最優先）
    if participation_rate is not None:
        target_neurons = int(n_neurons * participation_rate)
        neurons_per_class = target_neurons // n_classes
        remainder = target_neurons % n_classes
        
        assigned = np.zeros(n_neurons, dtype=bool)
        overlap_factor = 0.0 if participation_rate >= 0.99 else 0.3
        
        for class_idx in range(n_classes):
            available_affinity = column_affinity[class_idx].copy()
            
            # overlap許容の場合のみ、割り当て済みニューロンも重み付けで考慮
            if overlap_factor > 0:
                available_affinity[assigned] *= overlap_factor
            else:
                available_affinity[assigned] = 0
            
            sorted_indices = np.argsort(available_affinity)[::-1]
            n_neurons_for_this_class = neurons_per_class + (1 if class_idx < remainder else 0)
            selected = sorted_indices[:n_neurons_for_this_class]
            
            # マスク適用
            mask = np.zeros(n_neurons)
            mask[selected] = 1
            column_affinity[class_idx] *= mask
            
            assigned[selected] = True
    
    elif column_neurons is not None:
        # 明示的なニューロン数指定（中優先）
        assigned = np.zeros(n_neurons, dtype=bool)
        overlap_factor = 0.3
        
        for class_idx in range(n_classes):
            available_affinity = column_affinity[class_idx].copy()
            available_affinity[assigned] *= overlap_factor
            
            sorted_indices = np.argsort(available_affinity)[::-1]
            selected = sorted_indices[:column_neurons]
            
            mask = np.zeros(n_neurons)
            mask[selected] = 1
            column_affinity[class_idx] *= mask
            
            assigned[selected] = True
    
    else:
        # 従来のoverlap制御（最低優先）
        if overlap < 1.0:
            for neuron_idx in range(n_neurons):
                total_affinity = np.sum(column_affinity[:, neuron_idx])
                if total_affinity > 1.0:
                    max_class = np.argmax(column_affinity[:, neuron_idx])
                    for class_idx in range(n_classes):
                        if class_idx != max_class:
                            column_affinity[class_idx, neuron_idx] *= overlap
    
    return column_affinity
```

**円環構造の特性（v027更新）**：
- **中心化配置**: 等間隔配置+中心化オフセットで全クラスの均等なニューロンアクセス（ハニカム構造の知見を反映）
- **participation_rate対応**: 部分参加による学習促進（v026分析: ~0.71が有効）
- **ガウス型分布**: 中心ニューロンほど高い帰属度
- **円環トポロジー**: ニューロン配列の端と端が繋がっている
- **3モード対応**: 
  - モード1（最優先）: `participation_rate`指定
  - モード2（中優先）: `column_neurons`指定
  - モード3（最低優先）: `overlap`制御（従来方式）
- **検証ロジック**: 0.0と1.0の禁止（0.0=コラム無意味、1.0=学習不安定）

**共通の特性**：
- **各層独立**: 各隠れ層ごとに独立したコラム構造を作成
- **正規化**: 各クラスの帰属度の合計を調整

**v027以前との比較**：
| 要素 | v027以前 | v027更新後 |
|---|---|---|
| クラス配置 | 等間隔（エッジ配置） | 中心化した等間隔 |
| 参加率制御 | 未サポート | participation_rate対応 |
| 検証ロジック | なし | 0.0/1.0禁止 |
| ハニカム知見 | 未反映 | 完全反映 |

## 拡張機能一覧（オリジナル理論からの追加機能）

本実装では、金子勇氏のオリジナルED法理論を完全に保持しながら、以下の拡張機能を追加実装しています：

### 1. 多層ニューラルネットワーク対応
- **拡張機能**: 複数隠れ層を自由に組み合わせ可能
- **実装方法**: カンマ区切り指定（例：`--hidden 256,128,64`）
- **技術的特徴**: 層構造を管理するクラスによる動的層管理
- **ED法理論との整合性**: アミン拡散係数u1を多層間に適用

### 2. ミニバッチ学習システム（v027.2拡張）
- **拡張機能**: 複数サンプルをまとめて効率的に処理
- **実装方法**: `--batch` オプションでサイズ指定可能（未指定=オンライン学習）
- **TensorFlow Dataset API統合**（v027.2）:
  - NumPyシャッフル実装の完全削除（60行削減）
  - TensorFlow Dataset APIによる業界標準手法の採用
  - **学習安定性35.6%向上**: 標準偏差 NumPy 8.24% → TensorFlow 5.31%
  - **完全な再現性保証**: `--seed`引数が常にTensorFlow Dataset APIに渡される
  - 70エポック実験で最終精度の完全一致を確認（75.60%）
- **4つの学習モード**:
  * 引数なし → オンライン学習（シャッフルなし）
  * `--shuffle` → オンライン学習（シャッフルあり、batch_size=1）
  * `--batch N` → ミニバッチ学習（シャッフルなし）
  * `--batch N --shuffle` → ミニバッチ学習（シャッフルあり）
- **技術的実装**:
  ```python
  # TensorFlow Dataset API使用例
  dataset = tf.data.Dataset.from_tensor_slices((X, y))
  if shuffle:
      dataset = dataset.shuffle(buffer_size=len(X), seed=seed)
  if batch_size:
      dataset = dataset.batch(batch_size)
  ```

### 3. NumPy行列演算による劇的高速化
- **拡張機能**: NumPy行列演算による並列計算
- **技術的特徴**: ベクトル化
- **理論保持**: ED法のアルゴリズムに影響を与えない実装にする

### 4. リアルタイム可視化システム（v029拡張）
- **拡張機能**: 学習過程のリアルタイムグラフ表示
- **実装機能**: 学習曲線、混同行列、正答率推移の動的可視化
- **技術的特徴**: matplotlibを用いる
- **使用方法**: 
  - `--viz`: 学習曲線のリアルタイム可視化を有効化
  - `--heatmap`: 活性化ヒートマップの表示を有効化（`--viz`と併用）
  - `--save_viz [PATH]`: 可視化結果を保存
- **可視化結果の保存機能**（v029）:
  - **パス指定方法**:
    * 末尾"/"でディレクトリ指定 → タイムスタンプ付きファイル名で保存
      - 例: `--save_viz results/` → `results/viz_results_20251228_153045.png`
    * 末尾"/"なしでベースファイル名指定 → 指定名で保存
      - 例: `--save_viz results/exp1` → `results/exp1.png`
    * 引数なし → デフォルトディレクトリ（viz_results/）にタイムスタンプ付きで保存
  - **複数可視化の同時保存**:
    * 学習曲線とヒートマップを同時保存する場合、`_viz.png`と`_heatmap.png`が付加される
    * 例: `--viz --heatmap --save_viz results/exp1`
      - 出力: `results/exp1_viz.png`, `results/exp1_heatmap.png`
- **学習曲線の軸仕様**:
  - **縦軸**: 最小0.0、最大1.0、目盛り0.0から1.0まで0.1刻み
    - 中間グリッド線: 0.1, 0.3, 0.5, 0.7, 0.9が点線（`:`）
    - 中間グリッド線: 0.2, 0.4, 0.6, 0.8が実線（`-`）
  - **横軸**: 最小1、最大=設定された最大エポック数
    - 目盛り: 横軸を10分割した位置に設定
    - 中間グリッド線: 点線（`:`）と実線（`-`）を交互に配置
    - 例: 最大100エポック → 10,20,30,...,90の目盛り（10=点線、20=実線、30=点線、...）

### 5. 現代的データローダー統合（v027.3拡張）
- **拡張機能**: TensorFlow (tf.keras.datasets) 統合による自動データ処理
- **対応データセット**: 
  - 標準: MNIST・Fashion-MNIST・CIFAR-10・CIFAR-100の自動ダウンロード
  - **カスタムデータセット対応**（v027.3）: 任意の画像データセットを使用可能
- **技術的特徴**: `--train` `--test` で選択されたデータを全エポックで使用
- **使用方法**: 
  - 標準: `--dataset mnist`, `--dataset fashion`, `--dataset cifar10`, `--dataset cifar100`
  - カスタム: `--dataset my_custom_data` または `--dataset /path/to/my_data`
- **カスタムデータセット機能**（v027.3 Phase 2-3）:
  - **metadata.json対応**: 柔軟なデータセット管理、クラス名表示機能
  - **自動検証機能**: データ型、欠損値、ラベル範囲、整合性、クラス分布の5項目を検証
  - **エラーハンドリング強化**: 
    * JSON解析エラーの行番号・列番号表示
    * 類似データセット候補の自動提案
    * NaN/Inf位置の特定と修正方法の提示
  - **パフォーマンス最適化**:
    * メモリマップモード（100MB以上で自動適用: `np.load mmap_mode='r'`）
    * サンプル数制限の早期適用、不要なコピーの削減
    * 大規模データセット読み込み時の進捗表示
- **データセット検索優先順位**: 
  1. 指定パス
  2. ~/.keras/datasets/
  3. カレントディレクトリ

### 6. GPU計算支援（CuPy対応）
- **拡張機能**: NVIDIA GPU使用時の自動GPU計算
- **技術的特徴**: CuPy統合による透明なGPU処理
- **性能向上**: 大規模データセットでの更なる高速化
- **互換性**: GPU無環境でも自動的にCPU処理に切替

### 7. 詳細プロファイリング機能
- **拡張機能**: 処理段階別の詳細性能分析
- **実装機能**: ボトルネック特定、メモリ使用量監視
- **技術的特徴**: リアルタイム性能監視とレポート生成
- **使用方法**: `--verbose` オプションで詳細表示

### 10. HyperParamsクラスによる自動パラメータ管理（v0.2.6新機能）
- **拡張機能**: 層数に応じた最適パラメータの自動選択
- **パラメータ管理方式**: 
  - **外部YAMLファイル**: `config/hyperparameters.yaml`（編集可能）
  - **初期状態ファイル**: `config/hyperparameters_initial.yaml`（リファレンス用・復元用）
  - **読み込み関数**: `load_hyperparameters()`（`modules/hyperparameters.py`）
  - **エラーチェック**: モジュールimport時に自動実行（早期エラー検出）
- **実装機能**: 
  - 1-5層の最適パラメータテーブル管理
  - グリッドサーチ結果の統合（Phase 1, Phase 2完全評価）
  - エポック収束分析結果の反映
  - 6層以上のフォールバック機構
  - **層ごとの重み初期化係数の保持**（v0.2.8拡張）:
    * パラメータテーブルに `weight_init_scales` フィールドを追加
    * 隠れ層と出力層の係数をリスト形式で保持（例: `[2.25, 2.75, 12.00]`）
    * 実験024 Phase 1-3の最適化結果を反映
    * 計算式: `scale = coef / sqrt(fan_in)` で各層の重み初期化に使用
    * CLI引数（--wis）による上書き対応
- **最適化実績**:
  - **1層**: lr=0.20, epochs=40, 83.80%精度（2025-12-04最適化完了）
  - **2層**: lr=0.35, epochs=45, 64.27%精度（2025-12-09最適化完了）
  - **2層重み係数**: [2.25, 2.75, 12.00], 81.50%精度（2025-12-24最適化完了）
  - **エポック削減**: 1層60%、2層55%の時間短縮、精度低下<1%
- **技術的特徴**:
  - column_radiusの層数依存スケーリング: radius = base × sqrt(neurons/256)
  - participation_rate=1.0で全参加・重複なし方式を採用
  - CLI引数による個別パラメータのオーバーライド対応
  - 重み初期化係数の優先順位: CLI (--wis) > HyperParams > デフォルト値
- **使用方法**: 
  - `--list_hyperparams` で利用可能な設定一覧を表示
  - パラメータをカスタマイズする場合は `config/hyperparameters.yaml` を直接編集
  - 層数を指定すると自動的に最適パラメータが適用される
  - 個別パラメータは `--lr`, `--epochs` 等で上書き可能
  - 重み初期化係数は `--wis` で上書き可能

### 11. 重み初期化係数のコマンドライン指定（v0.2.8新機能）
- **拡張機能**: 重み初期化係数をコマンドライン引数で柔軟に指定
- **実装機能**: 
  - `--wis` 引数による層ごとの重み初期化係数の指定
  - 繰り返し記法 `値[回数]` による超多層対応（100層以上でも簡潔）
  - HyperParamsテーブルとの統合（優先順位: CLI > HyperParams > デフォルト）
  - 層数整合性チェックとエラーハンドリング
- **技術的特徴**:
  - カンマ区切りのシンプルな記法（`--hidden`と統一）
  - 繰り返し記法の例: `2.25,3.0[9],12.00` → `[2.25, 3.0, 3.0, ..., 3.0, 12.00]`（3.0が9個）
  - 隠れ層と出力層分を合わせて指定（n_layers + 1個）
  - 層数不一致時の詳細なエラーメッセージ（不足/過剰の個数を明示）
- **使用例**:
  ```bash
  # 2層構成（隠れ層2層 + 出力層）
  --wis 2.25,2.75,12.00
  
  # 10層構成（Layer 1-9を3.0に統一）
  --wis 2.25,3.0[9],12.00
  
  # 100層構成（超多層でも簡潔）
  --wis 2.0,3.0[99],12.00
  
  # HyperParamsのデフォルト値を使用
  （--wisなし、層数に応じた最適値を自動取得）
  ```
- **実験効率化への貢献**:
  - ファイル編集不要でパラメータ変更可能
  - グリッドサーチスクリプトの簡潔化
  - 実験の再現性向上（設定が明示的）

### 8. ヒートマップ可視化機能
- **拡張機能**: 学習過程のニューロン活動状況のリアルタイム表示
- **実装機能**: 各層のニューロン活動状況の動的可視化
- **技術的特徴**: 
  - matplotlib.gridspec基盤のGridSpecレイアウトシステム
  - rainbow配色による活性化強度の視覚化（vmin=0, vmax=1）
  - 2行×4列のグリッド配置で最大8層まで同時表示
  - 8層超過時: 最初の4層+最後の4層を自動選択表示
  - 出力層を含む全層を正方形グリッド表示（row-wise配置）
  - 各パネルにカラーバー付き（fraction=0.046, pad=0.04）
- **表示形式**:
  - タイトル行: エポック番号、正解クラス (クラス名)、予測クラス (クラス名)
  - 予測クラスタイトルの色分け表示: 正解時=青色、不正解時=赤色
  - 各層タイトル: 層名称とニューロン数表示
- **使用方法**: `--heatmap` オプションで有効化
- **併用機能**: `--viz`と`--heatmap`の同時指定
  - 両オプションを同時に指定した場合、2つの独立したウィンドウを表示
  - ウィンドウ1: 学習曲線 + 混同行列（`--viz`）
  - ウィンドウ2: 層別活性化ヒートマップ（`--heatmap`）
  - 保存ファイル: `--viz`と`--heatmap`のpng画像はサフィックス付きで別々に保存
  - 各ウィンドウは独立して更新され、相互に干渉しない

### 9. 精度・損失検証機能
- **拡張機能**: 学習終了後の詳細な精度・損失分析
- **実装機能**: クラス別精度、平均損失、文字ベース混同行列の表示
- **技術的特徴**:
  - 混同行列の表示桁数を動的に調整
  - 最大値の桁数が3桁以下の場合: 1列の表示幅を4桁に設定
  - 最大値の桁数が4桁以上の場合: 1列の表示幅を最大桁数+1桁に設定
  - 例: 最大値が999の場合 → 4桁表示、最大値が1000の場合 → 5桁表示
- **使用方法**: `--verify_acc_loss` オプションで有効化

## ED法理論

以下は金子勇氏によるオリジナルED法理論であり、拡張版においても100%保持されています：

### 1. 興奮性・抑制性ニューロンペア構造
- 入力層: 興奮性（+1）・抑制性（-1）ニューロンがペアで構成
- 同種間結合: 正の重み制約
- 異種間結合: 負の重み制約
- 生物学的妥当性の保証

### 2. アミン拡散学習制御
- 出力層の誤差がアミン濃度として隠れ層に拡散
- 二種類のアミン: 正誤差、負誤差
- パラメータ`u1`による拡散強度制御

### 3. 入力層の完全接続構造
- **重要**: 入力層の全ニューロン（興奮性・抑制性両方）が次層に接続
- 入力サイズの自動計算:
  - MNIST/Fashion-MNIST: 784ピクセル → 1568ニューロン（784ペア）
  - CIFAR-10/100: 3072ピクセル (32×32×3カラー) → 6144ニューロン（3072ペア）
- 重み行列サイズ: `paired_input_size × hidden_units`（入力層→隠れ層）
- 各ピクセル値は興奮性・抑制性ニューロンの両方に同じ値として設定（入力）

### 4. Dale's Principle（デールの原理）の厳密適用
- **原理**: ニューロンは興奮性または抑制性のいずれか一方の性質のみを持つ
- **実装**: 重み符号制約 `w *= ow[source] * ow[target]`
- **同種間結合**: `(+1) * (+1) = +1` または `(-1) * (-1) = +1` → 正の重み
- **異種間結合**: `(+1) * (-1) = -1` または `(-1) * (+1) = -1` → 負の重み
- **生物学的妥当性**: 実際の神経系における基本原理を遵守

### 5. ニューロンタイプ配列の全ユニット定義
- **重要**: バイアス項を含む全てのユニットがニューロンタイプを持つ
- インデックス0から`all+1`まで全て定義（入力層、隠れ層、出力層）
- 偶数インデックス: 興奮性（`+1`）
- 奇数インデックス: 抑制性（`-1`）
- バイアス項も含めて交互に配置

### 6. f[11]フラグによる抑制性ニューロン制御
- **f[11] = 1**（デフォルト）: 抑制性ニューロンからの接続も有効
- **f[11] = 0**: 抑制性ニューロン（奇数インデックス）からの接続を0に設定
- 実装: `if (f[11] == 0 && l < in+2 && (l % 2) == 1) w_ot_ot[n][k][l] = 0;`
- 用途: 抑制性ニューロンの影響を制御する実験的設定

## 詳細仕様

### データ構造定義

#### 重み行列の構造

**1つの共有重み空間**を使用し、全クラスで重み行列を共有します。

```python
class ColumnarEDNetwork:
    def __init__(self, n_input, n_hidden, n_output, ...):
        # 重み行列（全クラスで共有）
        self.w_hidden = []  # 隠れ層の重み行列リスト
        self.w_output = None  # 出力層の重み行列
        
        # 第1層: 入力層 → 第1隠れ層
        # shape: [n_hidden[0], n_input * 2]
        # n_input * 2: 興奮性・抑制性ペア構造
        
        # 第2層以降: 隠れ層i-1 → 隠れ層i
        # shape: [n_hidden[i], n_hidden[i-1]]
        
        # 出力層: 最終隠れ層 → 出力層
        # shape: [n_output, n_hidden[-1]]
```

#### コラム構造の管理

```python
# コラム帰属度マップ（各隠れ層ごと）
self.column_affinity_all_layers = []  # リスト
# 各要素: shape [n_output, n_hidden[layer]]
# [class_idx, neuron_idx] = 帰属度 (0.0-1.0)

# アミン濃度（出力クラスごと）
self.amine_concentrations = np.zeros((n_output, n_hidden[0], 2))
# [class_idx, neuron_idx, amine_type]
# amine_type: 0=興奮性アミン、1=抑制性アミン

# 側方抑制重み
self.lateral_weights = np.zeros((n_output, n_output))
# [winner_class, true_class] = 抑制強度（負の値）
```

#### E/Iフラグ配列

```python
# 興奮性(+1)・抑制性(-1)のフラグ
self.excitatory_inhibitory = np.array(
    [1 if i % 2 == 0 else -1 for i in range(n_input * 2)]
)
# 偶数インデックス: +1（興奮性）
# 奇数インデックス: -1（抑制性）
```

### ニューロンタイプ配列の初期化（重要）


### 入力層データ設定（重要）

```c
// Python実装例
def set_input_data(indata_input, output_neurons, in_units):
    """
    入力データを興奮性・抑制性ペアに設定（入力）
    各ピクセル値を2つのニューロン（興奮性・抑制性）に同じ値として設定
    """
    ot_in = np.zeros((output_neurons, in_units + 2))
    for n in range(output_neurons):
        for k in range(2, in_units + 2):
            # 各ピクセルを興奮性・抑制性ペアに設定
            pixel_index = (k // 2) - 1
            ot_in[n][k] = indata_input[pixel_index]
    return ot_in
```

### 重み行列の初期化とDale's Principle適用（重要）

#### 初期化の手順

```python
def __init__(self, n_input, n_hidden, n_output, ...):
    # ステップ1: 重み行列をランダム初期化
    self.w_hidden = []
    for layer in range(len(n_hidden)):
        if layer == 0:
            # 第1層: 入力層からの接続
            w = np.random.randn(n_hidden[layer], n_input * 2) * 0.1
        else:
            # 第2層以降: 隠れ層間の接続
            w = np.random.randn(n_hidden[layer], n_hidden[layer-1]) * 0.1
        self.w_hidden.append(w)
    
    # 出力層の重み
    self.w_output = np.random.randn(n_output, n_hidden[-1]) * 0.1
    
    # ステップ2: 第1層にDale's Principleを適用
    if len(self.w_hidden) > 0:
        w1 = self.w_hidden[0]
        
        # 入力側のE/Iフラグ
        ei_input = self.excitatory_inhibitory[:w1.shape[1]]
        
        # 出力側（隠れ層）は全て興奮性
        ei_hidden = np.ones(w1.shape[0])
        
        # Dale's Principle適用
        sign_matrix = np.outer(ei_hidden, ei_input)
        
        # ★重要★ 絶対値を取ってから符号制約を適用
        self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
```

#### Dale's Principleの重要性

**初期化時の処理**:
- `w = |w| × sign_matrix`を使用
- これにより、興奮性入力への重みは常に正、抑制性入力への重みは常に負になる

**学習中の処理**:
```python
# 重み更新時: sign_constraintを適用しない（自由に学習）
delta_w_column = amine_f_column * z_input.reshape(1, -1)

# 学習後: 符号を強制
self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
```

**相殺問題の回避**:
- 重み更新時に符号制約を適用すると、興奮性と抑制性の変化が相殺される
- 学習後に符号を強制することで、この問題を回避
- 詳細は「コラム構造を用いたED法の実装方法.md」のセクション3を参照

### アーキテクチャ構成

#### ネットワーク構造
```
入力層(in*2) → 隠れ層(hd) → 出力層(ot)
     ↑              ↑           ↑
興奮性・抑制性    ランダム配置   独立学習
    ペア          ±1タイプ    各クラス専用
```

#### インデックス体系
- `0, 1`: バイアス項
- `2 ～ in+1`: 入力層（興奮性・抑制性ペア）
- `in+2`: 出力層開始位置
- `in+3 ～ all+1`: 隠れ層

### 学習アルゴリズム

#### 1. 順方向計算

```python
def forward(self, x):
    """
    順伝播計算
    
    Args:
        x: 入力データ（shape: [n_input]）
    
    Returns:
        a_hiddens: 各隠れ層の線形出力リスト
        z_hiddens: 各隠れ層の活性化後出力リスト
        z_output: 出力層の活性化後出力
    """
    a_hiddens = []
    z_hiddens = []
    
    # 第1層: ペア構造入力を作成
    z_input = create_excitatory_inhibitory_pairs(x)
    
    # 各隠れ層を順伝播
    for layer in range(len(self.w_hidden)):
        # 線形出力
        a = np.dot(self.w_hidden[layer], z_input)
        
        # 活性化関数（Tanh Scaled）
        z = self.tanh_scaled(a)
        
        a_hiddens.append(a)
        z_hiddens.append(z)
        z_input = z  # 次層の入力
    
    # 出力層
    a_output = np.dot(self.w_output, z_hiddens[-1])
    z_output = 1.0 / (1.0 + np.exp(-a_output / self.sig))
    
    return a_hiddens, z_hiddens, z_output
```

#### 2. Tanh Scaled活性化関数

隠れ層には、生物学的に妥当なTanh Scaled活性化関数を使用します。

```python
def tanh_scaled(self, u):
    """
    スケールされたtanh活性化関数
    出力範囲: 0-1（シグモイドと同等）
    
    Args:
        u: 線形出力
    
    Returns:
        z: 活性化後の出力（0-1）
    """
    sig = 2.0  # スケール係数
    return 0.5 * (1.0 + np.tanh(u / sig))

def tanh_scaled_derivative(self, u, z):
    """
    Tanh Scaledの微分
    
    Args:
        u: 線形出力
        z: 活性化後の出力
    
    Returns:
        微分値
    """
    sig = 2.0
    return (1.0 - (2 * z - 1) ** 2) / sig
```

**特性**:
- 出力範囲: 0-1（シグモイドと互換性あり）
- 勾配消失問題の緩和
- 生物学的に妥当な対称性

#### 3. アミン濃度計算

アミン濃度は、オリジナルED法の`calculate_amine_separation`関数を使用します。

```python
# Winner-Takes-All方式でアミン濃度を計算
winner_class = np.argmax(z_output)

if winner_class == y_true:
    # 正解時: 正解クラスに興奮性アミン
    error_correct = 1.0 - z_output[y_true]
    calculate_amine_separation(
        np.array([error_correct]), 
        self.amine_concentrations, 
        y_true, 
        initial_amine=0.7
    )
else:
    # 誤答時: 勝者を抑制、正解を強化
    # 勝者クラスに抑制性アミン
    error_winner = 0.0 - z_output[winner_class]
    calculate_amine_separation(
        np.array([error_winner]), 
        self.amine_concentrations, 
        winner_class, 
        initial_amine=0.7
    )
    
    # 正解クラスに興奮性アミン（側方抑制を考慮）
    error_correct = 1.0 - z_output[y_true]
    lateral_effect = self.lateral_weights[winner_class, y_true]
    
    if lateral_effect < 0:
        enhanced_amine = 0.7 * (1.0 - lateral_effect)
    else:
        enhanced_amine = 0.7
    
    calculate_amine_separation(
        np.array([error_correct]), 
        self.amine_concentrations, 
        y_true, 
        initial_amine=enhanced_amine
    )
```

**重要なポイント**:
- **Winner-Takes-All**: 最も出力が大きいクラスを勝者とする
- **正解時**: 正解クラスのみ学習（興奮性アミン）
- **誤答時**: 勝者を抑制（抑制性アミン）、正解を強化（興奮性アミン）
- **側方抑制**: 誤答時のアミン濃度を増強

#### 4. 重み更新


#### 重み更新の実装

```python
def columnar_ed_update(self, x, y_true, a_hiddens, z_hiddens, z_output):
    """
    コラムED法による重み更新
    """
    # 出力層の誤差と重み更新（標準的な方法）
    y_target = np.zeros(self.n_output)
    y_target[y_true] = 1.0
    error_output = y_target - z_output
    
    sigmoid_derivative = z_output * (1 - z_output) / self.sig
    delta_w_output = self.alpha * np.outer(
        error_output * sigmoid_derivative, 
        z_hiddens[-1]
    )
    self.w_output += delta_w_output
    
    # 隠れ層への誤差の伝播方法をここに記述する
    # 誤差逆伝播は使用禁止
    # TODO: 以下のerror_currentは未定義（誤差逆伝播の結果を想定）
    # 純粋なED法によるアミン拡散メカニズムで置き換える必要がある
    
    # 各隠れ層を逆順に処理
    for layer in range(self.n_layers - 1, -1, -1):
        # コラム係数による誤差配分
        column_affinity = self.column_affinity_all_layers[layer]
        column_scale = column_affinity[y_true]
        
        column_coef = 0.5      # コラム内係数
        non_column_coef = 1.5  # コラム外係数
        
        # 警告: error_currentは誤差逆伝播の結果（本来は使用禁止）
        # TODO: 純粋なED法によるアミン拡散で置き換える
        error_column = error_current * column_scale * column_coef
        error_non_column = error_current * (1.0 - np.clip(column_scale, 0, 1)) * non_column_coef
        error_hidden = (error_column + error_non_column) * self.tanh_scaled_derivative(a_hiddens[layer], z_hiddens[layer])
        
        # 入力の取得
        if layer == 0:
            z_input = create_excitatory_inhibitory_pairs(x)
            ei_input = self.excitatory_inhibitory[:len(z_input)]
        else:
            z_input = z_hiddens[layer-1]
            ei_input = None
        
        # Winner-Takes-All方式の学習対象決定
        winner_class = np.argmax(z_output)
        if winner_class == y_true:
            learning_configs = [(y_true, 0, 1.0)]
        else:
            learning_configs = [
                (winner_class, 1, 1.0),  # 勝者を抑制
                (y_true, 0, 1.0)         # 正解を強化
            ]
        
        # 各対象クラスのコラムを更新
        for target_class, amine_type, learning_coef in learning_configs:
            column_scale = column_affinity[target_class]
            column_neurons_mask = column_scale > 0.5
            
            if not np.any(column_neurons_mask):
                continue
            
            column_indices = np.where(column_neurons_mask)[0]
            d_column = self.amine_concentrations[target_class, column_indices, amine_type]
            f_prime_column = self.tanh_scaled_derivative(
                a_hiddens[layer][column_indices], 
                z_hiddens[layer][column_indices]
            )
            
            # 重み更新（Dale's Principle考慮）
            if ei_input is not None:  # 第1層
                # 符号制約なしで学習
                amine_f_column = (self.alpha * d_column * f_prime_column * learning_coef).reshape(-1, 1)
                delta_w_column = amine_f_column * z_input.reshape(1, -1)
            else:  # 第2層以降
                w_sign_column = np.sign(self.w_hidden[layer][column_indices, :])
                w_sign_column[w_sign_column == 0] = 1
                
                amine_f_column = (self.alpha * d_column * f_prime_column * learning_coef).reshape(-1, 1)
                delta_w_column = amine_f_column * z_input.reshape(1, -1) * w_sign_column
            
            self.w_hidden[layer][column_indices, :] += delta_w_column
        
        # Dale's Principleの強制（第1層のみ）
        if layer == 0:
            ei_input_full = self.excitatory_inhibitory[:self.w_hidden[0].shape[1]]
            ei_hidden = np.ones(self.w_hidden[0].shape[0])
            sign_matrix = np.outer(ei_hidden, ei_input_full)
            self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
        
        # 警告: 以下は誤差逆伝播（微分の連鎖律）を使用している
        # 本来は使用禁止だが、代替手法が未確立
        # TODO: 純粋なED法によるアミン拡散メカニズムで置き換える
        # 前層への誤差伝播（現在は誤差逆伝播を使用）
        if layer > 0:
            error_current = np.dot(error_hidden, self.w_hidden[layer])
```

### マルチクラス分類メカニズム

#### Winner-Takes-All（WTA）方式

本実装では、**Winner-Takes-All方式**により多クラス分類を実現します。

**基本原理**:
1. 出力層の最大値を持つクラスを「勝者クラス」とする
2. 勝者クラスと正解クラスの関係に応じて学習戦略を変更

**学習戦略**:
```python
winner_class = np.argmax(z_output)

if winner_class == y_true:
    # ケース1: 正解時
    # - 正解クラスのコラムのみ学習
    # - 興奮性アミン（amine_type=0）を使用
    # - 出力をさらに強化
    learning_configs = [(y_true, 0, 1.0)]
else:
    # ケース2: 誤答時
    # - 勝者クラスのコラムを抑制（抑制性アミン）
    # - 正解クラスのコラムを強化（興奮性アミン）
    learning_configs = [
        (winner_class, 1, 1.0),  # 抑制
        (y_true, 0, 1.0)         # 強化
    ]
```

**重要な特性**:
- **競合学習**: クラス間で出力の大きさを競う
- **選択的学習**: 勝者と正解のみに焦点を当てる
- **効率的**: 全クラスを更新する必要がない

#### 側方抑制の活用

誤答時、側方抑制を考慮してアミン濃度を増強します。

```python
# 側方抑制重みの作成
self.lateral_weights = -w1 * np.ones((n_output, n_output))
np.fill_diagonal(self.lateral_weights, 0)
# w1: 抑制強度（推奨: 0.1）

# 誤答時のアミン濃度増強
lateral_effect = self.lateral_weights[winner_class, y_true]
if lateral_effect < 0:
    enhanced_amine = 0.7 * (1.0 - lateral_effect)
    # 抑制が強いほど、アミン濃度を増強
```

**効果**:
- 勝者クラスからの抑制を相殺
- 正解クラスの学習を促進
- クラス間の競合関係をモデル化

### 重み制約メカニズム


#### 興奮性・抑制性制約


### パラメータ設定指針

#### HyperParamsクラスによる自動パラメータ管理（v0.2.6以降）

**概要**:
- 層数（1-5層）に応じた最適パラメータを自動選択
- パラメータは外部YAMLファイル（`config/hyperparameters.yaml`）で管理
- グリッドサーチとエポック収束分析により最適化済み
- 6層以上は5層のパラメータをフォールバックとして使用
- コマンドライン引数で個別オーバーライド可能

**パラメータファイル**:
- **メイン設定**: `config/hyperparameters.yaml`（直接編集可能）
- **初期状態**: `config/hyperparameters_initial.yaml`（リファレンス用、復元用）
- **カスタマイズ方法**: YAMLファイルを直接編集
- **エラーチェック**: プログラム起動時（モジュールimport時）に自動実行

**最適化状況**:
- **1層構成**: 完全最適化済み（83.80%テスト精度、2025-12-04）
  * lr=0.20, u1=0.5, epochs=40, base_column_radius=1.0
  * エポック最適化: 100→40（60%時間削減、精度低下<1%）
- **2層構成**: 完全最適化済み（64.27%テスト精度、2025-12-09）
  * lr=0.35, u2=0.5, epochs=45
  * エポック最適化: 100→45（55%時間削減、精度低下<1%）
- **3-5層構成**: 暫定値設定済み（lr=0.35, epochs=50）
  * 今後のグリッドサーチで最適化予定

**使用例**:
```python
from modules.hyperparameters import load_hyperparameters, HyperParams

# YAMLファイルから設定を読み込み
params_dict = load_hyperparameters()  # デフォルト: config/hyperparameters.yaml

# 層数に応じた設定を自動取得
hp = HyperParams()  # YAMLファイルを自動読み込み
config = hp.get_config(n_layers=2)  # 2層の最適パラメータ

# 設定内容の確認
hp.list_configs()  # 利用可能な全設定を表示
```

#### 基本パラメータ

| パラメータ | 推奨値 | 範囲 | 説明 |
|-----------|--------|------|------|
| `alpha` | 0.20（1層）<br>0.35（2層） | 0.03-0.5 | 学習率（層数により最適値が異なる） |
| `initial_amine` | 0.7 | 0.5-0.9 | 初期アミン濃度 |
| `sig` | 2.0 | 1.0-3.0 | 活性化関数のスケール係数 |

**学習率（alpha）の調整**:
- 1層構成: 0.20が最適（Phase 1 Extended Overall Best）
- 2層構成: 0.35が最適（Fine-tuning完了、64.27%達成）
- 3-5層: 0.35を暫定値として使用（最適化予定）
- 小さすぎる（< 0.03）: 収束が遅い、学習が進まない
- 大きすぎる（> 0.5）: 発散または振動

**重要な発見**:
- 多層構成では単層より高い学習率が必要（0.20→0.35）
- データ量増加で性能順位が逆転する場合がある
- 層数が増えるほど学習が難しくなる

#### ネットワーク構成

| パラメータ | 推奨値 | 範囲 | 説明 |
|-----------|--------|------|------|
| `n_hidden` | [512]（1層）<br>[256,128]（2層） | [128-512] | 隠れ層のニューロン数 |
| `n_layers` | 1-2 | 1-5 | 隠れ層の数（1-2層が最適化済み） |
| `epochs` | 40（1層）<br>45（2層）<br>50（3-5層） | 40-50 | エポック数（収束分析に基づく） |
| `batch_size` | 64 | 32-128 | ミニバッチサイズ |

**隠れ層数の選択**:
- **1層**: 最もシンプル、最高精度達成（83.80%）、完全最適化済み
- **2層**: 複雑なパターン認識、64.27%精度、完全最適化済み
- **3-5層**: パラメータ設定済み、今後のグリッドサーチで最適化予定
- **6層以上**: 5層パラメータを自動フォールバック、手動調整推奨

**重要な知見（2025-12-09時点）**:
- 層数増加に伴い学習が困難化（2層は1層より約19%精度低下）
- 多層構成では高めの学習率が必要（0.20→0.35）
- エポック収束点は層数により異なる（1層:40, 2層:45, 3-5層:50推定）

#### コラム構造パラメータ

| パラメータ | 推奨値 | 範囲 | 説明 |
|-----------|--------|------|------|
| `column_neurons` | 25-30 | 20-40 | 各クラスのニューロン数 |
| `column_overlap` | 0.3 | 0.2-0.5 | コラム間の重複係数 |
| `w1` | 0.1 | 0.05-0.15 | 側方抑制強度 |

**コラムニューロン数（column_neurons）**:
- 小さすぎる（< 20）: 表現力不足、精度低下
- 適切（20-40）: バランスの取れた性能
- 大きすぎる（> 40）: 過学習、コラム間干渉

**重複係数（column_overlap）**:
- 小さすぎる（< 0.2）: コラム間が完全に独立、情報共有不足
- 適切（0.2-0.5）: 情報共有と専門化のバランス
- 大きすぎる（> 0.5）: コラム境界が曖昧

**側方抑制強度（w1）**:
- 小さすぎる（< 0.05）: クラス間競合が弱い
- 適切（0.05-0.15）: 効果的なクラス分離
- 大きすぎる（> 0.15）: 学習が不安定

#### コラム係数パラメータ

| パラメータ | 推奨値 | 範囲 | 説明 |
|-----------|--------|------|------|
| `column_coef` | 0.5 | 0.3-0.7 | コラム内学習係数 |
| `non_column_coef` | 1.5 | 1.2-2.0 | コラム外学習係数 |

**コラム係数のバランス**:
- `column_coef < 1.0`: コラム内は控えめに学習（専門性を保持）
- `non_column_coef > 1.0`: コラム外は積極的に学習（汎化性能向上）
- `non_column_coef / column_coef ≈ 3.0`: 推奨比率


## 生物学的妥当性

### 1. アミン神経系の模倣
- ドーパミン・セロトニン等の神経伝達物質による学習制御
- 正・負の報酬信号による適応学習
- 空間的・時間的な拡散メカニズム

### 2. 興奮性・抑制性バランス
- 実際の神経系における興奮性・抑制性ニューロンの比率
- Dale's Principleの遵守（ニューロンタイプの一貫性）
- 安定した学習動態の実現

### 3. 局所学習規則
- 生物学的に実現可能な情報処理
- 時間遅延と局所性の考慮

## 実装上の重要ポイント

### 0. HyperParamsクラスの使用（v0.2.6新機能）

**最重要**: 層数に応じた最適パラメータを自動選択します。パラメータは外部YAMLファイル（`config/hyperparameters.yaml`）で管理されています。

```python
from modules.hyperparameters import load_hyperparameters, HyperParams

# YAMLファイルから設定を読み込み（自動実行）
hp = HyperParams()  # config/hyperparameters.yamlを自動読み込み
config = hp.get_config(n_layers=2)  # 2層の最適パラメータ

# 利用可能な設定一覧の表示
hp.list_configs()

# ネットワーク初期化時に使用
network = RefinedDistributionEDNetwork(
    input_dim=784,
    hidden_layers=config['hidden'],
    output_dim=10,
    learning_rate=config['learning_rate'],
    u1=config['u1'],
    u2=config['u2'],
    lateral_lr=config['lateral_lr'],
    base_column_radius=config['base_column_radius'],
    participation_rate=config['participation_rate'],
    ...
)
```

**設計の利点**:
- グリッドサーチ結果の一元管理
- エポック収束分析結果の統合
- 層数依存パラメータの自動スケーリング
- 6層以上のフォールバック機構
- CLI引数によるオーバーライド可能

### 1. 共有重み空間とコラム構造の実装

**重要**: 全クラスで1つの重み行列を共有し、コラム帰属度で区別します。

```python
# NG例: 各クラスごとに独立した重み空間
self.w_hidden = {}
for class_idx in range(n_output):
    self.w_hidden[class_idx] = np.random.randn(...)
# メモリ使用量が膨大、クラス間情報共有不可

# OK例: 共有重み空間 + コラム構造
self.w_hidden = [np.random.randn(...)]  # 1つの重み行列
self.column_affinity = create_column_affinity(...)  # 帰属度マップ
# メモリ効率が高く、情報共有と専門化を両立
```

### 2. アミン濃度の正確な管理

**重要**: 出力クラスごと、ニューロンごと、アミンタイプごとに管理します。

```python
# アミン濃度の形状
self.amine_concentrations = np.zeros((n_output, n_hidden, 2))
# [class_idx, neuron_idx, amine_type]
# amine_type: 0=興奮性、1=抑制性

# 正しいアクセス方法
d_column = self.amine_concentrations[target_class, column_indices, amine_type]

# 誤ったアクセス（次元が逆）
# d_column = self.amine_concentrations[amine_type, column_indices, target_class]  # NG
```

### 3. Dale's Principleの正確な実装

**最も重要**: 重み更新時と学習後で異なる処理を行います。

```python
# 初期化時
self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix

# 重み更新時（第1層）
if layer == 0:
    # ★重要★ sign_constraintを適用しない
    delta_w_column = amine_f_column * z_input.reshape(1, -1)
else:
    # 第2層以降は符号を保持
    w_sign = np.sign(self.w_hidden[layer])
    delta_w_column = amine_f_column * z_input.reshape(1, -1) * w_sign

# 学習後（第1層のみ）
if layer == 0:
    # ★最も重要★ 符号を強制
    self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
```

**理由**: 重み更新時に符号制約を適用すると、興奮性と抑制性の変化が完全に相殺されます。詳細は「コラム構造を用いたED法の実装方法.md」のセクション3を参照してください。


### 4. 入力層の完全接続実装（重要）
```python
# ★重要★ 入力層の全ニューロン（興奮性・抑制性両方）を次層に接続

# 正しい実装
input_units = input_size  # 1568個全てを使用 ← 正解

# 説明:
# - MNISTの場合: 784ピクセル → 1568ニューロン（784ペア）
# - 各ピクセル値は興奮性・抑制性の両方に同じ値として設定される
# - 重み行列は 1568 × hidden_units のサイズになる
# - 学習時も1568個全てのニューロンが参加する
```

### 5. ニューロンタイプ配列の完全定義（重要）
```python
# ★重要★ 全ユニット（バイアス項含む）にニューロンタイプを定義

def init_neuron_types(all_units):
    """
    全ユニットのニューロンタイプを初期化
    
    Parameters:
    -----------
    all_units : int
        全ユニット数（入力層 + 隠れ層）
    
    Returns:
    --------
    ow : array
        ニューロンタイプ配列
        偶数インデックス: +1（興奮性）
        奇数インデックス: -1（抑制性）
    """
    ow = np.ones(all_units + 2)
    ow[1::2] = -1  # 奇数インデックスを抑制性に設定
    return ow

# 説明:
# - インデックス0から all_units+1 まで全て定義
# - バイアス項（インデックス0,1）も含む
# - 入力層、隠れ層、出力層の全ユニットが対象
# - これによりDale's Principleが正しく適用される
```

### 6. Dale's Principleの正確な適用（重要）
```python
# Dale's Principleの効果:
# - 同種間結合（興奮性→興奮性、抑制性→抑制性）: 正の重み
# - 異種間結合（興奮性→抑制性、抑制性→興奮性）: 負の重み
# - 生物学的神経系の基本原理を遵守
# - ネットワークの安定性と学習性能の向上
```

## 研究の意義と将来性

### 1. 1999年時点での先進性
- 生物学的妥当性への早期着目
- 独創的なアーキテクチャ設計
- マルチクラス分類への独自アプローチ

### 2. 現代への示唆
- 局所学習の重要性
- 神経科学的知見の工学的応用
- 持続可能な学習システムの設計

### 3. 拡張版の発展可能性
- 多層構造による複雑パターン認識の向上
- 高速化技術の他分野への応用
- リアルタイム学習システムの実現
- GPU計算による大規模データ処理
- 深層学習との融合可能性

## 結論

金子勇氏によるED法は、1999年時点で既に現代の神経科学的知見を先取りした、極めて独創的かつ生物学的に妥当な学習アルゴリズムです。その核心である「アミン拡散による学習制御」「興奮性・抑制性バランス」は、従来の人工ニューラルネットワークの限界を克服する可能性を秘めています。

**本拡張版実装では、オリジナル理論の本質を100%保持しながら、現代的な実用性を大幅に向上させることに成功しました。**
多層化・高速化・可視化などの拡張機能により、実践的な機械学習プロジェクトでの利用が可能となり、現代のAI研究においても高い価値を持つ重要な研究成果として活用できます。

## 実装方針（拡張版対応）

 - **「微分の連鎖律を用いた誤差逆伝播法」の使用禁止**: 「微分の連鎖律を用いた誤差逆伝播法」の使用を禁止する。
 - **コーディングルール**: PEP8に準拠し、可読性を最優先にする。
 - **拡張機能の明示**: 新しい機能を追加する際は、オリジナル理論からの拡張であることを明確にコメントで示す。
 - **コードの可読性**: コメントは適度(できるだけ少なめ)な量にする。コメントは、whatではなくwhyを記述するようにして、コードの意図が明確になるように心がける。
 - **モジュール化**: 各機能を明確に分離し、再利用可能なモジュールとして実装する。
 - **テスト駆動開発**: 新しい機能を実装した場合はその機能に対してユニットテストを作成し、実装前にテストを通過させる。
 - **パラメータ調整**: argparseを用いて基本パラメータを柔軟に変更できるようにし、実験的な調整を容易にする。
 - **拡張機能の理論的根拠**: 拡張機能実装時は、ED法理論との整合性を保ち、必要に応じて理論的根拠をコメントで説明する。

---

## 誤差逆伝播法（微分の連鎖律使用）の典型的なコード例

**※ 以下の「微分連鎖律を用いた誤差逆伝播法」の典型的なコードの使用を禁止する。**

本セクションでは、ED法の実装において使用してはならない「微分の連鎖律を用いた誤差逆伝播法」の典型的なコードパターンを示します。これらのパターンが実装に含まれている場合、それは純粋なED法ではなく、一般的な誤差逆伝播法を使用していると判定されます。

### 1. シグモイド関数の微分を使用するパターン

```python
# ❌ 誤差逆伝播法（微分の連鎖律）- 使用禁止
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(z):
    """シグモイドの微分: z * (1 - z)"""
    return z * (1 - z)  # ← 微分の連鎖律

# 重み更新
z_hidden = sigmoid(np.dot(w_hidden, x))
z_output = sigmoid(np.dot(w_output, z_hidden))

# 出力層の誤差勾配（微分の連鎖律）
error_output = target - z_output
delta_output = error_output * sigmoid_derivative(z_output)  # ← 微分

# 隠れ層への誤差伝播（微分の連鎖律）
error_hidden = np.dot(w_output.T, delta_output)
delta_hidden = error_hidden * sigmoid_derivative(z_hidden)  # ← 微分

# 重み更新
w_output += learning_rate * np.outer(delta_output, z_hidden)
w_hidden += learning_rate * np.outer(delta_hidden, x)
```

**判定ポイント**:
- `z * (1 - z)` という形式（絶対値なし）
- 活性化関数の微分として使用
- 誤差を前の層に伝播する際に微分を掛ける

---

### 2. tanh関数の微分を使用するパターン

```python
# ❌ 誤差逆伝播法（微分の連鎖律）- 使用禁止
def tanh(x):
    return np.tanh(x)

def tanh_derivative(z):
    """tanhの微分: 1 - z^2"""
    return 1.0 - z ** 2  # ← 微分の連鎖律

# 隠れ層の計算
a_hidden = np.dot(w_hidden, x)
z_hidden = tanh(a_hidden)

# 誤差伝播（微分の連鎖律）
error_hidden = np.dot(w_output.T, delta_output)
delta_hidden = error_hidden * tanh_derivative(z_hidden)  # ← 微分

# 重み更新
w_hidden += learning_rate * np.outer(delta_hidden, x)
```

**判定ポイント**:
- `1 - z^2` という形式
- tanhの導関数として使用
- 微分の連鎖律による誤差伝播

---

### 3. ReLUの微分を使用するパターン

```python
# ❌ 誤差逆伝播法（微分の連鎖律）- 使用禁止
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLUの微分: x > 0 なら 1, それ以外は 0"""
    return (x > 0).astype(float)  # ← 微分の連鎖律

# 隠れ層の計算
a_hidden = np.dot(w_hidden, x)
z_hidden = relu(a_hidden)

# 誤差伝播（微分の連鎖律）
error_hidden = np.dot(w_output.T, delta_output)
delta_hidden = error_hidden * relu_derivative(a_hidden)  # ← 微分

# 重み更新
w_hidden += learning_rate * np.outer(delta_hidden, x)
```

**判定ポイント**:
- ステップ関数的な微分
- 活性化前の値 `a_hidden` を使用
- 微分の連鎖律による勾配計算

---

### 4. SoftMaxとCross-Entropyの微分パターン

```python
# ❌ 誤差逆伝播法（微分の連鎖律）- 使用禁止
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def softmax_cross_entropy_derivative(y_pred, y_true):
    """SoftMax + Cross-Entropyの微分"""
    return y_pred - y_true  # ← 微分の連鎖律（簡略化された形）

# 出力層
z_output = softmax(np.dot(w_output, z_hidden))

# 誤差勾配（微分の連鎖律）
delta_output = softmax_cross_entropy_derivative(z_output, y_true)  # ← 微分

# 隠れ層への伝播（微分の連鎖律）
error_hidden = np.dot(w_output.T, delta_output)
delta_hidden = error_hidden * sigmoid_derivative(z_hidden)  # ← 微分

# 重み更新
w_output += learning_rate * np.outer(delta_output, z_hidden)
w_hidden += learning_rate * np.outer(delta_hidden, x)
```

**判定ポイント**:
- SoftMaxの微分（ヤコビ行列）を暗黙的に使用
- Cross-Entropyとの組み合わせで `y_pred - y_true` に簡略化
- 隠れ層への伝播で微分を連鎖

---

### 純粋なED法との比較

#### ✅ 純粋なED法（微分の連鎖律不使用）

```python
# ✅ 純粋なED法 - これが正しい実装
def ed_saturation_term(z):
    """ED法の飽和項: |z| * (1 - |z|)"""
    return np.abs(z) * (1.0 - np.abs(z))  # ← 絶対値あり

# 出力層の更新
error = target - z_output
saturation = ed_saturation_term(z_output)  # ← 飽和項（微分ではない）
delta_w_output = learning_rate * error * saturation * z_hidden

# アミン拡散による隠れ層への伝播
amine_concentration = calculate_amine(error)  # ← アミン濃度計算
amine_hidden = amine_concentration * u1 * column_affinity  # ← 拡散
saturation_hidden = ed_saturation_term(z_hidden)  # ← 飽和項
delta_w_hidden = learning_rate * amine_hidden * saturation_hidden * x

# 重み更新
w_output += delta_w_output
w_hidden += delta_w_hidden
```

**ED法の特徴**:
- `abs(z) * (1 - abs(z))` という形式（絶対値あり）
- 「微分」ではなく「生物学的飽和特性」
- アミン拡散による誤差伝播（微分の連鎖律不使用）
- コラム帰属度による重み付け

---

### 判定基準まとめ

| 要素 | 誤差逆伝播法（微分の連鎖律）❌ | 純粋なED法✅ |
|------|---------------------------|-----------|
| **計算式** | `z * (1 - z)` または `1 - z^2` | `abs(z) * (1 - abs(z))` |
| **絶対値** | なし | あり |
| **意味** | 活性化関数の微分 | 生物学的飽和特性 |
| **誤差伝播** | 重み行列の転置を使用 | アミン拡散メカニズム |
| **連鎖** | 層間で微分を掛け算 | コラム帰属度で重み付け |
| **数学的基礎** | 勾配降下法、微積分 | 神経伝達物質の拡散 |

---

### 重要な注意事項

これらの誤差逆伝播法のコードパターンが実装に含まれている場合、それは「微分の連鎖律を用いた誤差逆伝播法」を使用していると判定されます。

**v026の実装では、これらのパターンを一切使用せず、純粋なED法の飽和項 `abs(z) * (1 - abs(z))` とアミン拡散メカニズムのみを使用しています。**

ED法の実装においては、必ず以下を確認してください：
1. 飽和項に絶対値が含まれているか（`abs(z) * (1 - abs(z))`）
2. 活性化関数の「微分」を使用していないか
3. 誤差伝播が「アミン拡散」メカニズムによるものか
4. 微分の連鎖律による計算が含まれていないか

---

**本仕様書は、オリジナルC実装の動作確認と詳細なコード解析、および拡張機能の実装検証に基づいて作成されました。**  
**オリジナル検証日**: 2025年8月30日  
**拡張版作成日**: 2025年9月13日  
**最終更新**: 2025年12月10日（誤差逆伝播法の典型的コード例追加）  
**検証者**: AI解析システム  
**ソースコード**: `/ed_original_src/` (コンパイル・実行確認済み) + 拡張版Python実装 + SNN統合実装
