# ED法にコラム構造を導入したネットワーク学習方法の仕様書

**ED法 原作発案者**: 金子勇（Isamu Kaneko）  
**原作開発年**: 1999年  
**ED法 オリジナルソース**: C言語実装（動作確認済み）

**拡張実装**: 2025年 by yoiwa0714  
**拡張版**: ED法へのコラム構造の導入

## 本プロジェクトの目的

 - 生物学的に妥当なニューラルネットワーク学習アルゴリズムの研究と開発を促進する
 - ED法は生物学的に妥当な理論であるため、金子勇氏オリジナルのED法理論にコラム構造を導入し、オリジナルED法理論を拡張する

### 本プロジェクトの背景

 - ED法は基本的に「二値分類問題には対応できるが、多分類問題は不得手」という特性を持っている
 - この二値分類しかできないという制限のあるED法を使って多クラス分類に対応させたい
 - そのためのアイデアとして、出力クラスと関係を持つコラム構造を設け、コラム構造と二値分類の機能を合わせることで、ED法による多クラス分類ができるようにしたい

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

ED法の基本原理はプロジェクトディレクトリのdocs/ED法_解説資料.mdに詳細に記載されています。
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

#### コラム帰属度マップの作成

コラム構造は、各ニューロンが各クラスにどの程度帰属するかを表す**帰属度マップ**で実装します。

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
    
    for class_idx in range(n_classes):
        # コラムの中心をランダムに選択
        center = np.random.randint(0, n_neurons)
        
        # 各ニューロンとの距離を計算（円環トポロジー）
        distances = np.minimum(
            np.abs(np.arange(n_neurons) - center),
            n_neurons - np.abs(np.arange(n_neurons) - center)
        )
        
        # ガウス型の帰属度を計算
        sigma = column_neurons / 3.0
        affinity = np.exp(-0.5 * (distances / sigma) ** 2)
        
        # 閾値以下を0に設定
        threshold = np.exp(-0.5 * 9)  # 3σ点
        affinity[affinity < threshold] = 0
        
        # 重複を考慮
        affinity *= (1.0 - overlap * np.sum(column_affinity, axis=0))
        affinity = np.clip(affinity, 0, 1)
        
        column_affinity[class_idx, :] = affinity
    
    return column_affinity
```

**重要な特性**：
- **ガウス型分布**: 中心ニューロンほど高い帰属度
- **円環トポロジー**: ニューロン配列の端と端が繋がっている
- **重複制御**: `overlap`パラメータで調整可能
- **各層独立**: 各隠れ層ごとに独立したコラム構造を作成

## 拡張機能一覧（オリジナル理論からの追加機能）

本実装では、金子勇氏のオリジナルED法理論を完全に保持しながら、以下の拡張機能を追加実装しています：

### 1. 多層ニューラルネットワーク対応
- **拡張機能**: 複数隠れ層を自由に組み合わせ可能
- **実装方法**: カンマ区切り指定（例：`--hidden 256,128,64`）
- **技術的特徴**: 層構造を管理するクラスによる動的層管理
- **ED法理論との整合性**: アミン拡散係数u1を多層間に適用

### 2. ミニバッチ学習システム
- **拡張機能**: 複数サンプルをまとめて効率的に処理
- **実装方法**: `--batch` オプションでサイズ指定可能

### 3. NumPy行列演算による劇的高速化
- **拡張機能**: NumPy行列演算による並列計算
- **技術的特徴**: ベクトル化
- **理論保持**: ED法のアルゴリズムに影響を与えない実装にする

### 4. リアルタイム可視化システム
- **拡張機能**: 学習過程のリアルタイムグラフ表示
- **実装機能**: 学習曲線、混同行列、正答率推移の動的可視化
- **技術的特徴**: matplotlibを用いる
- **使用方法**: `--viz` `--save_fig`オプションで有効化
- **学習曲線の軸仕様**:
  - **縦軸**: 最小0.0、最大1.0、目盛り0.0から1.0まで0.1刻み
    - 中間グリッド線: 0.1, 0.3, 0.5, 0.7, 0.9が点線（`:`）
    - 中間グリッド線: 0.2, 0.4, 0.6, 0.8が実線（`-`）
  - **横軸**: 最小1、最大=設定された最大エポック数
    - 目盛り: 横軸を10分割した位置に設定
    - 中間グリッド線: 点線（`:`）と実線（`-`）を交互に配置
    - 例: 最大100エポック → 10,20,30,...,90の目盛り（10=点線、20=実線、30=点線、...）

### 5. 現代的データローダー統合
- **拡張機能**: TensorFlow (tf.keras.datasets) 統合による自動データ処理
- **対応データセット**: MNIST・Fashion-MNIST・CIFAR-10・CIFAR-100の自動ダウンロード
- **技術的特徴**: `--train` `--test` で選択されたデータを全エポックで使用
- **使用方法**: `--mnist`, `--fashion`, `--cifar10`, `--cifar100` オプションでデータセット切替

### 6. GPU計算支援（CuPy対応）
- **拡張機能**: NVIDIA GPU使用時の自動GPU計算
- **技術的特徴**: CuPy統合による透明なGPU処理
- **性能向上**: 大規模データセットでの更なる高速化
- **互換性**: GPU無環境でも自動的にCPU処理に切替

### 7. 詳細プロファイリング機能
- **拡張機能**: 処理段階別の詳細性能分析
- **実装機能**: ボトルネック特定、メモリ使用量監視
- **技術的特徴**: リアルタイム性能監視とレポート生成微分の連鎖律を用いた誤差逆伝播法」の使用禁止
- **使用方法**: `--verbose` オプションで詳細表示

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

#### 基本パラメータ

| パラメータ | 推奨値 | 範囲 | 説明 |
|-----------|--------|------|------|
| `alpha` | 0.05 | 0.03-0.1 | 学習率 |
| `initial_amine` | 0.7 | 0.5-0.9 | 初期アミン濃度 |
| `sig` | 2.0 | 1.0-3.0 | 活性化関数のスケール係数 |

**学習率（alpha）の調整**:
- 小さすぎる（< 0.03）: 収束が遅い、学習が進まない
- 適切（0.03-0.1）: 安定した収束
- 大きすぎる（> 0.1）: 発散または振動

#### ネットワーク構成

| パラメータ | 推奨値 | 範囲 | 説明 |
|-----------|--------|------|------|
| `n_hidden` | [256] | [128-512] | 隠れ層のニューロン数 |
| `n_layers` | 1 | 1-3 | 隠れ層の数 |
| `epochs` | 10-20 | 5-50 | エポック数 |
| `batch_size` | 64 | 32-128 | ミニバッチサイズ |

**隠れ層数の選択**:
- 1層: 最もシンプル、小規模データに適する
- 2層: 複雑なパターン認識が可能
- 3層以上: 過学習のリスクが高い

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

**本仕様書は、オリジナルC実装の動作確認と詳細なコード解析、および拡張機能の実装検証に基づいて作成されました。**  
**オリジナル検証日**: 2025年8月30日  
**拡張版作成日**: 2025年9月13日  
**検証者**: AI解析システム  
**ソースコード**: `/ed_original_src/` (コンパイル・実行確認済み) + 拡張版Python実装 + SNN統合実装
