# ED法 - SNN(スパイキングニューラルネットワーク)の新しい学習法

[**日本語**](README.md) | [English](README_EN.md)

金子勇氏オリジナルError-Diffusion（ED）法をスパイキングニューラルネットワーク（SNN）に適用した実装です。<br>
本実装では、**3つの異なるバージョン**を用意しています。

 1. ed_multi_lif_snn.py         100％のLIFニューロンで構成されるネットワーク
 2. ed_multi_lif_snn_simple.py  基本的な機能のみの実装にしたネットワーク
 3. ed_mulit_frelu_snn.py       活性化関数にFReLUを用いたネットワーク

## ED法の詳細

ED法(Error-Diffusion法)とは、故金子勇氏が1999年に考案された、「微分の連鎖律を用いた誤差逆伝播法」を用いない生物学的に妥当な多層ニューラルネットワークの学習方法です。<br>
ED法の詳細については、[ED法_解説資料.md](docs/ja/ED法_解説資料.md)をご覧ください。

## ED法の技術的メリット

### 🚀 並列計算による高速化

誤差逆伝播法では後ろの層から順次計算が必要ですが、ED法では各層が独立してモノアミン濃度に基づいて重み更新を行うため、**層間の並列計算が可能**です。これにより、特に深いネットワークでの学習速度向上が期待できます。

- 各層が独立してアミン濃度に基づく重み更新
- 誤差逆伝播の順次計算制約を回避
- 深いネットワークでの学習速度向上

### 🛡️ 勾配消失問題の回避

微分の連鎖律を使用しないため、深い層でも**勾配消失問題が発生しません**。アミン拡散による局所学習により、各層が直接的に学習信号を受け取ることができます。

- 微分の連鎖律を使用しない局所学習
- 層の深さに関わらず安定した学習信号
- 生物学的妥当性と実用性の両立

## 概要

本実装は、生物学的な妥当性が低い「微分の連鎖律を用いた誤差逆伝播法」を使用せず、金子勇氏が考案された生物学的な妥当性の高い「ED法」による学習アルゴリズムを用いてMNISTとFashion-MNISTの画像分類を実現します。<br>
なお、本実装ではすべてのニューロンがLIFニューロンで構成されている、完全なスパイキングニューラルネットワークをベースとしています。

## ed_multi_lif_snn.pyによる学習結果例

ed_multi_lif_snn.pyによる、MNISTデータとFashion-MNISTデータの学習結果例を以下に示します。

### MNISTデータでの学習例

<img src="viz_results_for_public/lif_mnist_256_lr0.15_e20/realtime_viz_result_20251102_113203.png" alt="MNIST学習結果" width="60%">

・最高正答率: 87.60%

・学習実行コマンド

```bash
python ed_multi_lif_snn.py --mnist --train 1000 --test 500 --spike_max_rate 150 --spike_sim_time 50 --spike_dt 1.0 --viz --heatmap --save_fig viz_results_for_public/lif_mnist_256_lr0.15_e20 --epochs 20 --hidden 256 --lr 0.15
```

### Fashion-MNISTデータでの学習例

<img src="viz_results_for_public/lif_fashion_256_lr0.15_e20/realtime_viz_result_20251102_113256.png" alt="Fashion-MNIST学習結果" width="60%">

・最高正答率: 78.20%

・学習実行コマンド

```bash
python ed_multi_lif_snn.py --fashion --train 1000 --test 500 --spike_max_rate 150 --spike_sim_time 50 --spike_dt 1.0 --viz --heatmap --save_fig viz_results_for_public/lif_fashion_256_lr0.15_e20 --epochs 20 --hidden 256 --lr 0.15
```

## ed_multi_frelu_snn.pyによる学習結果例

ed_multi_frelu_snn.pyはed_multi_lif_snn.pyの活性化関数を、試験的にFReLUに置き換えたコードです。
ed_multi_frelu_snn.pyによる、Fashion-MNISTデータの学習結果例を以下に示します。

### FReLU版 Fashion-MNISTデータでの学習例

<img src="viz_results_for_public/frelu_snn_fashion_hid2048_128_dif1.5_epo20/realtime_viz_result_20251102_062334.png" alt="FReLU Fashion-MNIST学習結果" width="60%">

・最高正答率: 76.71%

・学習実行コマンド

```bash
python ed_multi_frelu_snn.py --viz --heatmap --fashion --seed 42 --train 2048 --test 2048 --batch 128 --save_fig viz_results_for_public/frelu_snn_fashion_hid2048_128_dif1.5_epo20 --hidden 2048,128 --epochs 20 --dif 1.5
```

## 試行例(参考)

実際の学習試行の詳細な記録とパラメータ設定の参考例を提供しています。

### 📊 LIF版学習試行記録

[**lif_snn_learning_test_results.md**](test_results/lif_snn_learning_test_results.md)

- **完全LIF版の学習結果**: Fashion-MNIST データセットでの複数試行記録
- **最高達成正答率**: 80.83% (隠れ層4096,128,128ニューロン, 30エポック)
- **パラメータ調整例**: 学習率、アミン濃度、隠れ層構造の試行錯誤過程
- **実行設定詳細**: 各試行でのコマンドライン引数と結果の対応関係

### 🚀 FReLU版学習試行記録

[**frelu_snn_learning_test_results.md**](test_results/frelu_snn_learning_test_results.md)

- **FReLU版の実験結果**: ED法とFReLU活性化関数の組み合わせ検証
- **ハイブリッド構成**: 入力層LIF + 隠れ層・出力層FReLUの性能評価
- **最適化実験**: 異なるパラメータでの性能比較と考察
- **実装特性**: FReLU版特有の設定と動作特性の記録

これらの試行記録は、実際にシステムを使用する際のパラメータ選択やトラブルシューティングの参考として活用できます。

## 提供バージョン

### 🧠 LIF版（完全SNN）- `ed_multi_lif_snn.py`

**完全なスパイキングニューラルネットワーク実装 + 実験用機能**

- ✅ **全層LIF化**: すべてのニューロンがLeaky Integrate-and-Fire（LIF）モデル
- ✅ **生物学的妥当性**: 最も現実の脳に近い実装
- ✅ **実験用途**: 多種な引数によるハイパーパラメータ類の容易な変更が可能
- 🎯 **達成精度**: MNIST 85.0%, Fashion-MNIST 82.0%

### 📚 Simple版（実装学習用）- `ed_multi_lif_snn_simple.py`

**ED法やSNNの実装を理解しやすい基本的な機能による実装**

- ✅ **シンプル設計**: コード理解が容易
- ✅ **学習特化**: ED法とSNNの学習に最適
- ✅ **コメント充実**: 詳細な説明付き
- 🎯 **目的**: アルゴリズム理解・学習用

### 🚀 FReLU版（FReLUの実装試験用）- `ed_multi_frelu_snn.py`

**FReLUを試験的に実装**

- ✅ **FReLU活性化**: 隠れ層・出力層にFlexible ReLU活性化関数を適用（FReLUの実装は試験目的）
- ✅ **ハイブリッド構成**: 入力層SNN + FReLU隠れ層・出力層の最適バランス
- ✅ **試験性**: ED法とFReLUの相性の検証試験

## 主要な特徴

- ✅ **純粋ED法**: 「微分の連鎖律を用いた誤差逆伝播法」を用いない生物学的学習
- ✅ **スパイキングニューラルネットワーク**: LIFニューロンによるスパイキングニューラルネットワーク
- ✅ **スパイク符号化**: ポアソン符号化による入力層のスパイク生成
- ✅ **E/Iペア構造**: 興奮性(E)・抑制性(I)ニューロンペアによる生物学的妥当性
- ✅ **Dale's Principle**: ニューロンの重み符号保持（興奮性≥0、抑制性≤0）
- ✅ **多層対応**: 任意の隠れ層構造（単層・多層）をサポート
- ✅ **高速化**: NumPy行列演算による効率的な実装
- ✅ **GPU対応**: CuPy使用時の自動GPU計算
- ✅ **可視化対応**: 正答率/エラー(誤答)率や各層のニューロンの発火状況をリアルタイム表示

## ファイル構成

```text
ed_multi_snn/
├── README.md                      # このファイル
├── README_EN.md                   # 英語版README
├── docs/                          # ドキュメント
│   ├── ja/                        # 日本語ドキュメント
│   │   ├── ed_multi_snn.prompt.md # ED法実装仕様書
│   │   ├── ED法_解説資料.md       # ED法理論解説
│   │   ├── EDLA_金子勇氏.md       # 金子勇氏の功績
│   │   └── PROJECT_OVERVIEW.md    # プロジェクト概要
│   └── en/                        # English Documentation
│       ├── ed_multi_snn.prompt_EN.md
│       ├── ED_Method_Explanation.md
│       ├── EDLA_Isamu_Kaneko.md
│       └── PROJECT_OVERVIEW.md
│
├── src/                           # ソースコード
│   ├── ja/                        # 日本語コメント版
│   │   ├── ed_multi_lif_snn.py        # LIF版メインプログラム
│   │   ├── ed_multi_lif_snn_simple.py # Simple版
│   │   ├── ed_multi_frelu_snn.py      # FReLU版
│   │   └── modules/               # 必要最小限のモジュール群
│   └── en/                        # English commented version
│       ├── ed_multi_lif_snn.py
│       ├── ed_multi_lif_snn_simple.py
│       ├── ed_multi_frelu_snn.py
│       └── modules/
│
└── test_results/                  # 学習結果レポート
    ├── lif_snn_learning_test_results.md    # LIF版学習結果
    └── frelu_snn_learning_test_results.md  # FReLU版学習結果
```

## 基本的な使い方

### 🧠 LIF版（完全SNN）

#### MNIST学習 (--mnistを指定)

```bash
python src/ja/ed_multi_lif_snn.py --mnist --train 1000 --test 100 --epochs 10
```

#### Fashion-MNIST学習 (--fashionを指定)

```bash
python src/ja/ed_multi_lif_snn.py --fashion --train 1000 --test 100 --epochs 10
```

#### 多層構造での学習 (--hiddenで層数とニューロン数を指定)

```bash
python src/ja/ed_multi_lif_snn.py --mnist --train 1000 --test 100 --epochs 10 --hidden 256,128,64
```

#### リアルタイム可視化付き (--vizや --heatmapを指定)

```bash
python src/ja/ed_multi_lif_snn.py --mnist --train 1000 --test 100 --epochs 10 --viz --heatmap
```

### 📚 Simple版（実装学習用）(コード内容を確認して実装方法を学習することが主目的)

#### 基本的な学習

```bash
python src/ja/ed_multi_lif_snn_simple.py --mnist --train 1000 --test 100 --epochs 10
```

#### 詳細ログ付き学習

```bash
python src/ja/ed_multi_lif_snn_simple.py --mnist --train 1000 --test 100 --epochs 10 --verbose
```

### 🚀 FReLU版（FReLUの試験的実装）

#### MNIST学習

```bash
python src/ja/ed_multi_frelu_snn.py --mnist --train 5000 --test 1000 --epochs 20
```

#### Fashion-MNIST学習

```bash
python src/ja/ed_multi_frelu_snn.py --fashion --train 10000 --test 2000 --epochs 30 --hidden 512,256,128
```

### 🛠️ 共通オプション

#### GPU使用 (機能は実装していますが、目立った高速化効果は体感できないかもしれません。)

```bash
# GPU自動検出・使用（CuPyインストール時）
python src/ja/[ファイル名] --mnist --train 1000 --test 100 --epochs 10
```

#### 図表保存付き学習 (--save_figの後に保存先ディレクトリを指定します。ディレクトリが指定されていない場合にはviz_resultsディレクトリ下に保存します。ディレクトリが存在しない場合には作成します。)

```bash
python src/ja/[ファイル名] --mnist --train 1000 --test 100 --epochs 10 --viz --heatmap --save_fig results/
```

#### バッチサイズ調整

```bash
python src/ja/[ファイル名] --mnist --train 1000 --test 100 --epochs 10 --batch_size 256
```

## 必要条件・インストール

### 基本環境

- Python 3.8以上
- pip (Python パッケージマネージャー)

### 必要パッケージ

```bash
pip install numpy tensorflow matplotlib tqdm psutil
```

### GPU使用時（推奨）

```bash
pip install cupy-cuda12x  # CUDA 12.x用
# または適切なCUDAバージョンに対応したcupyパッケージ
```

### バージョン別の特徴

| パッケージ | LIF版 | Simple版 | FReLU版 | 用途 |
|-----------|-------|----------|---------|------|
| numpy | ✅ | ✅ | ✅ | 基本数値計算 |
| tensorflow | ✅ | ✅ | ✅ | データローダー |
| cupy | 🚀 | 🚀 | 🚀 | GPU高速化（推奨） |
| matplotlib | 📊 | 📊 | 📊 | 可視化（--viz時） |

### クイックスタート

```bash
# リポジトリクローン
git clone https://github.com/yoiwa0714/ed_multi_snn.git
cd ed_multi_snn

# 依存関係インストール
pip install numpy tensorflow matplotlib tqdm psutil cupy-cuda12x

# 日本語コメント版のコードを使用（src/ja/）

# LIF版実行（完全SNN）
python src/ja/ed_multi_lif_snn.py --mnist --train 1000 --test 100 --epochs 10

# Simple版実行（学習用）
python src/ja/ed_multi_lif_snn_simple.py --mnist --train 1000 --test 100 --epochs 10

# FReLU版実行（FReLUの実験用実装）
python src/ja/ed_multi_frelu_snn.py --mnist --train 1000 --test 100 --epochs 10
```

> **💡 言語選択**: 
> - **日本語コメント版**: `src/ja/` ディレクトリを使用
> - **English commented version**: Use `src/en/` directory ([English Guide](README_EN.md))

## 主要なコマンドライン引数

### データセット

- `--mnist`: MNISTデータセット使用（デフォルト）
- `--fashion`: Fashion-MNISTデータセット使用

### 学習設定

- `--train N`: 訓練サンプル数（デフォルト: 512）
- `--test N`: テストサンプル数（デフォルト: 512）
- `--epochs N`: エポック数（デフォルト: 10）
- `--hidden N1,N2,...`: 隠れ層構造（デフォルト: 128）
- `--batch N`: ミニバッチサイズ（デフォルト: 128）
- `--seed N`: ランダムシード（デフォルト: ランダム）
- `--no_shuffle`: データシャッフル無効化

### ED法ハイパーパラメータ

- `--lr FLOAT`: 学習率 (alpha) - ニューロンの学習強度を制御（デフォルト: 0.1）
- `--ami FLOAT`: アミン濃度 (beta) - 初期誤差信号の強度（デフォルト: 0.25）
- `--dif FLOAT`: 拡散係数 (u1) - アミン（誤差信号）の拡散率（デフォルト: 0.5）
- `--sig FLOAT`: シグモイド閾値 (u0) - 活性化関数の感度（デフォルト: 1.2）
- `--w1 FLOAT`: 重み初期値1 - 興奮性ニューロンの初期重み（デフォルト: 0.3）
- `--w2 FLOAT`: 重み初期値2 - 抑制性ニューロンの初期重み（デフォルト: 0.5）

### LIFニューロンパラメータ

- `--v_rest FLOAT`: 静止膜電位（デフォルト: -65.0 mV）
- `--v_threshold FLOAT`: 発火閾値（デフォルト: -60.0 mV）
- `--v_reset FLOAT`: リセット電位（デフォルト: -70.0 mV）
- `--tau_m FLOAT`: 膜時定数（デフォルト: 20.0 ms）
- `--tau_ref FLOAT`: 不応期（デフォルト: 2.0 ms）
- `--simulation_time FLOAT`: シミュレーション時間（デフォルト: 50.0 ms）
- `--dt FLOAT`: 時間ステップ（デフォルト: 1.0 ms）
- `--R_m FLOAT`: 膜抵抗（デフォルト: 10.0 MΩ）
- `--spike_encoding METHOD`: スパイク符号化方法（デフォルト: poisson）
- `--spike_max_rate FLOAT`: 最大発火率 Hz（デフォルト: 100.0）
- `--spike_sim_time FLOAT`: スパイクシミュレーション時間 ms（デフォルト: 50.0）
- `--spike_dt FLOAT`: スパイク時間刻み ms（デフォルト: 1.0）

### 可視化

- `--viz`: リアルタイム学習進捗表示
- `--heatmap`: スパイク活動ヒートマップ表示
- `--save_fig DIR`: 図表保存ディレクトリ指定

### その他

- `--cpu`: CPU強制実行（GPU環境でも）
- `--verbose`: 詳細ログ表示
- `--verify_acc_loss`: 精度・誤差の検証レポートを表示

## 必要な環境

### 必須

- Python 3.8以上
- NumPy
- TensorFlow（データセット読み込み用）
- Matplotlib（可視化用）
- tqdm（進捗表示用）
- psutil（性能監視用）

### オプション

- CuPy（GPU高速化用）
- NVIDIA GPU + CUDA（GPU使用時）

## インストール

```bash
# 基本パッケージ
pip install numpy tensorflow matplotlib tqdm psutil

# GPU使用時（オプション）
pip install cupy-cuda11x  # CUDA 11.x用
# または
pip install cupy-cuda12x  # CUDA 12.x用
```

## ed_multi_snn.prompt.md完全準拠

本実装は金子勇氏のオリジナルED法理論を完全に保持しながら、以下の拡張機能を実装しています：

### 1. 純粋ED法の保持

- **重要**: 「微分の連鎖律を用いた誤差逆伝播法」を一切使用しない
- アミン拡散メカニズムによる生物学的学習
- 出力層中心のエラー拡散型重み更新

### 2. E/Iペア構造

- E: 興奮性ニューロン、I: 抑制性ニューロン
- 各入力ピクセル → 興奮性(+1) + 抑制性(-1) ニューロンペア
- MNIST: 784ピクセル → 1568ニューロン（784ペア）
- 生物学的妥当性の保証

### 3. Dale's Principle（デールの原理）

- 同種間結合: 正の重み制約
- 異種間結合: 負の重み制約
- 重み符号制約: `w *= ow[source] * ow[target]`

### 4. 独立出力ニューロンアーキテクチャ

- 各クラスが完全に独立した重み空間を保持
- 3次元重み配列: `w_ot_ot[出力ニューロン][送信先][送信元]`

### 5. 全層LIF化

- 入力層・隠れ層・出力層のすべてがLIFニューロン
- スパイク駆動型の完全なSNN実装
- 生物学的リアリズムの追求

### 6. 多層ニューラルネットワーク対応

- オリジナル仕様: 単一隠れ層
- 拡張機能: 複数隠れ層を自由に組み合わせ可能
- アミン拡散係数u1を多層間に適用

### 7. 高速化・GPU対応

- NumPy行列演算による並列計算
- CuPy統合によるGPU処理
- CPU/GPU自動切り替え

## 開発履歴

- **非公開版**:
- **v019**: 金子勇氏オリジナルED法完全準拠達成
- **v020**: リアルタイム可視化システム完成
- **v021-v023**: GPU最適化、過学習問題修正
- **v024**: GPU/CPU強制実行オプション対応
- **v025**: 全LIF化完成、スパイク符号化最適化、目標正答率達成（85.0%）

- **公開版（2025-10-26）**:
  - 全層LIF化完成版
  - MNIST/Fashion-MNIST専用に最適化
  - ed_multi_snn.prompt.md 100%準拠

## 技術的詳細

### LIFニューロンモデル

膜電位の時間発展:

```python
dV/dt = (V_rest - V + I_syn) / τ_m
```

- V: 膜電位
- V_rest: 静止膜電位
- I_syn: シナプス電流
- τ_m: 膜時定数

発火条件: `V ≥ V_threshold` → スパイク発火 → `V = V_reset`

### ED法学習則

重み更新:

```python
Δw = α × amine × input × output_error
```

- α: 学習率
- amine: アミン濃度（誤差信号強度）
- input: 入力ニューロン活性
- output_error: 出力誤差

アミン拡散:

```python
amine_hidden = u1 × amine_output
```

- u1: 拡散係数（0-1の範囲）

## 参考文献

- 金子勇 (1999): Error-Diffusion法の原論文
- ed_multi_snn.prompt.md: 本実装の設計指針・仕様書
- LIF neuron model: Leaky Integrate-and-Fire ニューロンモデル
- Dale's Principle: 神経伝達物質の単一性原理

## ライセンス

Original ED method by Isamu Kaneko (1999)  
Multi-layer & SNN extension implementation (2025)  
Public Release Version (2025-10-31)

## 開発情報

- **リポジトリ**: <https://github.com/yoiwa0714/ed_multi_snn>
- **作成日**: 2025年10月31日
- **公開バージョン**: 3ファイル統合版
  - LIF版: 完全SNNによる生物学的妥当性追求
  - Simple版: 実装学習用
  - FReLU版: 試験的な実装

## サポート

問題や質問がある場合は、GitHubのIssuesでお知らせください。<br>
可能な範囲内で対応いたします。

## 関連ドキュメント

- 📖 [技術仕様書](docs/ja/ed_multi_snn.prompt.md) - ED法実装仕様とアルゴリズム解説
- 📚 [ED法_解説資料](docs/ja/ED法_解説資料.md) - ED法理論の詳細解説
- 👨‍💻 [金子勇氏について](docs/ja/EDLA_金子勇氏.md) - ED法開発者の功績

---

**🧠 ED-Multi SNN - 生物学的知能から学ぶ次世代AI**
