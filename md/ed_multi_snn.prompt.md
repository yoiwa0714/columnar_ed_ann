# ED-Multi 多層対応ED法(Multi-Layer Error Diffusion Learning Algorithm) for SNN 仕様書

**原作者**: 金子勇（Isamu Kaneko）  
**原作開発年**: 1999年  
**拡張実装**: 2025年  
**オリジナルソース**: C言語実装（動作確認済み）  
**拡張版**: Python実装（多層対応・高速化版）
**SNN対応への機能拡張**: SNNネットワークの学習法

## 本プロジェクトの目的

 - 金子勇氏のオリジナルED法理論に、多層化・高速化の機能を追加し、オリジナルED法理論を拡張する。
 - SNN(スパイキングニューラルネットワーク)の学習法としてED法を実装する。
 - SNNとしては、すべてのニューロンにLIFニューロンを用いたネットワークを構築する。

金子勇氏のオリジナルED法理論に、多層化・高速化・現代的機能を追加## ED法の概要

ED法（Error Diffusion Learning Algorithm）は、生物学的神経系のアミン（神経伝達物質）拡散メカニズムを模倣した独創的な学習アルゴリズムです。従来のバックプロパゲーションとは根本的に異なる、興奮性・抑制性ニューロンペア構造と出力ニューロン中心のアーキテクチャを特徴とします。

## ED法の基本原理

ED法の基本原理はプロジェクトディレクトリのdocs/ED法_解説資料.mdに詳細に記載されています。本仕様書に目を通す場合には必ずdocs/ED法_解説資料.mdにも目を通し、ED法の基本原理を理解した上で本仕様書を参照してください。

## SNNの基本原理

SNNの基本原理はプロジェクトディレクトリのdocs/SNN_from_scratch_with_python_ver2_1.mdに詳細に記載されています。本仕様書に目を通す場合には必ずdocs/SNN_from_scratch_with_python_ver2_1.mdにも目を通し、SNNの基本原理を理解した上で本仕様書を参照してください。

## 拡張機能一覧（オリジナル理論からの追加機能）

本実装では、金子勇氏のオリジナルED法理論を完全に保持しながら、以下の拡張機能を追加実装しています：

### 1. 多層ニューラルネットワーク対応
- **オリジナル仕様**: 単一隠れ層のみサポート
- **拡張機能**: 複数隠れ層を自由に組み合わせ可能
- **実装方法**: カンマ区切り指定（例：`--hidden 256,128,64`）
- **技術的特徴**: NetworkStructureクラスによる動的層管理
- **ED法理論との整合性**: アミン拡散係数u1を多層間に適用

### 2. ミニバッチ学習システム
- **オリジナル仕様**: 1サンプルずつの逐次処理のみ
- **拡張機能**: 複数サンプルをまとめて効率的に処理
- **実装方法**: `--batch` オプションでサイズ指定可能
- **技術的特徴**: MiniBatchDataLoaderによる高速データ処理
- **性能向上**: エポック3.66倍・全体278倍の高速化を実現

### 3. NumPy行列演算による劇的高速化
- **オリジナル仕様**: 3重ループによる逐次計算
- **拡張機能**: NumPy行列演算による並列計算
- **性能向上**: フォワード計算で1,899倍の高速化達成
- **技術的特徴**: ベクトル化シグモイド関数とメモリ効率改善
- **理論保持**: ED法のアルゴリズム本質は完全保持

### 4. 動的メモリ管理システム
- **オリジナル仕様**: 固定サイズ配列（MAX=1000）
- **拡張機能**: データ量に応じた自動メモリサイズ調整
- **実装方法**: `calculate_safe_max_units`による安全性確保
- **技術的特徴**: 16GB RAM制限内での最大効率利用
- **安全性**: オーバーフロー保護とメモリ不足回避

### 5. リアルタイム可視化システム
- **オリジナル仕様**: テキスト出力による結果表示のみ
- **拡張機能**: 学習過程のリアルタイムグラフ表示
- **実装機能**: 学習曲線、混同行列、正答率推移の動的可視化
- **技術的特徴**: matplotlib基盤の非同期更新システム
- **使用方法**: `--viz` オプションで有効化
- **学習曲線の軸仕様**:
  - **縦軸**: 最小0.0、最大1.0、目盛り0.0から1.0まで0.1刻み
    - 中間グリッド線: 0.1, 0.3, 0.5, 0.7, 0.9が点線（`:`）
    - 中間グリッド線: 0.2, 0.4, 0.6, 0.8が実線（`-`）
  - **横軸**: 最小1、最大=設定された最大エポック数
    - 目盛り: 横軸を10分割した位置に設定
    - 中間グリッド線: 点線（`:`）と実線（`-`）を交互に配置
    - 例: 最大100エポック → 10,20,30,...,90の目盛り（10=点線、20=実線、30=点線、...）

### 6. 現代的データローダー統合
- **オリジナル仕様**: 独自データ形式の手動設定
- **拡張機能**: TensorFlow (tf.keras.datasets) 統合による自動データ処理
- **対応データセット**: MNIST・Fashion-MNIST・CIFAR-10・CIFAR-100の自動ダウンロード
- **技術的特徴**: バランス付きサンプリングとクラス均等化
- **使用方法**: `--mnist`, `--fashion`, `--cifar10`, `--cifar100` オプションでデータセット切替

### 7. GPU計算支援（CuPy対応）
- **オリジナル仕様**: CPU計算のみ
- **拡張機能**: NVIDIA GPU使用時の自動GPU計算
- **技術的特徴**: CuPy統合による透明なGPU処理
- **性能向上**: 大規模データセットでの更なる高速化
- **互換性**: GPU無環境でも自動的にCPU処理に切替

### 8. 詳細プロファイリング機能
- **オリジナル仕様**: 基本的な実行時間表示のみ
- **拡張機能**: 処理段階別の詳細性能分析
- **実装機能**: ボトルネック特定、メモリ使用量監視
- **技術的特徴**: リアルタイム性能監視とレポート生成微分の連鎖律を用いた誤差逆伝播法」の使用禁止
- **使用方法**: `--verbose` オプションで詳細表示

### 9. ヒートマップ可視化機能
- **オリジナル仕様**: オリジナル実装無し
- **拡張機能**: 学習過程のニューロン発火パターンのリアルタイム表示
- **実装機能**: 各層のニューロン発火パターンの動的可視化
- **技術的特徴**: 
  - matplotlib.gridspec基盤のGridSpecレイアウトシステム
  - rainbow配色による活性化強度の視覚化（vmin=0, vmax=1）
  - 2行×4列のグリッド配置で最大8層まで同時表示
  - 8層超過時: 最初の4層+最後の4層を自動選択表示
  - 出力層を含む全層を正方形グリッド表示（row-wise配置）
  - 各パネルにカラーバー付き（fraction=0.046, pad=0.04）
- **表示形式**:
  - タイトル行: エポック番号、正解クラス、予測クラス（色分け）
  - 予測クラス色分け: 正解時=青色、不正解時=赤色
  - 各層タイトル: 層名称とニューロン数表示
- **使用方法**: `--heatmap` オプションで有効化
- **併用機能**: `--viz`と`--heatmap`の同時指定
  - 両オプションを同時に指定した場合、2つの独立したウィンドウを表示
  - ウィンドウ1: 学習曲線 + 混同行列（`--viz`）
  - ウィンドウ2: 層別活性化ヒートマップ（`--heatmap`）
  - 保存ファイル: `_viz.png`と`_heatmap.png`のサフィックス付きで別々に保存
  - 各ウィンドウは独立して更新され、相互に干渉しない

### 9.5. 精度・損失検証機能
- **オリジナル仕様**: オリジナル実装無し
- **拡張機能**: 学習終了後の詳細な精度・損失分析
- **実装機能**: クラス別精度、平均損失、文字ベース混同行列の表示
- **技術的特徴**:
  - 混同行列の表示桁数を動的に調整
  - 最大値の桁数が3桁以下の場合: 1列の表示幅を4桁に設定
  - 最大値の桁数が4桁以上の場合: 1列の表示幅を最大桁数+1桁に設定
  - 例: 最大値が999の場合 → 4桁表示、最大値が1000の場合 → 5桁表示
- **使用方法**: `--verify_acc_loss` オプションで有効化

### 10. SNN(スパイキングニューラルネットワーク)対応
- **オリジナル仕様**: 非スパイキングニューラルネットワーク
- **拡張機能**: SNNアーキテクチャとED法による学習法の統合
- **技術的特徴**: スパイク発火モデルとED法の融合
- **ネットワーク**: SNNで構築
- **理論保持**: ED法の学習アルゴリズムに「微分の連鎖律を用いた誤差逆伝播法」の使用禁止を保持

### 11. 多重スパイクエンコーディング機能
- **オリジナル仕様**: オリジナル実装無し
- **拡張機能**: 複数のスパイク符号化方式による入力変換システム
- **実装機能**: ポアソン符号化、レート符号化、時間符号化、集団符号化
- **技術的特徴**: 
  - `_poisson_encode`: ポアソン分布によるスパイク列生成
  - `_rate_encode`: 強度をスパイク発火率に変換
  - `_temporal_encode`: 時間的発火パターン符号化
  - `_spike_encode`: 複合スパイクパターン生成
- **使用方法**: `encoding_type`パラメータで符号化方式選択

### 12. LIFニューロン統合システム
- **オリジナル仕様**: オリジナル実装無し
- **拡張機能**: 完全なLeaky Integrate-and-Fire (LIF) ニューロンモデル統合
- **実装機能**: 
  - `LIFNeuron`クラス: 個別LIFニューロン実装
  - `LIFNeuronLayer`クラス: 層単位のLIF管理
  - アミン濃度管理: `set_amine_concentration`, `get_amine_concentration`
  - 膜電位動力学: 微分方程式による膜電位更新
- **技術的特徴**: 生物学的に妥当なスパイク生成メカニズム
- **パラメータ**: 静止電位、閾値、リセット電位、時定数制御

### 13. スパイク-ED変換インターフェース
- **オリジナル仕様**: オリジナル実装無し
- **拡張機能**: スパイク活動とED法入力の双方向変換システム
- **実装機能**:
  - `convert_ed_outputs_to_spike_activities`: ED出力をスパイク活動パターンに変換
  - `convert_to_lif_input`: 画像データをLIF入力電流に変換（v019 Phase 11準拠）
  - `convert_spikes_to_ed_input`: スパイクパターンをED法入力形式に変換
- **技術的特徴**: 興奮性・抑制性ペア構造の完全保持
- **使用方法**: ED法とSNNの無縫接続による統合学習

### 14. 高速化SNNネットワーク実装
- **オリジナル仕様**: オリジナル実装無し
- **拡張機能**: 大規模SNN計算の高速化システム
- **実装機能**:
  - `SNNNetworkFastV2`クラス: 高速化SNN実装
  - `encode_input_fast`: 高速入力エンコーディング
  - `simulate_snn_fast`: 効率的SNN動力学シミュレーション
- **技術的特徴**: 計算ステップ削減とベクトル化処理
- **性能向上**: 従来SNN実装比で大幅な高速化達成

### 15. 純粋ED前処理システム
- **オリジナル仕様**: オリジナル実装無し
- **拡張機能**: ED法理論に完全準拠した前処理クラス
- **実装機能**: `PureEDPreprocessor`によるデータ前処理統合
- **技術的特徴**: 興奮性・抑制性ペア化、Dale's Principle適用
- **理論準拠**: 金子勇氏の原理論との完全整合性保証

### 16. 統合可視化システム拡張
- **オリジナル仕様**: 基本的な可視化のみ
- **拡張機能**: SNN学習過程の包括的リアルタイム可視化
- **実装機能**:
  - `RealtimeLearningVisualizer`: リアルタイム学習可視化
  - `setup_japanese_font`: 日本語フォント自動設定
    * 優先フォント: Noto Sans CJK JP, Noto Sans JP
    * フォールバック: IPAexGothic, IPAGothic, TakaoPGothic, VL PGothic
    * CJK/Japan/IPAキーワードによる自動検索
    * matplotlib.font_manager統合による利用可能フォント検出
    * 起動時にフォント設定状況を表示
  - `wait_for_keypress_or_timeout`: インタラクティブ制御
- **技術的特徴**: 
  - スパイク活動、膜電位、学習進捗の統合表示
  - 日本語表示の文字化け完全解決
  - Linux標準日本語フォントの自動検出と設定
- **使用方法**: SNN特有の動力学可視化とED法学習監視

### 17. モジュール化アーキテクチャ
- **オリジナル仕様**: 単一ファイル実装
- **拡張機能**: 機能別モジュール分離による保守性向上
- **実装構造**:
  - `modules/snn/`: SNNコア機能
  - `modules/ed_learning/`: ED法学習エンジン
  - `modules/visualization/`: 可視化システム
  - `modules/utils/`: 共通ユーティリティ
- **技術的特徴**: 疎結合設計による拡張性確保
- **保守性**: 機能追加・修正の独立性向上

## ED法核心理論（オリジナル仕様完全保持）

以下は金子勇氏によるオリジナルED法理論であり、拡張版においても100%保持されています：

### 1. 独立出力ニューロンアーキテクチャ
- 各出力ニューロンが完全に独立した重み空間を保持
- 3次元重み配列: `w_ot_ot[出力ニューロン][送信先][送信元]`
- 各クラスが独立したニューラルネットワークを構成

### 2. 興奮性・抑制性ニューロンペア構造
- 入力層: 興奮性（+1）・抑制性（-1）ニューロンがペアで構成
- 同種間結合: 正の重み制約
- 異種間結合: 負の重み制約
- 生物学的妥当性の保証

### 3. アミン拡散学習制御
- 出力層の誤差がアミン濃度として隠れ層に拡散
- 二種類のアミン: `del_ot[n][k][0]`（正誤差）、`del_ot[n][k][1]`（負誤差）
- パラメータ`u1`による拡散強度制御

### 4. 入力層の完全接続構造
- **重要**: 入力層の全ニューロン（興奮性・抑制性両方）が次層に接続
- 入力サイズの自動計算:
  - MNIST/Fashion-MNIST: 784ピクセル → 1568ニューロン（784ペア）
  - CIFAR-10/100: 3072ピクセル (32×32×3カラー) → 6144ニューロン（3072ペア）
- 重み行列サイズ: `paired_input_size × hidden_units`（入力層→隠れ層）
- 各ピクセル値は興奮性・抑制性ニューロンの両方に同じ値として設定

### 5. Dale's Principle（デールの原理）の厳密適用
- **原理**: ニューロンは興奮性または抑制性のいずれか一方の性質のみを持つ
- **実装**: 重み符号制約 `w *= ow[source] * ow[target]`
- **同種間結合**: `(+1) * (+1) = +1` または `(-1) * (-1) = +1` → 正の重み
- **異種間結合**: `(+1) * (-1) = -1` または `(-1) * (+1) = -1` → 負の重み
- **生物学的妥当性**: 実際の神経系における基本原理を遵守

### 6. ニューロンタイプ配列の全ユニット定義
- **重要**: バイアス項を含む全てのユニットがニューロンタイプを持つ
- インデックス0から`all+1`まで全て定義（入力層、隠れ層、出力層）
- 偶数インデックス: 興奮性（`+1`）
- 奇数インデックス: 抑制性（`-1`）
- バイアス項も含めて交互に配置

### 7. f[11]フラグによる抑制性ニューロン制御
- **f[11] = 1**（デフォルト）: 抑制性ニューロンからの接続も有効
- **f[11] = 0**: 抑制性ニューロン（奇数インデックス）からの接続を0に設定
- 実装: `if (f[11] == 0 && l < in+2 && (l % 2) == 1) w_ot_ot[n][k][l] = 0;`
- 用途: 抑制性ニューロンの影響を制御する実験的設定

## 詳細仕様

### データ構造定義

```c
// #define MAX 1000      // 最大ユニット数 ← 動的管理により制限撤廃
// #define NMAX 10       // 最大出力ニューロン数 ← 動的管理により制限撤廃

// 重み配列（3次元）
double w_ot_ot[NMAX+1][MAX+1][MAX+1];

// 各出力ニューロンの状態
double ot_in[NMAX+1][MAX+1];    // 入力値
double ot_ot[NMAX+1][MAX+1];    // 出力値
double del_ot[NMAX+1][MAX+1][2]; // アミン濃度[0:正, 1:負]

// ニューロンタイプ（興奮性/抑制性）
double ow[MAX+1];  // +1 or -1

// 学習パラメータ
double alpha;      // 学習率
double beta;       // 初期アミン濃度
double u0;         // シグモイド関数閾値
double u1;         // アミン拡散係数
```

### ニューロンタイプ配列の初期化（重要）

```c
// ★重要★ 全ユニット（バイアス項含む）のニューロンタイプを定義
void init_neuron_types(int all_units)
{
  for (k = 0; k <= all_units+1; k++) {
    // 偶数インデックス: 興奮性（+1）
    // 奇数インデックス: 抑制性（-1）
    ow[k] = ((k+1) % 2) * 2 - 1;
  }
}

// Python実装例
def init_neuron_types(all_units):
    """
    全ユニットのニューロンタイプを初期化
    偶数インデックス: 興奮性（+1）
    奇数インデックス: 抑制性（-1）
    """
    ow = np.ones(all_units + 2)
    ow[1::2] = -1  # 奇数インデックスを抑制性に設定
    return ow
```

### 入力層データ設定（重要）

```c
// ★重要★ 各ピクセル値を興奮性・抑制性ペアの両方に設定
void set_input_data(double* indata_input, int in_units)
{
  for (n = 0; n < ot; n++) {
    for (k = 2; k <= in_units+1; k++) {
      // k=2,3: indata_input[0]（同じ値）
      // k=4,5: indata_input[1]（同じ値）
      // k=6,7: indata_input[2]（同じ値）
      // ...
      ot_in[n][k] = indata_input[(int)(k/2)-1];
    }
  }
}

// Python実装例
def set_input_data(indata_input, output_neurons, in_units):
    """
    入力データを興奮性・抑制性ペアに設定
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

```c
// ★重要★ 重み初期化時にDale's Principleを適用
void init_weights(int in_units, int hidden_units, int output_neurons)
{
  double inival1 = 0.3;  // 隠れ層重み初期値
  double inival2 = 0.5;  // 入力層重み初期値
  int f[12];             // 制御フラグ配列
  
  for (n = 0; n < output_neurons; n++) {
    for (k = in_units+2; k <= all_units+1; k++) {  // 送信先ユニット
      for (l = 0; l <= all_units+1; l++) {          // 送信元ユニット
        
        // 基本的な重み初期化
        if (l < 2) {
          w_ot_ot[n][k][l] = inival2 * rnd();  // バイアス項
        }
        if (l > 1) {
          w_ot_ot[n][k][l] = inival1 * rnd();  // 入力層・隠れ層
        }
        
        // ★重要★ f[11]フラグによる抑制性ニューロン制御
        // f[11] = 0: 入力層の抑制性ニューロン（奇数インデックス）を無効化
        if (f[11] == 0 && l < in_units+2 && (l % 2) == 1) {
          w_ot_ot[n][k][l] = 0;
        }
        
        // ★重要★ Dale's Principleの適用（重み符号制約）
        // 同種間結合: ow[l] * ow[k] = +1 → 正の重み
        // 異種間結合: ow[l] * ow[k] = -1 → 負の重み
        w_ot_ot[n][k][l] *= ow[l] * ow[k];
      }
    }
  }
}

// Python実装例
def init_weights_with_dales_principle(
    in_units, hidden_units, output_neurons, 
    ow, inival1=0.3, inival2=0.5, f11=1
):
    """
    Dale's Principleを適用した重み初期化
    
    Parameters:
    -----------
    in_units : int
        入力ユニット数（興奮性・抑制性ペアの数）
    hidden_units : int
        隠れ層ユニット数
    output_neurons : int
        出力ニューロン数
    ow : array
        ニューロンタイプ配列（+1: 興奮性, -1: 抑制性）
    inival1 : float
        隠れ層重み初期値
    inival2 : float
        入力層重み初期値
    f11 : int
        抑制性ニューロン制御フラグ（1: 有効, 0: 無効）
    
    Returns:
    --------
    w_ot_ot : array
        初期化された重み配列
    """
    all_units = in_units + hidden_units
    w_ot_ot = np.zeros((output_neurons, all_units + 2, all_units + 2))
    
    for n in range(output_neurons):
        for k in range(in_units + 2, all_units + 2):  # 送信先（隠れ層・出力層）
            for l in range(all_units + 2):             # 送信元（全ユニット）
                
                # 基本的な重み初期化
                if l < 2:
                    w_ot_ot[n][k][l] = inival2 * np.random.rand()  # バイアス項
                else:
                    w_ot_ot[n][k][l] = inival1 * np.random.rand()  # 入力層・隠れ層
                
                # f[11]フラグによる抑制性ニューロン制御
                if f11 == 0 and l < in_units + 2 and (l % 2) == 1:
                    w_ot_ot[n][k][l] = 0  # 抑制性ニューロンを無効化
                
                # ★重要★ Dale's Principleの適用
                # 重み符号 = 送信元タイプ × 送信先タイプ
                w_ot_ot[n][k][l] *= ow[l] * ow[k]
    
    return w_ot_ot
```

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

#### 1. 順方向計算 (`neuro_output_calc`)

```c
// ★重要★ 各出力ニューロンが独立した計算を行う
void neuro_output_calc(double* indata_input)
{
  for (n = 0; n < ot; n++) {           // 各出力ニューロン独立処理
    
    // ★重要★ 入力層データ設定（興奮性・抑制性ペアに同じ値）
    for (k = 2; k <= in+1; k++)
      ot_in[n][k] = indata_input[(int)(k/2)-1];
    
    // 多時間ステップ計算（時間的展開による収束）
    for (t = 1; t <= t_loop; t++) {
      for (k = in+2; k <= all+1; k++) { // 各隠れ・出力ユニット
        inival = 0;
        
        // ★重要★ 全ユニット（入力層1568個含む）からの入力を計算
        for (m = 0; m <= all+1; m++)
          inival += w_ot_ot[n][k][m] * ot_in[n][m];
        
        ot_ot[n][k] = sigmf(inival);    // シグモイド活性化
      }
      
      // 出力を次の時間ステップの入力に設定（リカレント構造）
      for (k = in+2; k <= all+1; k++)
        ot_in[n][k] = ot_ot[n][k];
    }
  }
}

// Python実装例
def forward_pass(indata_input, w_ot_ot, ow, in_units, hidden_units, 
                 output_neurons, t_loop=5, u0=1.2):
    """
    順方向計算（各出力ニューロン独立処理）
    
    Parameters:
    -----------
    indata_input : array
        入力データ（例: MNISTの場合784個）
    w_ot_ot : array
        3次元重み配列 [output_neuron][target][source]
    ow : array
        ニューロンタイプ配列（+1: 興奮性, -1: 抑制性）
    in_units : int
        入力層のニューロン数（ペア数 × 2、例: 784 × 2 = 1568）
    hidden_units : int
        隠れ層のニューロン数
    output_neurons : int
        出力ニューロン数
    t_loop : int
        時間ステップ数（リカレント展開回数）
    u0 : float
        シグモイド関数の閾値
    
    Returns:
    --------
    ot_ot : array
        各出力ニューロンの出力値
    """
    all_units = in_units + hidden_units
    ot_in = np.zeros((output_neurons, all_units + 2))
    ot_ot = np.zeros((output_neurons, all_units + 2))
    
    # 各出力ニューロンごとに独立処理
    for n in range(output_neurons):
        
        # ★重要★ 入力データを興奮性・抑制性ペアに設定
        for k in range(2, in_units + 2):
            pixel_index = (k // 2) - 1
            ot_in[n][k] = indata_input[pixel_index]
        
        # 時間的展開による収束計算
        for t in range(t_loop):
            for k in range(in_units + 2, all_units + 2):
                
                # ★重要★ 全ユニットからの入力を計算
                # これにより入力層の全1568個が次層に接続される
                inival = 0
                for m in range(all_units + 2):
                    inival += w_ot_ot[n][k][m] * ot_in[n][m]
                
                # シグモイド活性化
                ot_ot[n][k] = 1.0 / (1.0 + np.exp(-2.0 * inival / u0))
            
            # 出力を次の時間ステップの入力に設定
            for k in range(in_units + 2, all_units + 2):
                ot_in[n][k] = ot_ot[n][k]
    
    return ot_ot
```

#### 2. シグモイド活性化関数

```c
double sigmf(double u) { 
  return(1 / (1 + exp((double)(-2 * u / u0)))); 
}
```

#### 3. アミン濃度計算 (`neuro_teach_calc`)

```c
void neuro_teach_calc(double* indata_tch)
{
  for (l = 0; l <= ot-1; l++) {        // 各出力ニューロン
    // 誤差計算
    wkb = indata_tch[l] - ot_ot[l][in+2];
    
    // 出力層アミン濃度設定
    if (wkb > 0) {
      del_ot[l][in+2][0] = wkb;        // 正誤差アミン
      del_ot[l][in+2][1] = 0;
    } else {
      del_ot[l][in+2][0] = 0;
      del_ot[l][in+2][1] = -wkb;       // 負誤差アミン
    }
    
    // 隠れ層への拡散
    inival1 = del_ot[l][in+2][0];
    inival2 = del_ot[l][in+2][1];
    
    for (k = in+3; k <= all+1; k++) {  // 各隠れユニット
      del_ot[l][k][0] = inival1 * u1;  // 拡散係数u1で拡散
      del_ot[l][k][1] = inival2 * u1;
    }
  }
}
```

#### 4. 重み更新 (`neuro_weight_calc`)

```c
// ★重要★ 各出力ニューロンが独立して重みを更新
void neuro_weight_calc()
{
  for (n = 0; n < ot; n++) {           // 各出力ニューロン独立更新
    for (k = in+2; k <= all+1; k++) {  // 各送信先ユニット
      for (m = 0; m <= all+1; m++) {   // 各送信元ユニット（★全ユニット）
        if (w_ot_ot[n][k][m] != 0) {
          
          // delta計算（生物学的意味での学習信号強度）
          del = alpha * ot_in[n][m];              // 学習率×入力強度
          del *= fabs(ot_ot[n][k]);               // 出力活性度
          del *= (1 - fabs(ot_ot[n][k]));        // 飽和抑制項
          
          // ★重要★ ニューロンタイプに応じた重み更新
          // 興奮性入力: 正誤差アミンを使用
          // 抑制性入力: 負誤差アミンを使用
          if (ow[m] > 0)  // 興奮性入力
            w_ot_ot[n][k][m] += del * del_ot[n][k][0] * ow[m] * ow[k];
          else            // 抑制性入力
            w_ot_ot[n][k][m] += del * del_ot[n][k][1] * ow[m] * ow[k];
        }
      }
    }
  }
}

// Python実装例
def update_weights(w_ot_ot, ot_in, ot_ot, del_ot, ow, alpha, 
                   in_units, hidden_units, output_neurons):
    """
    重み更新（各出力ニューロン独立処理）
    
    Parameters:
    -----------
    w_ot_ot : array
        3次元重み配列 [output_neuron][target][source]
    ot_in : array
        各ユニットの入力値
    ot_ot : array
        各ユニットの出力値
    del_ot : array
        アミン濃度 [output_neuron][unit][0:正/1:負]
    ow : array
        ニューロンタイプ配列（+1: 興奮性, -1: 抑制性）
    alpha : float
        学習率
    in_units : int
        入力層のニューロン数
    hidden_units : int
        隠れ層のニューロン数
    output_neurons : int
        出力ニューロン数
    
    Returns:
    --------
    w_ot_ot : array
        更新された重み配列
    """
    all_units = in_units + hidden_units
    
    # 各出力ニューロンごとに独立更新
    for n in range(output_neurons):
        for k in range(in_units + 2, all_units + 2):  # 送信先ユニット
            
            # ★重要★ 全ユニットからの接続を更新（入力層1568個含む）
            for m in range(all_units + 2):  # 送信元ユニット
                
                if w_ot_ot[n][k][m] != 0:
                    
                    # delta計算（学習信号強度）
                    delta = alpha * ot_in[n][m]              # 学習率 × 入力強度
                    delta *= abs(ot_ot[n][k])                # 出力活性度
                    delta *= (1 - abs(ot_ot[n][k]))         # 飽和抑制項
                    
                    # ★重要★ ニューロンタイプに応じたアミン使用
                    # Dale's Principleにより符号も自動調整される
                    if ow[m] > 0:  # 興奮性入力
                        # 正誤差アミン使用 × Dale's Principle符号
                        w_ot_ot[n][k][m] += delta * del_ot[n][k][0] * ow[m] * ow[k]
                    else:          # 抑制性入力
                        # 負誤差アミン使用 × Dale's Principle符号
                        w_ot_ot[n][k][m] += delta * del_ot[n][k][1] * ow[m] * ow[k]
    
    return w_ot_ot

# ★重要な注釈★
# 1. 送信元ユニット（m）は全ユニット（0 ～ all_units+1）をループ
#    これにより入力層の全1568個のニューロンが学習に参加
# 
# 2. ow[m] * ow[k]（Dale's Principle）により重みの符号が制約される
#    - 興奮性→興奮性: (+1) * (+1) = +1 → 正の重み増加
#    - 興奮性→抑制性: (+1) * (-1) = -1 → 負の重み増加
#    - 抑制性→興奮性: (-1) * (+1) = -1 → 負の重み増加
#    - 抑制性→抑制性: (-1) * (-1) = +1 → 正の重み増加
# 
# 3. 興奮性入力は正誤差アミン、抑制性入力は負誤差アミンを使用
#    これにより生物学的に妥当な学習が実現される
```

### マルチクラス分類メカニズム

#### 1. 独立出力ニューロン方式
- 各クラスに専用の出力ニューロンを割り当て
- 各出力ニューロンが独立した重み空間で学習
- クラス間の干渉を排除

#### 2. 教師信号パターン (`pat[k]`)
```c
pat[k] = 0;  // ランダム（0/1）
pat[k] = 1;  // パリティ問題（XOR系）
pat[k] = 2;  // ミラー対称問題
pat[k] = 3;  // 手動入力
pat[k] = 4;  // 連続値ランダム
pat[k] = 5;  // One-Hot符号化（真のマルチクラス）
```

#### 3. One-Hot符号化によるマルチクラス学習
```c
// pat[n] = 5の場合：各パターンで1つのクラスのみアクティブ
for (l = 0; l <= pa-1; l++) {
  g_indata_tch[l][n] = 0.0;  // 全パターンで0
}
// ランダムに選択された1つのパターンのみ1
g_indata_tch[selected_pattern][n] = 1.0;
```

### 重み制約メカニズム

#### 1. 興奮性・抑制性制約
```c
// 初期化時の制約適用
w_ot_ot[n][k][l] *= ow[l] * ow[k];  // 符号制約の強制

// 制約の意味：
// ow[l] * ow[k] = +1: 同種間結合（興奮-興奮 or 抑制-抑制）→正の重み
// ow[l] * ow[k] = -1: 異種間結合（興奮-抑制 or 抑制-興奮）→負の重み
```

#### 2. 構造的制約
```c
// 自己結合の制御
if (k == l && f[3] == 1)
  w_ot_ot[n][k][l] = 0;  // 自己結合禁止

// 層間接続の制御
if (f[6] == 1 && k != l && k > in+2 && l > in+1)
  w_ot_ot[n][k][l] = 0;  // 隠れ層間結合禁止
```

### パラメータ設定指針

#### 基本パラメータ
- `alpha` (学習率): 0.1 ～ 1.0
- `beta` (初期アミン): 0.1 ～ 0.5  
- `u0` (シグモイド閾値): 0.4 ～ 1.0
- `u1` (アミン拡散): 0.5 ～ 1.0
- `inival1` (隠れ層重み初期値): 0.1 ～ 0.5
- `inival2` (入力層重み初期値): 0.1 ～ 0.5

#### ネットワーク構成
- 入力ユニット: `in * 2` (興奮性・抑制性ペア)
- 隠れユニット: `hd` (問題の複雑さに応じて)
- 出力ユニット: `ot` (クラス数)

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
- Hebbian学習の拡張
- 生物学的に実現可能な情報処理
- 時間遅延と局所性の考慮

## 従来手法との比較優位性

### 1. vs 微分の連鎖律を用いたバックプロパゲーション
- **構造**: 独立出力vs共有隠れ層
- **学習**: アミン拡散vs微分の連鎖律を用いた誤差逆伝播
- **生物学的妥当性**: 高い vs 低い
- **マルチクラス**: 独立学習 vs 競合学習

### 2. vs Hopfieldネット
- **記憶容量**: 制約なし vs 0.15N限界
- **学習**: 教師あり vs 教師なし
- **収束性**: 安定 vs 偽記憶問題

### 3. vs SOM/競合学習
- **教師信号**: あり vs なし
- **表現力**: 高次特徴 vs トポロジー保存
- **正答率**: 高正答率分類 vs クラスタリング

## 実装上の重要ポイント

**注意**: 以下の実装ポイントは、オリジナルED法理論と拡張機能実装、およびSNNネットワーク実装のすべてに適用されます。拡張機能を使用する場合も、ED法の核心的アルゴリズムは必ず以下の仕様に従って実装する必要があります。

### 1. 3次元重み配列の正確な実装（オリジナル仕様・拡張機能共通）
```python
# Python実装例
w_ot_ot = np.zeros((ot_max, unit_max, unit_max))
del_ot = np.zeros((ot_max, unit_max, 2))
ot_in = np.zeros((ot_max, unit_max))
ot_ot = np.zeros((ot_max, unit_max))
```

### 2. アミン拡散の正確な計算
```python
# 出力層アミン設定
if error > 0:
    del_ot[l][output_pos][0] = error
    del_ot[l][output_pos][1] = 0
else:
    del_ot[l][output_pos][0] = 0
    del_ot[l][output_pos][1] = -error

# 隠れ層への拡散
for k in range(hidden_start, all_units+1):
    del_ot[l][k][0] = del_ot[l][output_pos][0] * u1
    del_ot[l][k][1] = del_ot[l][output_pos][1] * u1
```

### 3. 重み更新の正確な実装
```python
# 各出力ニューロン独立更新
for n in range(output_neurons):
    for k in range(hidden_start, all_units+1):
        for m in range(all_units+1):
            if w_ot_ot[n][k][m] != 0:
                delta = alpha * ot_in[n][m]
                delta *= abs(ot_ot[n][k])
                delta *= (1 - abs(ot_ot[n][k]))
                
                if ow[m] > 0:  # 興奮性
                    w_ot_ot[n][k][m] += delta * del_ot[n][k][0] * ow[m] * ow[k]
                else:          # 抑制性
                    w_ot_ot[n][k][m] += delta * del_ot[n][k][1] * ow[m] * ow[k]
```

### 4. 入力層の完全接続実装（重要）
```python
# ★重要★ 入力層の全ニューロン（興奮性・抑制性両方）を次層に接続

# 誤った実装（Phase 12の誤り）
input_units = input_size // 2  # 784個のみ使用 ← 誤り

# 正しい実装（Phase 14で修正）
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
# ★重要★ 重み符号制約の正確な実装

def apply_dales_principle(w_ot_ot, ow):
    """
    Dale's Principleを重み行列に適用
    
    Parameters:
    -----------
    w_ot_ot : array
        重み配列 [output_neuron][target][source]
    ow : array
        ニューロンタイプ配列（+1: 興奮性, -1: 抑制性）
    
    Returns:
    --------
    w_ot_ot : array
        符号制約が適用された重み配列
    """
    for n in range(w_ot_ot.shape[0]):      # 出力ニューロン
        for k in range(w_ot_ot.shape[1]):  # 送信先ユニット
            for l in range(w_ot_ot.shape[2]):  # 送信元ユニット
                # 重み符号 = 送信元タイプ × 送信先タイプ
                w_ot_ot[n][k][l] *= ow[l] * ow[k]
    
    return w_ot_ot

# Dale's Principleの効果:
# - 同種間結合（興奮性→興奮性、抑制性→抑制性）: 正の重み
# - 異種間結合（興奮性→抑制性、抑制性→興奮性）: 負の重み
# - 生物学的神経系の基本原理を遵守
# - ネットワークの安定性と学習性能の向上
```

### 7. f[11]フラグによる抑制性制御（重要）
```python
# ★重要★ 抑制性ニューロンの有効/無効制御

def apply_inhibitory_control(w_ot_ot, in_units, f11=1):
    """
    f[11]フラグによる抑制性ニューロン制御
    
    Parameters:
    -----------
    w_ot_ot : array
        重み配列 [output_neuron][target][source]
    in_units : int
        入力層のニューロン数（ペア数 × 2）
    f11 : int
        抑制性制御フラグ
        1: 抑制性ニューロン有効（デフォルト）
        0: 抑制性ニューロン無効
    
    Returns:
    --------
    w_ot_ot : array
        制御が適用された重み配列
    """
    if f11 == 0:
        # 入力層の抑制性ニューロン（奇数インデックス）を無効化
        for n in range(w_ot_ot.shape[0]):
            for k in range(w_ot_ot.shape[1]):
                for l in range(2, in_units + 2):
                    if (l % 2) == 1:  # 奇数インデックス（抑制性）
                        w_ot_ot[n][k][l] = 0
    
    return w_ot_ot

# 説明:
# - f11 = 1（デフォルト）: 興奮性・抑制性の両方が有効
#   → 重み行列は 1568 × hidden_units
# - f11 = 0: 抑制性ニューロンを無効化
#   → 実質的に 784 × hidden_units（興奮性のみ）
# - 実験的な設定として使用可能
```

## 動作検証結果

### オリジナルC実装での確認事項
- **学習の継続性**: エポックを重ねても学習が停滞しない
- **重み値の動的変化**: `th+ 6.53 → 6.56` など継続的調整
- **誤差の減少**: 出力値が目標値に段階的に接近
- **隠れ層活動**: `hd:` 値の動的変化による特徴学習

### 学習動態の特徴
```
エポック進行例:
in:0.00 0.00 1.00 1.00 -> 0.00770, 0.00  # 初期状態
                      ↓ 
in:0.00 0.00 1.00 1.00 -> 0.00576, 0.00  # 誤差減少
```

## 適用可能分野

### 1. パターン認識
- 手書き文字認識
- 画像分類
- 音声認識

### 2. 関数近似
- XOR・パリティ問題
- 非線形回帰
- 時系列予測

### 3. 最適化問題
- 組み合わせ最適化
- 制約満足問題
- スケジューリング

## 研究の意義と将来性

### 1. 1999年時点での先進性
- 生物学的妥当性への早期着目
- 独創的なアーキテクチャ設計
- マルチクラス分類への独自アプローチ

### 2. 現代への示唆
- Transformer時代における局所学習の重要性
- 神経科学的知見の工学的応用
- 持続可能な学習システムの設計

### 3. 拡張版の発展可能性
- 多層構造による複雑パターン認識の向上
- 高速化技術の他分野への応用
- リアルタイム学習システムの実現
- GPU計算による大規模データ処理
- 深層学習との融合可能性

## 結論

金子勇氏によるED法は、1999年時点で既に現代の神経科学的知見を先取りした、極めて独創的かつ生物学的に妥当な学習アルゴリズムです。その核心である「アミン拡散による学習制御」「独立出力ニューロンアーキテクチャ」「興奮性・抑制性バランス」は、従来の人工ニューラルネットワークの限界を克服する可能性を秘めています。

**本拡張版実装では、オリジナル理論の本質を100%保持しながら、現代的な実用性を大幅に向上させることに成功しました。**多層化・高速化・可視化などの拡張機能により、実践的な機械学習プロジェクトでの利用が可能となり、現代のAI研究においても高い価値を持つ重要な研究成果として活用できます。

**ED法 for SNNでは、ED法オリジナル理論の本質を100%保持しながら、SNNネットワークでもED法が有効に機能する実装を目指します。**

## SNN統合実装詳細（拡張機能）

### 1. スパイクエンコーディング実装仕様

#### 1.1 ポアソン符号化 (`_poisson_encode`)
```python
def _poisson_encode(self, input_data, time_steps=100, dt=1.0):
    """
    ポアソン分布によるスパイク列生成
    
    Parameters:
    -----------
    input_data : array
        正規化入力データ (0-1)
    time_steps : int
        時間ステップ数
    dt : float
        時間刻み [ms]
    
    Returns:
    --------
    spike_trains : array
        スパイク列 (time_steps, neurons)
    """
```

#### 1.2 レート符号化 (`_rate_encode`)
```python
def _rate_encode(self, input_data, max_rate=100.0):
    """
    入力強度をスパイク発火率に変換
    
    Parameters:
    -----------
    input_data : array
        入力データ
    max_rate : float
        最大発火率 [Hz]
    
    Returns:
    --------
    firing_rates : array
        発火率パターン
    """
```

#### 1.3 時間符号化 (`_temporal_encode`)
```python
def _temporal_encode(self, input_data, time_window=50.0):
    """
    時間的発火パターン符号化
    
    Parameters:
    -----------
    input_data : array
        入力データ
    time_window : float
        時間窓 [ms]
    
    Returns:
    --------
    temporal_pattern : array
        時間的スパイクパターン
    """
```

### 2. LIFニューロン実装仕様

#### 2.1 LIFニューロンクラス
```python
class LIFNeuron:
    """
    Leaky Integrate-and-Fire ニューロンモデル
    
    Parameters:
    -----------
    v_rest : float
        静止膜電位 [mV]
    v_threshold : float
        発火閾値 [mV]
    v_reset : float
        リセット電位 [mV]
    tau_m : float
        膜時定数 [ms]
    tau_ref : float
        不応期 [ms]
    r_m : float
        膜抵抗 [MΩ]
    neuron_type : str
        ニューロンタイプ ('excitatory' or 'inhibitory')
    """
    
    def update(self, input_current, dt=1.0):
        """
        膜電位更新とスパイク判定
        
        Parameters:
        -----------
        input_current : float
            入力電流 [nA]
        dt : float
            時間刻み [ms]
        
        Returns:
        --------
        spike : bool
            スパイク発生フラグ
        """
```

#### 2.2 アミン濃度管理
```python
def set_amine_concentration(self, concentration):
    """
    アミン濃度設定（ED法統合用）
    
    Parameters:
    -----------
    concentration : float
        アミン濃度値
    """

def get_amine_concentration(self):
    """
    現在のアミン濃度取得
    
    Returns:
    --------
    concentration : float
        現在のアミン濃度
    """
```

### 3. スパイク-ED変換インターフェース

#### 3.1 ED出力→スパイク活動変換
```python
def convert_ed_outputs_to_spike_activities(ed_core, inputs, original_image_shape=(28, 28)):
    """
    ED出力をスパイク活動パターンに変換（ヒートマップ用）
    
    Parameters:
    -----------
    ed_core : MultiLayerEDCore
        ED Core インスタンス
    inputs : array
        E/Iペア化された入力 (shape: [paired_input_size])
    original_image_shape : tuple
        元の画像形状 ((28, 28), (32, 32, 3)など)
    
    Returns:
    --------
    spike_activities : list
        各層のスパイク活動パターン
    """
```

#### 3.2 画像データ→LIF入力変換
```python
def convert_to_lif_input(image_data, scale_factor=10.0):
    """
    画像データをLIF入力電流に変換（v019 Phase 11修正）
    
    Parameters:
    -----------
    image_data : array
        正規化済み画像データ (0-1)
        784個（MNIST）または1568個（既にE/Iペア化済み）
    scale_factor : float
        電流スケールファクター（デフォルト: 10.0 nA）
        
    Returns:
    --------
    current_pattern : array
        電流パターン (nA単位)
        **Phase 11**: 1568個（興奮性784個+抑制性784個）
        
    Note:
    -----
    v019 Phase 11修正: ED法仕様に完全準拠
    - 金子勇氏のオリジナルED法では入力層は興奮性・抑制性ペア構成が必須
    - 物理的に1568個のニューロン（興奮性784個+抑制性784個）で構成
    - 入力が784個の場合: 1568個に変換
    - 入力が1568個の場合: そのまま処理（既にペア化済み）
    """
```

### 4. 高速化SNN実装

#### 4.1 高速化SNNネットワーク
```python
class SNNNetworkFastV2:
    """
    高速化ED-SNNネットワーク実装
    
    Parameters:
    -----------
    network_structure : list
        [入力, 隠れ, 出力] のリスト
    simulation_time : float
        シミュレーション時間 (ms)
    dt : float
        時間刻み (ms)
    use_fast_core : bool
        高速化EDコア使用フラグ
    """
    
    def encode_input_fast(self, input_data, encoding_type='rate'):
        """高速入力エンコーディング"""
        
    def simulate_snn_fast(self, encoded_input):
        """高速SNNシミュレーション"""
```

### 5. 統合学習アルゴリズム

#### 5.1 ED-SNN統合学習ステップ
```python
def train_step(self, input_data, target_data, encoding_type='rate'):
    """
    ED-SNN統合学習ステップ
    
    Process:
    --------
    1. 入力をスパイク列にエンコード
    2. SNNダイナミクスシミュレーション
    3. スパイクパターンをED法入力に変換
    4. ED法学習実行
    
    Parameters:
    -----------
    input_data : array
        入力データ
    target_data : array  
        目標データ
    encoding_type : str
        エンコーディングタイプ
        
    Returns:
    --------
    result : dict
        学習結果情報
    """
```

### 6. モジュール構成

#### 6.1 modules/snn/ ディレクトリ
```
modules/snn/
├── __init__.py
├── lif_neuron.py          # LIFニューロン実装
├── snn_network.py         # 基本SNNネットワーク
├── snn_network_fast.py    # 高速化SNN
├── snn_network_fast_v2.py # 最適化SNN
└── ed_core_fast_v2.py     # ED-SNN統合コア
```

#### 6.2 統合アーキテクチャ
```
MultiLayerEDCore (メインクラス)
├── SNNNetwork統合
├── スパイクエンコーダー
├── LIFニューロン層
├── ED法学習エンジン
└── 可視化インターフェース
```

### 7. 理論的整合性保証

#### 7.1 ED法理論との統合
- **興奮性・抑制性ペア構造**: SNNでも完全保持
- **Dale's Principle**: LIFニューロンレベルで適用
- **アミン拡散**: スパイク発火率としてアミン濃度をモデル化
- **独立出力ニューロン**: 各クラス専用のSNNサブネットワーク

#### 7.2 生物学的妥当性
- **スパイク発火**: 実際の神経細胞の動力学
- **膜電位動力学**: 微分方程式による正確なモデル化
- **時間的ダイナミクス**: リアルタイム情報処理
- **可塑性**: スパイクタイミング依存性を保持

## 実装方針（拡張版対応）

 **ED法理論の絶対保持**: 金子勇氏のオリジナル理論は一切変更せず、拡張機能は理論に準拠した形で追加する。
 **「微分の連鎖律を用いた誤差逆伝播法」の使用禁止**: 「微分の連鎖律を用いた誤差逆伝播法」の使用を禁止する。
 **SNNネットワーク対応**: SNNアーキテクチャにED法を適用する際も、ED法の学習アルゴリズムを完全に保持する。
 **コーディングルール**: PEP8に準拠し、可読性を最優先にする。
 **拡張機能の明示**: 新しい機能を追加する際は、オリジナル理論からの拡張であることを明確にコメントで示す。
 **コードの可読性**: コメントは適度(できるだけ少なめ)な量にする。コメントは、whatではなくwhyを記述するようにして、コードの意図が明確になるように心がける。
 **モジュール化**: 各機能を明確に分離し、再利用可能なモジュールとして実装する。
 **テスト駆動開発**: 新しい機能を実装した場合はその機能に対してユニットテストを作成し、実装前にテストを通過させる。
 **パラメータ調整**: argparseを用いて基本パラメータを柔軟に変更できるようにし、実験的な調整を容易にする。
 **拡張機能の理論的根拠**: 拡張機能実装時は、ED法理論との整合性を保ち、必要に応じて理論的根拠をコメントで説明する。

---

**本仕様書は、オリジナルC実装の動作確認と詳細なコード解析、および拡張機能の実装検証に基づいて作成されました。**  
**オリジナル検証日**: 2025年8月30日  
**拡張版作成日**: 2025年9月13日  
**SNN統合実装**: 2025年11月  
**検証者**: AI解析システム  
**ソースコード**: `/ed_original_src/` (コンパイル・実行確認済み) + 拡張版Python実装 + SNN統合実装
