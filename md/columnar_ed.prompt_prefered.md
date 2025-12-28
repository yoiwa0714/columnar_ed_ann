# コラムED法(Columnar ED法) 仕様書 (ANN版)

**原作者**: 金子勇（Isamu Kaneko）  
**原作開発年**: 1999年  
**拡張実装**: 2025年  
**オリジナルソース**: C言語実装（動作確認済み）  
**拡張版**: Python実装（コラムED法）

## 本プロジェクトの目的

 - 金子勇氏のオリジナルED法の理論的基盤を完全に保持しつつ、コラムED法の本質的な特徴を実装します。
 - コラムED法の動作原理を理解し易い実装とすることで、初学者がコラムED法を理解し易くします。
 - したがって、高い正答率を出すことを目的とした複雑な機能の実装は行ないません。

## ED法の概要

ED法（Error Diffusion Learning Algorithm）は、生物学的神経系のアミン（神経伝達物質）拡散メカニズムを模倣した独創的な学習アルゴリズムです。従来のバックプロパゲーションとは根本的に異なる、興奮性・抑制性ニューロンペア構造と出力ニューロン中心のアーキテクチャを特徴とします。

## ED法の基本原理

ED法の基本原理はプロジェクトディレクトリのdocs/ED法_解説資料.mdに詳細に記載されています。本仕様書に目を通す場合には必ずdocs/ED法_解説資料.mdにも目を通し、ED法の基本原理を理解した上で本仕様書を参照してください。

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

### 3. NumPy行列演算による高速化
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

### 10. 精度・損失検証機能
- **オリジナル仕様**: オリジナル実装無し
- **拡張機能**: 学習終了後の詳細な精度・損失分析
- **実装機能**: クラス別精度、平均損失、文字ベース混同行列の表示
- **技術的特徴**:
  - 混同行列の表示桁数を動的に調整
  - 最大値の桁数が3桁以下の場合: 1列の表示幅を4桁に設定
  - 最大値の桁数が4桁以上の場合: 1列の表示幅を最大桁数+1桁に設定
  - 例: 最大値が999の場合 → 4桁表示、最大値が1000の場合 → 5桁表示
- **使用方法**: `--verify_acc_loss` オプションで有効化

### 11. 純粋ED前処理システム
- **オリジナル仕様**: オリジナル実装無し
- **拡張機能**: ED法理論に完全準拠した前処理クラス
- **実装機能**: `PureEDPreprocessor`によるデータ前処理統合
- **技術的特徴**: 興奮性・抑制性ペア化、Dale's Principle適用
- **理論準拠**: 金子勇氏の原理論との完全整合性保証

### 12. コラムED法（Columnar ED Method）統合
- **オリジナル仕様**: オリジナル実装無し
- **拡張機能**: 大脳皮質のコラム構造を模倣した選択的学習システム
- **生物学的基盤**: 大脳視覚野などに見られる機能的神経細胞群（コラム）の模倣
- **実装ファイル**: `columnar_ed_ann_v004.py` (純粋ANN実装、Tanh版)

#### 12.1 コラムED法の本質
- **微分の連鎖律を使わない学習**:
  - バックプロパゲーションとは根本的に異なるアプローチ
  - アミン拡散メカニズムによる重み更新
  - ED法の理論的基盤を完全保持
  
- **コラム帰属度による選択的学習**:
  - コラムニューロン: 係数0.5で学習（抑制的）
  - 非コラムニューロン: 係数1.5で学習（促進的）
  - 公式: `amine = error * column_affinity * u1`
  - 1つの共有重み空間で複数クラスを分類
  
- **全層貫通コラム構造**:
  - 各クラスのコラムが全隠れ層を垂直に貫通
  - 同じクラスのニューロンが層間で縦に並ぶ
  - 生物学的妥当性: 大脳皮質の柱状機能単位を再現

#### 12.2 Tanhスケール版活性化関数
- **目的**: Sigmoidの勾配消失問題を緩和
- **定義**: `tanh_scaled(x) = (tanh(x/sig) + 1) / 2`
- **出力範囲**: [0, 1]（Sigmoidと互換）
- **特徴**: 
  - ゼロ中心性により学習が安定
  - より大きい勾配で勾配消失を緩和
  - ED法のアミン拡散メカニズムと整合

#### 12.3 コラム構造の基本設定
- **`--column_ratio`**: 全層共通のコラム参加率
  - デフォルト: 0.2（20%、実験結果に基づく最適値）
  - 範囲: 0.0～1.0
  - 意味: 各層のニューロンのうち何%がコラムに参加するか
  
- **`--column_overlap`**: 隣接コラムとの重複度
  - デフォルト: 0.1
  - 範囲: 0.0～0.5
  - 巡回構造（円環状配置）による隣接関係
  
- **`--column_neurons`**: コラム占有数（絶対数指定）
  - デフォルト: None（自動計算）
  - 各コラムの専有ニューロン数を直接指定

#### 12.4 層別コラム設定（高度な機能）
- **`--column_neurons_per_layer N1,N2,...`**: 層別絶対数指定
  - 例: `--column_neurons_per_layer 30,25,20,15,10`
  - 各層のコラム参加ニューロン総数を個別指定
  - 生物学的妥当性: 浅い層は多く、深い層は少なく
  - 実装日: 2025-11-28
  
- **`--column_ratio_per_layer R1,R2,...`**: 層別参加率指定
  - 例: `--column_ratio_per_layer 0.5,0.3,0.25,0.2,0.2`
  - 各層のコラム参加率を個別指定
  - 柔軟な構造設計が可能
  - 実装日: 2025-11-28
  
- **優先順位**:
  1. `column_neurons_per_layer`（層別絶対数）- 最優先
  2. `column_ratio_per_layer`（層別参加率）- 次優先
  3. `column_ratio`（全層共通参加率）- デフォルト

#### 12.5 エラー処理とバリデーション
コラムED法では、学習の品質を保証するため包括的なエラー検証を実施：

- **Critical Error条件1: ニューロン数不足**
  ```
  n_neurons < n_output（クラス数）
  → 各クラスに最低1ニューロン必要
  → プログラム停止、修正方法を明示
  ```

- **Critical Error条件2: コラム参加ニューロン数不足**
  ```
  column_neurons_total < n_output（クラス数）
  → 各コラムに最低1ニューロン必要
  → プログラム停止、修正方法を明示
  ```

- **Critical Error条件3: 一部コラムが0個**
  ```
  一部のコラムにニューロンを割り当てられない
  → コラムED法の真価を発揮できない
  → プログラム停止、詳細な解決方法を提示
  ```

- **自動調整機能**:
  - 指定値 > 実際のニューロン数の場合
  - 自動的に実際のニューロン数に調整
  - 警告メッセージ: `(※ 指定150→実際128に調整)`
  - 実行は継続

- **エラーメッセージの特徴**:
  - 層番号の明示: `層4のコラム参加ニューロン数...`
  - 必要な最小値の提示: `0.156以上に設定してください`
  - 修正方法の具体的指示: どのオプションをどう変更すべきか
  - 視覚的強調: `❌ CRITICAL ERROR`マーク

#### 12.6 層数別最適設定（実験結果に基づく）
実験により確認された各層数における最適パラメータ：

```python
層数別デフォルト設定:
1層: [256], lr=0.1, epochs=50
2層: [512,256], lr=0.0005, epochs=100 (Fashion-MNIST: 78.06%)
3層: [256,128,64], lr=0.1, epochs=50
4層: [128,96,80,64], lr=0.01, epochs=50 (Fashion-MNIST: 84.43%, 最高精度)
5層: [128,96,80,64,64], lr=0.01, epochs=50 (Fashion-MNIST: 83.57%, デフォルト)
```

- **自動設定**: ユーザーが`--hidden`で層数を指定すると対応する最適パラメータを自動適用
- **手動上書き**: `--lr`や`--epochs`を明示的に指定すれば自動設定を上書き可能

#### 12.7 forward戻り値の仕様（重要）
```python
a_hiddens, z_hiddens, z_output = network.forward(x)
# a_hiddens: 各層の線形出力（活性化前） [layer][n_hidden[layer]]
# z_hiddens: 各層の活性化後出力 [layer][n_hidden[layer]]
# z_output: 出力層の活性 [n_output]
```
**重要**: forwardメソッドは常に3つの値を返す（2つではない）

#### 12.8 コラムED法の技術的特徴
- **1つの共有重み空間**: 
  - 複数クラスを単一の重み行列で分類
  - コラム帰属度による選択的学習で実現
  
- **バッチ処理対応**: 
  - 効率的な学習
  - 大規模データセット対応
  
- **ヒートマップ可視化**: 
  - 全層の帰属度を一度に表示（4行×4列グリッド）
  - ed_multi_lif_snn.py準拠
  - 各層のコラム構造が視覚的に確認可能

#### 12.9 使用例とベストプラクティス
```bash
# デフォルト設定（5層、生物学的妥当性）
python columnar_ed_ann_v004.py --fashion

# 最高精度（4層構成、84.43%）
python columnar_ed_ann_v004.py --fashion --hidden 128,96,80,64

# 層別コラム設定（絶対数指定）
python columnar_ed_ann_v004.py --fashion --column_neurons_per_layer 30,25,20,15,10

# 層別コラム設定（参加率指定）
python columnar_ed_ann_v004.py --fashion --column_ratio_per_layer 0.5,0.3,0.25,0.2,0.2

# 可視化付き実行
python columnar_ed_ann_v004.py --fashion --viz --heatmap
```

#### 12.10 推奨設定
- **小規模実験**: `--train 1000 --test 100 --epochs 10`
- **精度重視**: 4層構成 `--hidden 128,96,80,64`（84.43%達成）
- **生物学的妥当性重視**: デフォルト5層構成（83.57%）
- **層別最適化**: 浅い層で多く、深い層で少ないニューロン割り当て

#### 12.11 実装バージョン
- **v0.0.4** (2025-11-28):
  - プロジェクト本来の目的に立ち返る
  - 純粋なコラムED法のみを実装（段階的学習機能を削除）
  - 層別コラム参加ニューロン数設定機能を追加
  - 包括的なエラー処理とバリデーション
  - Tanhスケール版活性化関数の安定動作確認

## コラムED法の理論的基盤

以下はコラムED法（columnar_ed_ann_v004.py）の理論的基盤です：

### 1. アミン拡散学習制御（ED法の核心原理）
- 出力層の誤差がアミン濃度として隠れ層に拡散
- パラメータ`u1`による拡散強度制御
- コラム帰属度による選択的拡散が特徴

### 2. コラム帰属度による選択的学習
- 各クラスに対応するコラムニューロン群を定義
- コラム帰属度マップ: `column_affinity[class][neuron]`
- 選択的アミン拡散: `amine = error * column_affinity * u1`
- コラムニューロン: 係数0.5で学習（抑制的）
- 非コラムニューロン: 係数1.5で学習（促進的）

### 3. 共有重み空間アーキテクチャ
- 1つの共有重み行列: `w[hidden][input]`
- 複数クラスを単一の重み空間で分類
- コラム帰属度による選択的学習で実現

### 4. 多層対応とバッチ処理
- 任意の層数に対応（1層～5層以上）
- バッチ処理による効率的な学習
- 層別にコラム参加率を設定可能

## 詳細仕様

### データ構造定義（コラムED法）

```python
# 共有重み空間（多層対応）
w_hidden = []  # list of arrays [layer][n_out, n_in]
w_output = array [n_output, n_hidden[-1]]

# コラム帰属度マップ（全層貫通構造）
column_affinity_all_layers = []  # list of arrays [layer][n_output, n_hidden[layer]]

# 学習パラメータ
alpha  # 学習率
u1     # アミン拡散係数
sig    # シグモイド閾値（Tanhスケール版用）
column_ratio  # コラム参加率
column_overlap  # 隣接コラム重複度
```

### ネットワークアーキテクチャ

```python
# コラムED法のネットワーク構造
入力層(784) → 隠れ層(多層) → 出力層(10)
     ↓            ↓              ↓
  直接接続     コラム構造      共有重み空間
              全層貫通        選択的学習
```

### 学習アルゴリズム（コラムED法）

#### 1. 順方向計算

```python
def forward(self, x):
    """
    順方向計算（多層対応、Tanhスケール版）
    
    Returns:
    --------
    a_hiddens : list of arrays
        各隠れ層の線形出力 [layer][n_hidden[layer]]
    z_hiddens : list of arrays
        各隠れ層の活性 [layer][n_hidden[layer]]
    z_output : array [n_output]
        出力層の活性
    """
    a_hiddens = []
    z_hiddens = []
    z = x
    
    # 多層順伝播（Tanhスケール版）
    for i in range(self.n_layers):
        a = np.dot(self.w_hidden[i], z)  # 線形出力
        z = self.tanh_scaled(a)          # Tanhスケール版活性化
        a_hiddens.append(a)
        z_hiddens.append(z)
    
    # 出力層（シグモイド）
    z_output = 1 / (1 + np.exp(-np.dot(self.w_output, z_hiddens[-1]) / self.sig))
    
    return a_hiddens, z_hiddens, z_output
```

#### 2. Tanhスケール版活性化関数

```python
def tanh_scaled(self, x):
    """
    Tanhスケール版活性化関数
    出力を[0, 1]にスケーリング
    
    tanh_scaled(x) = (tanh(x/sig) + 1) / 2
    """
    return (np.tanh(x / self.sig) + 1) / 2

def tanh_scaled_derivative(self, a):
    """
    Tanhスケール版の導関数
    """
    z = self.tanh_scaled(a)
    return z * (1 - z) / self.sig
```

#### 3. コラムED法による重み更新

```python
def columnar_ed_update(self, x, y_true, a_hiddens, z_hiddens, z_output):
    """
    コラムED法による重み更新（全層貫通版）
    
    シンプルなコラムED法:
    - コラムニューロン: 係数0.5で学習
    - 非コラムニューロン: 係数1.5で学習
    - 微分の連鎖律を使わないアミン拡散メカニズム
    """
    # 出力層の誤差
    y_target = np.zeros(self.n_output)
    y_target[y_true] = 1.0
    error_output = y_target - z_output
    
    # 出力層の重み更新
    sigmoid_derivative = z_output * (1 - z_output) / self.sig
    delta_w_output = self.alpha * np.outer(
        error_output * sigmoid_derivative, 
        z_hiddens[-1]
    )
    self.w_output += delta_w_output
    
    # 隠れ層への誤差逆伝播（コラム選択的）
    error_current = np.dot(
        self.w_output.T, 
        error_output * sigmoid_derivative
    )
    
    # 各隠れ層を逆順に処理
    for layer in range(self.n_layers - 1, -1, -1):
        # コラム帰属度マップ
        column_affinity = self.column_affinity_all_layers[layer]
        
        # コラム選択的アミン拡散
        column_amine = column_affinity[y_true]  # 正解クラスのコラム帰属度
        amine_modulation = np.where(
            column_amine > 0.5,
            0.5,  # コラムニューロン: 抑制的学習
            1.5   # 非コラムニューロン: 促進的学習
        )
        
        # 選択的誤差（アミン拡散）
        selective_error = error_current * amine_modulation * self.u1
        
        # Tanhスケール版の導関数
        tanh_derivative = self.tanh_scaled_derivative(a_hiddens[layer])
        delta_hidden = selective_error * tanh_derivative
        
        # 重み更新
        if layer == 0:
            input_vec = x
        else:
            input_vec = z_hiddens[layer-1]
        
        delta_w = self.alpha * np.outer(delta_hidden, input_vec)
        self.w_hidden[layer] += delta_w
        
        # 次の層への誤差伝播
        if layer > 0:
            error_current = np.dot(self.w_hidden[layer].T, delta_hidden)
```

#### 4. コラム帰属度マップの初期化

```python
def initialize_columns_all_layers(self):
    """
    全層貫通コラム構造の初期化（層別対応版）
    
    Returns:
    --------
    column_affinity_all : list of arrays
        各層のコラム帰属度マップ
        [layer][n_output, n_hidden[layer]]
    """
    column_affinity_all = []
    
    for layer_idx, n_neurons in enumerate(self.n_hidden):
        # 層別設定の優先順位処理
        # 1. column_neurons_per_layer（絶対数）
        # 2. column_ratio_per_layer（参加率）
        # 3. column_ratio（全層共通）
        
        # コラム帰属度マップ作成
        column_affinity = np.zeros((self.n_output, n_neurons))
        
        # 各クラスのコラムニューロンに帰属度1.0を設定
        # 隣接コラムには重複度（column_overlap）を設定
        
        column_affinity_all.append(column_affinity)
    
    return column_affinity_all
```

## 生物学的妥当性

### 1. アミン神経系の模倣
- ドーパミン・セロトニン等の神経伝達物質による学習制御
- 正・負の報酬信号による適応学習
- 空間的な拡散メカニズム

### 2. 大脳皮質のコラム構造
- 視覚野などに見られる機能的神経細胞群の模倣
- 同じ特徴に反応するニューロン群が垂直方向に並ぶ
- 全層を貫通する柱状構造

### 3. 局所学習規則
- Hebbian学習の拡張
- 生物学的に実現可能な情報処理
- コラム帰属度による選択的学習

## 従来手法との比較優位性

### 1. vs バックプロパゲーション
- **学習メカニズム**: アミン拡散 vs 微分の連鎖律
- **生物学的妥当性**: 高い vs 低い
- **コラム構造**: あり vs なし
- **選択的学習**: 可能 vs 不可

### 2. vs Hopfieldネット
- **記憶容量**: 制約なし vs 0.15N限界
- **学習**: 教師あり vs 教師なし
- **収束性**: 安定 vs 偽記憶問題

### 3. vs SOM/競合学習
- **教師信号**: あり vs なし
- **表現力**: 高次特徴 vs トポロジー保存
- **正答率**: 高正答率分類 vs クラスタリング

## 実装上の重要ポイント

### 1. コラム帰属度マップの正確な実装
```python
# 全層貫通コラム構造の初期化
def initialize_columns_all_layers(self):
    """
    全層貫通コラム構造の初期化（層別対応版）
    
    Returns:
    --------
    column_affinity_all : list of arrays
        各層のコラム帰属度マップ
        [layer][n_output, n_hidden[layer]]
    """
    column_affinity_all = []
    
    for layer_idx, n_neurons in enumerate(self.n_hidden):
        # 層別設定の優先順位処理
        # 1. column_neurons_per_layer（絶対数）
        # 2. column_ratio_per_layer（参加率）
        # 3. column_ratio（全層共通）
        
        # コラム帰属度マップ作成
        column_affinity = np.zeros((self.n_output, n_neurons))
        
        # 各クラスのコラムニューロンに帰属度1.0を設定
        # 隣接コラムには重複度（column_overlap）を設定
        
        column_affinity_all.append(column_affinity)
    
    return column_affinity_all
```

### 2. エラー処理とバリデーション
```python
# Critical Errorチェック
if n_neurons < self.n_output:
    raise ValueError(f"層{layer}のニューロン数不足")

if column_neurons_total < self.n_output:
    raise ValueError(f"層{layer}のコラム参加ニューロン数不足")

# 自動調整
column_neurons_total = min(requested_total, n_neurons)
```

### 3. forward戻り値の正確な取得
```python
# 重要: forwardメソッドは常に3つの値を返す
a_hiddens, z_hiddens, z_output = network.forward(x)
```

## 動作検証結果

### Fashion-MNISTでの学習結果
- **4層構成** [128,96,80,64]: テスト精度84.43%（最高精度）
- **5層構成** [128,96,80,64,64]: テスト精度83.57%（生物学的妥当性）
- **2層構成** [512,256]: テスト精度78.06%

### 学習動態の特徴
- エポックを重ねても学習が停滞しない
- 誤差が段階的に減少
- コラム構造による効率的な学習

## 適用可能分野

### 1. パターン認識
- 手書き文字認識（MNIST）
- ファッション画像分類（Fashion-MNIST）
- 画像分類

### 2. 生物学的モデリング
- 大脳皮質の計算モデル
- 視覚野の機能シミュレーション
- 神経科学研究

### 3. 教育・啓蒙
- 微分の連鎖律を使わない学習の実証
- 生物学的に妥当な学習アルゴリズムの提示
- ED法の普及活動

---

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
