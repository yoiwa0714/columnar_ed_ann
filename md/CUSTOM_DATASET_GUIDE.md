# カスタムデータセット対応ガイド

## 概要

v027.3のPhase 2では、カスタムデータセットに対応しました。任意の画像データセット（または数値データセット）を使って、コラムED法で学習を実行できます。

## カスタムデータセットの準備

### 1. ディレクトリ構造

カスタムデータセットは以下の構造で準備してください：

```
my_custom_data/
├── metadata.json       # メタデータ（必須）
├── x_train.npy        # 訓練データ（必須）
├── y_train.npy        # 訓練ラベル（必須）
├── x_test.npy         # テストデータ（必須）
└── y_test.npy         # テストラベル（必須）
```

### 2. metadata.jsonの形式

```json
{
    "name": "my_custom_data",
    "n_classes": 10,
    "input_shape": [28, 28],
    "normalize": true,
```json
{
    "name": "my_custom_data",
    "n_classes": 10,
    "input_shape": [28, 28],
    "normalize": true,
    "class_names": ["class0", "class1", "class2", "class3", "class4",
                    "class5", "class6", "class7", "class8", "class9"],
    "description": "データセットの説明（オプション）"
}
```

**フィールド説明:**
- `name`: データセット名（文字列、必須）
- `n_classes`: クラス数（整数、必須）
- `input_shape`: 入力データの形状（配列、必須）
  - 例: `[28, 28]` (グレースケール画像)
  - 例: `[32, 32, 3]` (カラー画像)
- `normalize`: 正規化が必要か（boolean、必須）
  - `true`: データは0-255の範囲で、0-1に正規化する
  - `false`: データは既に正規化済み（0-1）
- `class_names`: クラス名のリスト（配列、オプション）
  - 長さはn_classesと同じである必要があります
  - 指定すると学習結果表示時にクラス名が使用されます
- `description`: データセットの説明（文字列、オプション）

### 3. データファイル (.npy) の形式

**訓練データ (x_train.npy):**
- 形状: `[n_samples, height, width]` または `[n_samples, height, width, channels]`
- 型: `uint8` (0-255) または `float32` (0-1)

**訓練ラベル (y_train.npy):**
- 形状: `[n_samples]` または `[n_samples, 1]`
- 型: `int32` または `int64`
- 値: 0からn_classes-1の整数

**テストデータ (x_test.npy) / テストラベル (y_test.npy):**
- 訓練データと同じ形式

## データセットの配置場所

カスタムデータセットは以下のいずれかに配置できます：

### オプション1: 標準ディレクトリに配置（推奨）

```bash
~/.keras/datasets/my_custom_data/
```

この場合、データセット名だけで指定できます：

```bash
python columnar_ed_ann_v027_3.py --dataset my_custom_data
```

### オプション2: 任意のパスに配置

任意の場所に配置し、フルパスで指定：

```bash
python columnar_ed_ann_v027_3.py --dataset /path/to/my_custom_data
```

### オプション3: カレントディレクトリに配置

プロジェクトディレクトリ内に配置：

```bash
./my_custom_data/
```

データセット名で指定可能：

```bash
python columnar_ed_ann_v027_3.py --dataset my_custom_data
```

## カスタムデータセットの検証機能

v027.3以降、カスタムデータセット読み込み時に以下の検証が自動的に実行されます。

### 検証項目

1. **データ型チェック**
   - データとラベルがNumPy配列であることを確認

2. **欠損値チェック**
   - データにNaNやInfが含まれていないかチェック
   - ラベルにNaNが含まれていないかチェック

3. **ラベル範囲チェック**
   - すべてのラベルが0からn_classes-1の範囲内であることを確認
   - 範囲外のラベルがあればエラーを表示

4. **整合性チェック**
   - 訓練データとラベルのサンプル数が一致するか確認
   - テストデータとラベルのサンプル数が一致するか確認
   - 訓練データとテストデータの次元が一致するか確認

5. **クラス分布の表示**
   - 各クラスのサンプル数を表示
   - 存在しないクラスがあれば警告を表示

### 検証出力例

```
カスタムデータセットの検証中...
  ✓ データ型: OK
  ✓ 欠損値: なし
  ✓ ラベル範囲: [0, 9]
  ✓ 整合性: OK

  クラス分布:
    訓練データ: 10クラス (Class [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    テストデータ: 10クラス (Class [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

  各クラスのサンプル数:
    Class 0: Train= 100, Test=  20
    Class 1: Train= 100, Test=  20
    ...

データセット 'test_custom_mnist' の検証完了
```

### 検証のスキップ

検証は**カスタムデータセットのみ**で実行され、標準データセット（MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100）ではスキップされます。これは標準データセットが既にTensorFlow/Kerasによって検証されているためです。

## 使用例

### 1. テスト用データセットの作成

付属のスクリプトを使用：

```bash
python create_test_dataset.py
```

これにより、`~/.keras/datasets/test_custom_mnist/` にMNISTのサブセットが作成されます。

### 2. カスタムデータセットでの学習

**名前指定（標準ディレクトリから自動検索）:**

```bash
python columnar_ed_ann_v027_3.py --dataset test_custom_mnist --hidden 64 --epochs 10 --seed 42
```

**パス指定:**

```bash
python columnar_ed_ann_v027_3.py --dataset ~/.keras/datasets/test_custom_mnist --hidden 64 --epochs 10 --seed 42
```

### 3. データセット自動検出の動作確認

以下のような出力が表示されれば成功です：

```
データ読み込み中... (訓練:None, テスト:None, データセット:test_custom_mnist)
カスタムデータセット 'test_custom_mnist' を読み込みました
  入力形状: [28, 28]
  クラス数: 10
データセット情報: 入力次元=784, クラス数=10
```

## 既存データセットからのカスタムデータセット作成

### MNISTから作成する例

```python
import os
import json
import numpy as np
from tensorflow import keras

# データ読み込み
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 保存先
output_dir = os.path.expanduser('~/.keras/datasets/my_mnist_subset')
os.makedirs(output_dir, exist_ok=True)

# サブセット作成（例: 最初の5000サンプル）
np.save(os.path.join(output_dir, 'x_train.npy'), x_train[:5000])
np.save(os.path.join(output_dir, 'y_train.npy'), y_train[:5000])
np.save(os.path.join(output_dir, 'x_test.npy'), x_test[:1000])
np.save(os.path.join(output_dir, 'y_test.npy'), y_test[:1000])

# metadata.json作成
metadata = {
    "name": "my_mnist_subset",
    "n_classes": 10,
    "input_shape": [28, 28],
    "normalize": True,
    "description": "MNIST subset for testing"
}

with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
```

### CIFAR-10から作成する例

```python
import os
import json
import numpy as np
from tensorflow import keras

# データ読み込み
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 保存先
output_dir = os.path.expanduser('~/.keras/datasets/my_cifar10_subset')
os.makedirs(output_dir, exist_ok=True)

# サブセット作成
np.save(os.path.join(output_dir, 'x_train.npy'), x_train[:10000])
np.save(os.path.join(output_dir, 'y_train.npy'), y_train[:10000])
np.save(os.path.join(output_dir, 'x_test.npy'), x_test[:2000])
np.save(os.path.join(output_dir, 'y_test.npy'), y_test[:2000])

# metadata.json作成
metadata = {
    "name": "my_cifar10_subset",
    "n_classes": 10,
    "input_shape": [32, 32, 3],  # カラー画像
    "normalize": True,
    "description": "CIFAR-10 subset for testing"
}

with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
```

## パス解決の優先順位

`--dataset`引数で指定された値は、以下の優先順位で解決されます：

1. **標準データセット名** (`mnist`, `fashion`, `cifar10`, `cifar100`)
   - TensorFlow/Kerasの標準データセットとして読み込み
   
2. **指定されたパスそのまま**
   - 相対パスまたは絶対パスとして存在するか確認
   
3. **標準ディレクトリ内**
   - `~/.keras/datasets/{dataset_name}` に存在するか確認
   
4. **カレントディレクトリ内**
   - `./{dataset_name}` に存在するか確認

見つからない場合はエラーメッセージが表示されます。

## トラブルシューティング

### metadata.json関連エラー

#### エラー: "metadata.json が見つかりません"

**原因:** カスタムデータセットディレクトリにmetadata.jsonがない

**解決策:**
```bash
# metadata.jsonを作成
cat > ~/.keras/datasets/my_data/metadata.json << 'EOF'
{
    "name": "my_data",
    "n_classes": 10,
    "input_shape": [28, 28],
    "normalize": true
}
EOF
```

#### エラー: "metadata.jsonのJSON形式が不正です"

**原因:** JSON構文エラー（カンマ、括弧、引用符の不備）

**解決策:**
1. JSONフォーマッタで確認
2. 以下をチェック：
   - 全ての文字列が`""`で囲まれているか
   - 末尾のカンマが余分に付いていないか
   - 括弧が正しく閉じられているか
   - コメント（//）が含まれていないか（JSONではコメント不可）

**正しい例:**
```json
{
    "name": "my_data",
    "n_classes": 10,
    "input_shape": [28, 28],
    "normalize": true
}
```

**誤った例:**
```json
{
    "name": "my_data",
    "n_classes": 10,
    "input_shape": [28, 28],
    "normalize": true,  // ← 末尾のカンマは不可
}
```

#### エラー: "metadata.jsonに必須フィールドがありません"

**原因:** `name`, `n_classes`, `input_shape`のいずれかが欠けている

**解決策:** 必須フィールドを全て含めてください：
```json
{
    "name": "データセット名",
    "n_classes": クラス数,
    "input_shape": [高さ, 幅] または [高さ, 幅, チャンネル]
}
```

#### エラー: "metadata.jsonのフィールドの型が不正です"

**原因:** フィールドの型が期待値と異なる

**解決策:** 型を修正：
- `name`: 文字列（例: `"my_dataset"`）
- `n_classes`: 整数（例: `10`）
- `input_shape`: リスト（例: `[28, 28]` または `[32, 32, 3]`）
- `normalize`: 真偽値（`true` または `false`）
- `class_names`: 文字列のリスト（例: `["class0", "class1", ...]`）

#### エラー: "class_namesの要素数がn_classesと一致しません"

**原因:** class_namesの長さがn_classesと異なる

**解決策:**
```json
{
    "n_classes": 3,
    "class_names": ["class0", "class1", "class2"]  // ← 3個必要
}
```

### データセットパス関連エラー

#### エラー: "データセット 'xxx' が見つかりません"

**原因:** 指定したデータセットが検索パスに存在しない

**エラーメッセージから候補を確認:**
```
類似する利用可能なデータセット:
  - test_custom_mnist
  - my_custom_data

使用方法:
  --dataset test_custom_mnist
```

**解決策:**
1. エラーメッセージの候補を使用
2. フルパスで指定してみる
3. データセットが正しい場所にあるか確認：
   ```bash
   ls -la ~/.keras/datasets/
   ```
4. ディレクトリ名のスペルミスを確認

### データファイル関連エラー

#### エラー: "x_train.npy が見つかりません"

**原因:** 必須ファイルが不足している

**解決策:** 以下の5つのファイルが全て存在することを確認
```bash
ls -la ~/.keras/datasets/my_data/
# 必要:
# - metadata.json
# - x_train.npy
# - y_train.npy
# - x_test.npy
# - y_test.npy
```

#### エラー: "データファイルの読み込み中にエラーが発生しました"

**原因1:** ファイルが破損している  
**解決策:** ファイルを再作成

**原因2:** NumPy形式で保存されていない  
**解決策:** 正しい方法で保存：
```python
import numpy as np
np.save('x_train.npy', x_train)  # これが正しい
```

**原因3:** ファイルサイズが0バイト  
**解決策:** データが正しく保存されているか確認：
```bash
ls -lh ~/.keras/datasets/my_data/*.npy
```

### データ検証エラー

#### エラー: "データはNumPy配列である必要があります"

**原因:** データの型が不正

**解決策:**
```python
import numpy as np
x_train = np.array(x_train)  # リストから変換
np.save('x_train.npy', x_train)
```

#### エラー: "データにNaNが含まれています"

**原因:** データに欠損値（NaN）が含まれている

**解決策（3つの方法）:**
```python
# 方法1: NaNを0で埋める
x_train = np.nan_to_num(x_train, nan=0.0)

# 方法2: NaNを平均値で埋める
x_train[np.isnan(x_train)] = np.nanmean(x_train)

# 方法3: NaNのある行を削除
mask = ~np.isnan(x_train).any(axis=1)
x_train = x_train[mask]
y_train = y_train[mask]
```

#### エラー: "データにInfが含まれています"

**原因:** データに無限大（Inf）が含まれている

**解決策:**
```python
# 方法1: Infを0で置き換える
x_train = np.nan_to_num(x_train, posinf=0.0, neginf=0.0)

# 方法2: Infを最大値で置き換える
x_train[np.isinf(x_train)] = np.finfo(np.float32).max
```

#### エラー: "訓練ラベルが範囲外です"

**原因:** ラベルが0からn_classes-1の範囲外

**エラー例:**
```
期待範囲: [0, 9]
実際の範囲: [1, 10]
範囲外ラベル: [10] （100個）
```

**解決策:**
```python
# ラベルを0から始まるように調整
y_train = y_train - 1
y_test = y_test - 1

# または、metadata.jsonのn_classesを修正
# n_classes: 11 (ラベルが1-10の場合)
```

#### エラー: "訓練データとラベルのサンプル数が一致しません"

**原因:** データとラベルの行数が異なる

**エラー例:**
```
x_train: 5000サンプル （shape: (5000, 784)）
y_train: 4800サンプル （shape: (4800,)）
```

**解決策:**
```python
# 短い方に合わせる
min_samples = min(len(x_train), len(y_train))
x_train = x_train[:min_samples]
y_train = y_train[:min_samples]
```

#### エラー: "訓練データとテストデータの次元が一致しません"

**原因:** フラット化方法が異なる、またはinput_shapeが不正

**エラー例:**
```
x_train: 784次元 （shape: (5000, 784)）
x_test: 1024次元 （shape: (1000, 1024)）
```

**解決策:**
```python
# 両方を同じ方法でreshape
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# metadata.jsonのinput_shapeを確認
# 28×28=784, 32×32=1024
```

### パフォーマンス関連

#### 大規模データセット（100MB以上）の読み込みが遅い

**症状:** データ読み込みに時間がかかる

**自動最適化:** v027.3では100MB以上のデータセットで自動的にメモリマップモードを使用します
```
大規模データセット検出 (250.5 MB)
  メモリマップモードで読み込み中...
  ✓ データファイル読み込み完了（メモリマップモード）
```

**さらに高速化したい場合:**
1. サンプル数を制限して使用：
   ```bash
   python columnar_ed_ann_v027_3.py --dataset my_data --train 10000 --test 2000
   ```

2. データを圧縮形式ではなく.npy形式で保存（既に最適）

#### メモリ不足エラー

**原因:** データセットが大きすぎてメモリに収まらない

**解決策:**
1. サンプル数を制限：
   ```bash
   python columnar_ed_ann_v027_3.py --dataset my_data --train 5000 --test 1000
   ```

2. メモリマップモードが有効か確認（自動適用）

3. 入力データを小さく（例: 画像サイズを縮小）

### その他のエラー

#### クラス分布の警告

**警告例:**
```
警告: 訓練データに存在しないクラス: [7, 8, 9]
```

**原因:** 一部のクラスのサンプルが存在しない

**影響:** 存在しないクラスは学習されない

**対応:**
- 意図的な場合: 無視可能
- 意図しない場合: データセット作成を見直し、全クラスのサンプルを含める

#### 実行例との出力の違い

**期待:** クラス名が表示される  
**実際:** クラス名が表示されない

**原因:** metadata.jsonにclass_namesフィールドがない

**解決策（オプション）:**
```json
{
    "name": "my_data",
    "n_classes": 10,
    "input_shape": [28, 28],
    "normalize": true,
    "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
}
```

## 機能の利点

### 1. 柔軟性
- 任意のデータセットを簡単に使用可能
- 標準データセットと同じインターフェースで扱える

### 2. 再現性
- metadata.jsonでデータ構造を明示
- 実験の再現が容易

### 3. 自動化
- 入力次元とクラス数を自動検出
- ネットワーク構造が自動調整される

### 4. 互換性
- 標準データセット（MNIST, CIFAR-10など）と同じ使い勝手
- 既存の学習パラメータがそのまま使用可能

## まとめ

Phase 2のカスタムデータセット対応により、v027.3は以下を実現しました：

✅ 任意のデータセットに対応可能  
✅ metadata.jsonによる柔軟な管理  
✅ 標準ディレクトリ検索の自動化  
✅ 入力次元・クラス数の完全自動検出  
✅ 標準データセットとの統一インターフェース  

これにより、研究者は独自のデータセットでも簡単にコラムED法を試すことができます。
