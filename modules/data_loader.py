#!/usr/bin/env python3
"""
データセット読み込み・前処理モジュール

役割:
  - MNIST/Fashion-MNISTの読み込み（TensorFlow/Keras統合）
  - データ正規化・フラット化
  - TensorFlow Dataset APIによるシャッフル・バッチ処理
  - クラス名取得

関数:
  - load_dataset: データ読み込みと前処理（NumPy配列を返す）
  - create_tf_dataset: TensorFlow Dataset API使用（tf.data.Datasetを返す）
  - get_class_names: データセット別クラス名取得

使用例:
    from modules.data_loader import load_dataset, create_tf_dataset, get_class_names
    
    # NumPy配列として読み込み（従来互換）
    (x_train, y_train), (x_test, y_test) = load_dataset(
        dataset='mnist',
        train_samples=3000, 
        test_samples=1000
    )
    
    # TensorFlow Datasetとして作成（シャッフル・バッチ対応）
    train_dataset = create_tf_dataset(
        x_train, y_train,
        batch_size=128,
        shuffle=True,
        seed=42
    )
    
    # クラス名取得
    class_names = get_class_names('fashion')
    # → ['T-shirt/top', 'Trouser', ...]
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_class_names(dataset_name, custom_class_names=None):
    """
    データセットのクラス名を取得
    
    Args:
        dataset_name: データセット名（'mnist', 'fashion', 'cifar10', 'cifar100'）
        custom_class_names: カスタムデータセットのクラス名（リスト、オプション）
    
    Returns:
        class_names: クラス名のリスト（クラス名情報がない場合はNone）
    """
    # カスタムクラス名が指定されていれば優先
    if custom_class_names is not None:
        return custom_class_names
    
    if dataset_name == 'mnist':
        # MNISTはクラス名情報を持たない（数字そのまま）
        return None
    elif dataset_name == 'fashion':
        # Fashion-MNISTのクラス名
        return [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    elif dataset_name == 'cifar10':
        # CIFAR-10のクラス名
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    elif dataset_name == 'cifar100':
        # CIFAR-100のクラス名（20のスーパークラス）
        # 注: CIFAR-100は100個の細かいクラスがあるため、必要に応じて拡張
        return None  # 100クラス分の名前は長いため、必要に応じて実装
    else:
        # 未知のデータセット
        return None


def validate_custom_dataset(x_train, y_train, x_test, y_test, metadata):
    """
    カスタムデータセットの検証
    
    Args:
        x_train: 訓練データ
        y_train: 訓練ラベル
        x_test: テストデータ
        y_test: テストラベル
        metadata: メタデータ辞書
    
    Raises:
        ValueError: データに問題がある場合
    
    検証項目:
        1. データ型チェック
        2. 欠損値チェック（NaN, Inf）
        3. ラベル範囲チェック（0からn_classes-1）
        4. 整合性チェック（trainとtestの次元一致）
        5. クラス分布の確認
    """
    print("\nカスタムデータセットの検証中...")
    
    n_classes = metadata.get('n_classes')
    dataset_name = metadata.get('name', 'unknown')
    
    # 1. データ型チェック
    if not isinstance(x_train, np.ndarray) or not isinstance(x_test, np.ndarray):
        raise ValueError(
            f"データはNumPy配列である必要があります\n"
            f"  x_train型: {type(x_train).__name__}\n"
            f"  x_test型: {type(x_test).__name__}\n"
            f"\n修正方法:\n"
            f"  import numpy as np\n"
            f"  x_train = np.array(x_train)  # リストから変換\n"
            f"  np.save('x_train.npy', x_train)  # NumPy形式で保存"
        )
    
    if not isinstance(y_train, np.ndarray) or not isinstance(y_test, np.ndarray):
        raise ValueError(
            f"ラベルはNumPy配列である必要があります\n"
            f"  y_train型: {type(y_train).__name__}\n"
            f"  y_test型: {type(y_test).__name__}\n"
            f"\n修正方法:\n"
            f"  import numpy as np\n"
            f"  y_train = np.array(y_train)  # リストから変換\n"
            f"  np.save('y_train.npy', y_train)  # NumPy形式で保存"
        )
    
    # 2. 欠損値チェック
    nan_info = []
    if np.any(np.isnan(x_train)):
        nan_count = np.sum(np.isnan(x_train))
        nan_indices = np.argwhere(np.isnan(x_train))
        nan_info.append(f"  x_train: {nan_count}個のNaN（最初の位置: {nan_indices[0] if len(nan_indices) > 0 else 'N/A'}）")
    if np.any(np.isnan(x_test)):
        nan_count = np.sum(np.isnan(x_test))
        nan_indices = np.argwhere(np.isnan(x_test))
        nan_info.append(f"  x_test: {nan_count}個のNaN（最初の位置: {nan_indices[0] if len(nan_indices) > 0 else 'N/A'}）")
    
    if nan_info:
        raise ValueError(
            f"データにNaNが含まれています:\n" + "\n".join(nan_info) +
            f"\n\n修正方法:\n"
            f"  1. 欠損値を0で埋める: x_train = np.nan_to_num(x_train, nan=0.0)\n"
            f"  2. 欠損値を平均値で埋める: x_train[np.isnan(x_train)] = np.nanmean(x_train)\n"
            f"  3. 欠損値のある行を削除: x_train = x_train[~np.isnan(x_train).any(axis=1)]"
        )
    
    inf_info = []
    if np.any(np.isinf(x_train)):
        inf_count = np.sum(np.isinf(x_train))
        inf_indices = np.argwhere(np.isinf(x_train))
        inf_info.append(f"  x_train: {inf_count}個のInf（最初の位置: {inf_indices[0] if len(inf_indices) > 0 else 'N/A'}）")
    if np.any(np.isinf(x_test)):
        inf_count = np.sum(np.isinf(x_test))
        inf_indices = np.argwhere(np.isinf(x_test))
        inf_info.append(f"  x_test: {inf_count}個のInf（最初の位置: {inf_indices[0] if len(inf_indices) > 0 else 'N/A'}）")
    
    if inf_info:
        raise ValueError(
            f"データにInfが含まれています:\n" + "\n".join(inf_info) +
            f"\n\n修正方法:\n"
            f"  1. Infを0で置き換える: x_train = np.nan_to_num(x_train, posinf=0.0, neginf=0.0)\n"
            f"  2. Infを最大値で置き換える: x_train[np.isinf(x_train)] = np.finfo(np.float32).max"
        )
    
    if np.any(np.isnan(y_train)) or np.any(np.isnan(y_test)):
        raise ValueError(
            f"ラベルにNaNが含まれています\n"
            f"  y_train NaN数: {np.sum(np.isnan(y_train))}\n"
            f"  y_test NaN数: {np.sum(np.isnan(y_test))}\n"
            f"\n修正方法:\n"
            f"  ラベルは整数である必要があります。NaNが含まれる場合は、\n"
            f"  データセットの作成過程を確認してください。"
        )
    
    # 3. ラベル範囲チェック
    if np.min(y_train) < 0 or np.max(y_train) >= n_classes:
        invalid_labels = y_train[(y_train < 0) | (y_train >= n_classes)]
        unique_invalid = np.unique(invalid_labels)
        raise ValueError(
            f"訓練ラベルが範囲外です\n"
            f"  期待範囲: [0, {n_classes-1}]\n"
            f"  実際の範囲: [{np.min(y_train)}, {np.max(y_train)}]\n"
            f"  範囲外ラベル: {unique_invalid.tolist()} （{len(invalid_labels)}個）\n"
            f"\n修正方法:\n"
            f"  1. metadata.jsonのn_classesを確認: 実際のクラス数と一致しているか\n"
            f"  2. ラベルが0から始まっているか確認: y_train = y_train - min(y_train)\n"
            f"  3. ラベルの最大値を確認: max(y_train) == n_classes - 1 であるべき"
        )
    
    if np.min(y_test) < 0 or np.max(y_test) >= n_classes:
        invalid_labels = y_test[(y_test < 0) | (y_test >= n_classes)]
        unique_invalid = np.unique(invalid_labels)
        raise ValueError(
            f"テストラベルが範囲外です\n"
            f"  期待範囲: [0, {n_classes-1}]\n"
            f"  実際の範囲: [{np.min(y_test)}, {np.max(y_test)}]\n"
            f"  範囲外ラベル: {unique_invalid.tolist()} （{len(invalid_labels)}個）\n"
            f"\n修正方法:\n"
            f"  1. metadata.jsonのn_classesを確認: 実際のクラス数と一致しているか\n"
            f"  2. ラベルが0から始まっているか確認: y_test = y_test - min(y_test)\n"
            f"  3. ラベルの最大値を確認: max(y_test) == n_classes - 1 であるべき"
        )
    
    # 4. 整合性チェック
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"訓練データとラベルのサンプル数が一致しません\n"
            f"  x_train: {x_train.shape[0]}サンプル （shape: {x_train.shape}）\n"
            f"  y_train: {y_train.shape[0]}サンプル （shape: {y_train.shape}）\n"
            f"\n修正方法:\n"
            f"  データとラベルのサンプル数を揃えてください:\n"
            f"  x_train = x_train[:min_samples]  # 短い方に合わせる\n"
            f"  y_train = y_train[:min_samples]"
        )
    
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"テストデータとラベルのサンプル数が一致しません\n"
            f"  x_test: {x_test.shape[0]}サンプル （shape: {x_test.shape}）\n"
            f"  y_test: {y_test.shape[0]}サンプル （shape: {y_test.shape}）\n"
            f"\n修正方法:\n"
            f"  データとラベルのサンプル数を揃えてください:\n"
            f"  x_test = x_test[:min_samples]  # 短い方に合わせる\n"
            f"  y_test = y_test[:min_samples]"
        )
    
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            f"訓練データとテストデータの次元が一致しません\n"
            f"  x_train: {x_train.shape[1]}次元 （shape: {x_train.shape}）\n"
            f"  x_test: {x_test.shape[1]}次元 （shape: {x_test.shape}）\n"
            f"\n考えられる原因:\n"
            f"  1. 訓練データとテストデータのフラット化方法が異なる\n"
            f"  2. metadata.jsonのinput_shapeと実際のデータ形状が一致していない\n"
            f"\n修正方法:\n"
            f"  データを同じ方法でreshapeしてください:\n"
            f"  x_train = x_train.reshape(x_train.shape[0], -1)\n"
            f"  x_test = x_test.reshape(x_test.shape[0], -1)"
        )
    
    # 5. クラス分布の確認
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)
    
    print(f"  ✓ データ型: OK")
    print(f"  ✓ 欠損値: なし")
    print(f"  ✓ ラベル範囲: [0, {n_classes-1}]")
    print(f"  ✓ 整合性: OK")
    print(f"\n  クラス分布:")
    print(f"    訓練データ: {len(unique_train)}クラス (Class {list(unique_train)})")
    print(f"    テストデータ: {len(unique_test)}クラス (Class {list(unique_test)})")
    
    # クラスごとのサンプル数を表示
    print(f"\n  各クラスのサンプル数:")
    for class_idx in range(n_classes):
        train_count = np.sum(y_train == class_idx)
        test_count = np.sum(y_test == class_idx)
        if train_count > 0 or test_count > 0:
            print(f"    Class {class_idx}: Train={train_count:4d}, Test={test_count:4d}")
    
    # 警告: 存在しないクラスがある場合
    missing_classes = set(range(n_classes)) - set(unique_train)
    if missing_classes:
        print(f"\n  警告: 訓練データに存在しないクラス: {sorted(missing_classes)}")
    
    missing_classes_test = set(range(n_classes)) - set(unique_test)
    if missing_classes_test:
        print(f"  警告: テストデータに存在しないクラス: {sorted(missing_classes_test)}")
    
    print(f"\nデータセット '{dataset_name}' の検証完了\n")


def resolve_dataset_path(dataset_spec):
    """
    データセット指定から実際のパスを解決
    
    Args:
        dataset_spec: データセット名（'mnist', 'fashion', etc.）またはパス
    
    Returns:
        resolved_path: 解決されたパス（標準データセット名の場合はそのまま、カスタムデータの場合は絶対パス）
        is_custom: カスタムデータセットかどうか（bool）
    
    動作:
        1. 標準データセット名（'mnist', 'fashion', 'cifar10', 'cifar100'）ならそのまま返す
        2. パス指定の場合、以下の順序で検索:
           a) 指定パスそのまま（絶対パス or 相対パス）
           b) ~/.keras/datasets/{dataset_spec}
           c) カレントディレクトリ/{dataset_spec}
        3. 見つかった場合は絶対パスとis_custom=Trueを返す
        4. 見つからない場合はエラー
    
    使用例:
        >>> path, is_custom = resolve_dataset_path('mnist')
        >>> # → ('mnist', False)
        
        >>> path, is_custom = resolve_dataset_path('my_custom_data')
        >>> # → ('/home/user/.keras/datasets/my_custom_data', True)
        
        >>> path, is_custom = resolve_dataset_path('/path/to/my_data')
        >>> # → ('/path/to/my_data', True)
    """
    # 標準データセット名の判定
    standard_datasets = ['mnist', 'fashion', 'cifar10', 'cifar100']
    if dataset_spec in standard_datasets:
        return dataset_spec, False
    
    # カスタムデータセットの場合、パスを探す
    search_paths = [
        dataset_spec,  # 指定されたパスそのまま
        os.path.expanduser(f'~/.keras/datasets/{dataset_spec}'),  # 標準ディレクトリ
        os.path.join(os.getcwd(), dataset_spec)  # カレントディレクトリ
    ]
    
    for path in search_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path, True
    
    # 見つからない場合、類似ディレクトリを探す
    keras_datasets_dir = os.path.expanduser('~/.keras/datasets')
    suggestions = []
    
    if os.path.exists(keras_datasets_dir):
        try:
            available_dirs = [
                d for d in os.listdir(keras_datasets_dir)
                if os.path.isdir(os.path.join(keras_datasets_dir, d))
            ]
            # 類似度が高い候補を抽出（前方一致または部分一致）
            for dir_name in available_dirs:
                if dataset_spec.lower() in dir_name.lower() or dir_name.lower() in dataset_spec.lower():
                    suggestions.append(dir_name)
        except Exception:
            pass  # エラーは無視（permissions等）
    
    # エラーメッセージの構築
    error_msg = f"データセット '{dataset_spec}' が見つかりません。\n\n"
    error_msg += "検索したパス:\n" + "\n".join(f"  - {p}" for p in search_paths)
    
    if suggestions:
        error_msg += f"\n\n類似する利用可能なデータセット:\n"
        error_msg += "\n".join(f"  - {s}" for s in suggestions[:5])  # 最大5個
        error_msg += f"\n\n使用方法:\n  --dataset {suggestions[0]}"
    
    error_msg += "\n\n標準データセット:\n"
    error_msg += "  - mnist: MNIST手書き数字\n"
    error_msg += "  - fashion: Fashion-MNIST衣類画像\n"
    error_msg += "  - cifar10: CIFAR-10自然画像（10クラス）\n"
    error_msg += "  - cifar100: CIFAR-100自然画像（100クラス）"
    
    error_msg += "\n\nカスタムデータセットの配置場所:\n"
    error_msg += f"  1. 推奨: ~/.keras/datasets/{{dataset_name}}/\n"
    error_msg += f"  2. カレントディレクトリ: ./{{dataset_name}}/\n"
    error_msg += f"  3. 絶対パス指定: /path/to/{{dataset_name}}/"
    
    error_msg += "\n\n詳細はCUSTOM_DATASET_GUIDE.mdを参照してください。"
    
    raise FileNotFoundError(error_msg)


def load_custom_dataset(dataset_path, train_samples=None, test_samples=None):
    """
    カスタムデータセットの読み込み（metadata.json対応）
    
    Args:
        dataset_path: データセットディレクトリの絶対パス
        train_samples: 訓練サンプル数（Noneなら全データ）
        test_samples: テストサンプル数（Noneなら全データ）
    
    Returns:
        (x_train, y_train), (x_test, y_test): 正規化・フラット化済みNumPy配列
    
    ディレクトリ構造:
        dataset_path/
        ├── metadata.json       # メタデータ（必須）
        ├── x_train.npy        # 訓練データ（必須）
        ├── y_train.npy        # 訓練ラベル（必須）
        ├── x_test.npy         # テストデータ（必須）
        └── y_test.npy         # テストラベル（必須）
    
    metadata.json形式:
        {
            "name": "dataset_name",
            "n_classes": 10,
            "input_shape": [28, 28],  // または [32, 32, 3]
            "normalize": true,         // 正規化が必要か（0-255 → 0-1）
            "description": "データセットの説明（オプション）"
        }
    
    使用例:
        >>> (x_train, y_train), (x_test, y_test) = load_custom_dataset(
        ...     '/home/user/.keras/datasets/my_data',
        ...     train_samples=3000,
        ...     test_samples=1000
        ... )
    
    Notes:
        - データは自動的にフラット化されます
        - metadata.jsonのnormalize=trueなら0-1正規化が実行されます
        - ラベルは自動的にflatten()されます
    """
    # metadata.json読み込み
    metadata_path = os.path.join(dataset_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"metadata.json が見つかりません: {metadata_path}\n"
            f"\nカスタムデータセットにはmetadata.jsonが必要です。\n"
            f"\nmetadata.json形式の例:\n"
            f"  {{\n"
            f"    \"name\": \"my_dataset\",\n"
            f"    \"n_classes\": 10,\n"
            f"    \"input_shape\": [28, 28],\n"
            f"    \"normalize\": true,\n"
            f"    \"class_names\": [\"class0\", \"class1\", ...],  // オプション\n"
            f"    \"description\": \"データセットの説明\"  // オプション\n"
            f"  }}\n"
            f"\n詳細はCUSTOM_DATASET_GUIDE.mdを参照してください。"
        )
    
    # JSONパース処理
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"metadata.jsonのJSON形式が不正です: {metadata_path}\n"
            f"エラー詳細: {e}\n"
            f"  行 {e.lineno}, 列 {e.colno}: {e.msg}\n"
            f"\nトラブルシューティング:\n"
            f"  1. JSONフォーマッタでファイルを確認してください\n"
            f"  2. 以下の点をチェックしてください:\n"
            f"     - 全ての文字列は\"\"で囲まれているか\n"
            f"     - 末尾のカンマが余分に付いていないか\n"
            f"     - 全ての括弧が正しく閉じられているか\n"
            f"     - コメント（//）が含まれていないか（JSONではコメント不可）"
        ) from e
    except Exception as e:
        raise IOError(
            f"metadata.jsonの読み込み中にエラーが発生しました: {metadata_path}\n"
            f"エラー詳細: {e}"
        ) from e
    
    # 必須フィールドの検証
    required_fields = {
        'name': str,
        'n_classes': int,
        'input_shape': list
    }
    
    missing_fields = []
    invalid_types = []
    
    for field, expected_type in required_fields.items():
        if field not in metadata:
            missing_fields.append(field)
        elif not isinstance(metadata[field], expected_type):
            actual_type = type(metadata[field]).__name__
            expected_type_name = expected_type.__name__
            invalid_types.append(
                f"  - '{field}': 期待される型={expected_type_name}, 実際の型={actual_type}"
            )
    
    if missing_fields:
        raise ValueError(
            f"metadata.jsonに必須フィールドがありません: {metadata_path}\n"
            f"不足フィールド: {', '.join(missing_fields)}\n"
            f"\n必須フィールド:\n"
            f"  - name: データセット名（文字列）\n"
            f"  - n_classes: クラス数（整数）\n"
            f"  - input_shape: 入力形状（リスト、例: [28, 28] または [32, 32, 3]）\n"
            f"\nオプションフィールド:\n"
            f"  - normalize: 正規化が必要か（真偽値、デフォルト: false）\n"
            f"  - class_names: クラス名リスト（文字列配列）\n"
            f"  - description: データセットの説明（文字列）"
        )
    
    if invalid_types:
        raise TypeError(
            f"metadata.jsonのフィールドの型が不正です: {metadata_path}\n"
            + "\n".join(invalid_types) +
            f"\n\n正しい型:\n"
            f"  - name: 文字列（例: \"my_dataset\"）\n"
            f"  - n_classes: 整数（例: 10）\n"
            f"  - input_shape: リスト（例: [28, 28] または [32, 32, 3]）\n"
            f"  - normalize: 真偽値（true または false）\n"
            f"  - class_names: 文字列のリスト（例: [\"class0\", \"class1\", ...]）"
        )
    
    # n_classesの範囲チェック
    if metadata['n_classes'] <= 0:
        raise ValueError(
            f"n_classesは正の整数である必要があります: {metadata['n_classes']}\n"
            f"metadata.json: {metadata_path}"
        )
    
    # input_shapeの検証
    if not all(isinstance(dim, int) and dim > 0 for dim in metadata['input_shape']):
        raise ValueError(
            f"input_shapeは正の整数のリストである必要があります: {metadata['input_shape']}\n"
            f"例: [28, 28] または [32, 32, 3]\n"
            f"metadata.json: {metadata_path}"
        )
    
    # class_namesの検証（存在する場合）
    if 'class_names' in metadata:
        if not isinstance(metadata['class_names'], list):
            raise TypeError(
                f"class_namesはリストである必要があります\n"
                f"実際の型: {type(metadata['class_names']).__name__}\n"
                f"metadata.json: {metadata_path}"
            )
        if len(metadata['class_names']) != metadata['n_classes']:
            raise ValueError(
                f"class_namesの要素数({len(metadata['class_names'])})が"
                f"n_classes({metadata['n_classes']})と一致しません\n"
                f"metadata.json: {metadata_path}"
            )
        if not all(isinstance(name, str) for name in metadata['class_names']):
            raise TypeError(
                f"class_namesの全要素は文字列である必要があります\n"
                f"metadata.json: {metadata_path}"
            )
    
    # データファイル読み込み
    data_files = {
        'x_train': os.path.join(dataset_path, 'x_train.npy'),
        'y_train': os.path.join(dataset_path, 'y_train.npy'),
        'x_test': os.path.join(dataset_path, 'x_test.npy'),
        'y_test': os.path.join(dataset_path, 'y_test.npy')
    }
    
    # 全ファイルが存在するか確認
    missing_files = []
    for name, path in data_files.items():
        if not os.path.exists(path):
            missing_files.append(f"  - {name}.npy")
    
    if missing_files:
        raise FileNotFoundError(
            f"必須データファイルが見つかりません:\n" +
            "\n".join(missing_files) +
            f"\n\nデータセットディレクトリ: {dataset_path}\n"
            f"\n必須ファイル:\n"
            f"  - x_train.npy: 訓練データ\n"
            f"  - y_train.npy: 訓練ラベル\n"
            f"  - x_test.npy: テストデータ\n"
            f"  - y_test.npy: テストラベル\n"
            f"\nトラブルシューティング:\n"
            f"  1. ディレクトリパスが正しいか確認してください\n"
            f"  2. ファイル名が正確か確認してください（拡張子.npyが必要）\n"
            f"  3. create_test_dataset.pyを参考にデータセットを作成してください"
        )
    
    # NumPy配列として読み込み（エラーハンドリング付き）
    # 大規模データセット対応: ファイルサイズをチェックしてmmap_modeを使用
    file_sizes = {name: os.path.getsize(path) for name, path in data_files.items()}
    total_size = sum(file_sizes.values())
    large_dataset_threshold = 100 * 1024 * 1024  # 100MB
    
    use_mmap = total_size > large_dataset_threshold
    
    if use_mmap:
        print(f"大規模データセット検出 ({total_size / (1024**2):.1f} MB)")
        print(f"  メモリマップモードで読み込み中...")
    
    try:
        # メモリマップモード: 大規模データセットでは'r'を使用
        # 注: reshape/flatten時にコピーが発生するため、サンプル数制限前にコピー
        mmap_mode = 'r' if use_mmap else None
        
        x_train = np.load(data_files['x_train'], mmap_mode=mmap_mode)
        y_train = np.load(data_files['y_train'], mmap_mode=mmap_mode)
        x_test = np.load(data_files['x_test'], mmap_mode=mmap_mode)
        y_test = np.load(data_files['y_test'], mmap_mode=mmap_mode)
        
        if use_mmap:
            print(f"  ✓ データファイル読み込み完了（メモリマップモード）")
        
    except Exception as e:
        raise IOError(
            f"データファイルの読み込み中にエラーが発生しました\n"
            f"エラー詳細: {e}\n"
            f"\nトラブルシューティング:\n"
            f"  1. .npyファイルが破損していないか確認してください\n"
            f"  2. NumPy形式で保存されたファイルか確認してください\n"
            f"     正しい保存方法: np.save('x_train.npy', x_train)\n"
            f"  3. ファイルサイズが0バイトでないか確認してください"
        ) from e
    
    # ラベルをflatten（CIFAR形式対応）
    if y_train.ndim > 1:
        y_train = y_train.flatten()
    if y_test.ndim > 1:
        y_test = y_test.flatten()
    
    # メモリ最適化: サンプル数制限を早期に適用
    # これにより、不要なデータを処理しない
    if train_samples is not None and train_samples < x_train.shape[0]:
        if use_mmap:
            # mmap使用時は明示的にコピー
            x_train = np.array(x_train[:train_samples])
            y_train = np.array(y_train[:train_samples])
        else:
            x_train = x_train[:train_samples]
            y_train = y_train[:train_samples]
        print(f"  訓練データを{train_samples}サンプルに制限しました")
    
    if test_samples is not None and test_samples < x_test.shape[0]:
        if use_mmap:
            # mmap使用時は明示的にコピー
            x_test = np.array(x_test[:test_samples])
            y_test = np.array(y_test[:test_samples])
        else:
            x_test = x_test[:test_samples]
            y_test = y_test[:test_samples]
        print(f"  テストデータを{test_samples}サンプルに制限しました")
    
    # mmap使用時、ここで明示的にコピー（reshape前）
    if use_mmap and (train_samples is None or test_samples is None):
        print(f"  メモリマップからコピー中...")
        if train_samples is None:
            x_train = np.array(x_train)
            y_train = np.array(y_train)
        if test_samples is None:
            x_test = np.array(x_test)
            y_test = np.array(y_test)
        print(f"  ✓ コピー完了")
    
    # 正規化（metadata指定に従う）
    # astype('float32')は新しい配列を作成するため、これ以降はmmap関係なし
    if metadata.get('normalize', False):
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    else:
        # すでに正規化済みと仮定
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
    
    # フラット化（reshape は in-place ではないが、メモリ効率的）
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    # データセット検証（カスタムデータセットのみ）
    # サンプル数制限後のデータで検証
    validate_custom_dataset(x_train, y_train, x_test, y_test, metadata)
    
    print(f"カスタムデータセット '{metadata.get('name', 'unknown')}' を読み込みました")
    print(f"  入力形状: {metadata.get('input_shape', 'unknown')}")
    print(f"  クラス数: {metadata.get('n_classes', 'unknown')}")
    
    # クラス名を返す（metadataから取得）
    class_names = metadata.get('class_names', None)
    
    return (x_train, y_train), (x_test, y_test), class_names


def load_dataset(dataset='mnist', train_samples=None, test_samples=None):
    """
    データセットの読み込みと前処理（TensorFlow/Keras使用）
    
    この関数はTensorFlow (tf.keras.datasets) を使用してデータを読み込みます。
    これは国際的に認知された標準的な手法であり、データ処理の信頼性を保証します。
    
    Args:
        dataset: データセット名（'mnist', 'fashion', 'cifar10', 'cifar100'）
        train_samples: 訓練サンプル数（Noneなら全データ）
        test_samples: テストサンプル数（Noneなら全データ）
    
    Returns:
        (x_train, y_train), (x_test, y_test): 正規化・フラット化済みNumPy配列
    
    Notes:
        - TensorFlow統合により、データ処理の透明性と再現性を確保
        - 返り値はNumPy配列（既存コードとの互換性維持）
        - シャッフル・バッチ処理にはcreate_tf_dataset()を使用
    """
    # データ読み込み（TensorFlow/Keras datasets API使用）
    if dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    else:  # mnist
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # 正規化（0-1）
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # フラット化
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    # サンプル数制限
    if train_samples is not None:
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]
    if test_samples is not None:
        x_test = x_test[:test_samples]
        y_test = y_test[:test_samples]
    
    return (x_train, y_train), (x_test, y_test)


def create_tf_dataset(x_data, y_data, batch_size=32, shuffle=True, seed=None, buffer_size=10000):
    """
    TensorFlow Dataset APIを使用したデータセット作成
    
    この関数はTensorFlow Data API (tf.data.Dataset) を使用します。
    これは業界標準の手法であり、以下の利点があります：
      1. データ処理パイプラインの信頼性が国際的に認知されている
      2. シャッフル機能が最適化され、再現性が保証される
      3. バッチ処理が効率的に実行される
      4. プリフェッチなど、パフォーマンス最適化が可能
    
    Args:
        x_data: 入力データ（NumPy配列）shape [n_samples, n_features]
        y_data: ラベルデータ（NumPy配列）shape [n_samples]
        batch_size: ミニバッチサイズ（デフォルト: 32）
        shuffle: エポックごとにシャッフルするか（デフォルト: True）
        seed: 乱数シード（再現性確保用、Noneなら非固定）
        buffer_size: シャッフルバッファサイズ（デフォルト: 10000）
    
    Returns:
        tf.data.Dataset: バッチ化・シャッフル済みTensorFlowデータセット
    
    使用例:
        >>> train_dataset = create_tf_dataset(
        ...     x_train, y_train, 
        ...     batch_size=128, 
        ...     shuffle=True, 
        ...     seed=42
        ... )
        >>> for x_batch, y_batch in train_dataset:
        ...     # x_batch.shape: (128, 784)
        ...     # y_batch.shape: (128,)
        ...     x_np = x_batch.numpy()  # TensorをNumPyに変換
        ...     y_np = y_batch.numpy()
    
    Notes:
        - tf.data.Dataset.from_tensor_slices(): NumPy配列からDataset作成
        - shuffle(): エポックごとのデータシャッフル（過学習防止）
        - batch(): ミニバッチ作成
        - prefetch(): バックグラウンドでデータ準備（パフォーマンス向上）
        - シード固定により完全な再現性を確保
    """
    # TensorFlow Datasetの作成（NumPy配列から変換）
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    
    # シャッフル（エポックごとにデータ順序をランダム化）
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=True  # 各エポックで再シャッフル
        )
    
    # バッチ化（指定サイズのミニバッチ作成）
    dataset = dataset.batch(batch_size)
    
    # プリフェッチ（バックグラウンドでデータ準備、パフォーマンス向上）
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset
