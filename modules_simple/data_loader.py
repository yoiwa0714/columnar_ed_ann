#!/usr/bin/env python3
"""
データセット読み込み・前処理モジュール

役割:
  - MNIST/Fashion-MNISTの読み込み（TensorFlow/Keras統合）
  - データ正規化・フラット化
  - クラス名取得
  - カスタムデータセット対応

関数:
  - load_dataset: データ読み込みと前処理（NumPy配列を返す）
  - get_class_names: データセット別クラス名取得
  - resolve_dataset_path: データセットパス解決
  - load_custom_dataset: カスタムデータセット読み込み
"""

import os
import json
import numpy as np
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
    if custom_class_names is not None:
        return custom_class_names
    
    if dataset_name == 'mnist':
        return None
    elif dataset_name == 'fashion':
        return [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    elif dataset_name == 'cifar10':
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    elif dataset_name == 'cifar100':
        return None
    else:
        return None


def validate_custom_dataset(x_train, y_train, x_test, y_test, metadata):
    """
    カスタムデータセットの検証
    
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
    
    if not isinstance(x_train, np.ndarray) or not isinstance(x_test, np.ndarray):
        raise ValueError(
            f"データはNumPy配列である必要があります\n"
            f"  x_train型: {type(x_train).__name__}\n"
            f"  x_test型: {type(x_test).__name__}"
        )
    
    if not isinstance(y_train, np.ndarray) or not isinstance(y_test, np.ndarray):
        raise ValueError(
            f"ラベルはNumPy配列である必要があります\n"
            f"  y_train型: {type(y_train).__name__}\n"
            f"  y_test型: {type(y_test).__name__}"
        )
    
    if np.any(np.isnan(x_train)) or np.any(np.isnan(x_test)):
        raise ValueError("データにNaNが含まれています")
    
    if np.any(np.isinf(x_train)) or np.any(np.isinf(x_test)):
        raise ValueError("データにInfが含まれています")
    
    if np.any(np.isnan(y_train)) or np.any(np.isnan(y_test)):
        raise ValueError("ラベルにNaNが含まれています")
    
    if np.min(y_train) < 0 or np.max(y_train) >= n_classes:
        raise ValueError(
            f"訓練ラベルが範囲外です（期待: [0, {n_classes-1}], "
            f"実際: [{np.min(y_train)}, {np.max(y_train)}]）"
        )
    
    if np.min(y_test) < 0 or np.max(y_test) >= n_classes:
        raise ValueError(
            f"テストラベルが範囲外です（期待: [0, {n_classes-1}], "
            f"実際: [{np.min(y_test)}, {np.max(y_test)}]）"
        )
    
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"訓練データとラベルのサンプル数が不一致: "
            f"x_train={x_train.shape[0]}, y_train={y_train.shape[0]}"
        )
    
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"テストデータとラベルのサンプル数が不一致: "
            f"x_test={x_test.shape[0]}, y_test={y_test.shape[0]}"
        )
    
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            f"訓練データとテストデータの次元が不一致: "
            f"x_train={x_train.shape[1]}, x_test={x_test.shape[1]}"
        )
    
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)
    
    print(f"  ✓ データ検証OK")
    print(f"  クラス分布: 訓練={len(unique_train)}クラス, テスト={len(unique_test)}クラス")
    
    for class_idx in range(n_classes):
        train_count = np.sum(y_train == class_idx)
        test_count = np.sum(y_test == class_idx)
        if train_count > 0 or test_count > 0:
            print(f"    Class {class_idx}: Train={train_count:4d}, Test={test_count:4d}")
    
    print(f"\nデータセット '{dataset_name}' の検証完了\n")


def resolve_dataset_path(dataset_spec):
    """
    データセット指定から実際のパスを解決
    
    Args:
        dataset_spec: データセット名またはパス
    
    Returns:
        resolved_path: 解決されたパス
        is_custom: カスタムデータセットかどうか
    """
    standard_datasets = ['mnist', 'fashion', 'cifar10', 'cifar100']
    if dataset_spec in standard_datasets:
        return dataset_spec, False
    
    search_paths = [
        dataset_spec,
        os.path.expanduser(f'~/.keras/datasets/{dataset_spec}'),
        os.path.join(os.getcwd(), dataset_spec)
    ]
    
    for path in search_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path, True
    
    error_msg = (
        f"データセット '{dataset_spec}' が見つかりません。\n\n"
        f"標準データセット: mnist, fashion, cifar10, cifar100\n"
        f"カスタムデータセットの配置場所:\n"
        f"  1. ~/.keras/datasets/{{dataset_name}}/\n"
        f"  2. カレントディレクトリ: ./{{dataset_name}}/\n"
        f"  3. 絶対パス指定\n\n"
        f"詳細はCUSTOM_DATASET_GUIDE.mdを参照してください。"
    )
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
    """
    metadata_path = os.path.join(dataset_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"metadata.json が見つかりません: {metadata_path}\n"
            f"詳細はCUSTOM_DATASET_GUIDE.mdを参照してください。"
        )
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    required_fields = ['name', 'n_classes', 'input_shape']
    missing = [f for f in required_fields if f not in metadata]
    if missing:
        raise ValueError(f"metadata.jsonに必須フィールドがありません: {', '.join(missing)}")
    
    data_files = {
        'x_train': os.path.join(dataset_path, 'x_train.npy'),
        'y_train': os.path.join(dataset_path, 'y_train.npy'),
        'x_test': os.path.join(dataset_path, 'x_test.npy'),
        'y_test': os.path.join(dataset_path, 'y_test.npy')
    }
    
    missing_files = [name for name, path in data_files.items() if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            f"必須データファイルが見つかりません: {', '.join(missing_files)}\n"
            f"ディレクトリ: {dataset_path}"
        )
    
    # 大規模データセット対応
    file_sizes = {name: os.path.getsize(path) for name, path in data_files.items()}
    total_size = sum(file_sizes.values())
    use_mmap = total_size > 100 * 1024 * 1024  # 100MB
    mmap_mode = 'r' if use_mmap else None
    
    x_train = np.load(data_files['x_train'], mmap_mode=mmap_mode)
    y_train = np.load(data_files['y_train'], mmap_mode=mmap_mode)
    x_test = np.load(data_files['x_test'], mmap_mode=mmap_mode)
    y_test = np.load(data_files['y_test'], mmap_mode=mmap_mode)
    
    if y_train.ndim > 1:
        y_train = y_train.flatten()
    if y_test.ndim > 1:
        y_test = y_test.flatten()
    
    # サンプル数制限
    if train_samples is not None and train_samples < x_train.shape[0]:
        x_train = np.array(x_train[:train_samples]) if use_mmap else x_train[:train_samples]
        y_train = np.array(y_train[:train_samples]) if use_mmap else y_train[:train_samples]
    
    if test_samples is not None and test_samples < x_test.shape[0]:
        x_test = np.array(x_test[:test_samples]) if use_mmap else x_test[:test_samples]
        y_test = np.array(y_test[:test_samples]) if use_mmap else y_test[:test_samples]
    
    if use_mmap:
        if train_samples is None:
            x_train = np.array(x_train)
            y_train = np.array(y_train)
        if test_samples is None:
            x_test = np.array(x_test)
            y_test = np.array(y_test)
    
    # 正規化
    if metadata.get('normalize', False):
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    else:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
    
    # フラット化
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    validate_custom_dataset(x_train, y_train, x_test, y_test, metadata)
    
    print(f"カスタムデータセット '{metadata.get('name', 'unknown')}' を読み込みました")
    
    class_names = metadata.get('class_names', None)
    return (x_train, y_train), (x_test, y_test), class_names


def load_dataset(dataset='mnist', train_samples=None, test_samples=None):
    """
    データセットの読み込みと前処理（TensorFlow/Keras使用）
    
    Args:
        dataset: データセット名（'mnist', 'fashion', 'cifar10', 'cifar100'）
        train_samples: 訓練サンプル数（Noneなら全データ）
        test_samples: テストサンプル数（Noneなら全データ）
    
    Returns:
        (x_train, y_train), (x_test, y_test): 正規化・フラット化済みNumPy配列
    """
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
