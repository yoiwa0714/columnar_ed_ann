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

import tensorflow as tf
from tensorflow import keras


def get_class_names(dataset_name):
    """
    データセットのクラス名を取得
    
    Args:
        dataset_name: データセット名（'mnist', 'fashion', 'cifar10', 'cifar100'）
    
    Returns:
        class_names: クラス名のリスト（クラス名情報がない場合はNone）
    """
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
