#!/usr/bin/env python3
"""
データセット読み込み・前処理モジュール

役割:
  - MNIST/Fashion-MNISTの読み込み
  - データ正規化・フラット化
  - クラス名取得

関数:
  - load_dataset: データ読み込みと前処理
  - get_class_names: データセット別クラス名取得

使用例:
    from modules.data_loader import load_dataset, get_class_names
    
    # データ読み込み
    x_train, y_train, x_test, y_test = load_dataset(
        n_train=3000, 
        n_test=1000, 
        dataset='mnist'
    )
    
    # クラス名取得
    class_names = get_class_names('fashion')
    # → ['T-shirt/top', 'Trouser', ...]
"""

from keras import datasets as keras_datasets


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
    データセットの読み込みと前処理
    
    Args:
        dataset: データセット名（'mnist', 'fashion', 'cifar10', 'cifar100'）
        train_samples: 訓練サンプル数（Noneなら全データ）
        test_samples: テストサンプル数（Noneなら全データ）
    
    Returns:
        (x_train, y_train), (x_test, y_test): 正規化・フラット化済みデータ
    """
    # データ読み込み
    if dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = keras_datasets.fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras_datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras_datasets.cifar100.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    else:  # mnist
        (x_train, y_train), (x_test, y_test) = keras_datasets.mnist.load_data()
    
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
