#!/usr/bin/env python3
"""
データセット読み込みモジュール（教育用シンプル版）

TensorFlow/KerasのデータセットAPIを使用して
MNIST, Fashion-MNIST, CIFAR-10等を読み込む。
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_dataset(dataset='mnist', train_samples=None, test_samples=None):
    """
    データセットの読み込みと前処理

    Args:
        dataset: データセット名（'mnist', 'fashion', 'cifar10'）
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
