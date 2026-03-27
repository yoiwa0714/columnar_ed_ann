#!/usr/bin/env python3
"""
活性化関数モジュール（教育用シンプル版）

ED法で使用する活性化関数:
  - tanh_activation: 隠れ層の活性化（tanhのスケール版）
  - softmax / softmax_batch: 出力層の確率分布
  - cross_entropy_loss: 損失計算
"""

import numpy as np


def tanh_activation(x):
    """
    隠れ層の活性化関数（tanh scaled）

    通常のtanhと同じだが、将来のスケーリング変更に対応するため関数として独立。
    出力範囲: [-1, 1]
    """
    return np.tanh(x)


def softmax(x):
    """
    出力層のSoftMax関数（単一サンプル）

    数値安定性のためmax減算を行う。
    """
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def softmax_batch(x_batch):
    """
    出力層のSoftMax関数（バッチ版）

    Args:
        x_batch: shape [batch_size, n_classes]

    Returns:
        shape [batch_size, n_classes] の確率分布
    """
    x_shifted = x_batch - np.max(x_batch, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """
    Cross-Entropy損失（単一サンプル）

    Args:
        y_pred: SoftMax出力（確率分布）
        y_true: 正解クラスインデックス（整数）

    Returns:
        損失値（スカラー）
    """
    return -np.log(y_pred[y_true] + 1e-10)
