#!/usr/bin/env python3
"""
活性化関数・損失関数モジュール

★ED法準拠★
役割:
  - ED法準拠の活性化関数（飽和特性あり）
  - SoftMax・Cross-Entropy（多クラス分類）
  - 数値安定性確保（overflow回避）

重要: このモジュールの関数は微分の連鎖律を使用しません

関数:
  - sigmoid: シグモイド関数
  - tanh_activation: tanh活性化
  - softmax: SoftMax確率分布
  - cross_entropy_loss: Cross-Entropy損失

使用例:
    from modules.activation_functions import sigmoid, tanh_activation, softmax
    import numpy as np
    
    # 隠れ層
    z_hidden = tanh_activation(np.dot(w, x))
    
    # 出力層
    z_output = softmax(np.dot(w_output, z_hidden))
"""

import numpy as np


def sigmoid(x):
    """
    シグモイド関数（overflow回避）
    
    Args:
        x: 入力値または配列
    
    Returns:
        シグモイド変換後の値（0-1の範囲）
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh_activation(x):
    """
    tanh活性化関数（双方向性、飽和特性あり）
    
    ★ED法準拠★
    - 飽和特性を持つため、ED法の飽和項と相性が良い
    - 双方向性（-1〜+1）で多クラス分類に適している
    
    Args:
        x: 入力値または配列
    
    Returns:
        tanh変換後の値（-1〜+1の範囲）
    """
    return np.tanh(np.clip(x, -500, 500))


def softmax(x):
    """
    SoftMax関数（多クラス分類用）
    
    確率分布を生成：
    - 各要素が [0, 1]
    - 全要素の合計が 1.0
    - クラス間の相対的な強弱が明確
    
    Args:
        x: 出力層の活性値（logits）
    
    Returns:
        確率分布（合計=1.0）
    """
    # 数値安定性のため最大値を引く
    x_shifted = x - np.max(x)
    exp_x = np.exp(np.clip(x_shifted, -500, 500))
    return exp_x / np.sum(exp_x)


def softmax_batch(x_batch):
    """
    バッチ対応SoftMax関数
    
    Args:
        x_batch: 出力層の活性値バッチ shape: [batch_size, n_output]
    
    Returns:
        確率分布バッチ shape: [batch_size, n_output] (各行の合計=1.0)
    """
    # 各行の最大値を引く（数値安定性）
    x_shifted = x_batch - np.max(x_batch, axis=1, keepdims=True)
    exp_x = np.exp(np.clip(x_shifted, -500, 500))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(probs, target_class):
    """
    Cross-Entropy損失関数
    
    Args:
        probs: SoftMax確率分布
        target_class: 正解クラスのインデックス
    
    Returns:
        損失値
    """
    # 数値安定性のため小さな値でクリップ
    prob_true = np.clip(probs[target_class], 1e-10, 1.0)
    return -np.log(prob_true)
