#!/usr/bin/env python3
"""
活性化関数・損失関数モジュール

★ED法準拠★
重要: このモジュールの関数は微分の連鎖律を使用しません

関数:
  - tanh_activation: tanh活性化
  - softmax: SoftMax確率分布
  - cross_entropy_loss: Cross-Entropy損失
"""

import numpy as np


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
    
    Args:
        x: 出力層の活性値（logits）
    
    Returns:
        確率分布（合計=1.0）
    """
    x_shifted = x - np.max(x)
    exp_x = np.exp(np.clip(x_shifted, -500, 500))
    return exp_x / np.sum(exp_x)


def cross_entropy_loss(probs, target_class):
    """
    Cross-Entropy損失関数
    
    Args:
        probs: SoftMax確率分布
        target_class: 正解クラスのインデックス
    
    Returns:
        損失値
    """
    prob_true = np.clip(probs[target_class], 1e-10, 1.0)
    return -np.log(prob_true)
