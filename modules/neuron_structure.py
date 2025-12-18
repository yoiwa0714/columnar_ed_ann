#!/usr/bin/env python3
"""
興奮性・抑制性ニューロンペア構造モジュール

★Dale's Principle実装の基礎★
役割:
  - Dale's Principle実装の基礎
  - ニューロンペア構造の生成
  - E/Iフラグ管理

関数:
  - create_ei_pairs: ニューロンペア生成
  - create_ei_flags: E/Iフラグ配列生成

使用例:
    from modules.neuron_structure import create_ei_pairs, create_ei_flags
    import numpy as np
    
    # 入力ペア生成
    x = np.array([0.1, 0.2, 0.3])
    x_paired = create_ei_pairs(x)
    # → [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
    
    # E/Iフラグ生成
    ei_flags = create_ei_flags(n_neurons=6, layer_idx=0)
    # → [1, 1, 1, -1, -1, -1] (前半が興奮性、後半が抑制性)
"""

import numpy as np


def create_ei_pairs(x):
    """
    入力データを興奮性・抑制性ペアに変換
    
    ★Dale's Principle★
    入力を前半（興奮性）と後半（抑制性）に複製することで、
    各ニューロンが両方の信号を受け取れるようにする
    
    Args:
        x: 入力データ shape [n_input]
    
    Returns:
        x_paired: ペア構造 shape [n_input * 2]
                 [x1, x2, ..., xn, x1, x2, ..., xn]
    """
    return np.concatenate([x, x])


def create_ei_pairs_batch(x_batch):
    """
    バッチ対応E/Iペア構造作成
    
    Args:
        x_batch: 入力データバッチ shape [batch_size, n_input]
    
    Returns:
        x_paired_batch: ペア構造バッチ shape [batch_size, n_input * 2]
    """
    return np.concatenate([x_batch, x_batch], axis=1)


def create_ei_flags(n_neurons, layer_idx=0):
    """
    興奮性・抑制性フラグ配列を生成
    
    Args:
        n_neurons: ニューロン数
        layer_idx: 層インデックス（0=第1層、1=第2層、...）
    
    Returns:
        ei_flags: フラグ配列 shape [n_neurons]
                 +1 = 興奮性ニューロン
                 -1 = 抑制性ニューロン
    
    Notes:
        - 第1層（layer_idx=0）: 全て興奮性（v017準拠）
        - 第2層以降: 全て興奮性
    """
    # v026の実装では全て興奮性
    return np.ones(n_neurons)
