#!/usr/bin/env python3
"""
興奮性・抑制性ニューロンペア構造モジュール

★Dale's Principle実装の基礎★
役割:
  - ニューロンペア構造の生成（入力を興奮性・抑制性ペアに変換）

関数:
  - create_ei_pairs: ニューロンペア生成
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
    """
    return np.concatenate([x, x])
