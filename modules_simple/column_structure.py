#!/usr/bin/env python3
"""
コラム構造モジュール（大脳皮質カラム風の組織化）

役割:
  - 六角格子ベースのコラムメンバーシップ計算

関数:
  - create_column_membership: ハニカム構造コラムメンバーシップ
"""

import numpy as np


def create_column_membership(n_hidden, n_classes, participation_rate=1.0, 
                             use_hexagonal=True, column_radius=0.4, column_neurons=None):
    """
    コラムメンバーシップフラグを作成
    
    各ニューロンがどのクラスのコラムに所属するかをブールフラグで管理。
    コラム構造を維持しつつ、重みは学習で獲得される。
    
    Args:
        n_hidden: 隠れ層のニューロン総数
        n_classes: 出力クラス数
        participation_rate: 各クラスに割り当てるニューロンの割合（0.0-1.0）
        use_hexagonal: Trueならハニカム配置、Falseなら順次割り当て
        column_radius: コラム半径（ハニカム配置時の参考値）
        column_neurons: 各クラスに割り当てるニューロン数（明示指定、優先度最高）
    
    Returns:
        membership: shape [n_classes, n_hidden] のブール配列
        neuron_positions: shape [n_hidden, 2] の2D座標配列
        class_coords: 各クラスのコラム中心座標辞書
    """
    membership = np.zeros((n_classes, n_hidden), dtype=bool)
    neuron_positions = None
    class_coords = None
    
    # 各クラスに割り当てるニューロン数（優先順位: column_neurons > participation_rate）
    if column_neurons is not None:
        neurons_per_class = column_neurons
    else:
        neurons_per_class = int(n_hidden * participation_rate / n_classes)
    
    if neurons_per_class == 0:
        neurons_per_class = 1
    
    if use_hexagonal:
        # ハニカム配置（2-3-3-2パターンで10クラスを中心化配置）
        grid_size = int(np.ceil(np.sqrt(n_hidden)))
        grid_center = grid_size / 2.0
        
        # スケール係数: コラムを隠れ層全体の約4/5の範囲に分散
        scale_factor = (grid_size * 0.8) / 4.0
        
        # 2-3-3-2配置の10クラス座標
        class_coords = {
            0: (grid_center + scale_factor * (-1), grid_center + scale_factor * (-1)), 
            1: (grid_center + scale_factor * (+1), grid_center + scale_factor * (-1)),
            2: (grid_center + scale_factor * (-2), grid_center + scale_factor * (0)), 
            3: (grid_center + scale_factor * (0),  grid_center + scale_factor * (0)),
            4: (grid_center + scale_factor * (+2), grid_center + scale_factor * (0)),
            5: (grid_center + scale_factor * (-2), grid_center + scale_factor * (+1)), 
            6: (grid_center + scale_factor * (0),  grid_center + scale_factor * (+1)),
            7: (grid_center + scale_factor * (+2), grid_center + scale_factor * (+1)),
            8: (grid_center + scale_factor * (-1), grid_center + scale_factor * (+2)), 
            9: (grid_center + scale_factor * (+1), grid_center + scale_factor * (+2))
        }
        
        neuron_positions = np.array([
            [i // grid_size, i % grid_size] for i in range(n_hidden)
        ])
        
        for class_idx in range(min(n_classes, len(class_coords))):
            center_row, center_col = class_coords[class_idx]
            
            distances = np.sqrt(
                (neuron_positions[:, 0] - center_row) ** 2 +
                (neuron_positions[:, 1] - center_col) ** 2
            )
            
            closest_indices = np.argsort(distances)[:neurons_per_class]
            membership[class_idx, closest_indices] = True
    else:
        # 順次割り当て
        grid_size = int(np.ceil(np.sqrt(n_hidden)))
        neuron_positions = np.array([
            [i // grid_size, i % grid_size] for i in range(n_hidden)
        ])
        class_coords = None
        
        for class_idx in range(n_classes):
            start_idx = class_idx * neurons_per_class
            end_idx = min(start_idx + neurons_per_class, n_hidden)
            membership[class_idx, start_idx:end_idx] = True
    
    return membership, neuron_positions, class_coords
