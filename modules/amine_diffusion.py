#!/usr/bin/env python3
"""
アミン拡散モジュール（報酬物質配分）

★ED法の核心メカニズム★
役割:
  - 出力層の誤差を隠れ層へ配分（ED法準拠）
  - 重みベースのアミン濃度計算
  - 活性度ベースの関係性マップ管理

関数:
  - initialize_activity_association: 関係性マップ初期化
  - update_activity_association: 競合学習ベースの更新
  - distribute_amine_by_output_weights: v017の成功メカニズムをED法で再現

使用例:
    from modules.amine_diffusion import (
        initialize_activity_association,
        update_activity_association, 
        distribute_amine_by_output_weights
    )
    import numpy as np
    
    # 関係性マップ初期化
    association_maps = initialize_activity_association(
        n_classes=10, 
        hidden_sizes=[250, 100]
    )
    
    # 学習中の更新
    association_maps[0] = update_activity_association(
        association_maps[0], 
        class_idx=3, 
        hidden_activations=z_hidden
    )
    
    # アミン配分
    exc_amine, inh_amine = distribute_amine_by_output_weights(
        w_output, 
        error_output, 
        z_hidden
    )
"""

import numpy as np


def initialize_activity_association(n_classes, hidden_sizes):
    """
    活性度ベース関係性マップの初期化
    
    Args:
        n_classes: 出力クラス数
        hidden_sizes: 各隠れ層のニューロン数のリスト
    
    Returns:
        association_maps: 各層の関係性マップのリスト
                         各マップのshape: [n_classes, hidden_size]
                         各要素は出力クラスとニューロンの関係性の強さ（0.0-1.0）
    """
    association_maps = []
    for hidden_size in hidden_sizes:
        # 小さなランダム値で初期化して正規化
        assoc_map = np.random.uniform(0.8, 1.2, (n_classes, hidden_size))
        
        # 各クラスごとに正規化（合計を1にする）
        for c in range(n_classes):
            assoc_map[c] /= np.sum(assoc_map[c])
        
        association_maps.append(assoc_map)
    
    return association_maps


def update_activity_association(association_map, class_idx, hidden_activations, tau=0.95, top_k_ratio=0.05):
    """
    活性度ベース関係性の更新（競合学習 + 正規化）
    
    Args:
        association_map: 現在の関係性マップ shape [n_classes, hidden_size]
        class_idx: 対象クラスのインデックス
        hidden_activations: 隠れ層の活性度 shape [hidden_size]
        tau: 過去の記憶の保持率（0.0-1.0）
        top_k_ratio: 上位何%のニューロンを強化するか（デフォルト5%）
    
    Returns:
        更新された関係性マップ
    """
    hidden_size = len(hidden_activations)
    
    # 競合学習: 上位k個のニューロンのみを強化
    k = max(1, int(hidden_size * top_k_ratio))
    top_k_indices = np.argsort(hidden_activations)[-k:]
    
    # 更新マスク: 上位kニューロンは強化、それ以外は減衰
    update_mask = np.zeros(hidden_size)
    update_mask[top_k_indices] = 1.0
    
    # 活性度を正規化（0-1の範囲に）
    act_normalized = hidden_activations / (np.max(hidden_activations) + 1e-8)
    
    # 競合学習ベースの更新
    # 上位k: tau * old + (1-tau) * 活性度
    # それ以外: tau * old (減衰)
    association_map[class_idx] = (
        tau * association_map[class_idx] +
        (1 - tau) * act_normalized * update_mask
    )
    
    # 正規化: 各クラスの関係性マップの合計を1に保つ
    total = np.sum(association_map[class_idx])
    if total > 1e-8:
        association_map[class_idx] /= total
    else:
        # 関係性がゼロの場合は均等に初期化
        association_map[class_idx] = np.ones(hidden_size) / hidden_size
    
    return association_map


def distribute_amine_by_output_weights(w_output, error_output, z_hidden, use_activation_weight=True, top_k_ratio=0.3):
    """
    出力重みベースの洗練されたアミン配分（v017の成功メカニズムを純粋なED法で再現）
    
    v017の誤差逆伝播: error_hidden = w_output.T @ (error_output * derivative)
    → これを純粋なED法で再現: amine_hidden = w_output.T @ error_output の正規化版
    
    重要な改良点:
    1. 重みの符号を保持（正の重み→興奮性寄与、負の重み→抑制性寄与）
    2. 活性度による重み付け（活性の高いニューロンを優先）
    3. 適切な正規化（数値安定性とスケール制御）
    4. 競合学習（上位k個のみにアミン配分）
    
    Args:
        w_output: 出力層の重み shape [n_classes, hidden_size]
        error_output: 出力層の誤差 shape [n_classes]
        z_hidden: 隠れ層の活性度 shape [hidden_size]
        use_activation_weight: 活性度による重み付けを使用するか
        top_k_ratio: 上位何%のニューロンにアミンを配分するか
    
    Returns:
        興奮性アミン濃度, 抑制性アミン濃度 (各 shape [hidden_size])
    """
    n_classes, hidden_size = w_output.shape
    
    # v017の誤差逆伝播を模倣: error_hidden ≈ w_output.T @ error_output
    # ただし、興奮性/抑制性を分離するため、正/負の誤差を別々に処理
    
    # 正の誤差（正解クラスの活性不足）と負の誤差（不正解クラスの過剰活性）を分離
    error_positive = np.maximum(error_output, 0)  # 正解クラスの誤差
    error_negative = np.maximum(-error_output, 0)  # 不正解クラスの誤差
    
    # 重みの転置行列を使って誤差を配分（v017の方式）
    # 各ニューロンへの寄与 = Σ(重み × 誤差)
    excitatory_contribution = np.dot(w_output.T, error_positive)  # [hidden_size]
    inhibitory_contribution = np.dot(w_output.T, error_negative)  # [hidden_size]
    
    # 活性度による重み付け（オプション）
    if use_activation_weight:
        # 活性度が高いニューロンを優先（ただし、過度にならないよう平方根を使用）
        activation_weight = np.sqrt(z_hidden + 1e-8)
        excitatory_contribution *= activation_weight
        inhibitory_contribution *= activation_weight
    
    # 競合学習: 上位k個のニューロンのみにアミンを配分
    k_exc = max(1, int(hidden_size * top_k_ratio))
    k_inh = max(1, int(hidden_size * top_k_ratio))
    
    # 興奮性アミン: 上位k個を選択
    exc_abs = np.abs(excitatory_contribution)
    top_k_exc_indices = np.argsort(exc_abs)[-k_exc:]
    excitatory_amine = np.zeros(hidden_size)
    excitatory_amine[top_k_exc_indices] = exc_abs[top_k_exc_indices]
    
    # 抑制性アミン: 上位k個を選択
    inh_abs = np.abs(inhibitory_contribution)
    top_k_inh_indices = np.argsort(inh_abs)[-k_inh:]
    inhibitory_amine = np.zeros(hidden_size)
    inhibitory_amine[top_k_inh_indices] = inh_abs[top_k_inh_indices]
    
    # スケール調整（最大値を1.0に）: 正規化せず、相対的な強さを保持
    exc_max = np.max(excitatory_amine)
    inh_max = np.max(inhibitory_amine)
    
    if exc_max > 1e-8:
        excitatory_amine /= exc_max
    
    if inh_max > 1e-8:
        inhibitory_amine /= inh_max
    
    return excitatory_amine, inhibitory_amine
