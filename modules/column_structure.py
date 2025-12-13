#!/usr/bin/env python3
"""
コラム構造モジュール（大脳皮質カラム風の組織化）

★機能特化の実現メカニズム★
役割:
  - 六角格子ベースのコラム帰属度計算
  - 円環構造（旧方式）のサポート
  - 側方抑制重み行列管理

関数:
  - hex_distance: 六角座標距離計算
  - create_hexagonal_column_affinity: ハニカム構造コラム（推奨）
  - create_column_affinity: 円環構造コラム（旧方式）
  - create_lateral_weights: 側方抑制重み初期化

使用例:
    from modules.column_structure import (
        create_hexagonal_column_affinity,
        create_lateral_weights
    )
    
    # モード1: 完全コラム化（全ニューロンを均等分割）
    affinity = create_hexagonal_column_affinity(
        n_hidden=250, 
        n_classes=10, 
        column_neurons=25
    )
    
    # モード2: 参加率指定
    affinity = create_hexagonal_column_affinity(
        n_hidden=250, 
        n_classes=10, 
        participation_rate=1.0
    )
    
    # 側方抑制
    lateral_weights = create_lateral_weights(n_classes=10)
"""

import numpy as np


def _create_direct_column_assignment(n_hidden, n_classes, participation_rate=1.0):
    """
    直接的な順次割り当てによるコラム帰属度マップの作成（Strategy B-simplified）
    
    特徴:
    - シンプルな実装（~20行）
    - 完全均等分配（各クラス51-52個）
    - 重複なし（overlap=0）
    - participation_rate完全対応（1.0, 0.8, 0.5など）
    
    Args:
        n_hidden: 隠れ層のニューロン数（例: 512）
        n_classes: クラス数（例: 10）
        participation_rate: コラム参加率（0.0-1.0）
    
    Returns:
        affinity: コラム帰属度マップ shape [n_classes, n_hidden]
                  各要素は0.0または1.0（明示的なクラス帰属）
    
    使用例:
        >>> affinity = _create_direct_column_assignment(512, 10, 1.0)
        >>> [np.sum(affinity[i]) for i in range(10)]
        [52, 52, 51, 51, 51, 51, 51, 51, 51, 51]  # 合計512
    """
    affinity = np.zeros((n_classes, n_hidden))
    
    # 参加するニューロン総数
    total_neurons = int(n_hidden * participation_rate)
    
    # 各クラスへの基本割当数と余り
    neurons_per_class = total_neurons // n_classes
    remainder = total_neurons % n_classes
    
    # 順次割り当て
    neuron_id = 0
    for class_idx in range(n_classes):
        # 余りを最初のremainder個のクラスに+1個ずつ分配
        n_neurons = neurons_per_class + (1 if class_idx < remainder else 0)
        
        for i in range(n_neurons):
            if neuron_id < n_hidden:
                affinity[class_idx, neuron_id] = 1.0
                neuron_id += 1
    
    return affinity


def hex_distance(q1, r1, q2, r2):
    """
    六角座標系での距離計算（axial coordinates）
    
    Args:
        q1, r1: 座標1の六角座標
        q2, r2: 座標2の六角座標
    
    Returns:
        distance: 六角格子上の距離
    """
    return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2


def create_hexagonal_column_affinity(n_hidden, n_classes=10, column_radius=3.0, 
                                      column_neurons=None, participation_rate=None):
    """
    ハニカム構造（六角格子）に基づくコラム帰属度マップの作成
    
    ★v027更新★ クラス座標の中心化により全クラスが均等なアフィニティ分布を獲得
    
    Args:
        n_hidden: 隠れ層のニューロン数（例: 250）
        n_classes: クラス数（デフォルト: 10）
        column_radius: コラムの影響半径（六角距離単位）- column_neuronsが指定されない場合に使用
        column_neurons: 各クラスのコラムに割り当てるニューロン数（指定時はradiusより優先）
        participation_rate: コラム参加率（0.01-0.99）。コラムに参加するニューロンの割合
                           0.0と1.0は禁止（0.0=コラム無意味、1.0=学習不安定）
                           column_neuronsが指定されていない場合のみ有効
    
    Returns:
        affinity: コラム帰属度マップ shape [n_classes, n_hidden]
                  各要素は0.0以上の値（ニューロンがそのクラスに帰属する度合い）
    
    使用例:
        # モード1: 完全コラム化（全ニューロンを均等分割）
        affinity = create_hexagonal_column_affinity(250, 10, column_neurons=25)
        # → 各クラス25個、重複なし、全ニューロン参加
        
        # モード2: 参加率指定（推奨）
        affinity = create_hexagonal_column_affinity(250, 10, participation_rate=0.71)
        # → 71%のニューロンがコラムに参加、残り29%は非コラム（柔軟性保持）
        
        # モード3: 従来のradius方式（互換性維持）
        affinity = create_hexagonal_column_affinity(250, 10, column_radius=3.0)
        # → ガウス分布ベース、重複あり
    """
    # participation_rate検証
    if participation_rate is not None:
        if participation_rate <= 0.0 or participation_rate >= 1.0:
            raise ValueError(
                f"participation_rate must be in range (0.0, 1.0), exclusive. "
                f"Got {participation_rate}. "
                f"0.0 makes columns meaningless, 1.0 causes learning instability."
            )
        if participation_rate < 0.01 or participation_rate > 0.99:
            print(f"⚠️  Warning: participation_rate={participation_rate:.2f} is outside recommended range.")
            print(f"    Recommended: 0.01 <= participation_rate <= 0.99")
            print(f"    (Based on v026 analysis: ~0.71 worked well)")
    
    # 1. ★v027改善★ 10クラスをグリッド中心に配置（全クラス均等なニューロンアクセス）
    grid_size = int(np.ceil(np.sqrt(n_hidden)))
    grid_center = grid_size / 2.0
    
    # 中心化した2-3-3-2配置（従来の相対配置を保持しつつ中心に移動）
    class_coords = {
        0: (grid_center - 1, grid_center - 1),   1: (grid_center + 1, grid_center - 1),  # 行1: 2個
        2: (grid_center - 2, grid_center),       3: (grid_center, grid_center),           # 行2: 3個
        4: (grid_center + 2, grid_center),
        5: (grid_center - 2, grid_center + 1),   6: (grid_center, grid_center + 1),       # 行3: 3個
        7: (grid_center + 2, grid_center + 1),
        8: (grid_center - 1, grid_center + 2),   9: (grid_center + 1, grid_center + 2)    # 行4: 2個
    }
    
    # 2. ニューロンを2次元格子に配置（16×16グリッド）
    grid_size = int(np.ceil(np.sqrt(n_hidden)))
    neuron_coords = []
    for i in range(n_hidden):
        x = i % grid_size
        y = i // grid_size
        # 六角座標に変換（offset coordinates → axial coordinates）
        q = x - (y - (y & 1)) // 2
        r = y
        neuron_coords.append((q, r))
    
    # 3. 六角距離に基づくガウス型帰属度を計算
    affinity = np.zeros((n_classes, n_hidden))
    
    for class_idx in range(n_classes):
        if class_idx not in class_coords:
            continue  # 10クラス超の場合はスキップ
        
        class_q, class_r = class_coords[class_idx]
        
        for neuron_idx in range(n_hidden):
            neuron_q, neuron_r = neuron_coords[neuron_idx]
            
            # 六角距離
            dist = hex_distance(class_q, class_r, neuron_q, neuron_r)
            
            # ガウス型帰属度（column_radiusが標準偏差σに相当）
            # ★v024修正★ 仕様書準拠: sigma = column_neurons / 3.0
            if column_neurons is not None:
                sigma = column_neurons / 3.0  # 仕様書: 3σ点で帰属度がほぼゼロ
            else:
                sigma = column_radius  # 旧方式互換性維持
            aff = np.exp(-0.5 * (dist / sigma) ** 2)
            
            affinity[class_idx, neuron_idx] = aff
    
    # 4. コラム参加ニューロンの決定（優先度: participation_rate > column_neurons > radius）
    # ★v026修正★ participation_rateを最優先に変更（デフォルト1.0で意図しない重複回避）
    if participation_rate is not None:
        # モード1（最優先）: 参加率指定
        # 全体で(n_hidden * participation_rate)個のニューロンが参加するように調整
        target_neurons = int(n_hidden * participation_rate)
        neurons_per_class = target_neurons // n_classes
        remainder = target_neurons % n_classes  # 余りのニューロン
        
        assigned = np.zeros(n_hidden, dtype=bool)
        # ★重要★ participation_rate=1.0（デフォルト）では重複なし（overlap_factor=0.0）
        # participation_rate<1.0では重複許容（overlap_factor=0.3）
        overlap_factor = 0.0 if participation_rate >= 0.99 else 0.3
        
        for class_idx in range(n_classes):
            available_mask = ~assigned
            available_affinity = affinity[class_idx].copy()
            
            # overlap許容の場合のみ、割り当て済みニューロンも重み付けで考慮
            if overlap_factor > 0:
                available_affinity[~available_mask] *= overlap_factor
            else:
                # overlap_factor=0の場合、割り当て済みニューロンは完全に除外
                available_affinity[~available_mask] = 0
            
            sorted_indices = np.argsort(available_affinity)[::-1]
            
            # 余りニューロンを最初のremainder個のクラスに+1個ずつ分配
            n_neurons_for_this_class = neurons_per_class + (1 if class_idx < remainder else 0)
            
            selected = sorted_indices[:n_neurons_for_this_class]
            
            # ★v027修正★ v026の元のマスク方式に戻す
            # 極小アフィニティ値は自然に閾値判定で除外される（意図的な部分参加）
            mask = np.zeros(n_hidden)
            mask[selected] = 1
            affinity[class_idx] *= mask
            
            assigned[selected] = True
    
    elif column_neurons is not None:
        # モード2（中優先）: 明示的なニューロン数指定
        # 各クラスに正確にcolumn_neurons個を割り当て（重複許容）
        assigned = np.zeros(n_hidden, dtype=bool)
        overlap_factor = 0.3  # 情報共有と専門化のバランス
        
        for class_idx in range(n_classes):
            available_mask = ~assigned
            available_affinity = affinity[class_idx].copy()
            
            # overlap許容: 割り当て済みニューロンも重み付けで考慮
            available_affinity[~available_mask] *= overlap_factor
            
            sorted_indices = np.argsort(available_affinity)[::-1]
            selected = sorted_indices[:column_neurons]
            
            mask = np.zeros(n_hidden)
            mask[selected] = 1
            affinity[class_idx] *= mask
            
            assigned[selected] = True
                    
    else:
        # モード3: 従来のradius方式（閾値処理のみ）
        threshold = np.exp(-0.5 * 9)  # 3σ点
        for class_idx in range(n_classes):
            for neuron_idx in range(n_hidden):
                if affinity[class_idx, neuron_idx] < threshold:
                    affinity[class_idx, neuron_idx] = 0
    
    # 5. 正規化（各クラスの帰属度合計を一定に）
    for class_idx in range(n_classes):
        total = np.sum(affinity[class_idx])
        if total > 1e-8:
            if column_neurons is not None:
                # 明示的ニューロン数の場合: 合計をcolumn_neuronsに正規化
                affinity[class_idx] *= (column_neurons / total)
            else:
                # radius方式: 合計をradius依存のスケールに正規化
                target_sum = column_radius * 10
                affinity[class_idx] *= (target_sum / total)
    
    return affinity


def create_column_affinity(n_hidden, n_classes, column_size=30, overlap=0.3, use_gaussian=True):
    """
    コラム帰属度マップの作成(ガウス型または固定型) - 旧実装（円環構造）
    
    Args:
        n_hidden: 隠れ層のニューロン数
        n_classes: クラス数
        column_size: 各コラムの基準サイズ(ニューロン数)
        overlap: コラム間の重複度(0.0-1.0)
        use_gaussian: Trueならガウス型、Falseなら固定型
    
    Returns:
        affinity: コラム帰属度マップ shape [n_classes, n_hidden]
                  各要素は0.0-1.0の値(ニューロンがそのクラスに帰属する度合い)
    """
    affinity = np.zeros((n_classes, n_hidden))
    
    if use_gaussian:
        # ガウス型コラム帰属度(ドキュメント提案方式)
        # 各クラスのコラム中心を等間隔に配置
        centers = np.linspace(0, n_hidden, n_classes, endpoint=False).astype(int)
        
        # ガウス分布の標準偏差
        sigma = column_size / 3.0  # 3σ点で帰属度がほぼゼロ
        
        for class_idx in range(n_classes):
            center = centers[class_idx]
            
            # 各ニューロンへの円環距離を計算
            for neuron_idx in range(n_hidden):
                # 円環距離(トーラス構造)
                distance = min(
                    abs(neuron_idx - center),
                    n_hidden - abs(neuron_idx - center)
                )
                
                # ガウス分布で帰属度を計算
                aff = np.exp(-0.5 * (distance / sigma) ** 2)
                
                # 閾値処理(3σ点以下はゼロ)
                threshold = np.exp(-0.5 * 9)  # exp(-4.5) ≈ 0.011
                if aff < threshold:
                    aff = 0
                
                affinity[class_idx, neuron_idx] = aff
        
        # 重複制御: 複数クラスに帰属するニューロンの帰属度を調整
        if overlap < 1.0:
            for neuron_idx in range(n_hidden):
                total_affinity = np.sum(affinity[:, neuron_idx])
                if total_affinity > 1.0:
                    # 最大帰属度を持つクラスを優先
                    max_class = np.argmax(affinity[:, neuron_idx])
                    for class_idx in range(n_classes):
                        if class_idx != max_class:
                            affinity[class_idx, neuron_idx] *= overlap
    else:
        # 固定型コラム帰属度(v011方式)
        neurons_per_column = n_hidden // n_classes
        for class_idx in range(n_classes):
            start = class_idx * neurons_per_column
            end = start + neurons_per_column
            if class_idx == n_classes - 1:
                end = n_hidden  # 最後のクラスは残り全部
            affinity[class_idx, start:end] = 1.0
    
    # 正規化: 各クラスの帰属度マップを正規化(合計が1.0前後になるように)
    for class_idx in range(n_classes):
        total = np.sum(affinity[class_idx])
        if total > 1e-8:
            affinity[class_idx] /= total
            affinity[class_idx] *= column_size  # スケール調整
    
    return affinity


def create_lateral_weights(n_classes, strength=0.1):
    """
    側方抑制重み行列の初期化（ゼロ初期化）
    
    側方抑制は学習過程で動的に更新される。
    誤答時に[winner_class, true_class]の関係を負の値として記録し、
    繰り返し誤答するクラスペアの抑制を強化する。
    
    Returns:
        lateral_weights: shape [n_classes, n_classes]
                        初期値は全てゼロ
    """
    lateral_weights = np.zeros((n_classes, n_classes))
    return lateral_weights
