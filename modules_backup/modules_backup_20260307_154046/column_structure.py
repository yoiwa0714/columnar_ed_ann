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


def create_column_membership(n_hidden, n_classes, participation_rate=1.0, 
                             use_hexagonal=True, column_radius=0.4, column_neurons=None):
    """
    コラムメンバーシップフラグを作成（Affinity代替、学習可能化対応）
    
    目的:
        - Affinityの静的な勝者固定化問題を解決
        - コラム所属情報をブールフラグで保持（値は重みで学習）
        - コラム構造を維持しつつ学習可能性を回復
    
    Args:
        n_hidden: 隠れ層のニューロン総数
        n_classes: 出力クラス数
        participation_rate: 各クラスに割り当てるニューロンの割合（0.0-1.0）
        use_hexagonal: Trueならハニカム配置、Falseなら順次割り当て
        column_radius: コラム半径（ハニカム配置時の参考値）
        column_neurons: 各クラスに割り当てるニューロン数（明示指定、優先度最高）
    
    Returns:
        membership: shape [n_classes, n_hidden] のブール配列
                   membership[c, i] = Trueなら、ニューロンiはクラスcのコラムメンバー
        neuron_positions: shape [n_hidden, 2] の2D座標配列（可視化用）
        class_coords: 各クラスのコラム中心座標辞書（可視化用）
    
    設計思想:
        - フラグは固定（所属は変わらない）
        - 候補内での実力（勝率）は重みで決まる（学習可能）
        - 上位K個学習と組み合わせて使用
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
        neurons_per_class = 1  # 最低1個は割り当て
    
    if use_hexagonal:
        # ハニカム配置（2-3-3-2パターンで10クラスを中心化配置）
        # コラム間距離を十分に確保し、隠れ層全体（約4/5の範囲）に分散
        grid_size = int(np.ceil(np.sqrt(n_hidden)))
        grid_center = grid_size / 2.0  # グリッドの中心座標
        
        # スケール係数: コラムを隠れ層全体に広げるため、基本パターンを拡大
        # 2-3-3-2パターンの基本幅は4（-2から+2）
        # これをgrid_sizeの約0.8倍（4/5の範囲）に拡大
        scale_factor = (grid_size * 0.8) / 4.0
        
        # 2-3-3-2配置の10クラス座標（中心化 + スケール拡大）
        row_patterns = [2, 3, 3, 2]
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
        
        # 2D グリッド配置
        neuron_positions = np.array([
            [i // grid_size, i % grid_size] for i in range(n_hidden)
        ])
        
        for class_idx in range(min(n_classes, len(class_coords))):
            center_row, center_col = class_coords[class_idx]
            
            # 各ニューロンとの距離を計算
            distances = np.sqrt(
                (neuron_positions[:, 0] - center_row) ** 2 +
                (neuron_positions[:, 1] - center_col) ** 2
            )
            
            # 距離が近い上位neurons_per_class個をメンバーに
            closest_indices = np.argsort(distances)[:neurons_per_class]
            membership[class_idx, closest_indices] = True
    else:
        # 順次割り当て（シンプル）
        grid_size = int(np.ceil(np.sqrt(n_hidden)))
        neuron_positions = np.array([
            [i // grid_size, i % grid_size] for i in range(n_hidden)
        ])
        class_coords = None  # 順次割り当ての場合はコラム中心なし
        
        for class_idx in range(n_classes):
            start_idx = class_idx * neurons_per_class
            end_idx = min(start_idx + neurons_per_class, n_hidden)
            membership[class_idx, start_idx:end_idx] = True
    
    return membership, neuron_positions, class_coords


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


def create_hexagonal_column_affinity(n_hidden, n_classes=10, 
                                      column_neurons=None, participation_rate=None,
                                      affinity_max=1.0, affinity_min=0.0):
    """
    ハニカム構造に基づくコラムaffinity mapの作成（実験用）
    
    Args:
        n_hidden: 隠れ層のニューロン数
        n_classes: クラス数（デフォルト: 10）
        column_neurons: 各クラスのコラムニューロン数（未指定時はparticipation_rate使用）
        participation_rate: コラム参加率（0.0-1.0、デフォルト: None）
        affinity_max: コラムニューロンのaffinity値（デフォルト: 1.0）
        affinity_min: 非コラムニューロンのaffinity値（デフォルト: 0.0）
    
    Returns:
        affinity: shape [n_classes, n_hidden]、各要素はaffinity_maxまたはaffinity_min
    
    設計:
        - Membership方式と同じコラム構造を使用
        - membershipフラグに基づいてaffinity値を割り当て
        - affinity_max/affinity_minで重み爆発を制御可能
    """
    # Membership方式で構造を決定
    membership, _, _ = create_column_membership(
        n_hidden=n_hidden,
        n_classes=n_classes,
        participation_rate=participation_rate if participation_rate is not None else 0.1,
        use_hexagonal=True,
        column_neurons=column_neurons
    )
    
    # Affinityマップに変換（bool → float）
    affinity = np.zeros((n_classes, n_hidden))
    affinity[membership] = affinity_max   # コラムニューロン
    affinity[~membership] = affinity_min  # 非コラムニューロン
    
    return affinity


def create_column_affinity(n_hidden, n_classes, column_size=30, overlap=0.3, use_gaussian=True,
                           column_neurons=None, participation_rate=None):
    """
    コラム帰属度マップの作成(ガウス型または固定型) - 円環構造（v027更新）
    
    後方互換性のため保持。新規実装には create_hexagonal_column_affinity() を推奨。
    """
    # シンプルな順次割り当てに簡略化
    affinity = np.zeros((n_classes, n_hidden))
    
    if column_neurons is not None:
        neurons_per_class = column_neurons
    else:
        neurons_per_class = int(n_hidden * (participation_rate if participation_rate else 0.1) / n_classes)
    
    neurons_per_class = max(1, neurons_per_class)
    
    for class_idx in range(n_classes):
        start_idx = class_idx * neurons_per_class
        end_idx = min(start_idx + neurons_per_class, n_hidden)
        affinity[class_idx, start_idx:end_idx] = 1.0
    
    return affinity


def create_receptive_fields(n_input, membership, column_neurons, 
                            rf_overlap=0.5, rf_mode='random', seed=None):
    """
    コラム内ニューロンごとに異なる受容野（入力マスク）を生成
    
    生物学的背景:
        大脳皮質V1のコラム内ニューロンは同じ方位角に応答するが、
        空間的に少しずつずれた受容野を持つ。これにより、コラム全体として
        入力空間を広くカバーしつつ、各ニューロンが特化した役割を果たす。
    
    Args:
        n_input: 入力次元数（E/Iペア適用後: 784*2=1568）
        membership: コラムメンバーシップ [n_classes, n_hidden] boolean
        column_neurons: 各クラスのコラムニューロン数
        rf_overlap: 受容野の重複率（0.0=重複なし完全分割, 1.0=全入力を見る）
        rf_mode: 受容野割り当て方式
            'random'  : ランダムに入力次元を選択（汎用）
            'spatial' : 画像の空間的領域で分割（MNIST等の画像向け）
        seed: 乱数シード（再現性確保）
    
    Returns:
        rf_masks: [n_hidden, n_input] boolean配列
                  rf_masks[neuron_i, input_j] = True → ニューロンiは入力jを受容
                  コラム非メンバーのニューロンはすべてTrue（全入力受容、従来通り）
    
    Notes:
        - column_neurons=1の場合はNoneを返す（全入力受容、従来動作と同一）
        - E/Iペアを考慮: 入力インデックスiとi+n_input//2は常にセットで受容
        - コラム非メンバー（リザバー）ニューロンは受容野マスクの影響を受けない
    """
    if column_neurons is None or column_neurons <= 1:
        return None  # column_neurons=1では受容野マスク不要
    
    n_classes, n_hidden = membership.shape
    n_original = n_input // 2  # E/Iペア前の次元数（例: 784）
    
    # デフォルト: 全入力を受容（コラム非メンバー含む）
    rf_masks = np.ones((n_hidden, n_input), dtype=bool)
    
    rng = np.random.RandomState(seed)
    
    # 各ニューロンが見る入力次元数を計算
    # rf_overlap=0.0: 各ニューロンが1/N の入力を見る（完全分割）
    # rf_overlap=1.0: 各ニューロンが全入力を見る（マスクなし）
    rf_size = int(n_original * (1.0 / column_neurons + rf_overlap * (column_neurons - 1) / column_neurons))
    rf_size = min(rf_size, n_original)  # 上限は全入力
    rf_size = max(rf_size, 1)  # 最低1次元
    
    if rf_mode == 'spatial':
        # 空間的分割（画像データ向け）
        # 入力を2D画像として扱い、空間的に隣接する領域を割り当て
        img_size = int(np.sqrt(n_original))  # 例: 28 for MNIST
        if img_size * img_size != n_original:
            # 正方形でない場合はrandomにフォールバック
            print(f"  [受容野] 入力が正方形画像でないため random モードにフォールバック "
                  f"(n_original={n_original})")
            rf_mode = 'random'
    
    if rf_mode == 'spatial':
        img_size = int(np.sqrt(n_original))
        
        for class_idx in range(n_classes):
            member_indices = np.where(membership[class_idx])[0]
            n_members = len(member_indices)
            if n_members <= 1:
                continue
            
            # ニューロンをグリッド上に配置（1D→2D配置）
            # column_neurons個のニューロンを空間的に分布させる
            n_grid = int(np.ceil(np.sqrt(n_members)))
            
            for local_rank, neuron_idx in enumerate(member_indices):
                # このニューロンの中心位置を計算
                grid_row = local_rank // n_grid
                grid_col = local_rank % n_grid
                center_y = int((grid_row + 0.5) * img_size / n_grid)
                center_x = int((grid_col + 0.5) * img_size / n_grid)
                
                # 受容野サイズからガウスマスクを生成
                rf_radius = int(np.sqrt(rf_size / np.pi))  # 円形受容野の半径
                rf_radius = max(rf_radius, 2)  # 最低半径2
                
                # 各ピクセルとの距離でマスク生成
                mask_2d = np.zeros((img_size, img_size), dtype=bool)
                for y in range(img_size):
                    for x in range(img_size):
                        dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                        if dist <= rf_radius:
                            mask_2d[y, x] = True
                
                # 2Dマスクを1Dに展開
                mask_1d = mask_2d.flatten()
                
                # E/Iペア対応: 前半と後半に同じマスクを適用
                rf_masks[neuron_idx, :n_original] = mask_1d
                rf_masks[neuron_idx, n_original:] = mask_1d
    
    else:  # rf_mode == 'random'
        # ランダム分割
        for class_idx in range(n_classes):
            member_indices = np.where(membership[class_idx])[0]
            n_members = len(member_indices)
            if n_members <= 1:
                continue
            
            # 入力次元をシャッフルして各ニューロンに分配
            all_input_indices = rng.permutation(n_original)
            
            for local_rank, neuron_idx in enumerate(member_indices):
                # このニューロンの受容野を決定
                # 各ニューロンが見始める開始位置をずらす（均等分割ベース）
                start = int(local_rank * n_original / n_members)
                
                # 受容野に含まれる入力インデックスを選択
                selected_indices = []
                for j in range(rf_size):
                    idx = (start + j) % n_original
                    selected_indices.append(all_input_indices[idx])
                
                selected = np.array(selected_indices)
                
                # マスクを作成（まずFalseで初期化）
                rf_masks[neuron_idx, :] = False
                # E/Iペア対応: 前半と後半の同じ位置をTrue
                rf_masks[neuron_idx, selected] = True              # 興奮性入力
                rf_masks[neuron_idx, selected + n_original] = True  # 抑制性入力
    
    # 統計表示
    column_neuron_mask = np.any(membership, axis=0)  # どのクラスにも属するニューロン
    n_column = np.sum(column_neuron_mask)
    n_reservoir = n_hidden - n_column
    
    if n_column > 0:
        column_rf_sizes = np.sum(rf_masks[column_neuron_mask], axis=1)
        print(f"\n[受容野（Receptive Field）初期化]")
        print(f"  - モード: {rf_mode}")
        print(f"  - 重複率: {rf_overlap:.2f}")
        print(f"  - コラムニューロンの受容野サイズ: "
              f"平均{np.mean(column_rf_sizes):.0f}/{n_input} "
              f"({np.mean(column_rf_sizes)/n_input*100:.1f}%)")
        print(f"  - コラム外ニューロン({n_reservoir}個): 全入力受容（マスクなし）")
    
    return rf_masks


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
    return np.zeros((n_classes, n_classes))
    
    # 1. ★v027改善 + 2026-01-11修正★ 10クラスをグリッド中心に配置（直交座標系で統一）
    # COLUMN_PLACEMENT_CENTERING_FIX_REPORT.md準拠の中心化実装
    grid_size = int(np.ceil(np.sqrt(n_hidden)))
    grid_center = grid_size / 2.0
    
    # 中心化した2-3-3-2配置（直交座標 (row, col) で定義）
    class_coords = {
        0: (grid_center - 1, grid_center - 1),   1: (grid_center + 1, grid_center - 1),  # 行1: 2個
        2: (grid_center - 2, grid_center),       3: (grid_center, grid_center),           # 行2: 3個
        4: (grid_center + 2, grid_center),
        5: (grid_center - 2, grid_center + 1),   6: (grid_center, grid_center + 1),       # 行3: 3個
        7: (grid_center + 2, grid_center + 1),
        8: (grid_center - 1, grid_center + 2),   9: (grid_center + 1, grid_center + 2)    # 行4: 2個
    }
    
    # 2. ニューロンを2次元格子に配置（直交座標系）
    neuron_positions = np.array([
        [i // grid_size, i % grid_size] for i in range(n_hidden)
    ])
    
    # 3. ユークリッド距離に基づくガウス型帰属度を計算
    affinity = np.zeros((n_classes, n_hidden))
    
    for class_idx in range(n_classes):
        if class_idx not in class_coords:
            continue  # 10クラス超の場合はスキップ
        
        center_row, center_col = class_coords[class_idx]
        
        # 各ニューロンとの距離を計算
        distances = np.sqrt(
            (neuron_positions[:, 0] - center_row) ** 2 +
            (neuron_positions[:, 1] - center_col) ** 2
        )
        
        # ガウス型帰属度を計算
        # ★v024修正★ 仕様書準拠: sigma = column_neurons / 3.0
        # ★v036調整★ sigma係数を調整可能にして最適化
        if column_neurons is not None:
            # デフォルト: 3.0（仕様書準拠）
            # 調整可能範囲: 2.0-5.0（より広い/狭い分布）
            sigma_coef = 3.0  # TODO: パラメータ化検討
            sigma = column_neurons / sigma_coef
        else:
            sigma = column_radius  # 旧方式互換性維持
        
        # ベクトル化されたガウス型帰属度計算
        affinity[class_idx] = np.exp(-0.5 * (distances / sigma) ** 2)
    
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


def create_column_affinity(n_hidden, n_classes, column_size=30, overlap=0.3, use_gaussian=True,
                           column_neurons=None, participation_rate=None):
    """
    コラム帰属度マップの作成(ガウス型または固定型) - 円環構造（v027更新）
    
    ★v027更新★ ハニカム構造の知見を反映:
    - participation_rate対応: 部分参加による学習促進
    - 中心化配置: 全クラスの均等なニューロンアクセス
    - 検証ロジック: 0.0と1.0の禁止
    
    Args:
        n_hidden: 隠れ層のニューロン数
        n_classes: クラス数
        column_size: 各コラムの基準サイズ(ニューロン数) - column_neurons/participation_rateが未指定時に使用
        overlap: コラム間の重複度(0.0-1.0) - participation_rate未指定時に使用
        use_gaussian: Trueならガウス型、Falseなら固定型
        column_neurons: 各クラスのコラムに割り当てるニューロン数（指定時はcolumn_sizeより優先）
        participation_rate: コラム参加率（0.01-0.99）。指定時はcolumn_neurons/column_sizeより優先
                           0.0と1.0は禁止（0.0=コラム無意味、1.0=学習不安定）
    
    Returns:
        affinity: コラム帰属度マップ shape [n_classes, n_hidden]
                  各要素は0.0-1.0の値(ニューロンがそのクラスに帰属する度合い)
    """
    # ★v027追加★ participation_rate検証（ハニカム構造と同じ）
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
    
    affinity = np.zeros((n_classes, n_hidden))
    
    if use_gaussian:
        # ガウス型コラム帰属度
        # ★v027改善 + 2026-01-11修正★ 2次元円環配置（円周上に等角度間隔配置）
        # columnar_ed.prompt.md準拠: 各クラスを円周上に配置してトーラストポロジーを表現
        grid_size = int(np.ceil(np.sqrt(n_hidden)))
        grid_center = grid_size / 2.0
        circle_radius = grid_size * 0.4  # グリッド中心からの円半径
        
        # ニューロンを2次元格子に配置（直交座標系）
        neuron_positions = np.array([
            [i // grid_size, i % grid_size] for i in range(n_hidden)
        ])
        
        # ガウス分布の標準偏差（優先度: column_neurons > column_size）
        if column_neurons is not None:
            sigma = column_neurons / 3.0  # 仕様書準拠: 3σ点で帰属度がほぼゼロ
        else:
            sigma = column_size / 3.0
        
        # 各クラスを円周上に等角度間隔で配置（10クラス→36度ごと）
        for class_idx in range(n_classes):
            angle = 2 * np.pi * class_idx / n_classes
            center_row = grid_center + circle_radius * np.cos(angle)
            center_col = grid_center + circle_radius * np.sin(angle)
            
            # 各ニューロンとの距離を計算
            distances = np.sqrt(
                (neuron_positions[:, 0] - center_row) ** 2 +
                (neuron_positions[:, 1] - center_col) ** 2
            )
            
            # ガウス分布で帰属度を計算
            affinity[class_idx] = np.exp(-0.5 * (distances / sigma) ** 2)
            
            # 閾値処理(3σ点以下はゼロ)
            threshold = np.exp(-0.5 * 9)  # exp(-4.5) ≈ 0.011
            affinity[class_idx][affinity[class_idx] < threshold] = 0
        
        # ★v027追加★ participation_rate対応（優先度: participation_rate > column_neurons > overlap）
        if participation_rate is not None:
            # モード1（最優先）: 参加率指定
            target_neurons = int(n_hidden * participation_rate)
            neurons_per_class = target_neurons // n_classes
            remainder = target_neurons % n_classes
            
            assigned = np.zeros(n_hidden, dtype=bool)
            overlap_factor = 0.0 if participation_rate >= 0.99 else 0.3
            
            for class_idx in range(n_classes):
                available_mask = ~assigned
                available_affinity = affinity[class_idx].copy()
                
                # overlap許容の場合のみ、割り当て済みニューロンも重み付けで考慮
                if overlap_factor > 0:
                    available_affinity[~available_mask] *= overlap_factor
                else:
                    available_affinity[~available_mask] = 0
                
                sorted_indices = np.argsort(available_affinity)[::-1]
                
                # 余りニューロンを最初のremainder個のクラスに+1個ずつ分配
                n_neurons_for_this_class = neurons_per_class + (1 if class_idx < remainder else 0)
                selected = sorted_indices[:n_neurons_for_this_class]
                
                # v026マスク方式（極小アフィニティは自然にフィルタリング）
                mask = np.zeros(n_hidden)
                mask[selected] = 1
                affinity[class_idx] *= mask
                
                assigned[selected] = True
        
        elif column_neurons is not None:
            # モード2（中優先）: 明示的なニューロン数指定
            assigned = np.zeros(n_hidden, dtype=bool)
            overlap_factor = 0.3
            
            for class_idx in range(n_classes):
                available_mask = ~assigned
                available_affinity = affinity[class_idx].copy()
                available_affinity[~available_mask] *= overlap_factor
                
                sorted_indices = np.argsort(available_affinity)[::-1]
                selected = sorted_indices[:column_neurons]
                
                mask = np.zeros(n_hidden)
                mask[selected] = 1
                affinity[class_idx] *= mask
                
                assigned[selected] = True
        
        else:
            # モード3（最低優先）: 従来のoverlap制御
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
    
    # 正規化: 各クラスの帰属度マップを正規化
    for class_idx in range(n_classes):
        total = np.sum(affinity[class_idx])
        if total > 1e-8:
            if column_neurons is not None:
                # 明示的ニューロン数の場合: 合計をcolumn_neuronsに正規化
                affinity[class_idx] *= (column_neurons / total)
            else:
                # radius方式: 合計をcolumn_sizeに正規化
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
