#!/usr/bin/env python3
"""
ED法ネットワークモジュール（統合クラス）

★Pure ED + Column Structure Network★
役割:
  - RefinedDistributionEDNetworkクラス
  - 多層多クラス分類対応
  - ED法準拠（微分の連鎖律不使用）
  - HyperParams統合対応

クラス:
  - RefinedDistributionEDNetwork: メインネットワーククラス

使用例:
    from modules.ed_network import RefinedDistributionEDNetwork
    from modules.hyperparameters import HyperParams
    
    # HyperParamsを使用
    hp = HyperParams()
    config = hp.get_config(n_layers=2)
    
    network = RefinedDistributionEDNetwork(
        n_input=784,
        n_hidden=config['hidden'],
        n_output=10,
        learning_rate=config['learning_rate'],
        base_column_radius=config['base_column_radius']
    )
    
    # 学習
    network.train_one_sample(x, y_true)
    
    # 評価
    accuracy, loss = network.evaluate(x_test, y_test)
"""

from __future__ import annotations
from typing import overload, Literal

import numpy as np

# CuPy対応（オプション）
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from .activation_functions import sigmoid, tanh_activation, softmax, softmax_batch, cross_entropy_loss
from .neuron_structure import create_ei_pairs, create_ei_pairs_batch, create_ei_flags
from .column_structure import (
    hex_distance,
    create_hexagonal_column_affinity,
    create_column_affinity,
    create_lateral_weights,
    create_column_membership,
    create_receptive_fields
)


class RefinedDistributionEDNetwork:
    """
    Pure ED + Column Structure Network (純粋なED法 + コラム構造)
    
    HyperParams統合版:
      - HyperParamsクラスのテーブル設定を活用可能
      - 個別パラメータ指定も維持（後方互換性）
      - 優先順位: 明示的指定 > HyperParams > デフォルト
    """
    
    def __init__(self, n_input=784, n_hidden=[250], n_output=10, 
                 learning_rate=0.20, lateral_lr=0.08, u1=0.5, u2=0.8,
                 column_radius=None, base_column_radius=1.0, column_neurons=None, participation_rate=1.0,
                 use_hexagonal=True, overlap=0.0, activation='tanh', leaky_alpha=0.01,
                 use_layer_norm=False, gradient_clip=0.0, 
                 lateral_cooperation=0.0, top_k_winners=None,
                 hyperparams=None, init_limit=0.08, sparsity=0.0, hidden_sparsity=None,
                 column_lr_factors=None, init_method='uniform',
                 use_affinity=False, affinity_max=1.0, affinity_min=0.0,
                 init_scales=None, layer_learning_rates=None, flat_value=0.05, flat_perturbation=0.001,
                 normalize_output_weights=False, balance_output_weights=False,
                 uniform_amine=False,
                 rf_overlap=0.5, rf_mode='random', seed=None,
                 rank_lut_mode='default',
                 amine_base_level=0.0,
                 column_decorrelation=0.0,
                 amine_diffusion_sigma=0.0,
                 output_spatial_bias=0.0,
                 hebbian_alignment=0.0,
                 lateral_inhibition=0.0,
                 enable_non_column_learning=False,
                 nc_sparse_k=0.0,
                 nc_amine_strength=0.5,
                 nc_nearest_learning=False,
                 competitive_inhibition=False,
                 inhibition_strength=0.01,
                 inhibition_topk=1,
                 column_lateral_inhibition=False,
                 cli_alpha=0.1):
        """
        初期化
        
        Args:
            n_input: 入力次元数（784 for MNIST）
            n_hidden: 隠れ層ニューロン数のリスト（例: [256] or [256, 128]）
            n_output: 出力クラス数
            learning_rate: 学習率（Phase 1 Extended Overall Best: 0.20）
            lateral_lr: 側方抑制の学習率（Phase 1 Extended Overall Best: 0.08）
            u1: アミン拡散係数（Phase 1 Extended Overall Best: 0.5）
            u2: アミン拡散係数（隠れ層間、デフォルト0.8）
            column_radius: コラム影響半径（Noneなら層ごとに自動計算、デフォルト: None）
            base_column_radius: 基準コラム半径（256ニューロン層での値、デフォルト1.0、推奨値）
            column_neurons: 各クラスのコラムに割り当てるニューロン数（明示指定、優先度：中）
            participation_rate: コラム参加率（0.0-1.0、デフォルト1.0=全ニューロン参加、優先度：最高）
            use_hexagonal: Trueならハニカム構造、Falseなら旧円環構造
            overlap: コラム間の重複度（0.0-1.0、円環構造でのみ有効、デフォルト0.0）
            lateral_cooperation: 側方協調学習の強度（0.0-1.0、0.0=無効、Phase A実証: 0.3で最適）
            top_k_winners: 学習参加ニューロン数の上限（None=全員参加、Phase C実証: 1で最適）
            hyperparams: HyperParamsインスタンス（Noneなら個別パラメータ使用）
            init_method: 出力層重み初期化手法（'uniform', 'xavier', 'he'、デフォルト'uniform'）
            use_affinity: Affinity方式を使用（デフォルト: False=Membership方式）
            affinity_max: コラムニューロンのaffinity値（デフォルト: 1.0、use_affinity=Trueで有効）
            affinity_min: 非コラムニューロンのaffinity値（デフォルト: 0.0、use_affinity=Trueで有効）
            init_scales: 層別初期化スケール係数のリスト（Noneなら層数依存デフォルト）
                         長さは len(hidden_layers)+1（Layer 0, Layer 1, ..., 出力層）
                         例: [0.3, 0.5, 1.0] for 2-layer network
                         デフォルト: 1層=[0.3, 1.0], 2層=[0.3, 0.5, 1.0], 3層=[0.3, 0.5, 0.7, 1.0]
            layer_learning_rates: 層ごとの学習率リスト（Noneなら全層で同一学習率）
                                 長さは len(hidden_layers)+1（入力層→第1層、第1層→第2層、...）
                                 例: [0.05, 0.1, 0.15] for 2-layer network
            hidden_sparsity: 隠れ層（非コラムニューロン）のスパース率
                            - List[float]: 層別に指定（len == len(hidden_layers)必須）
                              例: [0.2, 0.3] for 2-layer network (Layer0=0.2, Layer1=0.3)
                            - None: スパース化なし（全層0.0と同等）
                            コラム内ニューロンは密結合を維持、非コラムニューロンのみに適用
                            生物学的背景: 脳内コラム構造は密結合、コラム外は疎結合
            normalize_output_weights: 出力層重みをクラス別に正規化（公平な初期活性化を保証）
                                     Trueの場合、各クラスの重み絶対値平均を強制的に同一に設定
            balance_output_weights: 出力層重みの正負バランスを揃える（v043.1新機能）
                                   Trueの場合、各クラスの重みを正50%:負50%に調整
                                   これにより隠れ層活性化との内積が公平になり、
                                   勝者選択の偏りを防止（Class 5問題の解決策）
        
        HyperParams統合の使用例:
            # パターン1: HyperParamsを使用
            hp = HyperParams()
            config = hp.get_config(n_layers=2)
            network = RefinedDistributionEDNetwork(
                n_input=784,
                n_hidden=config['hidden'],
                learning_rate=config['learning_rate'],
                base_column_radius=config['base_column_radius'],
                hyperparams=hp  # HyperParamsインスタンスを渡す
            )
            
            # パターン2: 個別パラメータ指定（従来通り）
            network = RefinedDistributionEDNetwork(
                n_input=784,
                n_hidden=[256, 128],
                learning_rate=0.05,
                base_column_radius=1.0
            )
        """
        # HyperParamsから設定を取得（指定があれば）
        if hyperparams is not None:
            n_layers = len(n_hidden)
            try:
                config = hyperparams.get_config(n_layers)
                # 明示的に指定されていないパラメータをHyperParamsから取得
                if learning_rate == 0.20:  # デフォルト値の場合
                    learning_rate = config.get('learning_rate', learning_rate)
                if base_column_radius == 1.0:  # デフォルト値の場合
                    base_column_radius = config.get('base_column_radius', base_column_radius)
            except ValueError as e:
                print(f"Warning: {e}")
                print("個別パラメータを使用します。")
        
        # パラメータ保存
        self.n_input = n_input
        self.n_hidden = n_hidden if isinstance(n_hidden, list) else [n_hidden]
        self.n_layers = len(self.n_hidden)
        self.n_output = n_output
        self.learning_rate = learning_rate
        
        # ★v039新機能★ 層別学習率（静的パラメータ）
        # layer_learning_rates指定時: ユーザー指定値を使用（層数+1個必要）
        # layer_learning_rates=None時: 従来通り全層で同一学習率
        if layer_learning_rates is not None:
            # 検証: 層数+1（入力層用を含む）と一致するか
            expected_length = self.n_layers + 1
            if len(layer_learning_rates) != expected_length:
                raise ValueError(
                    f"layer_learning_rates must have {expected_length} values "
                    f"(got {len(layer_learning_rates)}). "
                    f"Expected: [lr_layer0, lr_layer1, ...] for {self.n_layers}-layer network."
                )
            self.layer_lrs = list(layer_learning_rates)
            print(f"\n[層別学習率モード有効]")
            for i, lr in enumerate(self.layer_lrs):
                print(f"  Layer {i}: learning_rate={lr:.4f}")
        else:
            # 従来通り: 全層で同一学習率 + 出力層用
            self.layer_lrs = [learning_rate] * (self.n_layers + 1)
        
        # ★v039.2新機能★ エポック依存学習率スケジューリング用のベース学習率を保存
        self.base_layer_lrs = list(self.layer_lrs)
        
        # ★旧実装★ 層ごとの適応的学習率（z_inputとsaturation_termのスケールに対応）
        # 分析結果：
        #   - Layer 1の平均|Δw|: 0.120520
        #   - Layer 0の平均|Δw|: 0.407937（3.54倍）
        #   - 原因1: 最大|z_input|の差（Layer 0: 1.0, Layer 1: 0.5-0.8）
        #   - 原因2: saturation_termの差（Layer 0が極端に小さい: 1/17）
        # 修正不要：saturation_termとz_inputの自然な違いを保持
        # → 層ごとの学習率は同じに戻す（layer_lrsに統合）
        self.layer_specific_lr = self.layer_lrs[:self.n_layers]  # 出力層を除く
        
        self.lateral_lr = lateral_lr  # 側方抑制の学習率
        self.u1 = u1  # アミン拡散係数（出力層→最終隠れ層）
        self.u2 = u2  # アミン拡散係数（隠れ層間）
        self.uniform_amine = uniform_amine  # 全層均一アミン拡散（青斑核モデル）
        self.initial_amine = 1.0  # 基準アミン濃度
        self.amine_base_level = amine_base_level  # 非コラムニューロンへの基本アミン拡散レベル
        self.amine_diffusion_sigma = amine_diffusion_sigma  # 空間的アミン拡散の標準偏差 (0.0=無効)
        self.enable_non_column_learning = enable_non_column_learning  # 非コラムニューロン学習参加フラグ
        self.nc_sparse_k = nc_sparse_k  # 非コラムの活性化抑制ゲート選択率 (0.0=無効, 0.02=上位2%)
        self.nc_amine_strength = nc_amine_strength  # 予約（現在未使用）
        self._nc_gate_mask = None  # forward()で計算、_compute_gradients()で再利用
        self.nc_nearest_learning = nc_nearest_learning  # NC最近傍クラス帰属学習フラグ
        self._nc_nearest_membership = []  # 拡張membership: コラム+NC最近傍 [n_layers][n_classes, n_neurons]
        
        # ★v047新機能★ 競合コラム間選択的抑制
        self.competitive_inhibition = competitive_inhibition  # 競合抑制有効フラグ
        self.inhibition_strength = inhibition_strength  # 抑制強度（正の学習に対する比率）
        self.inhibition_topk = inhibition_topk  # 抑制対象の不正解クラス数
        
        # ★v047新機能★ コラム間活性化側方抑制（Column Lateral Inhibition）
        self.column_lateral_inhibition = column_lateral_inhibition  # 活性化側方抑制フラグ
        self.cli_alpha = cli_alpha  # 抑制強度係数
        
        # ★層別column_lr_factors★
        if column_lr_factors is not None:
            self.column_lr_factors = column_lr_factors
        else:
            # デフォルト: 全層1.0（抑制なし）
            self.column_lr_factors = [1.0] * self.n_layers
        
        self.init_method = init_method  # 出力層重み初期化手法
        self.normalize_output_weights = normalize_output_weights  # ★v039.4新機能★ 出力層重み正規化
        
        # ★新機能★ 層依存のcolumn_radius（シンプルなsqrtスケーリング）
        self.base_column_radius = base_column_radius
        if column_radius is None:
            # 各層のニューロン数に応じて自動計算（基準: 256ニューロン = 1.2）
            self.column_radius_per_layer = [
                base_column_radius * np.sqrt(n / 256.0) for n in self.n_hidden
            ]
            print(f"\n[層依存column_radius自動計算]")
            for i, (n, r) in enumerate(zip(self.n_hidden, self.column_radius_per_layer)):
                print(f"  Layer {i}: {n}ニューロン → radius={r:.2f}")
        else:
            # ユーザー指定値を全層で使用
            self.column_radius_per_layer = [column_radius] * self.n_layers
            print(f"\n[column_radius固定値使用: {column_radius}]")
        
        self.column_neurons = column_neurons
        self.participation_rate = participation_rate
        self.use_hexagonal = use_hexagonal
        self.activation = activation  # 'sigmoid' or 'leaky_relu'
        self.leaky_alpha = leaky_alpha  # Leaky ReLUの負勾配
        self.use_layer_norm = use_layer_norm  # 層間正規化
        self.gradient_clip = gradient_clip  # 勾配クリッピング値
        # ★拡張★ 隠れ層（非コラム）スパース率を層別に管理
        # hidden_sparsityの正規化（List[float]に統一）
        if hidden_sparsity is None:
            self.hidden_sparsity = [0.0] * len(self.n_hidden)
        else:
            self.hidden_sparsity = list(hidden_sparsity)
            # 層数との整合性チェック
            if len(self.hidden_sparsity) != len(self.n_hidden):
                raise ValueError(
                    f"hidden_sparsityの数({len(self.hidden_sparsity)})が"
                    f"隠れ層数({len(self.n_hidden)})と一致しません。"
                    f"例: --hidden_sparsity 0.2,0.3 for 2-layer network"
                )
        
        # ★v036新機能★ Phase A/C実証済み改善機能
        self.lateral_cooperation = lateral_cooperation  # 側方協調学習の強度（0.0-1.0）
        self.top_k_winners = top_k_winners  # 学習参加ニューロン数の上限（None=全員参加）
        
        # デバッグモード（v036.1で追加）
        self.debug_lateral_cooperation = False  # 外部から設定可能
        self.lc_stats = {  # lateral_cooperation統計情報
            'avg_cooperation': 0.0,
            'avg_before': 0.0,
            'avg_after': 0.0
        }
        
        # ★v039.3新機能★ 勝者選択回数カウンター（エポック毎に統計表示）
        self.winner_selection_counts = np.zeros(n_output, dtype=int)
        self.total_training_samples = 0
        
        # 各クラスの学習実行回数
        self.class_training_counts = np.zeros(n_output, dtype=int)
        
        # デバッグモード（v036.1で追加）
        self.debug_lateral_cooperation = False  # 外部から設定可能
        self.lc_stats = {  # lateral_cooperation統計情報
            'total_samples': 0,
            'affected_neurons_per_class': [],
            'activation_changes': [],
            'members_count_per_class': []
        }
        
        # ★v039.3新機能★ 勝者選択回数の記録
        self.winner_selection_counts = np.zeros(n_output, dtype=int)  # クラス別勝者選択回数
        self.total_training_samples = 0  # 総学習サンプル数
        
        # 側方抑制（必須要素6）- ゼロ初期化、学習中に動的更新
        self.lateral_weights = create_lateral_weights(n_output)
        
        # ★v040新機能★ 層別初期化スケール係数
        # デフォルト値: Noneの場合は層数に応じた値を自動設定
        if init_scales is None:
            n_layers = len(self.n_hidden)
            if n_layers == 1:
                init_scales = [0.4, 1.0]  # [Layer 0, 出力層] (v044グリッドサーチ最適値)
            elif n_layers == 2:
                init_scales = [0.3, 0.5, 1.0]  # [Layer 0, Layer 1, 出力層]
            elif n_layers == 3:
                init_scales = [0.3, 0.5, 0.7, 1.0]
            elif n_layers == 4:
                init_scales = [0.3, 0.5, 0.7, 0.9, 1.0]
            else:  # 5層以上
                # 段階的に増加するスケールを生成
                init_scales = [0.3] + [0.3 + (0.7 * i / (n_layers - 1)) for i in range(1, n_layers)] + [1.0]
        self.init_scales = init_scales  # 保存（後で参照可能）
        
        # ★v046新機能★ コラム内重みベクトル脱相関強度
        self.column_decorrelation = column_decorrelation
        
        # ★新機能★ ランク依存学習率のルックアップテーブル（ベクトル化最適化用）
        # rank_lut_mode で学習率カーブを選択可能
        self.rank_lut_mode = rank_lut_mode
        max_rank_in_lut = 256  # 十分な長さ（最大コラムサイズを想定）
        self._learning_weight_lut = np.zeros(max_rank_in_lut, dtype=np.float32)
        if rank_lut_mode == 'equal':
            # 全コラムニューロンが均等に学習（cn>1の差別化促進）
            self._learning_weight_lut[:] = 1.0
        elif rank_lut_mode == 'gradual':
            # 緩やかな線形減衰（最低0.2、全ニューロンが有意に学習）
            for i in range(max_rank_in_lut):
                self._learning_weight_lut[i] = max(0.2, 1.0 - 0.1 * i)
        elif rank_lut_mode == 'classic':
            # v032方式（旧default）: 1位=100%, 2位=70%, 3位=40%, 4位=33%, 5位以降=0%
            self._learning_weight_lut[0] = 1.0
            self._learning_weight_lut[1] = 0.7
            self._learning_weight_lut[2] = 0.4
            self._learning_weight_lut[3] = 0.33
            # 4位以降は0.0（デフォルトで既に0）
        else:  # 'default' → cn依存型線形減衰
            # ★v046変更★ column_neuronsに応じた線形減衰LUT
            # LUT[r] = (cn - r) / cn で全コラムニューロンが学習に参加
            # cn=1: [1.0], cn=5: [1.0, 0.8, 0.6, 0.4, 0.2], cn=10: [1.0, 0.9, ..., 0.1]
            cn = self.column_neurons if self.column_neurons is not None else 1
            for i in range(max_rank_in_lut):
                if i < cn:
                    self._learning_weight_lut[i] = (cn - i) / cn
                else:
                    self._learning_weight_lut[i] = 0.0
        
        # ★重要★ コラム帰属度マップの初期化（ハニカム構造版）
        # ★v032移行★ Membership方式とAffinity方式を並行運用
        # ★v038.2追加★ use_affinityフラグでAffinity方式の再実験を可能に
        self.use_affinity = use_affinity  # Affinity方式使用フラグ
        self.affinity_max = affinity_max  # コラムニューロンのaffinity値
        self.affinity_min = affinity_min  # 非コラムニューロンのaffinity値
        self.column_membership_all_layers = []  # 新方式（学習可能化対応）
        self.column_affinity_all_layers = []  # 旧方式（後方互換性のため保持）
        self.neuron_positions_all_layers = []  # 可視化用2D座標
        self.class_coords_all_layers = []  # 可視化用コラム中心座標
        
        for layer_idx, layer_size in enumerate(self.n_hidden):
            # 各層に対応するradiusを取得
            layer_radius = self.column_radius_per_layer[layer_idx]
            
            # ★v038.2実装★ use_affinityに応じて処理を分岐
            if use_affinity:
                # Affinity方式（実験用）
                if use_hexagonal:
                    affinity = create_hexagonal_column_affinity(
                        n_hidden=layer_size,
                        n_classes=n_output,
                        column_neurons=column_neurons,
                        participation_rate=participation_rate,
                        affinity_max=affinity_max,
                        affinity_min=affinity_min
                    )
                else:
                    affinity = create_column_affinity(
                        n_hidden=layer_size,
                        n_classes=n_output,
                        column_size=int(layer_radius * 10),
                        overlap=overlap,
                        use_gaussian=True,
                        column_neurons=column_neurons,
                        participation_rate=participation_rate
                    )
                self.column_affinity_all_layers.append(affinity)
                
                # Membershipもaffinityから生成（整合性維持）
                membership = (affinity >= (affinity_max + affinity_min) / 2.0)
                self.column_membership_all_layers.append(membership)
                # Affinity方式では座標情報なし
                self.neuron_positions_all_layers.append(None)
                self.class_coords_all_layers.append(None)
            else:
                # Membership方式（標準）
                membership, neuron_positions, class_coords = create_column_membership(
                    n_hidden=layer_size,
                    n_classes=n_output,
                    participation_rate=participation_rate,
                    use_hexagonal=use_hexagonal,
                    column_radius=layer_radius,
                    column_neurons=column_neurons
                )
                self.column_membership_all_layers.append(membership)
                self.neuron_positions_all_layers.append(neuron_positions)
                self.class_coords_all_layers.append(class_coords)
                
                # Affinityは後方互換性のため空配列
                self.column_affinity_all_layers.append(np.zeros((n_output, layer_size)))
        
        print(f"\n[コラム構造初期化]")
        if use_affinity:
            print(f"  - ★v038.2実験★ Affinity方式（実験用）")
            print(f"  - affinity_max: {affinity_max} (コラムニューロン)")
            print(f"  - affinity_min: {affinity_min} (非コラムニューロン)")
            print(f"  - 学習係数: affinity値で直接制御")
        else:
            print(f"  - ★v032移行★ Membership方式（学習可能化対応）")
            print(f"  - コラム所属: ブールフラグ（固定）")
            print(f"  - 勝者決定: 活性値ランクベース（動的）")
        if use_hexagonal:
            print(f"  - 方式: ハニカム構造(2-3-3-2配置)")
            # 実際の優先順位: participation_rate > column_neurons > radius
            if participation_rate is not None:
                print(f"  - モード: 参加率指定（{participation_rate * 100:.0f}%）")
                print(f"  - 各クラス約{int(self.n_hidden[0] * participation_rate / n_output)}ニューロン")
            elif column_neurons is not None:
                print(f"  - モード: 完全コラム化（各クラス{column_neurons}ニューロン）")
                print(f"  - 参加率: {column_neurons * n_output / self.n_hidden[0] * 100:.1f}%")
            else:
                print(f"  - モード: 半径ベース（radius={self.column_radius_per_layer[0]:.2f}）")
        else:
            print(f"  - 方式: 円環構造（v027更新: 中心化配置+participation_rate対応）")
            # 実際の優先順位: participation_rate > column_neurons > radius
            if participation_rate is not None:
                print(f"  - モード: 参加率指定（{participation_rate * 100:.0f}%）")
                print(f"  - 各クラス約{int(self.n_hidden[0] * participation_rate / n_output)}ニューロン")
            elif column_neurons is not None:
                print(f"  - モード: 完全コラム化（各クラス{column_neurons}ニューロン）")
                print(f"  - 参加率: {column_neurons * n_output / self.n_hidden[0] * 100:.1f}%")
            else:
                print(f"  - モード: 従来方式（コラムサイズ: {int(self.column_radius_per_layer[0] * 10)}）")
        
        for layer_idx in range(len(self.column_affinity_all_layers)):
            if self.use_affinity:
                # Affinity方式: affinity_maxとaffinity_minの中間値を閾値とする
                affinity = self.column_affinity_all_layers[layer_idx]
                threshold = (self.affinity_max + self.affinity_min) / 2.0
                non_zero_counts = [int(np.count_nonzero(affinity[c] > threshold)) for c in range(n_output)]
            else:
                # Membership方式: membershipから直接カウント
                membership = self.column_membership_all_layers[layer_idx]
                non_zero_counts = [int(np.count_nonzero(membership[c])) for c in range(n_output)]
            print(f"  - 層{layer_idx+1}: 各クラスの帰属ニューロン数={non_zero_counts}")
        
        # ★v046新機能★ 空間的アミン拡散の距離重み行列の事前計算
        # コラムの学習時にamineが空間的に減衰しながら近傍の非コラムニューロンに拡散
        # 生物学的根拠: 神経修飾物質（ノルアドレナリン等）の空間的拡散
        self._spatial_amine_weights = []  # 各層の距離重み [n_classes, n_neurons]
        if self.amine_diffusion_sigma > 0.0:
            for layer_idx in range(len(self.n_hidden)):
                positions = self.neuron_positions_all_layers[layer_idx]
                coords = self.class_coords_all_layers[layer_idx]
                membership = self.column_membership_all_layers[layer_idx]
                n_neurons = self.n_hidden[layer_idx]
                
                if positions is not None and coords is not None:
                    # 各ニューロンと各コラム中心の距離を計算
                    spatial_weights = np.zeros((self.n_output, n_neurons), dtype=np.float32)
                    for class_idx in range(self.n_output):
                        if class_idx in coords:
                            center = np.array(coords[class_idx])  # [2]
                            # 全ニューロンとの距離
                            dists = np.sqrt(np.sum((positions - center[np.newaxis, :]) ** 2, axis=1))  # [n_neurons]
                            # ガウシアン減衰
                            spatial_weights[class_idx] = np.exp(-dists ** 2 / (2 * self.amine_diffusion_sigma ** 2))
                    
                    # コラムメンバーの空間重みは使わない（LUTで制御）ので0にする
                    is_column = np.any(membership, axis=0)  # [n_neurons]
                    spatial_weights[:, is_column] = 0.0
                    
                    # ★重要★ 空間分布パターンを正規化（各クラスの最大値を1.0に）
                    # 実際の強度はamine_base_levelで制御する分離設計
                    # これにより:
                    #   - σ: 空間的な拡散範囲のみを制御
                    #   - amine_base_level: 非コラムの学習強度のみを制御
                    for class_idx in range(self.n_output):
                        max_w = spatial_weights[class_idx].max()
                        if max_w > 1e-8:
                            spatial_weights[class_idx] /= max_w
                    
                    # ★性能最適化★ 微小な重みを閾値でカット
                    # 理由: 重み0.01未満のニューロンまでactive扱いになると
                    # 勾配計算の対象が数百個に膨れ、計算量が数十倍に増加する
                    spatial_threshold = 0.01
                    spatial_weights[spatial_weights < spatial_threshold] = 0.0
                    
                    # amine_base_levelでスケーリング（ピーク強度制御）
                    # spatial_weights[c,i] ∈ [0, 1] × amine_base_level → [0, amine_base_level]
                    spatial_weights *= self.amine_base_level
                    
                    self._spatial_amine_weights.append(spatial_weights)
                    
                    # 統計表示
                    nc_mask = ~is_column
                    nc_weights = spatial_weights[:, nc_mask]  # [n_classes, n_non_column]
                    actively_learning = nc_weights > 1e-8
                    nonzero_per_class = np.sum(actively_learning, axis=1)
                    print(f"  [空間的アミン拡散] 層{layer_idx}: σ={self.amine_diffusion_sigma:.1f}, "
                          f"amine_base_level={self.amine_base_level}, "
                          f"有効NC/クラス: 平均{np.mean(nonzero_per_class):.0f}, "
                          f"ピーク重み: {nc_weights.max():.6f}")
                else:
                    # 座標情報がない場合はゼロ行列（無効）
                    self._spatial_amine_weights.append(np.zeros((self.n_output, n_neurons), dtype=np.float32))
        else:
            # sigma=0.0: 全層ゼロ行列（従来動作と同一）
            for layer_idx in range(len(self.n_hidden)):
                n_neurons = self.n_hidden[layer_idx]
                self._spatial_amine_weights.append(np.zeros((self.n_output, n_neurons), dtype=np.float32))        
        # 興奮性・抑制性フラグ（必須要素7）
        # 入力ペア: 前半が興奮性(+1)、後半が抑制性(-1)
        n_input_paired = n_input * 2
        self.ei_flags_input = np.array([1 if i < n_input else -1 
                                        for i in range(n_input_paired)])
        
        # 各隠れ層のE/Iフラグ（第1層以外は全て興奮性）
        self.ei_flags_hidden = []
        for layer_idx, layer_size in enumerate(self.n_hidden):
            if layer_idx == 0:
                # 第1層のみ: 交互配置（ただしv017では全て興奮性）
                # ★重要★ v017に合わせて全て興奮性にする
                ei_flags = np.ones(layer_size)
            else:
                # 第2層以降: 全て興奮性
                ei_flags = np.ones(layer_size)
            self.ei_flags_hidden.append(ei_flags)
        
        # 重みの初期化（適応的スケーリング - 飽和問題対策）
        # ★v039改善★ 隠れ層にも--init_method適用
        # Layer 0: 入力次元が大きい(784*2=1568) → 小さいスケールで飽和を防ぐ
        # Layer 1+: 標準的な初期化スケール
        self.w_hidden = []
        for layer_idx in range(self.n_layers):
            if layer_idx == 0:
                # 第1層: 入力→隠れ層
                n_in = n_input_paired
                n_out = self.n_hidden[0]
            else:
                # 第2層以降: 隠れ層→隠れ層
                n_in = self.n_hidden[layer_idx - 1]
                n_out = self.n_hidden[layer_idx]
            
            # v039: --init_methodに応じた初期化スケール選択
            if init_method == 'he':
                # He初期化（ReLU系活性化関数向け、Tanhでも有効）
                # 理論: Var(W) = 2 / fan_in
                base_scale = np.sqrt(2.0 / n_in)
            elif init_method == 'xavier':
                # Xavier初期化（Tanh/Sigmoid向け）
                # 理論: Var(W) = 1 / fan_in
                base_scale = np.sqrt(1.0 / n_in)
            elif init_method == 'flat':
                # フラット初期化（全重み同一値、デバッグ用）
                # 初期化の影響を完全に排除し、純粋な学習ダイナミクスを観察
                base_scale = flat_value  # パラメータ指定値
            else:  # 'uniform'
                # 従来のXavier風（デフォルト）
                base_scale = np.sqrt(1.0 / n_in)
            
            # 層ごとの適応的係数
            if init_method == 'flat':
                # フラット初期化: ベース値 + 小さなランダム摂動
                # 摂動により重みに最小限の分散を与え、ED法の学習を可能にする
                scale = base_scale
                w = scale + np.random.uniform(-flat_perturbation, +flat_perturbation, (n_out, n_in))
                print(f"  [重み初期化] Layer {layer_idx}: init_method=flat, "
                      f"base={scale:.6f}, perturbation=±{flat_perturbation:.6f}, "
                      f"range=[{scale-flat_perturbation:.6f}, {scale+flat_perturbation:.6f}]")
            else:
                # 通常初期化（He/Xavier/Uniform）
                # ★v040新機能★ 層別スケール係数を適用
                layer_scale = self.init_scales[layer_idx]
                scale = base_scale * layer_scale
                w = np.random.randn(n_out, n_in) * scale
                print(f"  [重み初期化] Layer {layer_idx}: init_method={init_method}, "
                      f"scale={scale:.4f} (base={base_scale:.4f}×{layer_scale})")
            
            # ★拡張★ 隠れ層のスパース化（非コラムニューロンのみ、層別指定対応）
            # 生物学的背景: 脳内コラム構造は密結合、コラム外は疎結合
            layer_sparsity = self.hidden_sparsity[layer_idx] if layer_idx < len(self.hidden_sparsity) else 0.0
            if layer_sparsity > 0.0 and layer_idx < len(self.column_membership_all_layers):
                membership = self.column_membership_all_layers[layer_idx]
                # 非コラムニューロンを特定（どのクラスにも属さないニューロン）
                is_column_neuron = np.any(membership, axis=0)  # [n_hidden]
                non_column_mask = ~is_column_neuron
                
                # 非コラムニューロンの重みにのみスパース化を適用
                n_non_column = np.sum(non_column_mask)
                n_zeros_hidden = 0
                if n_non_column > 0:
                    non_column_indices = np.where(non_column_mask)[0]
                    for neuron_idx in non_column_indices:
                        # このニューロンへの入力重みをランダムにゼロ化
                        sparsity_mask = np.random.rand(n_in) < layer_sparsity
                        w[neuron_idx, sparsity_mask] = 0.0
                        n_zeros_hidden += np.sum(sparsity_mask)
                    
                    total_weights = n_non_column * n_in
                    n_column = n_out - n_non_column
                    print(f"  [隠れ層スパース化] Layer {layer_idx}: "
                          f"コラム{n_column}個(密結合維持), 非コラム{n_non_column}個, "
                          f"{n_zeros_hidden}/{total_weights}重みを0に設定 "
                          f"(実効率{n_zeros_hidden/total_weights*100:.1f}%, 指定率{layer_sparsity*100:.1f}%)")
            
            self.w_hidden.append(w)
        
        # 出力層の重み初期化（v036.5: Simple Uniform、v036.6: スパース化、v038.1: Xavier/He追加）
        # ★v040新機能★ 出力層にもinit_scalesの最後の値を適用
        # 初期化手法の選択
        fan_in = self.n_hidden[-1]
        fan_out = n_output
        output_scale = self.init_scales[-1]  # 出力層のスケール係数
        
        if init_method == 'uniform':
            # Simple Uniform初期化（従来手法、生物学的観点重視）
            # ★v040改善★ スケール係数を適用
            scaled_limit = init_limit * output_scale
            self.w_output = np.random.uniform(-scaled_limit, +scaled_limit, (n_output, fan_in))
            print(f"  [重み初期化] Output Layer: init_method=uniform, "
                  f"limit=±{scaled_limit:.4f} (base=±{init_limit}×{output_scale})")
        elif init_method == 'xavier':
            # Xavier初期化（Glorot初期化）
            # 理論: Var(W) = 2 / (fan_in + fan_out)
            # 一様分布版: W ~ U(-limit, +limit), limit = sqrt(6 / (fan_in + fan_out))
            limit_xavier = np.sqrt(6.0 / (fan_in + fan_out)) * output_scale
            self.w_output = np.random.uniform(-limit_xavier, limit_xavier, (n_output, fan_in))
            print(f"  [重み初期化] Output Layer: init_method=xavier, "
                  f"limit=±{limit_xavier:.4f} (xavier_base×{output_scale})")
        elif init_method == 'he':
            # He初期化（Kaiming初期化）
            # 理論: Var(W) = 2 / fan_in
            # 正規分布版: W ~ N(0, sqrt(2 / fan_in))
            std_he = np.sqrt(2.0 / fan_in) * output_scale
            self.w_output = np.random.randn(n_output, fan_in) * std_he
            print(f"  [重み初期化] Output Layer: init_method=he, "
                  f"std={std_he:.4f} (he_base×{output_scale})")
        elif init_method == 'flat':
            # フラット初期化（ベース値 + 小さなランダム摂動）
            # ★v040改善★ スケール係数を適用
            scaled_flat_value = flat_value * output_scale
            self.w_output = scaled_flat_value + np.random.uniform(-flat_perturbation, +flat_perturbation, (n_output, fan_in))
            print(f"  [重み初期化] Output Layer: init_method=flat, "
                  f"base={scaled_flat_value:.6f} (flat_base×{output_scale}), perturbation=±{flat_perturbation:.6f}")
        else:
            raise ValueError(f"Unknown init_method: {init_method}. Choose from 'uniform', 'xavier', 'he', 'flat'.")
        
        # スパース化（v036.6新規）- 全初期化手法に適用可能
        n_zeros = 0
        if sparsity > 0.0:
            # ランダムに指定率の重みを0に設定
            mask = np.random.rand(n_output, fan_in) < sparsity
            n_zeros = np.sum(mask)
            self.w_output[mask] = 0.0
        
        # 初期化確認（正規化なしでクラス間の偏りを検証）
        actual_abs_means = [np.abs(self.w_output[c]).mean() for c in range(n_output)]
        non_zero_counts = [np.count_nonzero(self.w_output[c]) for c in range(n_output)]
        
        # 初期化手法の表示
        if init_method == 'uniform':
            method_str = f"Simple Uniform, limit=±{init_limit:.4f}"
        elif init_method == 'xavier':
            limit_xavier = np.sqrt(6.0 / (fan_in + fan_out))
            method_str = f"Xavier (Glorot), limit=±{limit_xavier:.4f}"
        elif init_method == 'he':
            std_he = np.sqrt(2.0 / fan_in)
            method_str = f"He (Kaiming), std={std_he:.4f}"
        elif init_method == 'flat':
            method_str = f"Flat (base={flat_value:.6f}, perturbation=±{flat_perturbation:.6f})"
        else:
            method_str = f"{init_method}"
        
        if sparsity > 0.0:
            print(f"  [重み初期化] 出力層: {method_str}, sparsity={sparsity:.2f} ({n_zeros}/{self.w_output.size}個を0に設定)")
        else:
            print(f"  [重み初期化] 出力層: {method_str}")
        print(f"  [重み初期化] クラス別絶対値平均: min={min(actual_abs_means):.6f}, max={max(actual_abs_means):.6f}, mean={np.mean(actual_abs_means):.6f}")
        
        # ★v039.4新機能★ 出力層重みの正規化（公平な初期活性化を保証）
        if normalize_output_weights:
            # 各クラスの重み絶対値平均を計算
            class_abs_means = np.array([np.abs(self.w_output[c]).mean() for c in range(n_output)])
            target_mean = class_abs_means.mean()  # 全クラスの平均を目標値とする
            
            # 各クラスの重みをスケーリング（符号を保持）
            for c in range(n_output):
                if class_abs_means[c] > 1e-10:  # ゼロ除算回避
                    scale_factor = target_mean / class_abs_means[c]
                    self.w_output[c] *= scale_factor
            
            # 正規化後の統計
            normalized_abs_means = np.array([np.abs(self.w_output[c]).mean() for c in range(n_output)])
            print(f"  [★正規化実行★] 各クラスの重み絶対値平均を {target_mean:.6f} に統一")
            print(f"  [正規化後] クラス別絶対値平均: min={normalized_abs_means.min():.6f}, max={normalized_abs_means.max():.6f}, mean={normalized_abs_means.mean():.6f}")
            print(f"  [正規化後] 標準偏差: {normalized_abs_means.std():.8f} (理想: 0.000000)")
        
        if sparsity > 0.0:
            print(f"  [重み初期化] クラス別非ゼロ数: min={min(non_zero_counts)}, max={max(non_zero_counts)}, mean={np.mean(non_zero_counts):.1f}")
        
        # ★v043.1新機能★ 出力層重みの正負バランス調整（勝者選択の公平性を保証）
        if balance_output_weights:
            n_weights = self.w_output.shape[1]
            
            # まず絶対値平均を統一（normalize_output_weightsが無効でも実行）
            if not normalize_output_weights:
                class_abs_means = np.array([np.abs(self.w_output[c]).mean() for c in range(n_output)])
                target_mean = class_abs_means.mean()
                for c in range(n_output):
                    if class_abs_means[c] > 1e-10:
                        scale_factor = target_mean / class_abs_means[c]
                        self.w_output[c] *= scale_factor
            
            # 各クラスの正負バランスを50%:50%に調整
            for c in range(n_output):
                weights = self.w_output[c]
                
                # 絶対値でソートしたインデックスを取得（大きい順）
                abs_sorted_idx = np.argsort(np.abs(weights))[::-1]
                
                # 交互に正負を割り当て（大きい重みから）
                new_weights = np.abs(weights)
                for i, idx in enumerate(abs_sorted_idx):
                    if i % 2 == 0:
                        new_weights[idx] = np.abs(weights[idx])  # 正
                    else:
                        new_weights[idx] = -np.abs(weights[idx])  # 負
                
                self.w_output[c] = new_weights
            
            # バランス調整後の統計
            pos_ratios = [np.mean(self.w_output[c] > 0) * 100 for c in range(n_output)]
            w_sums = [self.w_output[c].sum() for c in range(n_output)]
            print(f"  [★バランス調整★] 各クラスの重みを正50%:負50%に調整")
            print(f"  [バランス後] 正の重み比率: min={min(pos_ratios):.1f}%, max={max(pos_ratios):.1f}%")
            print(f"  [バランス後] W_sum: min={min(w_sums):.6f}, max={max(w_sums):.6f}")
        
        # ★Idea A★ Hebbian Weight Alignment の事前計算
        # 生物学的背景: V1の方位コラムのように、空間的に近いニューロンは類似した
        # 特徴に反応する。非コラムニューロンの重みを最も近いクラスコラムの
        # 重み重心に向けてエポック毎にドリフトさせる。
        self.hebbian_alignment_alpha = hebbian_alignment
        self._hebbian_nearest_class_per_layer = []  # 各層の非コラムニューロン→最近傍クラスマップ
        self._hebbian_non_column_indices_per_layer = []  # 各層の非コラムニューロンインデックス
        
        if hebbian_alignment > 0.0 and column_neurons is not None and not use_affinity:
            for layer_idx in range(self.n_layers):
                membership = self.column_membership_all_layers[layer_idx]  # (n_classes, n_neurons)
                positions = self.neuron_positions_all_layers[layer_idx]  # (n_neurons, 2)
                class_centers = self.class_coords_all_layers[layer_idx]  # dict
                
                is_column_any = np.any(membership, axis=0)  # (n_neurons,)
                non_col_idx = np.where(~is_column_any)[0]
                
                if len(non_col_idx) > 0 and positions is not None:
                    centers = np.array([class_centers[c] for c in range(n_output)])  # (n_classes, 2)
                    non_col_pos = positions[non_col_idx]  # (n_non_column, 2)
                    diffs = non_col_pos[:, np.newaxis, :] - centers[np.newaxis, :, :]
                    distances = np.sqrt(np.sum(diffs ** 2, axis=2))  # (n_non_column, n_classes)
                    nearest_class = np.argmin(distances, axis=1)  # (n_non_column,)
                    
                    self._hebbian_non_column_indices_per_layer.append(non_col_idx)
                    self._hebbian_nearest_class_per_layer.append(nearest_class)
                    
                    class_counts = np.bincount(nearest_class, minlength=n_output)
                    print(f"  [★Hebbian Alignment★] Layer {layer_idx}: "
                          f"alpha={hebbian_alignment:.4f}, 非コラム{len(non_col_idx)}個")
                    print(f"  [Hebbian] 最近傍クラス分布: {class_counts.tolist()}")
                else:
                    self._hebbian_non_column_indices_per_layer.append(np.array([], dtype=int))
                    self._hebbian_nearest_class_per_layer.append(np.array([], dtype=int))
        else:
            for _ in range(self.n_layers if hasattr(self, 'n_layers') else len(self.n_hidden)):
                self._hebbian_non_column_indices_per_layer.append(np.array([], dtype=int))
                self._hebbian_nearest_class_per_layer.append(np.array([], dtype=int))
        
        # ★Idea D★ 側方抑制（Lateral Inhibition）の事前計算
        # 生物学的背景: バスケット細胞（GABAergic抑制性介在ニューロン）は異なる
        # コラムの発火率を制御。非コラムニューロンの活性が高い時、
        # 最近傍クラス以外の出力スコアを抑制してコントラストを強化。
        # V1の周辺抑制: 活動の70%低下、スパースコーディング支援。
        self.lateral_inhibition_strength = lateral_inhibition
        self._li_non_column_indices = None
        self._li_inhibition_matrix = None  # (n_output, n_non_column) 正規化済み
        
        if lateral_inhibition > 0.0 and column_neurons is not None and not use_affinity:
            last_layer_idx = self.n_layers - 1
            membership_last = self.column_membership_all_layers[last_layer_idx]
            positions = self.neuron_positions_all_layers[last_layer_idx]
            class_centers = self.class_coords_all_layers[last_layer_idx]
            
            is_column_any = np.any(membership_last, axis=0)
            non_col_idx = np.where(~is_column_any)[0]
            
            if len(non_col_idx) > 0 and positions is not None:
                self._li_non_column_indices = non_col_idx
                centers = np.array([class_centers[c] for c in range(n_output)])
                non_col_pos = positions[non_col_idx]
                diffs = non_col_pos[:, np.newaxis, :] - centers[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diffs ** 2, axis=2))  # (n_non_col, n_classes)
                nearest_class = np.argmin(distances, axis=1)  # (n_non_col,)
                
                # 抑制マトリクス: mask[c, i] = 1 if nearest[i] != c
                # ⇒ クラスcを抑制する非コラムニューロンのマスク
                raw_mask = np.ones((n_output, len(non_col_idx)))
                for i, nc in enumerate(nearest_class):
                    raw_mask[nc, i] = 0.0  # 最近傍クラスは抑制しない
                
                # 各クラスの抑制ニューロン数で正規化（平均抑制量に）
                inhibitor_counts = raw_mask.sum(axis=1, keepdims=True)  # (n_output, 1)
                inhibitor_counts = np.maximum(inhibitor_counts, 1.0)  # ゼロ除算回避
                self._li_inhibition_matrix = raw_mask / inhibitor_counts
                
                class_counts = np.bincount(nearest_class, minlength=n_output)
                print(f"  [★側方抑制★] lateral_inhibition={lateral_inhibition:.4f}, "
                      f"非コラム{len(non_col_idx)}個")
                print(f"  [側方抑制] 最近傍クラス分布: {class_counts.tolist()}")
                print(f"  [側方抑制] クラス別抑制ニューロン数: "
                      f"min={int(raw_mask.sum(axis=1).min())}, max={int(raw_mask.sum(axis=1).max())}")
        
        # Idea B: 出力層空間バイアスマスク（勾配変調用、output_spatial_bias>0で有効）
        self.output_spatial_mask = None
        if output_spatial_bias > 0.0 and column_neurons is not None and not use_affinity:
            last_layer_idx = self.n_layers - 1
            membership_last = self.column_membership_all_layers[last_layer_idx]
            positions = self.neuron_positions_all_layers[last_layer_idx]
            class_centers = self.class_coords_all_layers[last_layer_idx]
            is_column_any = np.any(membership_last, axis=0)
            non_column_indices = np.where(~is_column_any)[0]
            if len(non_column_indices) > 0:
                self.output_spatial_mask = np.ones((n_output, self.n_hidden[-1]))
                centers = np.array([class_centers[c] for c in range(n_output)])
                non_col_positions = positions[non_column_indices]
                diffs = non_col_positions[:, np.newaxis, :] - centers[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diffs ** 2, axis=2))
                nearest_class = np.argmin(distances, axis=1)
                bias = output_spatial_bias
                suppress = bias / (n_output - 1)
                for i, neuron_idx in enumerate(non_column_indices):
                    nc = nearest_class[i]
                    self.output_spatial_mask[nc, neuron_idx] = 1.0 + bias
                    for c in range(n_output):
                        if c != nc:
                            self.output_spatial_mask[c, neuron_idx] = 1.0 - suppress
        
        # ★NC最近傍クラス帰属学習★ 各非コラムニューロンを最近傍マイクロコラムに帰属
        # 生物学的背景: 「局所的な接続：脳のコラムが近隣の細胞と密接に連携」
        # 各NCが最近傍のクラスコラムからのみ学習信号を受け取る→多クラス干渉を原理的に解消
        self._nc_nearest_membership = []
        if nc_nearest_learning and column_neurons is not None and not use_affinity:
            for layer_idx in range(self.n_layers):
                membership = self.column_membership_all_layers[layer_idx]
                positions = self.neuron_positions_all_layers[layer_idx]
                class_centers = self.class_coords_all_layers[layer_idx]
                n_neurons_layer = self.n_hidden[layer_idx]
                
                # 拡張membership: コラムmembershipのコピーから開始
                extended = membership.copy()  # [n_classes, n_neurons]
                
                if positions is not None and class_centers is not None:
                    is_column_any = np.any(membership, axis=0)  # [n_neurons]
                    nc_indices = np.where(~is_column_any)[0]
                    
                    if len(nc_indices) > 0:
                        centers = np.array([class_centers[c] for c in range(n_output)])
                        nc_positions = positions[nc_indices]
                        diffs = nc_positions[:, np.newaxis, :] - centers[np.newaxis, :, :]
                        distances = np.sqrt(np.sum(diffs ** 2, axis=2))  # [n_nc, n_classes]
                        nearest = np.argmin(distances, axis=1)  # [n_nc]
                        
                        # 最近傍クラスにmembership設定
                        for i, neuron_idx in enumerate(nc_indices):
                            extended[nearest[i], neuron_idx] = True
                        
                        # 診断情報出力
                        class_nc_counts = np.bincount(nearest, minlength=n_output)
                        col_counts = np.sum(membership, axis=1)
                        total_counts = np.sum(extended, axis=1)
                        print(f"  [★NC最近傍帰属★] Layer {layer_idx}: "
                              f"NC {len(nc_indices)}個を最近傍クラスに割当")
                        print(f"    コラムニューロン数: {col_counts.tolist()}")
                        print(f"    NC帰属数: {class_nc_counts.tolist()}")
                        print(f"    合計(コラム+NC): {total_counts.tolist()}")
                
                self._nc_nearest_membership.append(extended)
        else:
            # 無効時: 空リスト
            self._nc_nearest_membership = []

        # Dale's Principleの初期化（必須要素1）- 第1層のみ
        # ★P0最適化★ sign_matrixをキャッシュ（不変なので毎サンプル再計算不要）
        self._sign_matrix_layer0 = np.outer(self.ei_flags_hidden[0], self.ei_flags_input)
        self.w_hidden[0] = np.abs(self.w_hidden[0]) * self._sign_matrix_layer0
        
        # ★新機能★ 受容野（Receptive Field）マスクの生成（第1層、column_neurons>=2のみ）
        # 生物学的背景: V1コラム内ニューロンは異なる受容野を持ち、入力空間を分担
        self.receptive_field_masks = []
        for layer_idx in range(self.n_layers):
            if (layer_idx == 0 and column_neurons is not None and column_neurons >= 2
                    and not use_affinity):
                rf_mask = create_receptive_fields(
                    n_input=n_input * 2,  # E/Iペア後の次元数
                    membership=self.column_membership_all_layers[0],
                    column_neurons=column_neurons,
                    rf_overlap=rf_overlap,
                    rf_mode=rf_mode,
                    seed=seed
                )
                self.receptive_field_masks.append(rf_mask)
                if rf_mask is not None:
                    # 初期重みに受容野マスクを適用（マスク外の重みをゼロに）
                    self.w_hidden[0] *= rf_mask
                    # Dale's Principleの再適用（マスク後）
                    self.w_hidden[0] = np.abs(self.w_hidden[0]) * self._sign_matrix_layer0
            else:
                self.receptive_field_masks.append(None)
        
        # アミン濃度の記憶領域（必須要素3）- 各層ごと
        self.amine_concentrations = []
        for layer_size in self.n_hidden:
            amine = np.zeros((n_output, layer_size, 2))
            self.amine_concentrations.append(amine)
        
        # ★Step 2★ 初期重みを保存（デバッグ: 重み変化量の計算に必要）
        self._initial_weights = [w.copy() for w in self.w_hidden]
        self._initial_w_output = self.w_output.copy()
        
        if self.enable_non_column_learning:
            print(f"\n[★非コラムニューロン学習★] enable_non_column_learning=True")
            print(f"  非コラムニューロンがamine=1.0で学習に参加します")
    
    def forward(self, x):
        """
        順伝播（多クラス分類版）
        
        変更点:
          - 隠れ層: sigmoid → tanh（双方向性、飽和特性）
          - 出力層: sigmoid → SoftMax（確率分布化）
        
        Args:
            x: 入力データ（shape: [n_input]）
        
        Returns:
            z_hiddens: 各隠れ層の出力のリスト
            z_output: 出力層の確率分布（SoftMax、合計=1.0）
            x_paired: 入力ペア
        """
        # 入力ペア構造（必須要素7）
        x_paired = create_ei_pairs(x)
        
        # 各隠れ層の順伝播
        z_hiddens = []
        z_current = x_paired
        
        for layer_idx in range(self.n_layers):
            a_hidden = np.dot(self.w_hidden[layer_idx], z_current)
            
            # 活性化関数の適用
            if self.activation == 'leaky_relu':
                # Leaky ReLU: max(0, u) + alpha * min(0, u)
                z_hidden = np.where(a_hidden > 0, a_hidden, self.leaky_alpha * a_hidden)
            else:  # tanh（デフォルト）
                z_hidden = tanh_activation(a_hidden)
            
            z_hiddens.append(z_hidden)
            z_current = z_hidden
            
            # ★v036新機能★ lateral_cooperation（affinity方式）
            if self.lateral_cooperation > 0.0 and layer_idx == self.n_layers - 1:
                z_current = self._apply_lateral_cooperation_affinity(z_current, layer_idx)
            
            # ★対策5★ 層間正規化（アミン伝播の安定化）
            if self.use_layer_norm:
                mean = np.mean(z_current)
                std = np.std(z_current) + 1e-8
                z_current = (z_current - mean) / std
        
        # ★v047新機能★ コラム間活性化側方抑制（Column Lateral Inhibition）
        # 生物学的根拠: 大脳皮質の抑制性介在ニューロン（GABAergic interneurons）が
        # 活性化されたコラムの隣接コラムの膜電位を一時的に抑制する。
        # 重みには触れず、活性化を一時的に修正するだけ → ED学習への干渉なし。
        if self.column_lateral_inhibition and hasattr(self, 'column_membership_all_layers') and len(self.column_membership_all_layers) > 0:
            last_layer_idx = self.n_layers - 1
            if last_layer_idx < len(self.column_membership_all_layers):
                membership = self.column_membership_all_layers[last_layer_idx]
                n_classes = membership.shape[0]
                
                # 各コラムの平均活性化を計算
                column_mean_acts = np.zeros(n_classes)
                for c in range(n_classes):
                    members = membership[c]
                    if np.any(members):
                        column_mean_acts[c] = np.mean(np.abs(z_hiddens[-1][members]))
                
                # 最大活性化コラムを特定
                max_col = np.argmax(column_mean_acts)
                max_act = column_mean_acts[max_col]
                
                if max_act > 1e-8:
                    # ライバルコラムの活性化を一時的に抑制
                    # 抑制量はmax_actとの比率に基づく（強いコラムほど強く抑制）
                    z_inhibited = z_hiddens[-1].copy()  # コピーして一時修正
                    for c in range(n_classes):
                        if c == max_col:
                            continue
                        members = membership[c]
                        if np.any(members):
                            # 抑制係数: 1.0 - alpha * (max_act / max_act) = 1.0 - alpha
                            # → 最大コラムに近い活性化のライバルほど強く抑制される
                            suppression = self.cli_alpha * (column_mean_acts[c] / max_act)
                            z_inhibited[members] *= (1.0 - suppression)
                    
                    # z_hiddensの最終層を抑制後の値で「出力計算用のみ」上書き
                    # 注意: z_hiddens[-1]自体は重み更新で参照されるためコピーを使う
                    z_hiddens[-1] = z_inhibited
        
        # 出力層の計算（★SoftMax活性化★）
        # ★HTM空間プーリング★ 非コラムニューロンの活性化抑制ゲート
        # 生物学的背景: 「最も活性の高いミニコラムを選択し、近傍を抑制」
        # 非コラムの隠れ層重みは凍結（リザバー）、上位K%の活性値のみ出力層に通過
        if self.nc_sparse_k > 0.0 and hasattr(self, 'column_membership_all_layers') and len(self.column_membership_all_layers) > 0:
            last_layer_idx = self.n_layers - 1
            membership_last = self.column_membership_all_layers[last_layer_idx]
            is_any_column = np.any(membership_last, axis=0)  # [n_neurons]
            nc_indices = np.where(~is_any_column)[0]
            n_nc = len(nc_indices)
            
            if n_nc > 0:
                top_k = max(1, int(n_nc * self.nc_sparse_k))
                nc_abs_acts = np.abs(z_hiddens[-1][nc_indices])
                # 上位K個の閾値を取得（partial sortで高速）
                threshold = np.partition(nc_abs_acts, -top_k)[-top_k]
                # ゲートマスク: コラム=1, NC上位K%=1, それ以外=0
                gate_mask = np.ones(len(z_hiddens[-1]), dtype=np.float64)
                suppress_nc = nc_indices[nc_abs_acts < threshold]
                gate_mask[suppress_nc] = 0.0
                self._nc_gate_mask = gate_mask
                
                # ゲート適用: 抑制されたNCの活性値をゼロに
                z_for_output = z_hiddens[-1] * gate_mask
            else:
                self._nc_gate_mask = None
                z_for_output = z_hiddens[-1]
        else:
            self._nc_gate_mask = None
            z_for_output = z_hiddens[-1]
        
        a_output = np.dot(self.w_output, z_for_output)
        
        # ★Idea D★ 活動依存の側方抑制（非コラムニューロン寄与の差異的増幅）
        # 非コラムニューロンの出力への寄与を計算し、クラス間の差異を増幅。
        # 生物学的背景: 介在ニューロンの活動がコラム間のコントラストを強化。
        if self._li_non_column_indices is not None:
            nc_acts = z_hiddens[-1][self._li_non_column_indices]  # (n_nc,) 符号つき
            # 各クラスへの非コラム寄与 = w_output[:,nc] @ nc_acts
            nc_contrib = self.w_output[:, self._li_non_column_indices] @ nc_acts  # (n_output,)
            # 平均寄与を引いて差分を計算（softmaxに影響する部分のみ）
            nc_differential = nc_contrib - nc_contrib.mean()
            # 差分を増幅してpre-softmaxスコアに加算
            a_output += self.lateral_inhibition_strength * nc_differential
        
        z_output = softmax(a_output)  # ★変更: sigmoid → softmax★
        
        # 注意: 側方抑制は順伝播では適用せず、学習時の確率ベース競合で実現
        
        return z_hiddens, z_output, x_paired
    
    def _apply_lateral_cooperation_affinity(self, z_hidden, layer_idx):
        """
        ★v036新機能★ affinity方式によるコラム内協調学習
        
        v035のmembership方式と異なり、affinity値（連続値）を使用して
        コラム内ニューロン間の情報共有を実現。
        
        メカニズム:
            - affinity値が高いニューロン同士（同一コラム内）で活性値を共有
            - affinity重み付き平均により、自己活性と協調活性をブレンド
            - 生物学的妥当性: コラム内の局所的結合による情報共有
        
        Args:
            z_hidden: 隠れ層の活性化値 [n_hidden]
            layer_idx: 層インデックス
        
        Returns:
            調整後の活性化値
        """
        z_adjusted = z_hidden.copy()
        affinity = self.column_affinity_all_layers[layer_idx]  # [n_classes, n_hidden]
        
        # デバッグモード用統計
        if self.debug_lateral_cooperation:
            sample_affected_neurons = []
            sample_activation_changes = []
            sample_members_count = []
        
        for class_idx in range(self.n_output):
            # このクラスへのaffinity値を取得
            class_affinity = affinity[class_idx]  # [n_hidden]
            
            # affinity閾値以上のニューロンを特定（実質的なコラムメンバー）
            # threshold=0.01: より多くのニューロンを協調対象に含める（緩和）
            threshold = 0.01
            members = np.where(class_affinity >= threshold)[0]
            
            if len(members) > 1:  # 2個以上のニューロンがいる場合のみ協調
                # affinity重み付き平均活性化
                member_activations = z_hidden[members]
                member_affinities = class_affinity[members]
                
                # affinity正規化
                affinity_sum = np.sum(member_affinities)
                if affinity_sum > 1e-8:
                    affinity_weights = member_affinities / affinity_sum
                    weighted_mean_activation = np.sum(member_activations * affinity_weights)
                    
                    # 自己活性化と重み付き平均活性化のブレンド
                    # lateral_cooperation=0.3なら、70%が自己、30%が重み付き平均
                    for member_idx in members:
                        original_activation = z_hidden[member_idx]
                        z_adjusted[member_idx] = (
                            (1.0 - self.lateral_cooperation) * original_activation + 
                            self.lateral_cooperation * weighted_mean_activation
                        )
                        
                        # デバッグ統計収集
                        if self.debug_lateral_cooperation:
                            activation_change = abs(z_adjusted[member_idx] - original_activation)
                            sample_activation_changes.append(activation_change)
                    
                    # デバッグ統計収集（クラス単位）
                    if self.debug_lateral_cooperation:
                        sample_affected_neurons.append(len(members))
                        sample_members_count.append(len(members))
        
        # デバッグ統計を蓄積
        if self.debug_lateral_cooperation:
            self.lc_stats['total_samples'] += 1
            if sample_affected_neurons:
                self.lc_stats['affected_neurons_per_class'].extend(sample_affected_neurons)
            if sample_activation_changes:
                self.lc_stats['activation_changes'].extend(sample_activation_changes)
            if sample_members_count:
                self.lc_stats['members_count_per_class'].extend(sample_members_count)
        
        return z_adjusted
    
    def print_lateral_cooperation_stats(self):
        """
        lateral_cooperation デバッグ統計の表示
        
        デバッグモード有効時に収集された統計情報を表示します。
        - 影響を受けたニューロン数
        - 活性化値の変化量
        - コラムメンバー数
        """
        if not self.debug_lateral_cooperation or self.lc_stats['total_samples'] == 0:
            print("[lateral_cooperation デバッグモード: 無効または未実行]")
            return
        
        stats = self.lc_stats
        print("\n" + "="*70)
        print("lateral_cooperation 影響度分析")
        print("="*70)
        print(f"分析サンプル数: {stats['total_samples']}")
        
        if stats['affected_neurons_per_class']:
            print(f"\nクラスごとの影響ニューロン数:")
            print(f"  平均: {np.mean(stats['affected_neurons_per_class']):.2f}個")
            print(f"  中央値: {np.median(stats['affected_neurons_per_class']):.1f}個")
            print(f"  最小-最大: {np.min(stats['affected_neurons_per_class'])}-{np.max(stats['affected_neurons_per_class'])}個")
        
        if stats['activation_changes']:
            print(f"\n活性化値の変化量:")
            print(f"  平均変化: {np.mean(stats['activation_changes']):.6f}")
            print(f"  最大変化: {np.max(stats['activation_changes']):.6f}")
            print(f"  変化>0.01のニューロン: {np.sum(np.array(stats['activation_changes']) > 0.01)}/{len(stats['activation_changes'])} ({np.sum(np.array(stats['activation_changes']) > 0.01)/len(stats['activation_changes'])*100:.1f}%)")
        
        if stats['members_count_per_class']:
            print(f"\nコラムメンバー数分布:")
            members = np.array(stats['members_count_per_class'])
            unique, counts = np.unique(members, return_counts=True)
            for n, c in zip(unique, counts):
                print(f"  {int(n)}個: {c}回 ({c/len(members)*100:.1f}%)")
        
        print("="*70 + "\n")
    
    def reset_lateral_cooperation_stats(self):
        """lateral_cooperation統計をリセット"""
        self.lc_stats = {
            'total_samples': 0,
            'affected_neurons_per_class': [],
            'activation_changes': [],
            'members_count_per_class': []
        }
    
    def _apply_top_k_winners(self, learning_signals_3d, active_neurons, layer_idx):
        """
        top_k_winners: affinity値が高いニューロンのみ学習参加（ベクトル化版）
        
        Args:
            learning_signals_3d: (n_classes, 2, n_active_neurons) 学習信号
            active_neurons: 活性ニューロンのインデックス
            layer_idx: 層インデックス
            
        Returns:
            learning_signals_3d: 上位K個のみ学習参加に調整された学習信号
        """
        if self.top_k_winners >= len(active_neurons):
            # 全員参加の場合は何もしない
            return learning_signals_3d
        
        affinity = self.column_affinity_all_layers[layer_idx]  # (n_classes, n_hidden)
        active_affinity = affinity[:, active_neurons]  # (n_classes, n_active_neurons)
        
        # 各クラスでtop_k_winnersのニューロンのみ学習参加
        # argsort降順でtop_k個のインデックスを取得
        sorted_indices = np.argsort(-active_affinity, axis=1)  # (n_classes, n_active_neurons)
        top_k_indices = sorted_indices[:, :self.top_k_winners]  # (n_classes, top_k_winners)
        
        # マスク作成（ベクトル化）
        mask = np.zeros_like(active_affinity, dtype=np.float32)  # (n_classes, n_active_neurons)
        for class_idx in range(self.n_output):
            for rank, neuron_idx in enumerate(top_k_indices[class_idx]):
                if rank == 0:
                    mask[class_idx, neuron_idx] = 1.0
                elif rank == 1:
                    mask[class_idx, neuron_idx] = 0.7
                else:
                    mask[class_idx, neuron_idx] = 0.4
        
        # 学習信号に適用
        learning_signals_3d *= mask[:, np.newaxis, :]  # broadcast: (n_classes, 1, n_active_neurons)
        
        return learning_signals_3d
    
    def set_epoch_lr_scale(self, epoch, schedule):
        """
        エポック依存学習率スケーリングを適用
        
        Args:
            epoch: 現在のエポック番号（0始まり）
            schedule: スケーリング係数のリスト（例: [0.7, 0.9, 1.0]）
                     Epoch 0から順に適用、最後の値を維持
        
        Example:
            # Gentle戦略: Epoch 0-1で0.7倍、Epoch 2で0.9倍、Epoch 3+で1.0倍
            network.set_epoch_lr_scale(epoch, [0.7, 0.9, 1.0])
        """
        if schedule is None or len(schedule) == 0:
            # スケジューリングなし: ベース学習率をそのまま使用
            self.layer_lrs = list(self.base_layer_lrs)
            return
        
        # 現在のエポックに対応する係数を取得（最後の値を維持）
        scale_idx = min(epoch, len(schedule) - 1)
        scale_factor = schedule[scale_idx]
        
        # 全層の学習率にスケーリング係数を適用
        self.layer_lrs = [base_lr * scale_factor for base_lr in self.base_layer_lrs]
    
    def _compute_gradients(self, x_paired, z_hiddens, z_output, y_true):
        """
        勾配の計算（ED法準拠、微分の連鎖律不使用）
        
        Args:
            x_paired: 入力ペア
            z_hiddens: 各隠れ層の出力のリスト
            z_output: 出力層の確率分布（SoftMax）
            y_true: 正解クラス
            
        Returns:
            gradients: {
                'w_output': 出力層の勾配,
                'w_hidden': 隠れ層の勾配リスト（★P1最適化★ スパース形式）,
                    各要素は (active_neurons, delta_w_batch) のタプル、
                    または active_neurons が空の場合は None,
                'lateral_weights': 側方抑制の勾配
            }
        """
        gradients = {
            'w_output': None,
            'w_hidden': [None] * self.n_layers,
            'lateral_weights': np.zeros_like(self.lateral_weights)
        }
        
        # ============================================
        # 1. 出力層の勾配計算
        # ============================================
        target_probs = np.zeros(self.n_output)
        target_probs[y_true] = 1.0
        error_output = target_probs - z_output
        
        saturation_output = np.abs(z_output) * (1.0 - np.abs(z_output))
        # ★v039変更★ 出力層にも層別学習率を適用（layer_lrs[-1]）
        # ★HTM空間プーリング★ 出力層勾配もゲート適用
        # 抑制されたNCの活性値をゼロにして出力層重み更新から除外
        z_for_output = z_hiddens[-1]
        if self._nc_gate_mask is not None:
            z_for_output = z_hiddens[-1] * self._nc_gate_mask
        
        output_lr = self.layer_lrs[-1] if len(self.layer_lrs) > self.n_layers else self.learning_rate
        gradients['w_output'] = output_lr * np.outer(
            error_output * saturation_output,
            z_for_output
        )
        
        # ============================================
        # 2. 出力層のアミン濃度計算
        # ============================================
        amine_concentration_output = np.zeros((self.n_output, 2))
        
        # ★純粋ED法★ 正解クラスのみ学習（v039で実証済み）
        # WTA誤答時の負の誤差拡散は学習を阻害するため削除
        error_correct = 1.0 - z_output[y_true]
        if error_correct > 0:
            amine_concentration_output[y_true, 0] = error_correct * self.initial_amine
        
        # ============================================
        # 3. 多層アミン拡散と勾配計算（逆順、微分の連鎖律不使用）
        # ============================================
        for layer_idx in range(self.n_layers - 1, -1, -1):
            # 入力の取得
            if layer_idx == 0:
                z_input = x_paired
            else:
                z_input = z_hiddens[layer_idx - 1]
            
            # 拡散係数の選択
            # ★提案B★ uniform_amine=Trueの場合、全層u1で均一拡散（青斑核モデル）
            if self.uniform_amine:
                diffusion_coef = self.u1  # 全層均一: 青斑核から全脳領域に均一投射
            elif layer_idx == self.n_layers - 1:
                diffusion_coef = self.u1
            else:
                diffusion_coef = self.u2
            
            # ★v032移行★ Membership方式によるアミン拡散
            # ★v038.2実験★ use_affinityフラグでAffinity方式の実験も可能
            amine_mask = amine_concentration_output >= 1e-8
            
            # ステップ1: アミン濃度に拡散係数を適用（全ニューロン一律）
            amine_diffused = amine_concentration_output * diffusion_coef
            
            if self.use_affinity:
                # ★v038.2実験★ Affinity方式（過去の重み爆発問題を再検証）
                amine_hidden_3d = (
                    amine_diffused[:, :, np.newaxis] *  # [n_classes, 2, 1]
                    self.column_affinity_all_layers[layer_idx][:, np.newaxis, :]  # [n_classes, 1, n_neurons]
                )
            else:
                # ★v032標準★ Membership方式 - 活性値ランクベースTop-K学習
                # ★v042最適化★ 完全ベクトル化（ループ削除）
                membership = self.column_membership_all_layers[layer_idx]  # [n_classes, n_neurons] boolean
                z_current = z_hiddens[layer_idx]  # 現在の隠れ層活性
                n_neurons = self.n_hidden[layer_idx]
                
                # アミン濃度が非ゼロのクラスを特定
                active_classes = np.where(np.any(amine_diffused >= 1e-8, axis=1))[0]
                n_active = len(active_classes)
                
                if n_active == 0:
                    # アクティブなクラスがない場合はゼロ行列
                    amine_hidden_3d = np.zeros((self.n_output, 2, n_neurons))
                else:
                    # ベクトル化実装
                    # 1. アクティブクラスのmembershipを取得
                    active_membership = membership[active_classes]  # [n_active, n_neurons]
                    
                    # 2. メンバーニューロンの活性値を取得（非メンバーは-inf）
                    masked_activations = np.where(active_membership, z_current, -np.inf)  # [n_active, n_neurons]
                    
                    # 3. 各クラス内でランク計算（argsort二回でランク取得、降順）
                    sorted_indices = np.argsort(-masked_activations, axis=1)
                    ranks = np.argsort(sorted_indices, axis=1)  # [n_active, n_neurons]
                    
                    # 4. ランクから学習率を取得（ルックアップテーブル参照）
                    clamped_ranks = np.minimum(ranks, len(self._learning_weight_lut) - 1)
                    learning_weights = self._learning_weight_lut[clamped_ranks]  # [n_active, n_neurons]
                    
                    # 5. 非メンバーの学習率を設定
                    # ★NC最近傍クラス帰属学習★
                    # 各NCが最近傍マイクロコラムに帰属し、そのクラスからのみ学習信号を受け取る
                    # 多クラス干渉を原理的に解消するハード割当方式
                    if (self.nc_nearest_learning and 
                        layer_idx < len(self._nc_nearest_membership)):
                        # 拡張membershipを使用: コラム+最近傍NC
                        ext_membership = self._nc_nearest_membership[layer_idx]
                        ext_active_membership = ext_membership[active_classes]  # [n_active, n_neurons]
                        
                        # コラムニューロン: 従来のLUTランク
                        # NC(最近傍クラスのみ): nc_amine_strength
                        # NC(他クラス): 0
                        is_original_column = active_membership  # [n_active, n_neurons]
                        is_nc_nearest = ext_active_membership & ~is_original_column  # [n_active, n_neurons]
                        
                        # NCの学習重みを設定
                        nc_weights = np.where(is_nc_nearest, self.nc_amine_strength, 0.0)
                        learning_weights = np.where(
                            is_original_column,
                            learning_weights,  # コラム: LUTランクベース
                            nc_weights           # NC: 最近傍クラスのみ
                        )
                    elif (self._nc_gate_mask is not None and 
                        self.nc_amine_strength > 0.0 and 
                        layer_idx == self.n_layers - 1):
                        # ゲートマスクからNC学習重みを構成
                        # gate_mask=1 かつ 非コラム → nc_amine_strength
                        nc_in_gate = (self._nc_gate_mask > 0.5) & ~np.any(membership, axis=0)  # [n_neurons] bool
                        nc_learning = np.where(nc_in_gate[np.newaxis, :], self.nc_amine_strength, 0.0)  # [1, n_neurons] → broadcast
                        nc_learning_broadcast = np.broadcast_to(nc_learning, (n_active, n_neurons))
                        learning_weights = np.where(
                            active_membership,
                            learning_weights,  # コラム: LUTランクベース
                            nc_learning_broadcast  # 非コラム: ゲート選択NCのみ
                        )
                    elif self.amine_diffusion_sigma > 0.0:
                        # 空間的距離に基づく重み（事前計算済み）
                        nc_weights = self._spatial_amine_weights[layer_idx][active_classes]  # [n_active, n_neurons]
                        learning_weights = np.where(
                            active_membership, 
                            learning_weights,
                            nc_weights  # 非コラム: 空間距離に応じたamine拡散
                        )
                    elif self.amine_base_level > 0.0:
                        # 旧方式: 全非コラムに均一amine（後方互換）
                        learning_weights = np.where(
                            active_membership, 
                            learning_weights,
                            self.amine_base_level
                        )
                    elif self.enable_non_column_learning:
                        # ★非コラム学習解除★ 非コラムニューロンもamine=1.0で学習参加
                        # オリジナルED法の「全ニューロン参加」原則に回帰
                        learning_weights = np.where(
                            active_membership, 
                            learning_weights,
                            1.0  # コラム最高ランクと同等の学習信号
                        )  # [n_active, n_neurons]
                    else:
                        # デフォルト: 非コラムはamine=0（学習しない）
                        learning_weights = np.where(
                            active_membership, 
                            learning_weights,
                            0.0
                        )  # [n_active, n_neurons]
                    
                    # 6. アミン拡散値に学習率を適用
                    # amine_diffused[active_classes]: [n_active, 2]
                    # learning_weights: [n_active, n_neurons]
                    amine_hidden_3d = np.zeros((self.n_output, 2, n_neurons))
                    amine_hidden_3d[active_classes] = (
                        amine_diffused[active_classes, :, np.newaxis] *  # [n_active, 2, 1]
                        learning_weights[:, np.newaxis, :]                # [n_active, 1, n_neurons]
                    )
            
            amine_hidden_3d = amine_hidden_3d * amine_mask[:, :, np.newaxis]
            
            # 活性ニューロンの特定
            neuron_mask = np.any(amine_hidden_3d >= 1e-8, axis=(0, 1))
            active_neurons = np.where(neuron_mask)[0]
            
            if len(active_neurons) == 0:
                gradients['w_hidden'][layer_idx] = None  # ★P1最適化★ ゼロ配列生成を省略
                continue
            
            # 活性化関数の勾配
            z_active = z_hiddens[layer_idx][active_neurons]
            if self.activation == 'leaky_relu':
                saturation_term_raw = np.where(z_active > 0, 1.0, self.leaky_alpha)
            else:
                saturation_term_raw = np.abs(z_active) * (1.0 - np.abs(z_active))
            saturation_term = np.maximum(saturation_term_raw, 1e-3)
            
            # 学習信号強度の計算
            # ★v039変更★ 層別学習率を使用（layer_lrs[layer_idx]）
            layer_lr = self.layer_lrs[layer_idx]
            learning_signals_3d = (
                layer_lr * 
                amine_hidden_3d[:, :, active_neurons] * 
                saturation_term[np.newaxis, np.newaxis, :]
            )
            
            # top_k_winners適用（最終層のみ）
            if self.top_k_winners is not None and layer_idx == self.n_layers - 1:
                learning_signals_3d = self._apply_top_k_winners(
                    learning_signals_3d, 
                    active_neurons,
                    layer_idx
                )
            
            # 勾配の計算
            # ★最適化★ 3D配列生成を回避し、信号合計×入力のouter productで計算
            # 旧: delta_w_3d[n_active, n_comb, n_input] → 256MB (n_active=2048時)
            # 新: signal_sum[n_active] × z_input[n_input] → 1.6MB (20倍削減)
            n_combinations = self.n_output * 2
            learning_signals_flat = learning_signals_3d.reshape(n_combinations, -1).T  # [n_active, n_comb]
            signal_sum = learning_signals_flat.sum(axis=1)  # [n_active]
            delta_w_batch = signal_sum[:, np.newaxis] * z_input[np.newaxis, :]  # [n_active, n_input]
            
            # 層ごとの符号制約
            if layer_idx > 0:
                w_sign = np.sign(self.w_hidden[layer_idx][active_neurons, :])
                w_sign[w_sign == 0] = 1
                delta_w_batch *= w_sign
            
            # gradient clipping
            if self.gradient_clip > 0:
                delta_w_norms = np.linalg.norm(delta_w_batch, axis=1, keepdims=True)
                clip_mask = delta_w_norms > self.gradient_clip
                delta_w_batch = np.where(
                    clip_mask,
                    delta_w_batch * (self.gradient_clip / delta_w_norms),
                    delta_w_batch
                )
            
            # ★新機能★ 隠れ層のコラムニューロン勾配抑制（層別対応）
            # columnar_ed.prompt.mdの「活性化バランスの問題」対策
            # コラムニューロンの過剰活性化（重み飽和）を防ぐため、学習率を抑制
            layer_lr_factor = self.column_lr_factors[layer_idx]  # 層別の係数を取得
            if layer_lr_factor < 1.0 and layer_idx < len(self.column_membership_all_layers):
                membership = self.column_membership_all_layers[layer_idx]  # [n_classes, n_neurons] boolean
                # 各ニューロンがいずれかのクラスのコラムに属しているか
                is_column_neuron = np.any(membership, axis=0)  # [n_neurons] boolean
                # 活性ニューロンのうちコラムに属するものを特定
                active_is_column = is_column_neuron[active_neurons]  # [n_active] boolean
                # コラムニューロンの勾配を抑制
                if np.any(active_is_column):
                    delta_w_batch[active_is_column, :] *= layer_lr_factor
            
            # ★P1最適化★ スパース形式で保存（full_gradient展開を省略）
            gradients['w_hidden'][layer_idx] = (active_neurons, delta_w_batch)
        
        return gradients
    
    def update_weights(self, x_paired, z_hiddens, z_output, y_true):
        """
        重みの更新（多層多クラス分類版 + ED法、微分の連鎖律不使用）
        
        v026変更点: 多層アミン拡散対応
        - u1: 出力層→最終隠れ層の拡散係数
        - u2: 隠れ層間の拡散係数
        - 各層で独立に重み更新（微分の連鎖律不使用）
        
        Args:
            x_paired: 入力ペア
            z_hiddens: 各隠れ層の出力のリスト
            z_output: 出力層の確率分布（SoftMax）
            y_true: 正解クラス
        """
        # 勾配計算
        gradients = self._compute_gradients(x_paired, z_hiddens, z_output, y_true)
        
        # ★新機能★ コラムニューロンの重み肥大化抑制
        # 出力層のコラムニューロンへの勾配を抑制（最終隠れ層の係数を使用）
        output_lr_factor = self.column_lr_factors[-1]  # 最終隠れ層の係数を出力層に適用
        if output_lr_factor < 1.0:
            # 最終隠れ層のaffinity
            final_layer_idx = self.n_layers - 1
            affinity = self.column_affinity_all_layers[final_layer_idx]  # [n_classes, n_hidden]
            # affinity > 0.1 のニューロンをコラムと判定
            column_mask = (affinity > 0.1)  # [n_classes, n_hidden]
            
            # 出力層の勾配抑制（各クラスのコラムニューロンからの重み）
            for class_idx in range(self.n_output):
                if np.any(column_mask[class_idx]):
                    # クラスclass_idxのコラムニューロンからの重みを抑制
                    gradients['w_output'][class_idx, column_mask[class_idx]] *= output_lr_factor
        
        # ★Idea B★ 出力層の空間的勾配変調
        # 非コラムニューロンの出力重み更新を空間的に変調。
        # 最近傍クラスへの結合をブースト、他クラスへの結合を抑制。
        # 順伝播は変更せず、学習速度のみを空間的に調整。
        if self.output_spatial_mask is not None:
            gradients['w_output'] *= self.output_spatial_mask
        
        # 勾配適用
        self.w_output += gradients['w_output']
        self.lateral_weights += gradients['lateral_weights']
        
        for layer_idx in range(self.n_layers):
            # ★P1最適化★ スパース形式の勾配適用（active_neurons行のみ更新）
            sparse_grad = gradients['w_hidden'][layer_idx]
            if sparse_grad is not None:
                active_neurons, delta_w_batch = sparse_grad
                self.w_hidden[layer_idx][active_neurons] += delta_w_batch
                
                # 第1層のDale's Principle強制（★P0+P1最適化★ active行のみ）
                if layer_idx == 0:
                    self.w_hidden[0][active_neurons] = (
                        np.abs(self.w_hidden[0][active_neurons]) *
                        self._sign_matrix_layer0[active_neurons]
                    )
                    # Receptive Field マスク再適用（active行のみ）
                    if (self.receptive_field_masks
                            and self.receptive_field_masks[0] is not None):
                        self.w_hidden[0][active_neurons] *= \
                            self.receptive_field_masks[0][active_neurons]
            # sparse_gradがNoneの場合: 活性ニューロンなし → 更新不要
        
        # ★v046新機能★ コラム内重みベクトル脱相関
        # 同一コラム内のニューロンが異なる特徴を学習するよう促進
        # 生物学的根拠: コラム内の側抑制による特徴分化
        if self.column_decorrelation > 0.0:
            self._apply_column_decorrelation()
        
        # 出力重みの正則化（★P0最適化★ 1演算に統合）
        self.w_output *= (1.0 - 0.00001)
    
    def _apply_column_decorrelation(self):
        """
        ★v046新機能★ コラム内重みベクトル脱相関
        
        同一コラム内のニューロンの重みベクトルが類似している場合、
        互いに反発する力を加えて異なる特徴への分化を促す。
        
        生物学的根拠: 大脳皮質コラム内の側抑制（lateral inhibition）
        による受容野特性の分化。
        
        数学: 各ペア(i,j)のcosine類似度が正の場合、
        Δw_i -= λ * cos_sim(w_i, w_j) * w_j_hat
        （w_j_hat は w_j の正規化ベクトル）
        
        ED法との互換性: 重み更新後の正則化ステップであり、
        ED法の学習メカニズム自体には手を加えない。
        """
        λ = self.column_decorrelation
        
        for layer_idx in range(self.n_layers):
            if layer_idx >= len(self.column_membership_all_layers):
                continue
            
            membership = self.column_membership_all_layers[layer_idx]  # [n_classes, n_neurons]
            w = self.w_hidden[layer_idx]  # [n_neurons, n_input]
            
            for class_idx in range(self.n_output):
                # このクラスのコラムメンバーを取得
                members = np.where(membership[class_idx])[0]
                
                if len(members) < 2:
                    continue  # 1個以下なら脱相関不要
                
                # メンバーの重みベクトルを取得
                w_members = w[members]  # [cn, n_input]
                
                # 各ベクトルのノルムを計算
                norms = np.linalg.norm(w_members, axis=1, keepdims=True)  # [cn, 1]
                norms = np.maximum(norms, 1e-8)  # ゼロ除算防止
                w_normalized = w_members / norms  # [cn, n_input]
                
                # ペアワイズcosine類似度行列
                cos_sim_matrix = w_normalized @ w_normalized.T  # [cn, cn]
                
                # 対角要素（自分自身）をゼロに
                np.fill_diagonal(cos_sim_matrix, 0.0)
                
                # 正の類似度のみ反発力を適用（負の類似度は既に分化している）
                cos_sim_positive = np.maximum(cos_sim_matrix, 0.0)  # [cn, cn]
                
                # 反発力の計算: Δw_i = -λ * Σ_j(cos_sim(i,j) * w_j_hat)
                repulsion = cos_sim_positive @ w_normalized  # [cn, n_input]
                
                # 重みに反発力を適用
                w[members] -= λ * repulsion
            
            # 第1層のDale's Principle再適用
            if layer_idx == 0 and hasattr(self, '_sign_matrix_layer0'):
                # 脱相関後に符号制約を再適用
                affected_neurons = set()
                for class_idx in range(self.n_output):
                    members = np.where(membership[class_idx])[0]
                    affected_neurons.update(members.tolist())
                if affected_neurons:
                    affected = np.array(list(affected_neurons))
                    self.w_hidden[0][affected] = (
                        np.abs(self.w_hidden[0][affected]) *
                        self._sign_matrix_layer0[affected]
                    )
    
    def update_weights_batch(self, x_paired_batch, z_hiddens_batch, z_output_batch, y_batch):
        """
        ミニバッチ重み更新（ED法準拠、サンプルごとに即座更新）
        
        Args:
            x_paired_batch: 入力ペアのバッチ [batch_size, n_input]
            z_hiddens_batch: 隠れ層出力のリスト（各要素は [batch_size, n_hidden]）
            z_output_batch: 出力層の確率分布 [batch_size, n_output]
            y_batch: 正解クラスのバッチ [batch_size]
        
        Notes:
            - ED法準拠: 各サンプルごとに重みを即座に更新
            - オンライン学習と完全に同じ処理（バッチ内でサンプルを順次処理）
            - 勾配を累積せず、サンプルごとに update_weights を呼び出し
        """
        batch_size = len(y_batch)
        
        # 各サンプルごとに重み更新（オンライン学習と同じ）
        for i in range(batch_size):
            self.update_weights(
                x_paired_batch[i],
                [z[i] for z in z_hiddens_batch],
                z_output_batch[i],
                y_batch[i]
            )
    
    def train_one_sample(self, x, y_true):
        """
        1サンプルの学習（オンライン学習）
        
        Args:
            x: 入力データ
            y_true: 正解クラス
        
        Returns:
            loss: 損失
            correct: 正解か否か
        """
        # 順伝播
        z_hiddens, z_output, x_paired = self.forward(x)
        
        # 予測
        y_pred = np.argmax(z_output)
        correct = (y_pred == y_true)
        
        # ★v039.3新機能★ 勝者選択回数を記録（y_predが勝者）
        self.winner_selection_counts[y_pred] += 1
        self.total_training_samples += 1
        
        # ◆変更◆ Cross-Entropy損失計算
        loss = cross_entropy_loss(z_output, y_true)
        
        # 純粋ED法: 正解クラスのみ学習（不正解クラスは何もしない）
        self.update_weights(x_paired, z_hiddens, z_output, y_true)
        self.class_training_counts[y_true] += 1
        
        # ★v047新機能★ 競合コラム間選択的抑制
        # 正解クラスのED学習後、最大出力の不正解クラスのコラムニューロンに微弱な抑制信号を送信
        # 生物学的対応: 大脳皮質の側方抑制（lateral inhibition）
        if self.competitive_inhibition and self.inhibition_strength > 0:
            self._apply_competitive_inhibition(
                x_paired, z_hiddens, z_output, y_true
            )
        
        return loss, correct
    
    def _apply_competitive_inhibition(self, x_paired, z_hiddens, z_output, y_true):
        """
        ★v047新機能★ 競合コラム間選択的抑制
        
        正解クラスのED学習後に呼ばれ、最大出力の不正解クラス（ライバル）の
        コラムニューロンに対して微弱な抑制信号を適用する。
        
        生物学的根拠:
            大脳皮質の側方抑制（lateral inhibition）。活性化したコラムが
            隣接する競合コラムを抑制するのは確立された神経科学的事実。
        
        ED法との整合性:
            - 微分の連鎖律は使用しない
            - アミン拡散機構を通じて抑制信号を伝達
            - 抑制対象はライバルクラスのコラムニューロンのみ（限定的）
            - 抑制強度は正の学習の1/100程度（微弱）
        
        Args:
            x_paired: 入力ペア
            z_hiddens: 各隠れ層の出力のリスト
            z_output: 出力層の確率分布（SoftMax後）
            y_true: 正解クラス
        """
        # ライバルクラスの特定（正解クラスを除いた最大出力クラス）
        outputs_for_rival = z_output.copy()
        outputs_for_rival[y_true] = -np.inf
        
        if self.inhibition_topk == 1:
            rival_classes = [np.argmax(outputs_for_rival)]
        else:
            # 上位k個の不正解クラスを抑制対象とする
            k = min(self.inhibition_topk, self.n_output - 1)
            rival_classes = np.argsort(outputs_for_rival)[-k:][::-1].tolist()
        
        # 各ライバルクラスに対して抑制を適用
        for rival_class in rival_classes:
            # ライバルの出力が正解より弱ければ抑制不要（マージンベース）
            # → 正解が既に勝っている場合でも微弱な抑制は有用なので、常に適用
            # ただし、ライバルの出力が極めて小さい場合はスキップ
            if z_output[rival_class] < 1e-6:
                continue
            
            # ============================================
            # 出力層の抑制勾配
            # ============================================
            # ライバルクラスの誤差: ライバルは0であるべき → 目標0 - 実際値 = -z_output[rival]
            # 抑制方向: ライバルの出力を下げる
            rival_error = -z_output[rival_class]  # 負の値（出力を減少させる方向）
            
            saturation_output = np.abs(z_output) * (1.0 - np.abs(z_output))
            
            # 出力層の抑制勾配（ライバルクラスの行のみ）
            z_for_output = z_hiddens[-1]
            if self._nc_gate_mask is not None:
                z_for_output = z_hiddens[-1] * self._nc_gate_mask
            
            # ライバルクラスのみの出力層勾配
            output_lr = self.layer_lrs[-1] if len(self.layer_lrs) > self.n_layers else self.learning_rate
            inhibition_grad_output = np.zeros_like(self.w_output)
            inhibition_grad_output[rival_class] = (
                output_lr * self.inhibition_strength *
                rival_error * saturation_output[rival_class] *
                z_for_output
            )
            
            # 出力層の抑制勾配を適用
            self.w_output += inhibition_grad_output
            
            # ============================================
            # 隠れ層の抑制（アミン拡散機構経由）
            # ============================================
            # ライバルクラスのコラムニューロンに対して抑制信号をアミン拡散
            for layer_idx in range(self.n_layers - 1, -1, -1):
                if layer_idx == 0:
                    z_input = x_paired
                else:
                    z_input = z_hiddens[layer_idx - 1]
                
                # 拡散係数
                if self.uniform_amine:
                    diffusion_coef = self.u1
                elif layer_idx == self.n_layers - 1:
                    diffusion_coef = self.u1
                else:
                    diffusion_coef = self.u2
                
                # ライバルクラスのアミン濃度（抑制方向）
                # rival_error is negative → 抑制信号
                inhibition_amine = abs(rival_error) * self.initial_amine * self.inhibition_strength
                
                if inhibition_amine < 1e-8:
                    continue
                
                # Membership方式: ライバルクラスのコラムニューロンのみ
                if layer_idx < len(self.column_membership_all_layers):
                    membership = self.column_membership_all_layers[layer_idx]
                    rival_members = membership[rival_class]  # [n_neurons] boolean
                    
                    if not np.any(rival_members):
                        continue
                    
                    active_neurons = np.where(rival_members)[0]
                    
                    # 活性化関数の飽和抑制項
                    z_active = z_hiddens[layer_idx][active_neurons]
                    if self.activation == 'leaky_relu':
                        saturation_term_raw = np.where(z_active > 0, 1.0, self.leaky_alpha)
                    else:
                        saturation_term_raw = np.abs(z_active) * (1.0 - np.abs(z_active))
                    saturation_term = np.maximum(saturation_term_raw, 1e-3)
                    
                    # 抑制勾配の計算（負方向 = 出力を弱める方向）
                    layer_lr = self.layer_lrs[layer_idx]
                    inhibition_signal = (
                        -layer_lr * diffusion_coef * inhibition_amine * saturation_term
                    )  # [n_active] 負の値
                    
                    # 重み更新量: signal * input
                    delta_w = inhibition_signal[:, np.newaxis] * z_input[np.newaxis, :]  # [n_active, n_input]
                    
                    # gradient clipping
                    if self.gradient_clip > 0:
                        delta_w_norms = np.linalg.norm(delta_w, axis=1, keepdims=True)
                        clip_mask = delta_w_norms > self.gradient_clip
                        delta_w = np.where(
                            clip_mask,
                            delta_w * (self.gradient_clip / (delta_w_norms + 1e-10)),
                            delta_w
                        )
                    
                    # 隠れ層重み更新
                    self.w_hidden[layer_idx][active_neurons] += delta_w
                    
                    # 第1層のDale's Principle強制
                    if layer_idx == 0:
                        self.w_hidden[0][active_neurons] = (
                            np.abs(self.w_hidden[0][active_neurons]) *
                            self._sign_matrix_layer0[active_neurons]
                        )
                        # Receptive Field マスク再適用
                        if (self.receptive_field_masks
                                and self.receptive_field_masks[0] is not None):
                            self.w_hidden[0][active_neurons] *= \
                                self.receptive_field_masks[0][active_neurons]
    
    def train_epoch(self, x_train, y_train, return_true_accuracy=True, progress_callback=None, collect_errors=False):
        """
        1エポックの学習
        
        Args:
            x_train: 訓練データ
            y_train: 訓練ラベル
            return_true_accuracy: Trueの場合、学習完了後に全データを再評価して真の訓練精度を返す
                                 Falseの場合、学習中の平均精度を返す（従来の動作、非推奨）
            progress_callback: 学習中に定期的に呼ばれるコールバック関数（ヒートマップ更新等）
                              callback(network, i, n_samples) の形式
        
        Returns:
            accuracy: 訓練精度
            loss: 平均損失
        
        注意:
            return_true_accuracy=True（デフォルト、推奨）:
                学習完了後の最終重みで全訓練データを再評価し、真の訓練精度を返します。
                これにより、テスト精度と公平な比較が可能になり、正しい学習曲線が得られます。
            
            return_true_accuracy=False（非推奨）:
                学習中の各サンプルの予測精度の平均を返します。
                初期サンプルは未学習の重み、後期サンプルは学習済みの重みで評価されるため、
                真の訓練精度より低く算出されます（特にEpoch 1で顕著）。
        """
        import time as _time
        n_samples = len(x_train)
        total_loss = 0.0
        n_correct = 0
        _last_callback_time = _time.time()
        
        # 学習ループ
        for i in range(n_samples):
            loss, correct = self.train_one_sample(x_train[i], y_train[i])
            total_loss += loss
            if correct:
                n_correct += 1
            
            # 約5秒ごとにコールバックを呼び出し（ヒートマップ更新等）
            if progress_callback is not None:
                now = _time.time()
                if now - _last_callback_time >= 5.0:
                    _last_callback_time = now
                    progress_callback(self, i, n_samples)
        
        # 学習中の平均精度と損失
        training_accuracy = n_correct / n_samples
        avg_loss = total_loss / n_samples
        
        if return_true_accuracy:
            # 学習完了後、全訓練データを最終重みで再評価（推奨）
            # これにより、Test精度と同じ条件で評価される
            # ★P2最適化★ 並列評価で高速化
            if collect_errors:
                # 最終エポック用: 評価と同時に不正解サンプルを収集（追加コストなし）
                true_accuracy, true_loss, error_list = self.evaluate_with_errors(x_train, y_train)
                return true_accuracy, true_loss, error_list
            else:
                true_accuracy, true_loss = self.evaluate_parallel(x_train, y_train)
                return true_accuracy, true_loss
        else:
            # 学習中の平均精度を返す（従来動作、非推奨）
            return training_accuracy, avg_loss
    
    def train_epoch_parallel(self, x_train, y_train, prefetch_size=64, use_cupy=False, 
                             return_true_accuracy=True, mode='prefetch'):
        """
        並列順伝播による高速化エポック学習（Phase 2: サンプル並列順伝播）
        
        戦略:
        1. 複数サンプルの順伝播をバッチで先に実行（並列化）
        2. 重み更新はサンプルごとに逐次実行（ED法原理に従う）
        3. 先読み結果は「古い重み」での計算だが、影響は限定的
        
        ★重要★ ED法の絶対原則（微分の連鎖律を用いた誤差逆伝播法を使用しない）は厳守
        
        Args:
            x_train: 訓練データ shape: (n_samples, n_input)
            y_train: 訓練ラベル shape: (n_samples,)
            prefetch_size: 先読みサンプル数（デフォルト64）
                          小さいほど精度への影響が少ない、大きいほど高速
            use_cupy: CuPy GPU高速化を使用（デフォルトFalse）
            return_true_accuracy: Trueなら学習後に再評価して真の精度を返す
            mode: 'prefetch' - 先読みパイプライン（デフォルト）
                  'batch_forward' - バッチ順伝播のみ（重み更新は逐次）
        
        Returns:
            accuracy: 訓練精度
            avg_loss: 平均損失
        
        使用例:
            # 基本使用
            acc, loss = network.train_epoch_parallel(x_train, y_train)
            
            # CuPy高速化
            acc, loss = network.train_epoch_parallel(x_train, y_train, use_cupy=True)
            
            # 精度重視（小さいprefetch_size）
            acc, loss = network.train_epoch_parallel(x_train, y_train, prefetch_size=16)
            
            # 速度重視（大きいprefetch_size）
            acc, loss = network.train_epoch_parallel(x_train, y_train, prefetch_size=128)
        
        注意:
            - prefetch_sizeが大きいほど、「古い重み」での順伝播結果を使用する頻度が増える
            - オリジナル実装と厳密に同じ結果は保証されない
            - 精度への影響は実験で検証する必要がある
        """
        n_samples = len(x_train)
        total_loss = 0.0
        n_correct = 0
        
        if use_cupy:
            if not CUPY_AVAILABLE:
                raise ImportError(
                    "CuPyがインストールされていません。\n"
                    "インストール: pip install cupy-cuda12x\n"
                    "または use_cupy=False で実行してください。"
                )
            return self._train_epoch_parallel_cupy(
                x_train, y_train, prefetch_size, return_true_accuracy
            )
        
        # NumPy版並列学習
        if mode == 'batch_forward':
            # モード1: バッチ順伝播 + 逐次重み更新
            return self._train_epoch_batch_forward(
                x_train, y_train, prefetch_size, return_true_accuracy
            )
        else:
            # モード2: 先読みパイプライン
            return self._train_epoch_prefetch_pipeline(
                x_train, y_train, prefetch_size, return_true_accuracy
            )
    
    def _train_epoch_batch_forward(self, x_train, y_train, batch_size, return_true_accuracy):
        """
        バッチ順伝播 + 逐次重み更新モード（修正版）
        
        重要な修正点:
        - バッチ順伝播は「予測」のみに使用（勝者決定・精度計算用）
        - 重み更新時は各サンプルで「現在の重みで再順伝播」して正しい活性値を得る
        
        これにより:
        - 予測計算はバッチ処理で高速化
        - 重み更新は逐次処理で正確な学習を維持
        """
        n_samples = len(x_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        total_loss = 0.0
        n_correct = 0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            current_batch_size = end_idx - start_idx
            
            # バッチ順伝播（予測計算用：バッチ開始時の重みで）
            _, z_output_batch, _ = self.forward_batch(x_batch)
            
            # サンプルごとに重み更新
            for i in range(current_batch_size):
                y_true = y_batch[i]
                x_sample = x_batch[i]
                
                # バッチ順伝播の予測を使用（精度計算用）
                z_output_batch_pred = z_output_batch[i]
                y_pred = np.argmax(z_output_batch_pred)
                if y_pred == y_true:
                    n_correct += 1
                
                # 統計情報の収集（バッチ予測ベース）
                self.winner_selection_counts[y_pred] += 1
                self.total_training_samples += 1
                self.class_training_counts[y_true] += 1
                
                # 重み更新用に現在の重みで再順伝播
                # これがED法の正確な学習に必要
                z_hiddens, z_output, x_paired = self.forward(x_sample)
                
                # 現在の重みでの損失を計算
                loss = cross_entropy_loss(z_output, y_true)
                total_loss += loss
                
                # 重み更新（現在の重みでの順伝播結果を使用）
                self.update_weights(x_paired, z_hiddens, z_output, y_true)
                
                # ★v047新機能★ 競合コラム間選択的抑制（並列学習パスでも適用）
                if self.competitive_inhibition and self.inhibition_strength > 0:
                    self._apply_competitive_inhibition(
                        x_paired, z_hiddens, z_output, y_true
                    )
        
        # 精度計算
        training_accuracy = n_correct / n_samples
        avg_loss = total_loss / n_samples
        
        if return_true_accuracy:
            true_accuracy, true_loss = self.evaluate_parallel(x_train, y_train)
            return true_accuracy, true_loss
        else:
            return training_accuracy, avg_loss
    
    def _train_epoch_prefetch_pipeline(self, x_train, y_train, prefetch_size, return_true_accuracy):
        """
        先読みパイプラインモード
        
        - 次のバッチの順伝播を先行実行
        - 現在のバッチの重み更新と並行して次のバッチを準備
        - より高度なパイプライン処理
        
        実装: 現時点では_train_epoch_batch_forwardと同じロジック
              将来的に真の非同期パイプラインに拡張可能
        """
        # 現時点ではbatch_forwardと同じ実装
        # 将来的にasyncio等で真のパイプライン処理を実装可能
        return self._train_epoch_batch_forward(
            x_train, y_train, prefetch_size, return_true_accuracy
        )
    
    def _train_epoch_parallel_cupy(self, x_train, y_train, prefetch_size, return_true_accuracy):
        """
        CuPy版並列学習（修正版）
        
        重要な修正点:
        - バッチ順伝播は「予測」のみに使用（勝者決定・精度計算用）
        - 重み更新時は各サンプルで「現在の重みで再順伝播」して正しい活性値を得る
        """
        n_samples = len(x_train)
        n_batches = (n_samples + prefetch_size - 1) // prefetch_size
        
        # データと重みをGPUに転送
        x_train_gpu = cp.array(x_train, dtype=cp.float32)
        
        w_hidden_gpu = [cp.array(w, dtype=cp.float32) for w in self.w_hidden]
        w_output_gpu = cp.array(self.w_output, dtype=cp.float32)
        
        total_loss = 0.0
        n_correct = 0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * prefetch_size
            end_idx = min((batch_idx + 1) * prefetch_size, n_samples)
            
            x_batch_gpu = x_train_gpu[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]  # CPU上で使用
            current_batch_size = end_idx - start_idx
            
            # バッチ順伝播（GPU）- 予測計算用のみ
            z_output_batch_gpu = self._forward_batch_cupy(
                x_batch_gpu, w_hidden_gpu, w_output_gpu
            )
            
            # CPU上で処理
            z_output_batch = cp.asnumpy(z_output_batch_gpu)
            x_batch = cp.asnumpy(x_batch_gpu)
            
            # サンプルごとに重み更新
            for i in range(current_batch_size):
                y_true = y_batch[i]
                x_sample = x_batch[i]
                
                # バッチ順伝播の予測を使用（精度計算用）
                z_output_batch_pred = z_output_batch[i]
                y_pred = np.argmax(z_output_batch_pred)
                if y_pred == y_true:
                    n_correct += 1
                
                # 統計情報（バッチ予測ベース）
                self.winner_selection_counts[y_pred] += 1
                self.total_training_samples += 1
                self.class_training_counts[y_true] += 1
                
                # 重み更新用に現在の重みで再順伝播（CPU側）
                z_hiddens, z_output, x_paired = self.forward(x_sample)
                
                # 現在の重みでの損失を計算
                loss = cross_entropy_loss(z_output, y_true)
                total_loss += loss
                
                # 重み更新
                self.update_weights(x_paired, z_hiddens, z_output, y_true)
            
            # GPU側の重みを更新（次バッチの予測用）
            for layer_idx in range(self.n_layers):
                w_hidden_gpu[layer_idx] = cp.array(self.w_hidden[layer_idx], dtype=cp.float32)
            w_output_gpu = cp.array(self.w_output, dtype=cp.float32)
        
        # 精度計算
        training_accuracy = n_correct / n_samples
        avg_loss = total_loss / n_samples
        
        if return_true_accuracy:
            true_accuracy, true_loss = self.evaluate_parallel(x_train, y_train, use_cupy=True)
            return true_accuracy, true_loss
        else:
            return training_accuracy, avg_loss

    def evaluate(self, x_test, y_test, return_per_class=False):
        """
        テストデータでの評価
        
        Args:
            x_test: テストデータ
            y_test: テストラベル
            return_per_class: クラス別精度を返すかどうか
        
        Returns:
            accuracy: テスト精度
            loss: 平均損失
            class_accuracies: (return_per_class=Trueの場合) クラス別精度のリスト
        """
        n_samples = len(x_test)
        total_loss = 0.0
        n_correct = 0
        
        # クラス別の統計
        if return_per_class:
            class_correct = np.zeros(self.n_output, dtype=int)
            class_total = np.zeros(self.n_output, dtype=int)
        
        for i in range(n_samples):
            # 順伝播のみ
            z_hiddens, z_output, _ = self.forward(x_test[i])
            
            # 予測
            y_pred = np.argmax(z_output)
            true_label = y_test[i]
            
            if y_pred == true_label:
                n_correct += 1
                if return_per_class:
                    class_correct[true_label] += 1
            
            if return_per_class:
                class_total[true_label] += 1
            
            # ◆変更◆ Cross-Entropy損失計算
            loss = cross_entropy_loss(z_output, true_label)
            total_loss += loss
        
        accuracy = n_correct / n_samples
        avg_loss = total_loss / n_samples
        
        if return_per_class:
            class_accuracies = []
            for c in range(self.n_output):
                if class_total[c] > 0:
                    class_acc = class_correct[c] / class_total[c]
                else:
                    class_acc = 0.0
                class_accuracies.append(class_acc)
            return accuracy, avg_loss, class_accuracies
        
        return accuracy, avg_loss

    def evaluate_with_errors(self, x_data, y_data, batch_size=256):
        """
        並列バッチ評価（誤り収集付き）

        evaluate_parallel と同じ効率で動作し、追加パスなしで不正解サンプルを収集する。
        最終エポックの train_epoch(collect_errors=True) から呼び出される。

        Args:
            x_data: 評価データ shape: (n_samples, n_input)
            y_data: ラベル shape: (n_samples,)
            batch_size: バッチサイズ（デフォルト256）

        Returns:
            accuracy: 精度
            avg_loss: 平均損失
            error_list: 不正解サンプルのリスト [(sample_idx, true_label, pred_label), ...]
        """
        n_samples = len(x_data)
        n_batches = (n_samples + batch_size - 1) // batch_size

        all_predictions = []
        all_losses = []

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            x_batch = x_data[start_idx:end_idx]
            y_batch = y_data[start_idx:end_idx]

            _, z_output_batch, _ = self.forward_batch(x_batch)
            y_pred_batch = np.argmax(z_output_batch, axis=1)
            all_predictions.extend(y_pred_batch)

            batch_losses = -np.log(
                z_output_batch[np.arange(len(y_batch)), y_batch] + 1e-10
            )
            all_losses.extend(batch_losses)

        all_predictions = np.array(all_predictions)
        all_losses = np.array(all_losses)

        n_correct = np.sum(all_predictions == y_data)
        accuracy = n_correct / n_samples
        avg_loss = np.mean(all_losses)

        error_list = [
            (int(i), int(y_data[i]), int(all_predictions[i]))
            for i in range(n_samples)
            if all_predictions[i] != y_data[i]
        ]

        return accuracy, avg_loss, error_list
    
    @overload
    def evaluate_parallel(self, x_test, y_test, batch_size: int = 256, return_per_class: Literal[True] = ..., use_cupy: bool = False) -> tuple[float, float, list]: ...
    @overload
    def evaluate_parallel(self, x_test, y_test, batch_size: int = 256, return_per_class: Literal[False] = ..., use_cupy: bool = False) -> tuple[float, float]: ...

    def evaluate_parallel(self, x_test, y_test, batch_size=256, return_per_class=False, use_cupy=False):
        """
        並列バッチ評価（Phase 1: サンプル並列順伝播）
        
        順伝播をバッチ処理で並列化し、評価時間を大幅に短縮。
        学習には影響しないため、精度は従来版と完全一致。
        
        Args:
            x_test: テストデータ shape: (n_samples, n_input)
            y_test: テストラベル shape: (n_samples,)
            batch_size: バッチサイズ（デフォルト256、大きいほど高速だがメモリ消費増）
            return_per_class: クラス別精度を返すかどうか
            use_cupy: CuPy GPU高速化を使用（デフォルトFalse）
        
        Returns:
            accuracy: テスト精度
            avg_loss: 平均損失
            class_accuracies: (return_per_class=Trueの場合) クラス別精度のリスト
        
        性能:
            - NumPy版: 従来版の5-10倍高速
            - CuPy版: 従来版の10-20倍高速（GPU使用時）
        
        使用例:
            # 基本使用
            acc, loss = network.evaluate_parallel(x_test, y_test)
            
            # CuPy高速化
            acc, loss = network.evaluate_parallel(x_test, y_test, use_cupy=True)
            
            # クラス別精度
            acc, loss, class_accs = network.evaluate_parallel(
                x_test, y_test, return_per_class=True
            )
        """
        n_samples = len(x_test)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # CuPy使用時の処理
        if use_cupy:
            if not CUPY_AVAILABLE:
                raise ImportError(
                    "CuPyがインストールされていません。\n"
                    "インストール: pip install cupy-cuda12x\n"
                    "または use_cupy=False で実行してください。"
                )
            return self._evaluate_parallel_cupy(x_test, y_test, batch_size, return_per_class)
        
        # NumPy版バッチ評価
        all_predictions = []
        all_losses = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            x_batch = x_test[start_idx:end_idx]
            y_batch = y_test[start_idx:end_idx]
            
            # バッチ順伝播
            _, z_output_batch, _ = self.forward_batch(x_batch)
            
            # バッチ予測
            y_pred_batch = np.argmax(z_output_batch, axis=1)
            all_predictions.extend(y_pred_batch)
            
            # バッチ損失計算（Cross-Entropy）
            # z_output_batch: [batch_size, n_output]
            # y_batch: [batch_size]
            batch_losses = -np.log(
                z_output_batch[np.arange(len(y_batch)), y_batch] + 1e-10
            )
            all_losses.extend(batch_losses)
        
        # 統計計算
        all_predictions = np.array(all_predictions)
        all_losses = np.array(all_losses)
        
        n_correct = np.sum(all_predictions == y_test)
        accuracy = n_correct / n_samples
        avg_loss = np.mean(all_losses)
        
        if return_per_class:
            class_correct = np.zeros(self.n_output, dtype=int)
            class_total = np.zeros(self.n_output, dtype=int)
            
            for i in range(n_samples):
                true_label = y_test[i]
                class_total[true_label] += 1
                if all_predictions[i] == true_label:
                    class_correct[true_label] += 1
            
            class_accuracies = []
            for c in range(self.n_output):
                if class_total[c] > 0:
                    class_acc = class_correct[c] / class_total[c]
                else:
                    class_acc = 0.0
                class_accuracies.append(class_acc)
            
            return accuracy, avg_loss, class_accuracies
        
        return accuracy, avg_loss
    
    def _evaluate_parallel_cupy(self, x_test, y_test, batch_size=256, return_per_class=False):
        """
        CuPy版並列バッチ評価（GPU高速化）
        
        Args:
            x_test: テストデータ
            y_test: テストラベル
            batch_size: バッチサイズ
            return_per_class: クラス別精度を返すかどうか
        
        Returns:
            accuracy, avg_loss, [class_accuracies]
        """
        n_samples = len(x_test)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # データと重みをGPUに転送
        x_test_gpu = cp.array(x_test, dtype=cp.float32)
        y_test_gpu = cp.array(y_test, dtype=cp.int32)
        
        w_hidden_gpu = [cp.array(w, dtype=cp.float32) for w in self.w_hidden]
        w_output_gpu = cp.array(self.w_output, dtype=cp.float32)
        
        all_predictions = []
        all_losses = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            x_batch_gpu = x_test_gpu[start_idx:end_idx]
            y_batch_gpu = y_test_gpu[start_idx:end_idx]
            
            # バッチ順伝播（GPU）
            z_output_batch_gpu = self._forward_batch_cupy(
                x_batch_gpu, w_hidden_gpu, w_output_gpu
            )
            
            # バッチ予測
            y_pred_batch_gpu = cp.argmax(z_output_batch_gpu, axis=1)
            all_predictions.append(cp.asnumpy(y_pred_batch_gpu))
            
            # バッチ損失計算
            batch_losses_gpu = -cp.log(
                z_output_batch_gpu[cp.arange(len(y_batch_gpu)), y_batch_gpu] + 1e-10
            )
            all_losses.append(cp.asnumpy(batch_losses_gpu))
        
        # CPU上で統計計算
        all_predictions = np.concatenate(all_predictions)
        all_losses = np.concatenate(all_losses)
        
        n_correct = np.sum(all_predictions == y_test)
        accuracy = n_correct / n_samples
        avg_loss = np.mean(all_losses)
        
        if return_per_class:
            class_correct = np.zeros(self.n_output, dtype=int)
            class_total = np.zeros(self.n_output, dtype=int)
            
            for i in range(n_samples):
                true_label = y_test[i]
                class_total[true_label] += 1
                if all_predictions[i] == true_label:
                    class_correct[true_label] += 1
            
            class_accuracies = []
            for c in range(self.n_output):
                if class_total[c] > 0:
                    class_acc = class_correct[c] / class_total[c]
                else:
                    class_acc = 0.0
                class_accuracies.append(class_acc)
            
            return accuracy, avg_loss, class_accuracies
        
        return accuracy, avg_loss
    
    def _forward_batch_cupy(self, x_batch_gpu, w_hidden_gpu, w_output_gpu):
        """
        CuPy版バッチ順伝播
        
        Args:
            x_batch_gpu: 入力バッチ（CuPy配列） [batch_size, n_input]
            w_hidden_gpu: 隠れ層重みリスト（CuPy配列）
            w_output_gpu: 出力層重み（CuPy配列）
        
        Returns:
            z_output_batch_gpu: 出力確率 [batch_size, n_output]
        """
        batch_size = len(x_batch_gpu)
        
        # 入力ペア構造（バッチ対応）
        x_paired_gpu = cp.concatenate([x_batch_gpu, x_batch_gpu], axis=1)
        
        # 各隠れ層の順伝播
        z_current = x_paired_gpu
        
        for layer_idx in range(self.n_layers):
            # 重み行列との積
            a_hidden = cp.dot(z_current, w_hidden_gpu[layer_idx].T)
            
            # 活性化関数
            if self.activation == 'leaky_relu':
                z_hidden = cp.where(a_hidden > 0, a_hidden, self.leaky_alpha * a_hidden)
            else:  # tanh
                z_hidden = cp.tanh(a_hidden)
            
            z_current = z_hidden
            
            # 層間正規化（必要な場合）
            if self.use_layer_norm:
                mean = cp.mean(z_current, axis=1, keepdims=True)
                std = cp.std(z_current, axis=1, keepdims=True) + 1e-8
                z_current = (z_current - mean) / std
        
        # 出力層（バッチ対応SoftMax）
        a_output = cp.dot(z_current, w_output_gpu.T)
        
        # SoftMax（数値安定性のためmax減算）
        a_output_shifted = a_output - cp.max(a_output, axis=1, keepdims=True)
        exp_a = cp.exp(a_output_shifted)
        z_output_batch_gpu = exp_a / cp.sum(exp_a, axis=1, keepdims=True)
        
        return z_output_batch_gpu

    def forward_batch(self, x_batch):
        """
        バッチ順伝播（ミニバッチ学習用）
        
        Args:
            x_batch: 入力データバッチ shape: [batch_size, n_input]
        
        Returns:
            z_hiddens_batch: 各隠れ層の出力リスト、各要素 shape: [batch_size, n_hidden]
            z_output_batch: 出力層の確率分布 shape: [batch_size, n_output]
            x_paired_batch: 入力ペアバッチ shape: [batch_size, n_input_paired]
        """
        batch_size = len(x_batch)
        
        # 入力ペア構造（バッチ対応）
        x_paired_batch = create_ei_pairs_batch(x_batch)  # [batch_size, 1568]
        
        # 各隠れ層の順伝播
        z_hiddens_batch = []
        z_current = x_paired_batch
        
        for layer_idx in range(self.n_layers):
            # 重み行列との積（バッチ対応）
            # W: [n_hidden, n_input], z_current: [batch_size, n_input]
            # 結果: [batch_size, n_hidden]
            a_hidden_batch = np.dot(z_current, self.w_hidden[layer_idx].T)
            
            # 活性化関数（要素ごとの演算なのでそのまま適用可能）
            if self.activation == 'leaky_relu':
                z_hidden_batch = np.where(a_hidden_batch > 0, 
                                          a_hidden_batch, 
                                          self.leaky_alpha * a_hidden_batch)
            else:  # tanh
                z_hidden_batch = tanh_activation(a_hidden_batch)
            
            z_hiddens_batch.append(z_hidden_batch)
            z_current = z_hidden_batch
            
            # 層間正規化（必要な場合）
            if self.use_layer_norm:
                mean = np.mean(z_current, axis=1, keepdims=True)
                std = np.std(z_current, axis=1, keepdims=True) + 1e-8
                z_current = (z_current - mean) / std
        
        # 出力層（バッチ対応SoftMax）
        a_output_batch = np.dot(z_current, self.w_output.T)  # [batch_size, n_output]
        z_output_batch = softmax_batch(a_output_batch)
        
        return z_hiddens_batch, z_output_batch, x_paired_batch
    
    def train_epoch_minibatch_tf(self, train_dataset, progress_callback=None):
        """
        ミニバッチ学習版エポック（TensorFlow Dataset API使用、勾配平均化方式）
        
        この関数はTensorFlow Data API (tf.data.Dataset) を使用します。
        これは業界標準の手法であり、以下の利点があります：
          1. データ処理パイプラインの信頼性が国際的に認知されている
          2. シャッフル機能が最適化され、再現性が保証される
          3. バッチ処理が効率的に実行される
        
        Args:
            train_dataset: tf.data.Dataset（バッチ化・シャッフル済み）
            progress_callback: 学習中に定期的に呼ばれるコールバック関数（ヒートマップ更新等）
                              callback(network, i, n_samples) の形式
        
        Returns:
            accuracy: 訓練精度
            avg_loss: 平均損失
        
        Notes:
            - TensorFlow Dataset APIで前処理（シャッフル・バッチ化）済み
            - **真のミニバッチ学習**: バッチ内全サンプルの勾配を平均化してから重み更新
            - v030で検証済みの正しい実装（v031の勾配合計方式は失敗、Test=9.03%）
            - Tensorをループ処理でNumPyに変換
        
        使用例:
            >>> from modules.data_loader import create_tf_dataset
            >>> train_dataset = create_tf_dataset(
            ...     x_train, y_train, batch_size=128, shuffle=True, seed=42
            ... )
            >>> train_acc, train_loss = network.train_epoch_minibatch_tf(train_dataset)
        """
        import time as _time
        total_loss = 0.0
        n_correct = 0
        n_samples = 0
        _last_callback_time = _time.time()
        
        # TensorFlow Datasetからバッチを取得
        for x_batch_tf, y_batch_tf in train_dataset:
            # TensorをNumPyに変換（既存コードとの互換性）
            x_batch = x_batch_tf.numpy()
            y_batch = y_batch_tf.numpy()
            batch_size = len(x_batch)
            
            # ============================================
            # 勾配蓄積（バッチ内全サンプルの勾配を計算）
            # ============================================
            accumulated_grads = None
            batch_predictions = []
            
            for i in range(batch_size):
                x_sample = x_batch[i]
                y_sample = y_batch[i]
                
                # 順伝播
                z_hiddens, z_output, x_paired = self.forward(x_sample)
                
                # 予測と損失（統計用）
                y_pred = np.argmax(z_output)
                batch_predictions.append((y_pred, y_sample))
                total_loss += cross_entropy_loss(z_output, y_sample)
                
                # ★v039.6新機能★ 統計情報の収集（オンライン学習と同じ）
                self.winner_selection_counts[y_pred] += 1
                self.total_training_samples += 1
                
                # クラス別学習回数の収集（純粋ED法: 正解クラスのみ学習）
                self.class_training_counts[y_sample] += 1
                
                # 勾配計算（重み更新はしない）
                grads = self._compute_gradients(x_paired, z_hiddens, z_output, y_sample)
                
                # ★P1互換★ スパース勾配をフル展開して蓄積（ミニバッチ用）
                if accumulated_grads is None:
                    accumulated_grads = {
                        'w_output': grads['w_output'].copy(),
                        'lateral_weights': grads['lateral_weights'].copy(),
                        'w_hidden': [np.zeros_like(self.w_hidden[li]) for li in range(self.n_layers)]
                    }
                    for layer_idx in range(self.n_layers):
                        sparse_grad = grads['w_hidden'][layer_idx]
                        if sparse_grad is not None:
                            active_neurons, delta_w = sparse_grad
                            accumulated_grads['w_hidden'][layer_idx][active_neurons] += delta_w
                else:
                    accumulated_grads['w_output'] += grads['w_output']
                    accumulated_grads['lateral_weights'] += grads['lateral_weights']
                    for layer_idx in range(self.n_layers):
                        sparse_grad = grads['w_hidden'][layer_idx]
                        if sparse_grad is not None:
                            active_neurons, delta_w = sparse_grad
                            accumulated_grads['w_hidden'][layer_idx][active_neurons] += delta_w
            
            # ============================================
            # 勾配平均化と重み更新（v030検証済み方式）
            # ============================================
            self.w_output += accumulated_grads['w_output'] / batch_size
            self.lateral_weights += accumulated_grads['lateral_weights'] / batch_size
            for layer_idx in range(self.n_layers):
                self.w_hidden[layer_idx] += accumulated_grads['w_hidden'][layer_idx] / batch_size
            
            # 統計集計
            for y_pred, y_sample in batch_predictions:
                n_correct += (y_pred == y_sample)
            n_samples += batch_size
            
            # 約5秒ごとにコールバックを呼び出し（ヒートマップ更新等）
            if progress_callback is not None:
                now = _time.time()
                if now - _last_callback_time >= 5.0:
                    _last_callback_time = now
                    progress_callback(self, n_samples, -1)
        
        accuracy = n_correct / n_samples if n_samples > 0 else 0.0
        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
        
        return accuracy, avg_loss
    
    def train_epoch_cupy_batch(self, x_train, y_train, batch_size=128, progress_callback=None):
        """
        ★CuPy版★ GPU最適化バッチ学習（多層ネットワーク対応）
        
        【推奨】60000サンプル以上で--batch_sizeを使用、
               60000サンプル未満の場合には--batch_sizeオプションを削除
        
        【注意】60000サンプル未満で--batch_sizeを使用すると
               テスト精度が10%程度低下する可能性があります
        
        戦略:
        - 重みとデータをGPU上に保持（データ転送最小化）
        - 全計算をCuPyで実行（1.93倍高速化）
        - 統計情報のみCPUに戻す
        
        多層対応:
        - _forward_cupy(): 全層の順伝播をGPU上で実行
        - _compute_gradients_cupy(): 多層アミン拡散をGPU上で実行
        - 2026-02-01: 多層対応を確認・ドキュメント更新
        
        Args:
            x_train: 訓練データ (n_samples, n_input)
            y_train: 訓練ラベル (n_samples,)
            batch_size: バッチサイズ（デフォルト: 128）
        
        Returns:
            accuracy: 訓練精度
            avg_loss: 平均損失
        """
        # ============================================
        # 警告の表示（データサイズに応じた注意喚起、1回のみ）
        # ============================================
        if not hasattr(self, '_batch_warning_shown'):
            self._batch_warning_shown = False
        
        if not self._batch_warning_shown:
            import warnings
            warnings.warn(
                "\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "⚠️  --batch_size オプション使用時の注意\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "\n"
                "【推奨】60000サンプル以上で--batch_sizeを使用、\n"
                "       60000サンプル未満の場合には--batch_sizeオプションを削除\n"
                "\n"
                "【注意】60000サンプル未満で--batch_sizeを使用すると\n"
                "       テスト精度が10%程度低下する可能性があります\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
                UserWarning,
                stacklevel=2
            )
            self._batch_warning_shown = True
        
        if not CUPY_AVAILABLE:
            raise ImportError(
                "CuPyがインストールされていません。\n"
                "インストール: pip install cupy-cuda12x"
            )
        
        # ============================================
        # データと重みをGPUに転送（1回のみ）
        # ============================================
        print("[CuPy] データをGPUに転送中...")
        x_train_gpu = cp.array(x_train, dtype=cp.float32)
        y_train_gpu = cp.array(y_train, dtype=cp.int32)
        
        w_hidden_gpu = [cp.array(w, dtype=cp.float32) for w in self.w_hidden]
        w_output_gpu = cp.array(self.w_output, dtype=cp.float32)
        lateral_weights_gpu = cp.array(self.lateral_weights, dtype=cp.float32)
        
        # EI flags
        ei_flags_input_gpu = cp.array(self.ei_flags_input, dtype=cp.float32)
        ei_flags_hidden_gpu = [cp.array(flags, dtype=cp.float32) for flags in self.ei_flags_hidden]
        
        # コラム親和性
        column_affinity_gpu = [cp.array(aff, dtype=cp.float32) for aff in self.column_affinity_all_layers]
        
        print("[CuPy] GPU転送完了")
        
        # ============================================
        # データの準備
        # ============================================
        n_samples = len(x_train)
        
        # シャッフル（GPU上で）
        indices = cp.arange(n_samples)
        cp.random.shuffle(indices)
        x_train_gpu = x_train_gpu[indices]
        y_train_gpu = y_train_gpu[indices]
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        import time as _time
        total_loss = 0.0
        n_correct = 0
        _last_callback_time = _time.time()
        
        print(f"[CuPy] GPU学習開始（{n_batches}バッチ）...")
        
        # ============================================
        # バッチ学習ループ（GPU上で実行）
        # ============================================
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            x_batch_gpu = x_train_gpu[start_idx:end_idx]
            y_batch_gpu = y_train_gpu[start_idx:end_idx]
            current_batch_size = end_idx - start_idx
            
            # ============================================
            # 勾配蓄積（GPU並列化）
            # ============================================
            accumulated_grads = None
            batch_predictions = []
            
            for i in range(current_batch_size):
                x_sample_gpu = x_batch_gpu[i]
                y_sample = int(y_batch_gpu[i])  # CPUに戻す（統計用）
                
                # 順伝播（GPU）
                z_hiddens_gpu, z_output_gpu, x_paired_gpu = self._forward_cupy(
                    x_sample_gpu, w_hidden_gpu, w_output_gpu
                )
                
                # 予測と損失（CPUに戻して計算）
                z_output_cpu = cp.asnumpy(z_output_gpu)
                y_pred = np.argmax(z_output_cpu)
                total_loss += cross_entropy_loss(z_output_cpu, y_sample)
                batch_predictions.append((y_pred, y_sample))
                
                # ★統計情報の収集
                self.winner_selection_counts[y_pred] += 1
                self.total_training_samples += 1
                
                # クラス別学習回数（純粋ED法: 正解クラスのみ学習）
                self.class_training_counts[y_sample] += 1
                
                # 勾配計算（GPU）
                grads = self._compute_gradients_cupy(
                    x_paired_gpu, z_hiddens_gpu, z_output_gpu, y_sample,
                    w_hidden_gpu, w_output_gpu, lateral_weights_gpu,
                    column_affinity_gpu
                )
                
                # 勾配を蓄積
                if accumulated_grads is None:
                    accumulated_grads = {
                        'w_output': grads['w_output'].copy(),
                        'lateral_weights': grads['lateral_weights'].copy(),
                        'w_hidden': [g.copy() for g in grads['w_hidden']]
                    }
                else:
                    accumulated_grads['w_output'] += grads['w_output']
                    accumulated_grads['lateral_weights'] += grads['lateral_weights']
                    for layer_idx in range(len(grads['w_hidden'])):
                        accumulated_grads['w_hidden'][layer_idx] += grads['w_hidden'][layer_idx]
            
            # ============================================
            # 勾配平均化と重み更新（GPU上で）
            # ============================================
            w_output_gpu += accumulated_grads['w_output'] / current_batch_size
            lateral_weights_gpu += accumulated_grads['lateral_weights'] / current_batch_size
            for layer_idx in range(self.n_layers):
                w_hidden_gpu[layer_idx] += accumulated_grads['w_hidden'][layer_idx] / current_batch_size
            
            # 統計集計
            for y_pred, y_sample in batch_predictions:
                n_correct += (y_pred == y_sample)
            
            # 約5秒ごとにコールバックを呼び出し（ヒートマップ更新等）
            if progress_callback is not None:
                now = _time.time()
                if now - _last_callback_time >= 5.0:
                    _last_callback_time = now
                    progress_callback(self, end_idx, n_samples)
        
        # ============================================
        # GPUからCPUに重みを戻す
        # ============================================
        print("[CuPy] 重みをCPUに転送中...")
        self.w_output = cp.asnumpy(w_output_gpu)
        self.lateral_weights = cp.asnumpy(lateral_weights_gpu)
        for layer_idx in range(self.n_layers):
            self.w_hidden[layer_idx] = cp.asnumpy(w_hidden_gpu[layer_idx])
        
        # Dale's Principle強制（第1層のみ）
        if self.n_layers > 0:
            sign_matrix = np.outer(self.ei_flags_hidden[0], self.ei_flags_input)
            self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
        
        print("[CuPy] GPU学習完了")
        
        # ============================================
        # 学習後の精度と損失
        # ============================================
        accuracy = n_correct / n_samples if n_samples > 0 else 0.0
        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
        
        return accuracy, avg_loss
    
    def _forward_cupy(self, x_sample_gpu, w_hidden_gpu, w_output_gpu):
        """
        CuPy版順伝播
        
        Args:
            x_sample_gpu: 入力サンプル (CuPy配列)
            w_hidden_gpu: 隠れ層重みのリスト (CuPy配列)
            w_output_gpu: 出力層重み (CuPy配列)
        
        Returns:
            z_hiddens_gpu: 隠れ層出力のリスト
            z_output_gpu: 出力層出力
            x_paired_gpu: 入力ペア
        """
        # 入力ペア（興奮性・抑制性）
        x_paired_gpu = cp.concatenate([x_sample_gpu, x_sample_gpu])
        
        # 隠れ層
        z_hiddens_gpu = []
        z_current = x_paired_gpu
        
        for layer_idx in range(self.n_layers):
            a_hidden = cp.matmul(w_hidden_gpu[layer_idx], z_current)
            
            if self.activation == 'leaky_relu':
                z_hidden = cp.where(a_hidden > 0, a_hidden, self.leaky_alpha * a_hidden)
            else:  # tanh
                z_hidden = cp.tanh(a_hidden)
            
            z_hiddens_gpu.append(z_hidden)
            z_current = z_hidden
        
        # 出力層
        a_output = cp.matmul(w_output_gpu, z_current)
        exp_a = cp.exp(a_output - cp.max(a_output))
        z_output_gpu = exp_a / cp.sum(exp_a)
        
        return z_hiddens_gpu, z_output_gpu, x_paired_gpu
    
    def _compute_gradients_cupy(self, x_paired_gpu, z_hiddens_gpu, z_output_gpu, y_true,
                                w_hidden_gpu, w_output_gpu, lateral_weights_gpu,
                                column_affinity_gpu):
        """
        CuPy版勾配計算（v040のNumpy版を完全移植）
        
        全計算をGPU上で実行し、結果もGPU配列で返す。
        """
        # CPUに変換が必要な部分のみ
        z_output_cpu = cp.asnumpy(z_output_gpu)
        lateral_weights_cpu = cp.asnumpy(lateral_weights_gpu)
        
        # ============================================
        # 1. 出力層の勾配計算
        # ============================================
        target_probs = cp.zeros(self.n_output, dtype=cp.float32)
        target_probs[y_true] = 1.0
        
        error_output_gpu = target_probs - z_output_gpu
        saturation_output_gpu = cp.abs(z_output_gpu) * (1.0 - cp.abs(z_output_gpu))
        
        output_lr = self.layer_lrs[-1] if len(self.layer_lrs) > self.n_layers else self.learning_rate
        delta_w_output_gpu = output_lr * cp.outer(
            error_output_gpu * saturation_output_gpu,
            z_hiddens_gpu[-1]
        )
        
        # ============================================
        # 2. アミン濃度計算（CPUで計算、結果をGPUに戻す）
        # ============================================
        delta_lateral_cpu = np.zeros_like(lateral_weights_cpu)
        amine_concentration_cpu = np.zeros((self.n_output, 2))
        
        # ★純粋ED法★ 正解クラスのみ学習（v039で実証済み）
        # WTA誤答時の負の誤差拡散は学習を阻害するため削除
        error_correct = 1.0 - z_output_cpu[y_true]
        if error_correct > 0:
            amine_concentration_cpu[y_true, 0] = error_correct * self.initial_amine
        
        # GPUに戻す
        amine_concentration_gpu = cp.array(amine_concentration_cpu, dtype=cp.float32)
        delta_lateral_gpu = cp.array(delta_lateral_cpu, dtype=cp.float32)
        
        # ============================================
        # 3. 隠れ層の勾配計算（GPU）
        # ============================================
        delta_w_hidden_gpu = []
        
        for layer_idx in range(self.n_layers - 1, -1, -1):
            # 入力の取得
            if layer_idx == 0:
                z_input_gpu = x_paired_gpu
            else:
                z_input_gpu = z_hiddens_gpu[layer_idx - 1]
            
            # 拡散係数
            # ★提案B★ uniform_amine=Trueの場合、全層u1で均一拡散（青斑核モデル）
            if self.uniform_amine:
                diffusion_coef = self.u1
            else:
                diffusion_coef = self.u1 if layer_idx == self.n_layers - 1 else self.u2
            
            # アミン拡散（GPU）- Numpy版と同様にuse_affinityフラグで分岐
            amine_mask_gpu = amine_concentration_gpu >= 1e-8
            
            # ステップ1: アミン濃度に拡散係数を適用（全ニューロン一律）
            amine_diffused_gpu = amine_concentration_gpu * diffusion_coef
            
            if self.use_affinity:
                # ★Affinity方式★（従来のCuPy実装）
                amine_hidden_3d_gpu = (
                    amine_diffused_gpu[:, :, cp.newaxis] *
                    column_affinity_gpu[layer_idx][:, cp.newaxis, :]
                )
            else:
                # ★Membership方式★（v042最適化: ベクトル化）
                membership = self.column_membership_all_layers[layer_idx]  # [n_classes, n_neurons] boolean (numpy)
                n_neurons = self.n_hidden[layer_idx]
                
                # CPUで処理（membershipはNumPy配列のため）
                amine_diffused_cpu = cp.asnumpy(amine_diffused_gpu)
                z_current_cpu = cp.asnumpy(z_hiddens_gpu[layer_idx])
                
                # アミン濃度が非ゼロのクラスを特定
                active_classes = np.where(np.any(amine_diffused_cpu >= 1e-8, axis=1))[0]
                n_active = len(active_classes)
                
                if n_active == 0:
                    # アクティブなクラスがない場合はゼロ行列
                    amine_hidden_3d_gpu = cp.zeros((self.n_output, 2, n_neurons), dtype=cp.float32)
                else:
                    # ベクトル化実装（Numpy版と同じロジック）
                    # 1. アクティブクラスのmembershipを取得
                    active_membership = membership[active_classes]  # [n_active, n_neurons]
                    
                    # 2. メンバーニューロンの活性値を取得（非メンバーは-inf）
                    masked_activations = np.where(active_membership, z_current_cpu, -np.inf)  # [n_active, n_neurons]
                    
                    # 3. 各クラス内でランク計算（argsort二回でランク取得、降順）
                    sorted_indices = np.argsort(-masked_activations, axis=1)
                    ranks = np.argsort(sorted_indices, axis=1)  # [n_active, n_neurons]
                    
                    # 4. ランクから学習率を取得（ルックアップテーブル参照）
                    clamped_ranks = np.minimum(ranks, len(self._learning_weight_lut) - 1)
                    learning_weights = self._learning_weight_lut[clamped_ranks]  # [n_active, n_neurons]
                    
                    # 5. 非メンバーの学習率を設定（CuPy版）
                    if self.amine_diffusion_sigma > 0.0:
                        nc_weights = self._spatial_amine_weights[layer_idx][active_classes]
                        learning_weights = np.where(
                            active_membership, 
                            learning_weights,
                            nc_weights
                        )
                    elif self.amine_base_level > 0.0:
                        learning_weights = np.where(
                            active_membership, 
                            learning_weights,
                            self.amine_base_level
                        )
                    else:
                        learning_weights = np.where(
                            active_membership, 
                            learning_weights,
                            0.0
                        )  # [n_active, n_neurons]
                    
                    # 6. アミン拡散値に学習率を適用
                    amine_hidden_3d_cpu = np.zeros((self.n_output, 2, n_neurons), dtype=np.float32)
                    amine_hidden_3d_cpu[active_classes] = (
                        amine_diffused_cpu[active_classes, :, np.newaxis] *  # [n_active, 2, 1]
                        learning_weights[:, np.newaxis, :]                    # [n_active, 1, n_neurons]
                    )
                    
                    # GPUに戻す
                    amine_hidden_3d_gpu = cp.array(amine_hidden_3d_cpu, dtype=cp.float32)
            
            amine_hidden_3d_gpu = amine_hidden_3d_gpu * amine_mask_gpu[:, :, cp.newaxis]
            
            # 活性ニューロン
            neuron_mask = cp.any(amine_hidden_3d_gpu >= 1e-8, axis=(0, 1))
            active_neurons = cp.where(neuron_mask)[0]
            
            if len(active_neurons) == 0:
                delta_w_hidden_gpu.insert(0, cp.zeros_like(w_hidden_gpu[layer_idx]))
                continue
            
            # 活性化関数の勾配
            z_active_gpu = z_hiddens_gpu[layer_idx][active_neurons]
            if self.activation == 'leaky_relu':
                saturation_term_raw_gpu = cp.where(z_active_gpu > 0, 1.0, self.leaky_alpha)
            else:
                saturation_term_raw_gpu = cp.abs(z_active_gpu) * (1.0 - cp.abs(z_active_gpu))
            saturation_term_gpu = cp.maximum(saturation_term_raw_gpu, 1e-3)
            
            # 学習信号強度
            layer_lr = self.layer_lrs[layer_idx]
            learning_signals_3d_gpu = (
                layer_lr *
                amine_hidden_3d_gpu[:, :, active_neurons] *
                saturation_term_gpu[cp.newaxis, cp.newaxis, :]
            )
            
            # 勾配計算
            # ★最適化★ 3D配列生成を回避（Normal版と同じ数学的簡略化）
            n_combinations = self.n_output * 2
            learning_signals_2d_gpu = learning_signals_3d_gpu.reshape(n_combinations, -1)
            signal_sum_gpu = learning_signals_2d_gpu.sum(axis=0)  # [n_active]
            delta_w_active_gpu = signal_sum_gpu[:, cp.newaxis] * z_input_gpu[cp.newaxis, :]  # [n_active, n_input]
            
            # 層ごとの符号制約（layer_idx > 0の場合、Numpy版と同様）
            if layer_idx > 0:
                w_sign_gpu = cp.sign(w_hidden_gpu[layer_idx][active_neurons, :])
                w_sign_gpu = cp.where(w_sign_gpu == 0, 1.0, w_sign_gpu)
                delta_w_active_gpu = delta_w_active_gpu * w_sign_gpu
            
            # gradient clipping
            if self.gradient_clip > 0:
                delta_w_norms_gpu = cp.linalg.norm(delta_w_active_gpu, axis=1, keepdims=True)
                clip_mask_gpu = delta_w_norms_gpu > self.gradient_clip
                delta_w_active_gpu = cp.where(
                    clip_mask_gpu,
                    delta_w_active_gpu * (self.gradient_clip / delta_w_norms_gpu),
                    delta_w_active_gpu
                )
            
            # ★新機能★ 隠れ層のコラムニューロン勾配抑制（CuPy版、層別対応）
            # columnar_ed.prompt.mdの「活性化バランスの問題」対策
            layer_lr_factor = self.column_lr_factors[layer_idx]  # 層別の係数を取得
            if layer_lr_factor < 1.0 and layer_idx < len(self.column_membership_all_layers):
                membership = self.column_membership_all_layers[layer_idx]  # [n_classes, n_neurons] boolean (numpy)
                is_column_neuron = np.any(membership, axis=0)  # [n_neurons] boolean (numpy)
                active_is_column = is_column_neuron[cp.asnumpy(active_neurons)]  # numpy boolean array
                if np.any(active_is_column):
                    # GPU上でコラムニューロンの勾配を抑制
                    active_is_column_gpu = cp.asarray(active_is_column)
                    delta_w_active_gpu[active_is_column_gpu, :] *= layer_lr_factor
            
            # フルサイズ勾配行列
            dw_gpu = cp.zeros_like(w_hidden_gpu[layer_idx])
            dw_gpu[active_neurons, :] = delta_w_active_gpu
            delta_w_hidden_gpu.insert(0, dw_gpu)
        
        return {
            'w_output': delta_w_output_gpu,
            'lateral_weights': delta_lateral_gpu,
            'w_hidden': delta_w_hidden_gpu
        }
    
    def get_non_column_debug_info(self):
        """
        ★Step 2★ 非コラムニューロンの学習動態デバッグ情報を取得
        
        コラムニューロン vs 非コラムニューロンの重みノルム、変化量、
        出力層寄与度を層ごとに比較して返す。
        
        Returns:
            dict: 層ごとのデバッグ統計情報
        """
        info = {}
        for layer_idx in range(self.n_layers):
            if layer_idx >= len(self.column_membership_all_layers):
                continue
            
            membership = self.column_membership_all_layers[layer_idx]
            is_column = np.any(membership, axis=0)  # [n_neurons]
            w = self.w_hidden[layer_idx]
            w_init = self._initial_weights[layer_idx]
            
            n_col = int(np.sum(is_column))
            n_nc = int(np.sum(~is_column))
            
            # (a) 重みノルム統計
            col_norms = np.linalg.norm(w[is_column], axis=1)
            nc_norms = np.linalg.norm(w[~is_column], axis=1)
            
            # (b) 重み変化量 (初期値との差分ノルム)
            col_delta = np.linalg.norm(w[is_column] - w_init[is_column], axis=1)
            nc_delta = np.linalg.norm(w[~is_column] - w_init[~is_column], axis=1)
            
            # (c) 出力層での寄与度（最終隠れ層のみ意味がある）
            if layer_idx == self.n_layers - 1:
                out_contrib_col = np.abs(self.w_output[:, is_column]).sum(axis=0)
                out_contrib_nc = np.abs(self.w_output[:, ~is_column]).sum(axis=0)
            else:
                out_contrib_col = np.zeros(n_col)
                out_contrib_nc = np.zeros(n_nc)
            
            # (d) 出力層重みの変化量
            w_out_delta = np.linalg.norm(
                self.w_output - self._initial_w_output, axis=0
            )  # [n_neurons]
            col_out_delta = w_out_delta[is_column]
            nc_out_delta = w_out_delta[~is_column]
            
            info[f'layer_{layer_idx}'] = {
                'n_col': n_col,
                'n_nc': n_nc,
                'col_weight_norm': (float(col_norms.mean()), float(col_norms.std())),
                'nc_weight_norm': (float(nc_norms.mean()), float(nc_norms.std())),
                'col_weight_delta': (float(col_delta.mean()), float(col_delta.std())),
                'nc_weight_delta': (float(nc_delta.mean()), float(nc_delta.std())),
                'col_output_contrib': (float(out_contrib_col.mean()), float(out_contrib_col.std())),
                'nc_output_contrib': (float(out_contrib_nc.mean()), float(out_contrib_nc.std())),
                'col_out_weight_delta': (float(col_out_delta.mean()), float(col_out_delta.std())),
                'nc_out_weight_delta': (float(nc_out_delta.mean()), float(nc_out_delta.std())),
            }
        return info
    
    def get_debug_info(self, monitor_classes=[0, 1]):
        """
        デバッグ情報の取得（Error-Weighted方式用に簡略化）
        
        Args:
            monitor_classes: モニタリング対象の出力クラスのリスト
        
        Returns:
            debug_info: デバッグ情報の辞書
        """
        debug_info = {}
        
        for class_idx in monitor_classes:
            class_info = {}
            
            # 出力層の重みの統計
            w_output = self.w_output[class_idx]
            class_info['w_output'] = {
                'mean': float(np.mean(w_output)),
                'std': float(np.std(w_output)),
                'min': float(np.min(w_output)),
                'max': float(np.max(w_output)),
                'abs_mean': float(np.mean(np.abs(w_output)))
            }
            
            # 出力重みの影響力が大きい上位5ニューロン
            abs_weights = np.abs(w_output)
            top5_indices = np.argsort(abs_weights)[-5:]
            class_info['top5_influential_neurons'] = {
                'indices': top5_indices.tolist(),
                'weights': w_output[top5_indices].tolist(),
                'abs_weights': abs_weights[top5_indices].tolist()
            }
            
            debug_info[f'class_{class_idx}'] = class_info
        
        return debug_info
    
    def diagnose_column_structure(self):
        """
        コラム構造の詳細診断
        各層のコラム参加率、重複度、帰属度統計を分析
        """
        print("\n" + "="*80)
        print("コラム構造診断")
        print("="*80)
        
        for layer_idx in range(self.n_layers):
            affinity = self.column_affinity_all_layers[layer_idx]
            n_neurons = self.n_hidden[layer_idx]
            n_classes = self.n_output
            
            print(f"\n【Layer {layer_idx} - {n_neurons}ニューロン】")
            
            # 1. 各クラスのコラム参加率
            print("\n1. 各クラスのコラム参加ニューロン数:")
            class_neuron_counts = []
            for class_idx in range(n_classes):
                # 非ゼロ（帰属度 > 1e-8）のニューロン数
                participating = np.count_nonzero(affinity[class_idx] > 1e-8)
                class_neuron_counts.append(participating)
                participation_rate = participating / n_neurons * 100
                print(f"  Class {class_idx}: {participating:3d}個 ({participation_rate:5.1f}%)")
            
            print(f"  平均: {np.mean(class_neuron_counts):.1f}個")
            print(f"  標準偏差: {np.std(class_neuron_counts):.1f}個")
            
            # 2. ニューロンごとの重複度（何個のクラスに参加しているか）
            print("\n2. ニューロンの重複度分析:")
            overlap_counts = np.zeros(n_neurons, dtype=int)
            for neuron_idx in range(n_neurons):
                # このニューロンが参加しているクラス数
                n_classes_for_neuron = np.count_nonzero(affinity[:, neuron_idx] > 1e-8)
                overlap_counts[neuron_idx] = n_classes_for_neuron
            
            # ヒストグラム
            unique_overlaps, counts = np.unique(overlap_counts, return_counts=True)
            print("  重複度分布:")
            for overlap, count in zip(unique_overlaps, counts):
                pct = count / n_neurons * 100
                print(f"    {overlap}クラス参加: {count:3d}個 ({pct:5.1f}%)")
            
            print(f"  平均重複度: {np.mean(overlap_counts):.2f}クラス/ニューロン")
            print(f"  未参加ニューロン数: {np.count_nonzero(overlap_counts == 0)}個")
            
            # 3. 帰属度の統計（非ゼロ値のみ）
            print("\n3. 帰属度統計（非ゼロ値のみ）:")
            non_zero_affinity = affinity[affinity > 1e-8]
            if len(non_zero_affinity) > 0:
                print(f"  平均: {np.mean(non_zero_affinity):.4f}")
                print(f"  中央値: {np.median(non_zero_affinity):.4f}")
                print(f"  標準偏差: {np.std(non_zero_affinity):.4f}")
                print(f"  最小値: {np.min(non_zero_affinity):.4f}")
                print(f"  最大値: {np.max(non_zero_affinity):.4f}")
                
                # 帰属度の分布（四分位）
                q1 = np.percentile(non_zero_affinity, 25)
                q3 = np.percentile(non_zero_affinity, 75)
                print(f"  第1四分位: {q1:.4f}")
                print(f"  第3四分位: {q3:.4f}")
            
            # 4. クラス間の重複ニューロン分析
            print("\n4. クラス間重複ニューロン数（上位5ペア）:")
            overlap_matrix = np.zeros((n_classes, n_classes), dtype=int)
            for i in range(n_classes):
                for j in range(i+1, n_classes):
                    # クラスiとjの両方に参加しているニューロン数
                    both_participate = np.count_nonzero(
                        (affinity[i] > 1e-8) & (affinity[j] > 1e-8)
                    )
                    overlap_matrix[i, j] = both_participate
            
            # 上位5ペアを表示
            overlap_pairs = []
            for i in range(n_classes):
                for j in range(i+1, n_classes):
                    overlap_pairs.append((i, j, overlap_matrix[i, j]))
            overlap_pairs.sort(key=lambda x: x[2], reverse=True)
            
            for i, j, count in overlap_pairs[:5]:
                print(f"  Class {i} ⇔ Class {j}: {count}個共有")
        
        print("\n" + "="*80)

    def diagnose_hidden_weights(self, initial_weights=None):
        """
        隠れ層の重み状態を詳細診断
        
        コラム構造部分の重みが極大化（飽和）していないかを確認するための診断機能。
        
        Args:
            initial_weights: 初期重みのリスト（比較用、省略可）
        """
        print("\n" + "="*80)
        print("隠れ層の重み診断")
        print("="*80)
        
        for layer_idx in range(self.n_layers):
            w_hidden = self.w_hidden[layer_idx]
            n_neurons = self.n_hidden[layer_idx]
            
            # 入力次元の取得
            if layer_idx == 0:
                n_input_dim = self.n_input
            else:
                n_input_dim = self.n_hidden[layer_idx - 1]
            
            print(f"\n【Layer {layer_idx} - {n_neurons}ニューロン × {n_input_dim}入力】")
            
            # ============================================
            # 1. 全体の重み統計
            # ============================================
            print("\n1. 全体の重み統計:")
            print(f"  Mean:    {np.mean(w_hidden):+.6f}")
            print(f"  Std:     {np.std(w_hidden):.6f}")
            print(f"  Min:     {np.min(w_hidden):+.6f}")
            print(f"  Max:     {np.max(w_hidden):+.6f}")
            print(f"  Abs Mean:{np.mean(np.abs(w_hidden)):.6f}")
            
            # 初期重みとの比較（提供された場合）
            if initial_weights is not None and layer_idx < len(initial_weights):
                w_init = initial_weights[layer_idx]
                w_change = w_hidden - w_init
                print(f"\n  [初期値からの変化]")
                print(f"  Change Mean:    {np.mean(w_change):+.6f}")
                print(f"  Change Std:     {np.std(w_change):.6f}")
                print(f"  Change Abs Max: {np.max(np.abs(w_change)):.6f}")
            
            # ============================================
            # 2. コラム vs 非コラムニューロンの重み比較
            # ============================================
            print("\n2. コラム vs 非コラムニューロンの重み比較:")
            membership = self.column_membership_all_layers[layer_idx]
            
            # 各クラスのコラムニューロンを集約
            all_column_neurons = set()
            for class_idx in range(self.n_output):
                column_neurons = np.where(membership[class_idx])[0]
                all_column_neurons.update(column_neurons)
            
            column_neuron_indices = np.array(list(all_column_neurons), dtype=int)
            non_column_neuron_indices = np.array([i for i in range(n_neurons) if i not in all_column_neurons], dtype=int)
            
            print(f"  コラムニューロン数:    {len(column_neuron_indices)}個 ({100*len(column_neuron_indices)/n_neurons:.1f}%)")
            print(f"  非コラムニューロン数:  {len(non_column_neuron_indices)}個 ({100*len(non_column_neuron_indices)/n_neurons:.1f}%)")
            
            if len(column_neuron_indices) > 0:
                w_column = w_hidden[column_neuron_indices, :]
                print(f"\n  [コラムニューロンの入力重み]")
                print(f"    Mean:     {np.mean(w_column):+.6f}")
                print(f"    Std:      {np.std(w_column):.6f}")
                print(f"    Abs Mean: {np.mean(np.abs(w_column)):.6f}")
                print(f"    Min:      {np.min(w_column):+.6f}")
                print(f"    Max:      {np.max(w_column):+.6f}")
            
            if len(non_column_neuron_indices) > 0:
                w_non_column = w_hidden[non_column_neuron_indices, :]
                print(f"\n  [非コラムニューロンの入力重み]")
                print(f"    Mean:     {np.mean(w_non_column):+.6f}")
                print(f"    Std:      {np.std(w_non_column):.6f}")
                print(f"    Abs Mean: {np.mean(np.abs(w_non_column)):.6f}")
                print(f"    Min:      {np.min(w_non_column):+.6f}")
                print(f"    Max:      {np.max(w_non_column):+.6f}")
            
            # ============================================
            # 3. クラス別コラムニューロンの重み統計
            # ============================================
            print("\n3. クラス別コラムニューロンの重み統計:")
            print(f"  Class  N_neurons   Mean       Std        Min        Max")
            print(f"  " + "-"*65)
            
            for class_idx in range(self.n_output):
                column_neurons = np.where(membership[class_idx])[0]
                if len(column_neurons) > 0:
                    w_class = w_hidden[column_neurons, :]
                    print(f"  {class_idx:5d}  {len(column_neurons):9d}   {np.mean(w_class):+.5f}  {np.std(w_class):.5f}  {np.min(w_class):+.5f}  {np.max(w_class):+.5f}")
            
            # ============================================
            # 4. 飽和度分析（重みの絶対値が閾値以上のニューロンの割合）
            # ============================================
            print("\n4. 飽和度分析:")
            thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
            
            # ニューロン単位での最大重み絶対値
            neuron_max_abs_weights = np.max(np.abs(w_hidden), axis=1)
            
            print(f"  [ニューロン単位の最大重み絶対値分布]")
            print(f"    Mean: {np.mean(neuron_max_abs_weights):.4f}")
            print(f"    Max:  {np.max(neuron_max_abs_weights):.4f}")
            
            print(f"\n  [閾値別の飽和ニューロン数]")
            for thresh in thresholds:
                n_saturated = np.sum(neuron_max_abs_weights >= thresh)
                pct = 100 * n_saturated / n_neurons
                # コラムニューロン中の飽和割合も計算
                if len(column_neuron_indices) > 0:
                    n_column_saturated = np.sum(neuron_max_abs_weights[column_neuron_indices] >= thresh)
                    pct_column = 100 * n_column_saturated / len(column_neuron_indices)
                    print(f"    |w| >= {thresh:.1f}: {n_saturated:4d}個 ({pct:5.1f}%)  [コラム: {n_column_saturated:3d}個 ({pct_column:5.1f}%)]")
                else:
                    print(f"    |w| >= {thresh:.1f}: {n_saturated:4d}個 ({pct:5.1f}%)")
            
            # ============================================
            # 5. 重み分布のヒストグラム（テキスト表示）
            # ============================================
            print("\n5. 重み分布（ヒストグラム）:")
            w_flat = w_hidden.flatten()
            hist_bins = [-np.inf, -1.0, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1.0, np.inf]
            bin_labels = ["<-1.0", "-1~-0.5", "-0.5~-0.1", "-0.1~-0.01", "-0.01~0.01", "0.01~0.1", "0.1~0.5", "0.5~1", ">1.0"]
            
            hist, _ = np.histogram(w_flat, bins=hist_bins)
            total = len(w_flat)
            
            for label, count in zip(bin_labels, hist):
                bar_len = int(50 * count / total)
                bar = "█" * bar_len
                print(f"  {label:>12s}: {count:8d} ({100*count/total:5.1f}%) {bar}")
        
        # ============================================
        # 出力層の重みも診断
        # ============================================
        print(f"\n【出力層 - {self.n_output}クラス × {self.n_hidden[-1]}入力】")
        
        print("\n1. 全体の重み統計:")
        print(f"  Mean:    {np.mean(self.w_output):+.6f}")
        print(f"  Std:     {np.std(self.w_output):.6f}")
        print(f"  Min:     {np.min(self.w_output):+.6f}")
        print(f"  Max:     {np.max(self.w_output):+.6f}")
        print(f"  Abs Mean:{np.mean(np.abs(self.w_output)):.6f}")
        
        print("\n2. クラス別出力重み統計:")
        print(f"  Class   Mean       Std        Min        Max")
        print(f"  " + "-"*55)
        for class_idx in range(self.n_output):
            w_class = self.w_output[class_idx]
            print(f"  {class_idx:5d}   {np.mean(w_class):+.5f}  {np.std(w_class):.5f}  {np.min(w_class):+.5f}  {np.max(w_class):+.5f}")
        
        print("\n" + "="*80)

    # ============================================================================
    # ★Idea A★ Hebbian Weight Alignment（エポック後実行）
    # ============================================================================
    
    def apply_hebbian_alignment(self):
        """
        非コラムニューロンの隠れ層重みを最近傍コラムの重み重心に向けてドリフト。
        
        生物学的背景:
        - V1方位コラム: 空間的に近いニューロンは類似特徴に反応
        - 介在ニューロンの可塑性: シナプス結合を局所回路に適応させる
        - ドリフト強度α: 小さな値で多様性を保ちつつタスク関連特徴に整列
        
        メカニズム:
        - 各非コラムニューロンの最近傍コラム（事前計算済み）のメンバーニューロン特定
        - コラムメンバーの重みベクトル重心を計算
        - w_nc += alpha * (centroid - w_nc) で重心方向にドリフト
        - 第1層のみDale's Principle再適用
        
        Returns:
            dict: 各層の統計情報 {'layer_idx': {'n_aligned': int, 'mean_drift': float}}
        """
        alpha = self.hebbian_alignment_alpha
        if alpha <= 0.0:
            return None
        
        stats = {}
        for layer_idx in range(self.n_layers):
            non_col_idx = self._hebbian_non_column_indices_per_layer[layer_idx]
            nearest_class = self._hebbian_nearest_class_per_layer[layer_idx]
            
            if len(non_col_idx) == 0:
                continue
            
            membership = self.column_membership_all_layers[layer_idx]  # (n_classes, n_neurons)
            w = self.w_hidden[layer_idx]  # (n_neurons, n_input_dim)
            
            # 各クラスのコラムニューロン重み重心を計算
            centroids = {}  # class_idx -> centroid_weight_vector
            for class_idx in range(self.n_output):
                col_members = np.where(membership[class_idx])[0]
                if len(col_members) > 0:
                    centroids[class_idx] = w[col_members].mean(axis=0)
            
            # 非コラムニューロンをドリフト
            total_drift = 0.0
            n_aligned = 0
            for i, neuron_idx in enumerate(non_col_idx):
                nc = nearest_class[i]
                if nc in centroids:
                    diff = centroids[nc] - w[neuron_idx]
                    drift = alpha * diff
                    w[neuron_idx] += drift
                    total_drift += np.linalg.norm(drift)
                    n_aligned += 1
            
            # 第1層のDale's Principle再適用
            if layer_idx == 0:
                self.w_hidden[0] = np.abs(self.w_hidden[0]) * self._sign_matrix_layer0
            
            mean_drift = total_drift / n_aligned if n_aligned > 0 else 0.0
            stats[layer_idx] = {'n_aligned': n_aligned, 'mean_drift': mean_drift}
        
        return stats
    
    # ============================================================================
    # ★v042新機能★ 動的シナプス刈り込み（生物学的発達的プルーニング）
    # ============================================================================
    
    def initialize_pruning(self, final_sparsity=0.4, total_epochs=20, start_epoch=None, end_epoch=None, verbose=False):
        """
        動的シナプス刈り込みの初期化
        
        生物学的背景:
        - 脳の発達過程で約40-50%のシナプスが刈り込まれる（Huttenlocher, 1979）
        - 補体タンパク質C1q/C3が弱いシナプスをタグ付け（Stevens et al., 2007）
        - ミクログリアが弱いシナプスを貪食除去（Schafer et al., 2012）
        - "Use it or lose it"原理: 活性化頻度の低いシナプスが除去される
        - 刈り込みの終了時期は脳の部位により異なる:
          - 視覚野: 約6歳まで
          - 前頭前野: 20代後半まで
        
        ED法での近似:
        - 重みの絶対値 ≈ 活性化頻度（ED法は活性化を通じて重みを増大）
        - 低重みの接続 = 低活性化頻度 = 刈り込み対象
        - 安定期以降に刈り込み開始（初期重みの影響が薄れた後）
        - end_epoch指定時: start_epoch〜end_epoch間で目標スパース率を100%達成
        
        Args:
            final_sparsity: 最終エポックでの目標スパース率（デフォルト: 0.4=40%刈り込み）
            total_epochs: 総エポック数（プログレッシブ刈り込みの計算用）
            start_epoch: 刈り込み開始エポック（Noneなら安定期検出で自動開始）
            end_epoch: 刈り込み終了エポック（Noneなら最終エポックまで）
                       指定時: start_epoch〜end_epoch間で目標を100%達成
            verbose: 詳細ログを出力するか
        """
        self.pruning_enabled = True
        self.pruning_final_sparsity = final_sparsity
        self.pruning_total_epochs = total_epochs
        self.pruning_start_epoch = start_epoch  # Noneなら自動検出
        self.pruning_end_epoch = end_epoch  # Noneなら最終エポックまで
        self.pruning_verbose = verbose
        
        # 刈り込み状態の初期化
        self.pruning_masks = []  # 各層の刈り込みマスク（1=保持、0=刈り込み済み）
        self.pruning_started = False
        self.pruning_stable_detected = False
        self.pruning_actual_start_epoch = None
        
        # 各層の刈り込みマスクを初期化（全て1=保持）
        for layer_idx in range(self.n_layers):
            mask = np.ones_like(self.w_hidden[layer_idx])
            self.pruning_masks.append(mask)
        
        # 出力層のマスクも追加
        self.pruning_masks.append(np.ones_like(self.w_output))
        
        # 安定期検出用の履歴
        self.pruning_accuracy_history = []
        
        # 統計情報
        self.pruning_stats = {
            'initial_connections': [np.size(self.w_hidden[i]) for i in range(self.n_layers)] + [np.size(self.w_output)],
            'pruned_connections': [0] * (self.n_layers + 1),
            'current_sparsity': [0.0] * (self.n_layers + 1)
        }
        
        if verbose:
            print("\n[動的シナプス刈り込み初期化]")
            print(f"  目標スパース率:     {final_sparsity*100:.1f}%")
            print(f"  総エポック数:       {total_epochs}")
            print(f"  開始エポック:       {'自動検出' if start_epoch is None else start_epoch}")
            print(f"  終了エポック:       {'最終エポック' if end_epoch is None else end_epoch}")
            for i, conn in enumerate(self.pruning_stats['initial_connections']):
                layer_name = f"Layer {i}" if i < self.n_layers else "Output"
                print(f"  {layer_name} 接続数:   {conn:,d}")
    
    def check_stability_for_pruning(self, test_accuracy, epoch, threshold=0.01):
        """
        安定期を検出して刈り込み開始を判定
        
        安定期の定義:
        - 直近3エポックのTest精度変化率が閾値以下
        
        Args:
            test_accuracy: 現在のTest精度
            epoch: 現在のエポック（0始まり）
            threshold: 精度変化率の閾値（デフォルト: 0.01=1%）
        
        Returns:
            bool: 刈り込みを開始すべきかどうか
        """
        if not hasattr(self, 'pruning_enabled') or not self.pruning_enabled:
            return False
        
        if self.pruning_started:
            return True  # すでに開始済み
        
        # 固定開始エポックが指定されている場合
        if self.pruning_start_epoch is not None:
            if epoch >= self.pruning_start_epoch:
                self.pruning_started = True
                self.pruning_actual_start_epoch = epoch
                if self.pruning_verbose:
                    print(f"\n[刈り込み開始] 指定エポック {self.pruning_start_epoch} に到達")
                return True
            return False
        
        # 自動安定期検出
        self.pruning_accuracy_history.append(test_accuracy)
        
        if len(self.pruning_accuracy_history) >= 3:
            recent = self.pruning_accuracy_history[-3:]
            changes = [abs(recent[i+1] - recent[i]) for i in range(2)]
            max_change = max(changes)
            
            if max_change < threshold:
                self.pruning_started = True
                self.pruning_stable_detected = True
                self.pruning_actual_start_epoch = epoch
                if self.pruning_verbose:
                    print(f"\n[刈り込み開始] 安定期検出 (epoch {epoch})")
                    print(f"  直近3エポックの精度変化: {changes}")
                    print(f"  最大変化: {max_change:.4f} < 閾値 {threshold:.4f}")
                return True
        
        return False
    
    def compute_pruning_threshold(self, epoch):
        """
        現在のエポックに対する刈り込み閾値を計算
        
        プログレッシブ刈り込み:
        - 開始時は小さい閾値（強い接続のみ刈り込み対象外）
        - 終了エポックに向けて閾値を徐々に上げる
        - 目標: 終了エポックまでにfinal_sparsity分の接続を100%刈り込み
        - end_epoch未指定時: 最終エポックまで
        - end_epoch指定時: start_epoch〜end_epoch間で100%達成
        
        Args:
            epoch: 現在のエポック
        
        Returns:
            float: 各層の刈り込み閾値（この値未満の重みを刈り込み）
        """
        if not self.pruning_started:
            return 0.0
        
        start_epoch = self.pruning_actual_start_epoch
        if start_epoch is None:
            return 0.0
        
        # 終了エポックの決定
        if self.pruning_end_epoch is not None:
            # end_epoch指定時: start_epoch〜end_epoch間で100%達成
            pruning_duration = self.pruning_end_epoch - start_epoch
        else:
            # end_epoch未指定時: 最終エポックまで（従来動作）
            pruning_duration = self.pruning_total_epochs - start_epoch
        
        current_progress = epoch - start_epoch
        
        if pruning_duration <= 0:
            progress_ratio = 1.0
        else:
            progress_ratio = min(1.0, current_progress / pruning_duration)
        
        # 現在の目標スパース率（線形増加）
        target_sparsity = self.pruning_final_sparsity * progress_ratio
        
        return target_sparsity
    
    def apply_pruning(self, epoch, verbose=None):
        """
        刈り込みを実行
        
        重みの絶対値が小さい接続を刈り込み（マスクを0に設定）
        
        Args:
            epoch: 現在のエポック
            verbose: 詳細ログを出力するか（Noneならself.pruning_verboseを使用）
        
        Returns:
            dict: 刈り込み統計
        """
        if not hasattr(self, 'pruning_enabled') or not self.pruning_enabled:
            return None
        
        if not self.pruning_started:
            return None
        
        # 終了エポック以降は刈り込みをスキップ（既に目標達成済み）
        if self.pruning_end_epoch is not None and epoch > self.pruning_end_epoch:
            if verbose or self.pruning_verbose:
                print(f"\n[刈り込みスキップ] Epoch {epoch}: 終了エポック({self.pruning_end_epoch})を超過、刈り込み完了済み")
            return None
        
        if verbose is None:
            verbose = self.pruning_verbose
        
        target_sparsity = self.compute_pruning_threshold(epoch)
        
        stats = {
            'epoch': epoch,
            'target_sparsity': target_sparsity,
            'actual_sparsity': [],
            'pruned_this_epoch': [],
            'total_pruned': [],
        }
        
        for layer_idx in range(self.n_layers + 1):  # 隠れ層 + 出力層
            if layer_idx < self.n_layers:
                weights = self.w_hidden[layer_idx]
                mask = self.pruning_masks[layer_idx]
            else:
                weights = self.w_output
                mask = self.pruning_masks[layer_idx]
            
            # 現在の有効な重みのみを対象
            active_weights = np.abs(weights) * mask
            
            # 目標スパース率に基づく閾値計算
            # 全接続数に対してtarget_sparsity%を刈り込む
            total_connections = np.size(weights)
            target_pruned = int(total_connections * target_sparsity)
            
            # 現在の刈り込み済み数
            current_pruned = np.sum(mask == 0)
            
            # 追加で刈り込む数
            additional_prune = max(0, target_pruned - int(current_pruned))
            
            if additional_prune > 0:
                # 有効な重みの中で最も小さいものから刈り込む
                active_abs_weights = np.abs(weights) * mask  # 既に刈り込まれたものは0
                
                # 閾値を計算（下位additional_prune個を刈り込み）
                flat_weights = active_abs_weights.flatten()
                # 既に0のもの（刈り込み済み）を除外
                non_zero_weights = flat_weights[flat_weights > 0]
                
                if len(non_zero_weights) > additional_prune:
                    # ソートして閾値を決定
                    sorted_weights = np.sort(non_zero_weights)
                    threshold = sorted_weights[additional_prune - 1]
                    
                    # 閾値以下の重みを刈り込み
                    new_prune_mask = (active_abs_weights <= threshold) & (active_abs_weights > 0)
                    mask[new_prune_mask] = 0
            
            # 統計更新
            pruned_count = np.sum(mask == 0)
            actual_sparsity = pruned_count / total_connections
            
            stats['actual_sparsity'].append(actual_sparsity)
            stats['pruned_this_epoch'].append(int(pruned_count - current_pruned))
            stats['total_pruned'].append(int(pruned_count))
            
            # マスクを重みに適用（刈り込まれた接続を0に）
            if layer_idx < self.n_layers:
                self.w_hidden[layer_idx] *= mask
                self.pruning_masks[layer_idx] = mask
            else:
                self.w_output *= mask
                self.pruning_masks[layer_idx] = mask
            
            self.pruning_stats['pruned_connections'][layer_idx] = int(pruned_count)
            self.pruning_stats['current_sparsity'][layer_idx] = actual_sparsity
        
        if verbose:
            print(f"\n[刈り込み実行] Epoch {epoch}")
            print(f"  目標スパース率: {target_sparsity*100:.1f}%")
            for i in range(self.n_layers + 1):
                layer_name = f"Layer {i}" if i < self.n_layers else "Output"
                init_conn = self.pruning_stats['initial_connections'][i]
                pruned = stats['total_pruned'][i]
                sparsity = stats['actual_sparsity'][i]
                this_epoch = stats['pruned_this_epoch'][i]
                print(f"  {layer_name}: {pruned:,d}/{init_conn:,d} 刈り込み済み ({sparsity*100:.1f}%) [+{this_epoch:,d}]")
        
        return stats
    
    def get_pruning_summary(self):
        """
        刈り込み統計のサマリーを取得
        
        Returns:
            dict: 刈り込み統計サマリー
        """
        if not hasattr(self, 'pruning_enabled') or not self.pruning_enabled:
            return None
        
        return {
            'enabled': self.pruning_enabled,
            'started': self.pruning_started,
            'start_epoch': self.pruning_actual_start_epoch,
            'end_epoch': self.pruning_end_epoch,
            'stable_detected': self.pruning_stable_detected,
            'final_sparsity_target': self.pruning_final_sparsity,
            'layer_stats': [
                {
                    'layer': f"Layer {i}" if i < self.n_layers else "Output",
                    'initial_connections': self.pruning_stats['initial_connections'][i],
                    'pruned_connections': self.pruning_stats['pruned_connections'][i],
                    'current_sparsity': self.pruning_stats['current_sparsity'][i]
                }
                for i in range(self.n_layers + 1)
            ]
        }

    def get_winner_selection_stats(self):
        """
        ★v039.3新機能★ 勝者選択統計を取得
        
        Returns:
            dict: {
                'counts': np.array,  # クラス別勝者選択回数
                'percentages': np.array,  # 勝者選択割合(%)
                'total_samples': int,  # 総学習サンプル数
                'expected_percentage': float  # 期待値（均等分布なら10.0%）
            }
        """
        if self.total_training_samples == 0:
            return {
                'counts': self.winner_selection_counts,
                'percentages': np.zeros(self.n_output),
                'total_samples': 0,
                'expected_percentage': 100.0 / self.n_output
            }
        
        percentages = (self.winner_selection_counts / self.total_training_samples) * 100.0
        return {
            'counts': self.winner_selection_counts,
            'percentages': percentages,
            'total_samples': self.total_training_samples,
            'expected_percentage': 100.0 / self.n_output
        }
    
    def reset_winner_selection_stats(self):
        """
        ★v039.3新機能★ 勝者選択統計をリセット（エポック開始時に使用）
        """
        self.winner_selection_counts = np.zeros(self.n_output, dtype=int)
        self.total_training_samples = 0
    
    def get_class_training_stats(self):
        """
        ★v039.5新機能★ 各クラスの学習実行回数統計を取得
        
        WTA無効化モード用。各クラスが実際に何回学習されたかを返す。
        
        Returns:
            dict: {
                'counts': np.array,  # クラス別学習実行回数
                'percentages': np.array,  # 学習実行割合(%)
                'total_samples': int,  # 総学習サンプル数
            }
        """
        total_training = np.sum(self.class_training_counts)
        if total_training == 0:
            return {
                'counts': self.class_training_counts,
                'percentages': np.zeros(self.n_output),
                'total_samples': 0
            }
        
        percentages = (self.class_training_counts / total_training) * 100.0
        return {
            'counts': self.class_training_counts,
            'percentages': percentages,
            'total_samples': total_training
        }
    
    def reset_class_training_stats(self):
        """
        ★v039.5新機能★ クラス学習回数統計をリセット
        """
        self.class_training_counts = np.zeros(self.n_output, dtype=int)
    
    # ============================================================================
    # ★v041新機能★ GPU完全バッチ処理版学習メソッド
    # ============================================================================
    
    def train_epoch_gpu_batch(self, x_train, y_train, batch_size=128):
        """
        ★v041新機能★ GPU完全バッチ処理版（v040 Numpy版の完全移植）
        
        v040のtrain_epoch_numpy_batch()の全ロジックをGPUに移植:
        - 統計収集（winner_selection_counts, class_training_counts）
        - 正確なアミン拡散（コラム構造、飽和抑制項）
        - 側方抑制の学習
        - WTA無効化モード完全対応
        
        Args:
            x_train: 訓練データ (n_samples, n_input)
            y_train: 訓練ラベル (n_samples,)
            batch_size: バッチサイズ（デフォルト: 128）
        
        Returns:
            accuracy: 訓練精度
            avg_loss: 平均損失
        """
        import tensorflow as tf
        
        # ============================================
        # 現在の制限事項チェック（v040と同じ）
        # ============================================
        if self.n_layers != 1:
            raise NotImplementedError(
                "train_epoch_gpu_batch() は現在1層ネットワークのみサポートしています。\n"
                f"現在の層数: {self.n_layers}\n"
                "多層ネットワークには従来のtrain_epoch_minibatch_tf()を使用してください。"
            )
        
        # ============================================
        # データの準備
        # ============================================
        n_samples = len(x_train)
        
        # シャッフル（epoch毎に異なる順序で学習）
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        total_loss = 0.0
        n_correct = 0
        
        # ============================================
        # バッチ学習ループ（v040と同じ構造、内部をGPU化）
        # ============================================
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            current_batch_size = end_idx - start_idx
            
            # ============================================
            # 勾配蓄積（GPU最適化版: バッチ順伝播）
            # ============================================
            
            # ★GPU最適化★ バッチ全体を一括で順伝播（GPU並列化）
            batch_grads, batch_stats = self._compute_batch_gradients_gpu(
                x_batch, y_batch
            )
            
            # 統計情報の収集
            for y_pred, y_true_val in batch_stats:
                self.winner_selection_counts[y_pred] += 1
                self.total_training_samples += 1
                
                # クラス別学習回数（純粋ED法: 正解のみ学習）
                self.class_training_counts[y_true_val] += 1
            
            # 損失と正解数の更新
            total_loss += batch_grads['loss']
            n_correct += batch_grads['n_correct']
            
            # ============================================
            # 勾配平均化と重み更新（v040検証済み方式）
            # ============================================
            self.w_output += batch_grads['w_output'] / current_batch_size
            self.lateral_weights += batch_grads['lateral_weights'] / current_batch_size
            for layer_idx in range(self.n_layers):
                self.w_hidden[layer_idx] += batch_grads['w_hidden'][layer_idx] / current_batch_size
        
        # Dale's Principle強制（第1層のみ）
        if self.n_layers > 0:
            sign_matrix = np.outer(self.ei_flags_hidden[0], self.ei_flags_input)
            self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
        
        # ============================================
        # 学習後の精度と損失
        # ============================================
        accuracy = n_correct / n_samples if n_samples > 0 else 0.0
        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
        
        return accuracy, avg_loss
    
    def _compute_batch_gradients_gpu(self, x_batch, y_batch):
        """
        バッチ全体の勾配を一括計算（GPU最適化版）
        
        各サンプルの勾配を個別に計算してから蓄積する方式。
        順伝播もGPUで並列化し、勾配計算もGPUで実行。
        統計情報も同時に収集。
        
        Args:
            x_batch: バッチデータ (batch_size, n_input)
            y_batch: バッチラベル (batch_size,)
        
        Returns:
            batch_grads: 蓄積された勾配の辞書
            batch_stats: 統計情報のリスト [(y_pred, y_true), ...]
        """
        import tensorflow as tf
        
        batch_size = len(x_batch)
        
        # バッチ全体を一括で順伝播（GPU並列化）
        x_batch_tf = tf.constant(x_batch, dtype=tf.float32)
        
        # 入力ペア（興奮性・抑制性）
        x_paired_batch_tf = tf.concat([x_batch_tf, x_batch_tf], axis=1)  # [batch, 2*n_input]
        
        # 隠れ層（1層のみ）
        w_hidden_tf = tf.constant(self.w_hidden[0], dtype=tf.float32)
        a_hidden_batch = tf.matmul(x_paired_batch_tf, w_hidden_tf, transpose_b=True)
        
        if self.activation == 'leaky_relu':
            z_hidden_batch = tf.where(
                a_hidden_batch > 0,
                a_hidden_batch,
                self.leaky_alpha * a_hidden_batch
            )
        else:  # tanh
            z_hidden_batch = tf.nn.tanh(a_hidden_batch)
        
        # 出力層
        w_output_tf = tf.constant(self.w_output, dtype=tf.float32)
        a_output_batch = tf.matmul(z_hidden_batch, w_output_tf, transpose_b=True)
        z_output_batch = tf.nn.softmax(a_output_batch)
        
        # Numpyに変換
        x_paired_batch_np = x_paired_batch_tf.numpy()
        z_hidden_batch_np = z_hidden_batch.numpy()
        z_output_batch_np = z_output_batch.numpy()
        
        # 損失・精度計算
        total_loss = 0.0
        n_correct = 0
        batch_stats = []
        
        # 勾配蓄積
        accumulated_grads = None
        
        for i in range(batch_size):
            x_paired = x_paired_batch_np[i]
            z_hiddens = [z_hidden_batch_np[i]]
            z_output = z_output_batch_np[i]
            y_true = y_batch[i]
            
            # 予測と損失
            y_pred = np.argmax(z_output)
            total_loss += cross_entropy_loss(z_output, y_true)
            n_correct += (y_pred == y_true)
            
            # 統計情報
            batch_stats.append((y_pred, y_true))
            
            # GPU勾配計算（個別サンプル）
            grads = self._compute_gradients_gpu(
                x_paired, z_hiddens, z_output, y_true
            )
            
            # 勾配を蓄積
            if accumulated_grads is None:
                accumulated_grads = {
                    'w_output': grads['w_output'].copy(),
                    'lateral_weights': grads['lateral_weights'].copy(),
                    'w_hidden': [g.copy() for g in grads['w_hidden']]
                }
            else:
                accumulated_grads['w_output'] += grads['w_output']
                accumulated_grads['lateral_weights'] += grads['lateral_weights']
                for layer_idx in range(len(grads['w_hidden'])):
                    accumulated_grads['w_hidden'][layer_idx] += grads['w_hidden'][layer_idx]
        
        # 結果を返す
        batch_grads = {
            'w_output': accumulated_grads['w_output'],
            'lateral_weights': accumulated_grads['lateral_weights'],
            'w_hidden': accumulated_grads['w_hidden'],
            'loss': total_loss,
            'n_correct': n_correct
        }
        
        return batch_grads, batch_stats
    
    def _compute_gradients_gpu(self, x_paired, z_hiddens, z_output, y_true):
        """
        GPU版勾配計算（v040のNumpy版を完全移植）
        
        TensorFlowを使用して勾配計算を高速化し、結果はNumpy配列で返す。
        アルゴリズムはv040の_compute_gradients()と完全に同一。
        
        Args:
            x_paired: 入力ベクトル（興奮性・抑制性ペア） - Numpy配列
            z_hiddens: 隠れ層出力のリスト - Numpy配列のリスト
            z_output: 出力層出力 - Numpy配列
            y_true: 正解ラベル - int
        
        Returns:
            gradients: 各層の勾配を含む辞書（Numpy配列）
        """
        import tensorflow as tf
        
        # Numpy配列をTensorFlowテンソルに変換
        x_paired_tf = tf.constant(x_paired, dtype=tf.float32)
        z_hiddens_tf = [tf.constant(z, dtype=tf.float32) for z in z_hiddens]
        z_output_tf = tf.constant(z_output, dtype=tf.float32)
        
        # ============================================
        # 1. 出力層の勾配計算（v040と同じ）
        # ============================================
        target_probs = np.zeros(self.n_output)
        target_probs[y_true] = 1.0
        target_probs_tf = tf.constant(target_probs, dtype=tf.float32)
        
        error_output_tf = target_probs_tf - z_output_tf
        
        saturation_output_tf = tf.abs(z_output_tf) * (1.0 - tf.abs(z_output_tf))
        
        # ★v039変更★ 出力層にも層別学習率を適用
        output_lr = self.layer_lrs[-1] if len(self.layer_lrs) > self.n_layers else self.learning_rate
        delta_w_output_tf = output_lr * tf.linalg.matmul(
            tf.expand_dims(error_output_tf * saturation_output_tf, 1),
            tf.expand_dims(z_hiddens_tf[-1], 0)
        )  # [n_output, n_hidden]
        
        # ============================================
        # 2. 出力層のアミン濃度計算と側方抑制（v040と同じ）
        # ============================================
        delta_lateral = np.zeros_like(self.lateral_weights)
        amine_concentration_output = np.zeros((self.n_output, 2))
        
        # 純粋ED法: 正解クラスのみ学習
        error_correct = 1.0 - z_output[y_true]
        if error_correct > 0:
            amine_concentration_output[y_true, 0] = error_correct * self.initial_amine
        
        # ============================================
        # 3. 隠れ層の勾配計算（多層アミン拡散、GPU化）
        # ============================================
        delta_w_hidden = []
        
        # TensorFlowテンソルに変換
        amine_concentration_output_tf = tf.constant(amine_concentration_output, dtype=tf.float32)
        
        for layer_idx in range(self.n_layers - 1, -1, -1):
            # 入力の取得
            if layer_idx == 0:
                z_input_tf = x_paired_tf
            else:
                z_input_tf = z_hiddens_tf[layer_idx - 1]
            
            # 拡散係数の選択
            # ★提案B★ uniform_amine=Trueの場合、全層u1で均一拡散（青斑核モデル）
            if self.uniform_amine:
                diffusion_coef = self.u1
            elif layer_idx == self.n_layers - 1:
                diffusion_coef = self.u1
            else:
                diffusion_coef = self.u2
            
            # コラム親和性をTensorFlowテンソルに変換
            column_affinity_tf = tf.constant(
                self.column_affinity_all_layers[layer_idx], 
                dtype=tf.float32
            )  # [n_output, n_hidden]
            
            # アミン拡散（GPU並列化）
            # amine_hidden_3d[class, amine_type, neuron]
            amine_mask_tf = amine_concentration_output_tf >= 1e-8
            amine_hidden_3d_tf = (
                tf.expand_dims(amine_concentration_output_tf, 2) *  # [n_output, 2, 1]
                diffusion_coef *
                tf.expand_dims(column_affinity_tf, 1)  # [n_output, 1, n_hidden]
            )
            amine_hidden_3d_tf = amine_hidden_3d_tf * tf.cast(
                tf.expand_dims(amine_mask_tf, 2),
                tf.float32
            )
            
            # 活性ニューロンの特定
            neuron_mask = tf.reduce_any(amine_hidden_3d_tf >= 1e-8, axis=[0, 1])
            active_neurons = tf.where(neuron_mask)[:, 0]
            
            if tf.size(active_neurons) == 0:
                delta_w_hidden.insert(0, np.zeros_like(self.w_hidden[layer_idx]))
                continue
            
            # 活性化関数の勾配（GPU並列化）
            z_active_tf = tf.gather(z_hiddens_tf[layer_idx], active_neurons)
            if self.activation == 'leaky_relu':
                saturation_term_raw_tf = tf.where(
                    z_active_tf > 0,
                    1.0,
                    self.leaky_alpha
                )
            else:
                saturation_term_raw_tf = tf.abs(z_active_tf) * (1.0 - tf.abs(z_active_tf))
            saturation_term_tf = tf.maximum(saturation_term_raw_tf, 1e-3)
            
            # 学習信号強度の計算（GPU並列化）
            layer_lr = self.layer_lrs[layer_idx]
            amine_hidden_active_tf = tf.gather(amine_hidden_3d_tf, active_neurons, axis=2)
            learning_signals_3d_tf = (
                layer_lr *
                amine_hidden_active_tf *
                tf.expand_dims(tf.expand_dims(saturation_term_tf, 0), 0)
            )
            
            # 勾配の計算（GPU並列化）
            n_combinations = self.n_output * 2
            n_active = tf.size(active_neurons)
            
            # learning_signals_2d: [n_active, n_combinations]
            learning_signals_2d_tf = tf.transpose(
                tf.reshape(learning_signals_3d_tf, [n_combinations, -1])
            )
            
            # z_input_tile: [n_active, n_input]
            z_input_tile_tf = tf.tile(
                tf.expand_dims(z_input_tf, 0),
                [n_active, 1]
            )
            
            # delta_w_active: [n_active, n_input]
            # learning_signals: [n_active, n_combinations]
            # z_input: [n_active, n_input]
            learning_signals_expanded_tf = tf.expand_dims(learning_signals_2d_tf, 2)  # [n_active, n_combinations, 1]
            z_input_expanded_tf = tf.expand_dims(z_input_tile_tf, 1)  # [n_active, 1, n_input]
            delta_w_active_tf = tf.reduce_sum(
                learning_signals_expanded_tf * z_input_expanded_tf,
                axis=1
            )  # [n_active, n_input]
            
            # フルサイズの勾配行列を構築（Numpyで実行）
            dw = np.zeros_like(self.w_hidden[layer_idx])
            active_neurons_np = active_neurons.numpy()
            delta_w_active_np = delta_w_active_tf.numpy()
            dw[active_neurons_np, :] = delta_w_active_np
            delta_w_hidden.insert(0, dw)
        
        return {
            'w_output': delta_w_output_tf.numpy(),
            'lateral_weights': delta_lateral,
            'w_hidden': delta_w_hidden
        }
    
    def _process_batch_gpu(self, x_batch, y_batch, w_hidden_tf, w_output_tf, lateral_weights_tf):
        """
        バッチ全体の順伝播・勾配計算を一括実行（GPU並列化）
        
        これがv040で諦めていた「バッチ順伝播」の実装！
        
        Args:
            x_batch: バッチデータ [batch, n_input]
            y_batch: バッチラベル [batch]
            w_hidden_tf: 隠れ層重み（TensorFlowテンソルのリスト）
            w_output_tf: 出力層重み [n_output, n_hidden]
            lateral_weights_tf: 側方重み [n_output, n_output]
        
        Returns:
            total_loss: バッチ全体の損失
            n_correct: 正解数
            grads: 勾配の辞書
        """
        import tensorflow as tf
        
        batch_size = tf.shape(x_batch)[0]
        
        # ============================================
        # 1. バッチ順伝播（完全並列化）
        # ============================================
        
        # 入力ペア（興奮性・抑制性）
        # ★重要★ create_ei_pairs()と同じく、両方に同じ値を設定
        # Dale's Principleが重みに適用されることで抑制効果が生まれる
        x_paired_batch = tf.concat([x_batch, x_batch], axis=1)  # [batch, 2*n_input]
        
        # 隠れ層の順伝播（多層対応）
        z_hiddens_batch = []
        z_current = x_paired_batch
        
        for layer_idx in range(self.n_layers):
            # 行列積（GPU並列化）
            a_hidden = tf.matmul(z_current, w_hidden_tf[layer_idx], transpose_b=True)
            
            # 活性化関数
            if self.activation == 'leaky_relu':
                z_hidden = tf.where(
                    a_hidden > 0,
                    a_hidden,
                    tf.constant(self.leaky_alpha, dtype=tf.float32) * a_hidden
                )
            else:  # tanh（デフォルト）
                z_hidden = tf.nn.tanh(a_hidden)
            
            z_hiddens_batch.append(z_hidden)
            z_current = z_hidden
        
        # 出力層
        a_output = tf.matmul(z_current, w_output_tf, transpose_b=True)
        z_output_batch = tf.nn.softmax(a_output)  # [batch, n_output]
        
        # ============================================
        # 2. 損失・精度計算
        # ============================================
        
        # 損失
        target_probs = tf.one_hot(y_batch, depth=self.n_output, dtype=tf.float32)
        cross_entropy = -tf.reduce_sum(
            target_probs * tf.math.log(z_output_batch + 1e-10),
            axis=1
        )
        total_loss = tf.reduce_sum(cross_entropy)
        
        # 精度
        y_pred_batch = tf.argmax(z_output_batch, axis=1)
        y_batch_int64 = tf.cast(y_batch, tf.int64)
        n_correct = tf.reduce_sum(
            tf.cast(tf.equal(y_pred_batch, y_batch_int64), tf.int32)
        )
        
        # ============================================
        # 3. 勾配計算（ED法準拠、バッチ版）
        # ============================================
        
        # 出力誤差
        error_output_batch = target_probs - z_output_batch  # [batch, n_output]
        
        # 純粋ED法: 正解クラスのみ学習
        # 正解クラス以外の誤差をゼロにする
        error_output_batch = error_output_batch * target_probs
        
        # 出力層勾配（バッチ合計）
        z_last = z_hiddens_batch[-1] if self.n_layers > 0 else x_paired_batch
        delta_w_output = tf.constant(self.learning_rate, dtype=tf.float32) * tf.matmul(
            error_output_batch,
            z_last,
            transpose_a=True
        )  # [n_output, n_hidden]
        
        # 側方抑制勾配（簡略版: 現時点では更新なし）
        delta_lateral = tf.zeros_like(lateral_weights_tf)
        
        # 隠れ層勾配（アミン拡散、逆方向に伝播）
        delta_w_hidden = []
        amine_concentration = error_output_batch  # [batch, n_output]
        
        for layer_idx in range(self.n_layers - 1, -1, -1):
            # アミン拡散（バッチ版）
            if layer_idx == self.n_layers - 1:
                amine_hidden = tf.matmul(amine_concentration, w_output_tf)
            else:
                amine_hidden = tf.matmul(amine_concentration, w_hidden_tf[layer_idx + 1])
            
            # 拡散係数
            # ★提案B★ uniform_amine=Trueの場合、全層u1で均一拡散
            if self.uniform_amine:
                u_coef = tf.constant(self.u1, dtype=tf.float32)
            else:
                u_coef = tf.constant(self.u1 if layer_idx == 0 else self.u2, dtype=tf.float32)
            amine_diffused = amine_hidden * u_coef  # [batch, n_hidden]
            
            # 勾配計算
            z_input = x_paired_batch if layer_idx == 0 else z_hiddens_batch[layer_idx - 1]
            layer_lr = self.layer_lrs[layer_idx] if self.layer_lrs is not None else self.learning_rate
            
            delta_w = tf.constant(layer_lr, dtype=tf.float32) * tf.matmul(
                amine_diffused,
                z_input,
                transpose_a=True
            )  # [n_hidden, n_input]
            
            delta_w_hidden.insert(0, delta_w)
            amine_concentration = amine_diffused
        
        # 勾配を辞書にまとめる
        grads = {
            'w_output': delta_w_output,
            'lateral_weights': delta_lateral,
            'w_hidden': delta_w_hidden
        }
        
        return total_loss, n_correct, grads
