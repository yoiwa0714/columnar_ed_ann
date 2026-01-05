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
        column_radius=config['column_radius']
    )
    
    # 学習
    network.train_one_sample(x, y_true)
    
    # 評価
    accuracy, loss = network.evaluate(x_test, y_test)
"""

import numpy as np
from .activation_functions import sigmoid, tanh_activation, softmax, softmax_batch, cross_entropy_loss
from .neuron_structure import create_ei_pairs, create_ei_pairs_batch, create_ei_flags
from .amine_diffusion import (
    initialize_activity_association,
    update_activity_association,
    distribute_amine_by_output_weights
)
from .column_structure import (
    hex_distance,
    create_lateral_weights,
    create_column_membership,
    create_column_membership_circular
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
                 learning_rate=0.20, u1=0.5, u2=0.8,
                 column_neurons=None, participation_rate=None,
                 use_hexagonal=True, use_circular=False, overlap=0.0,
                 use_layer_norm=False, hyperparams=None,
                 weight_init_scales=None, weight_init_source=None,
                 winner_suppression_factor=1.0, weight_decay=0.0, bias_diversity=0.0,
                 weight_scale_diversity=None, sign_pattern_diversity=False, weight_sparsity=0.0,
                 cluster_init_groups=0, top_k_winners=1, column_weight_diversity=0.0,
                 use_ridge_regression=False):
        """
        初期化
        
        Args:
            n_input: 入力次元数（784 for MNIST）
            n_hidden: 隠れ層ニューロン数のリスト（例: [256] or [256, 128]）
            n_output: 出力クラス数
            learning_rate: 学習率（Phase 1 Extended Overall Best: 0.20）
            u1: アミン拡散係数（Phase 1 Extended Overall Best: 0.5）
            u2: アミン拡散係数（隠れ層間、デフォルト0.8）
            column_neurons: 各クラスのコラムに割り当てるニューロン数（participation_rateと排他的、デフォルト：HyperParamsから1）
            participation_rate: コラム参加率（0.0-1.0、column_neuronsと排他的、デフォルト：None）
            use_hexagonal: Trueならハニカム構造、Falseなら旧円環構造
            overlap: コラム間の重複度（0.0-1.0、円環構造でのみ有効、デフォルト0.0）
            top_k_winners: 各コラムで学習する上位ニューロン数（デフォルト1=Winner-Take-All、3推奨=協調学習）
            hyperparams: HyperParamsインスタンス（Noneなら個別パラメータ使用）
            weight_init_scales: 重み初期化係数のリスト（[Layer0, Layer1, ..., 出力層]、Noneなら自動取得）
            weight_init_source: 値の出処（'CLI', 'HyperParams', 'デフォルト値'）
        
        HyperParams統合の使用例:
            # パターン1: HyperParamsを使用
            hp = HyperParams()
            config = hp.get_config(n_layers=2)
            network = RefinedDistributionEDNetwork(
                n_input=784,
                n_hidden=config['hidden'],
                learning_rate=config['learning_rate'],
                hyperparams=hp  # HyperParamsインスタンスを渡す
            )
            
            # パターン2: 個別パラメータ指定（従来通り）
            network = RefinedDistributionEDNetwork(
                n_input=784,
                n_hidden=[256, 128],
                learning_rate=0.05,
                column_neurons=1  # リザバーコンピューティング最適値
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
            except ValueError as e:
                print(f"Warning: {e}")
                print("個別パラメータを使用します。")
        
        # パラメータ保存
        self.n_input = n_input
        self.n_hidden = n_hidden if isinstance(n_hidden, list) else [n_hidden]
        self.n_layers = len(self.n_hidden)
        self.n_output = n_output
        self.learning_rate = learning_rate
        
        # ★新機能★ 層ごとの適応的学習率（z_inputとsaturation_termのスケールに対応）
        # 分析結果：
        #   - Layer 1の平均|Δw|: 0.120520
        #   - Layer 0の平均|Δw|: 0.407937（3.54倍）
        #   - 原因1: 最大|z_input|の差（Layer 0: 1.0, Layer 1: 0.5-0.8）
        #   - 原因2: saturation_termの差（Layer 0が極端に小さい: 1/17）
        # 修正不要：saturation_termとz_inputの自然な違いを保持
        # → 層ごとの学習率は同じに戻す
        self.layer_specific_lr = [learning_rate] * self.n_layers
        
        self.winner_suppression_factor = winner_suppression_factor  # 勝者からの抑制削減率（デッドニューロン対策）
        self.weight_decay = weight_decay  # 重み減衰率（L2正則化）
        self.bias_diversity = bias_diversity  # バイアス多様化範囲
        self.weight_scale_diversity = weight_scale_diversity  # マルチスケール初期化のスケールリスト
        self.sign_pattern_diversity = sign_pattern_diversity  # 符号パターン多様化
        self.weight_sparsity = weight_sparsity  # スパース初期化の稀疎度
        self.cluster_init_groups = cluster_init_groups  # クラスタベース初期化のグループ数
        self.top_k_winners = top_k_winners  # ★新機能★ 上位K個学習（協調学習対応）
        self.column_weight_diversity = column_weight_diversity  # ★新機能★ コラムメンバーごとの重み多様性（0.0=無効、0.3推奨）
        self.use_ridge_regression = use_ridge_regression  # ★最適化★ Ridge回帰使用（ELM/RC標準）
        self.ridge_lambda = 1e-3  # Ridge回帰の正則化パラメータ（固定値）
        self.u1 = u1  # アミン拡散係数（出力層→最終隠れ層）
        self.u2 = u2  # アミン拡散係数（隠れ層間）
        self.initial_amine = 1.0  # 基準アミン濃度
        
        self.column_neurons = column_neurons
        self.participation_rate = participation_rate
        self.use_hexagonal = use_hexagonal
        self.use_circular = use_circular
        self.use_layer_norm = use_layer_norm  # 層間正規化
        
        # 側方抑制（必須要素6）- ゼロ初期化、学習中に動的更新
        self.lateral_weights = create_lateral_weights(n_output)
        
        # ★新方式★ コラムメンバーシップフラグの初期化（学習可能化対応）
        # Membership方式: 所属情報のみ、重みは学習可能
        self.column_membership_all_layers = []
        
        for layer_idx, layer_size in enumerate(self.n_hidden):
            # メンバーシップフラグ作成
            if use_circular:
                # 2次元円環配置
                membership = create_column_membership_circular(
                    n_hidden=layer_size,
                    n_classes=n_output,
                    participation_rate=participation_rate,
                    column_neurons=column_neurons
                )
            else:
                # ハニカム構造
                membership = create_column_membership(
                    n_hidden=layer_size,
                    n_classes=n_output,
                    participation_rate=participation_rate,
                    use_hexagonal=use_hexagonal,
                    column_neurons=column_neurons
                )
            self.column_membership_all_layers.append(membership)
        
        print(f"\n[コラム構造初期化]")
        print(f"  - ★新方式★ メンバーシップフラグ使用（学習可能化対応）")
        print(f"  - コラム所属: ブールフラグ（固定）")
        print(f"  - 勝者決定: 重みベース（学習可能）")
        if use_circular:
            print(f"  - 方式: 2次元円環配置（円周上に等角度間隔配置）")
            if column_neurons is not None:
                total_column_neurons = column_neurons * n_output
                participation_pct = total_column_neurons / self.n_hidden[0] * 100
                print(f"  - モード: 完全コラム化（各クラス{column_neurons}ニューロン）")
                print(f"  - コラム化ニューロン数: {total_column_neurons}個（全{self.n_hidden[0]}個中、{participation_pct:.1f}%）")
            elif participation_rate is not None:
                print(f"  - モード: 参加率指定（{participation_rate * 100:.0f}%）")
                print(f"  - 各クラス約{int(self.n_hidden[0] * participation_rate / n_output)}ニューロン")
        else:
            print(f"  - 方式: ハニカム構造(2-3-3-2配置)")
            if column_neurons is not None:
                total_column_neurons = column_neurons * n_output
                participation_pct = total_column_neurons / self.n_hidden[0] * 100
                print(f"  - モード: 完全コラム化（各クラス{column_neurons}ニューロン）")
                print(f"  - コラム化ニューロン数: {total_column_neurons}個（全{self.n_hidden[0]}個中、{participation_pct:.1f}%）")
            elif participation_rate is not None:
                print(f"  - モード: 参加率指定（{participation_rate * 100:.0f}%）")
                print(f"  - 各クラス約{int(self.n_hidden[0] * participation_rate / n_output)}ニューロン")
            else:
                print(f"  - モード: 半径ベース（radius={self.column_radius_per_layer[0]:.2f}）")
        
        for layer_idx, membership in enumerate(self.column_membership_all_layers):
            member_counts = [int(np.sum(membership[c])) for c in range(n_output)]
            print(f"  - 層{layer_idx+1}: 各クラスのメンバーニューロン数={member_counts}")
            
            # デバッグ: 円環でcolumn_neurons=1の場合、選択ニューロン位置を表示
            if use_circular and column_neurons == 1:
                selected_indices = []
                for c in range(n_output):
                    idx = np.where(membership[c])[0]
                    if len(idx) > 0:
                        selected_indices.append(int(idx[0]))
                print(f"    → 選択ニューロン位置: {selected_indices}")
                
                # 2次元座標も表示
                grid_size = int(np.ceil(np.sqrt(self.n_hidden[layer_idx])))
                coords_2d = [(idx // grid_size, idx % grid_size) for idx in selected_indices]
                print(f"    → 2D座標: {coords_2d}")
        
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
        # ★対策1★ 層ごとに異なる初期化スケールを使用
        # Layer 0: 入力次元が大きい(784*2=1568) → 小さいスケールで飽和を防ぐ
        # Layer 1+: 標準的なXavier初期化
        
        # 重み初期化係数の解決（優先順位: 引数 > HyperParams > デフォルト値）
        resolved_scales = weight_init_scales
        resolved_source = weight_init_source if weight_init_source else "デフォルト値"
        
        if resolved_scales is None and hyperparams is not None:
            # 引数で指定されていない場合、HyperParamsから取得を試みる
            try:
                config = hyperparams.get_config(self.n_layers)
                resolved_scales = config.get('weight_init_scales', None)
                if resolved_scales is not None:
                    resolved_source = "HyperParams"
                    print(f"  [重み初期化] HyperParamsテーブルから係数を取得: {resolved_scales}")
            except Exception as e:
                print(f"  [重み初期化] HyperParams取得エラー: {e}、デフォルト値を使用")
        elif resolved_scales is not None:
            # 引数で指定されている場合
            print(f"  [重み初期化] {resolved_source}から係数を取得: {resolved_scales}")
        
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
            
            # 層ごとの適応的スケール
            if resolved_scales is not None:
                # CLI/HyperParamsから取得
                scale_coef = resolved_scales[layer_idx]
                scale = np.sqrt(1.0 / n_in) * scale_coef
                print(f"  [重み初期化] Layer {layer_idx}: scale={scale:.4f} (係数{scale_coef:.2f}, {resolved_source})")
            else:
                # デフォルト値（後方互換性）
                if layer_idx == 0:
                    # Layer 0: デフォルト 1.80
                    scale_coef = 1.80
                    scale = np.sqrt(1.0 / n_in) * scale_coef
                    print(f"  [重み初期化] Layer {layer_idx}: scale={scale:.4f} (係数{scale_coef:.2f}, デフォルト)")
                else:
                    # Layer 1+: デフォルト 3.00
                    scale_coef = 3.00
                    scale = np.sqrt(1.0 / n_in) * scale_coef
                    print(f"  [重み初期化] Layer {layer_idx}: scale={scale:.4f} (係数{scale_coef:.2f}, デフォルト)")
            
            w = np.random.randn(n_out, n_in) * scale
            self.w_hidden.append(w)
        
        # 出力層の重み
        if resolved_scales is not None:
            # CLI/HyperParamsから取得（最後の要素が出力層）
            output_scale_coef = resolved_scales[-1]
            output_scale = np.sqrt(1.0 / self.n_hidden[-1]) * output_scale_coef
            print(f"  [重み初期化] 出力層: scale={output_scale:.4f} (係数{output_scale_coef:.2f}, {resolved_source})")
        else:
            # デフォルト値（後方互換性）
            output_scale_coef = 12.00
            output_scale = np.sqrt(1.0 / self.n_hidden[-1]) * output_scale_coef
            print(f"  [重み初期化] 出力層: scale={output_scale:.4f} (係数{output_scale_coef:.2f}, デフォルト)")
        
        self.w_output = np.random.randn(n_output, self.n_hidden[-1]) * output_scale
        
        # マルチスケール初期化（隠れ層の重みに適用）
        if self.weight_scale_diversity is not None and len(self.weight_scale_diversity) > 0:
            print(f"\n[マルチスケール初期化]")
            for layer_idx in range(self.n_layers):
                n_neurons = self.n_hidden[layer_idx]
                scales = self.weight_scale_diversity
                n_groups = len(scales)
                neurons_per_group = n_neurons // n_groups
                
                for group_idx, scale in enumerate(scales):
                    start_idx = group_idx * neurons_per_group
                    if group_idx == n_groups - 1:
                        # 最後のグループは残り全部
                        end_idx = n_neurons
                    else:
                        end_idx = start_idx + neurons_per_group
                    
                    # グループのニューロンに対してスケール適用
                    self.w_hidden[layer_idx][start_idx:end_idx, :] *= scale
                
                print(f"  Layer {layer_idx}: {n_groups}グループ、スケール={scales}, 各グループ約{neurons_per_group}ニューロン")
        
        # 符号パターン多様化（隠れ層の重みに適用）
        if self.sign_pattern_diversity:
            print(f"\n[符号パターン多様化]")
            for layer_idx in range(self.n_layers):
                n_neurons = self.n_hidden[layer_idx]
                # 奇数番号のニューロン（インデックス 1, 3, 5, ...）の重みを反転
                for neuron_idx in range(1, n_neurons, 2):
                    self.w_hidden[layer_idx][neuron_idx, :] *= -1
                
                n_inverted = (n_neurons + 1) // 2  # 奇数番号の個数
                print(f"  Layer {layer_idx}: {n_neurons}ニューロン中 {n_inverted}個を反転 (奇数番号)")
        
        # スパース初期化(隠れ層の重みに適用)
        if self.weight_sparsity > 0:
            print(f"\n[スパース初期化]")
            for layer_idx in range(self.n_layers):
                n_neurons = self.n_hidden[layer_idx]
                n_inputs = self.w_hidden[layer_idx].shape[1]
                total_weights = n_neurons * n_inputs
                n_zero = int(total_weights * self.weight_sparsity)
                
                # ランダムに weight_sparsity の割合をゼロに
                mask = np.ones(total_weights, dtype=bool)
                zero_indices = np.random.choice(total_weights, n_zero, replace=False)
                mask[zero_indices] = False
                mask = mask.reshape(n_neurons, n_inputs)
                self.w_hidden[layer_idx] *= mask
                
                print(f"  Layer {layer_idx}: {total_weights}重み中 {n_zero}個をゼロ化 ({self.weight_sparsity*100:.0f}%)")
        
        # クラスタベース初期化(隠れ層の重みに適用)
        if self.cluster_init_groups > 0:
            print(f"\n[クラスタベース初期化]")
            for layer_idx in range(self.n_layers):
                n_neurons = self.n_hidden[layer_idx]
                neurons_per_group = n_neurons // self.cluster_init_groups
                
                for group_idx in range(self.cluster_init_groups):
                    start_idx = group_idx * neurons_per_group
                    if group_idx == self.cluster_init_groups - 1:
                        # 最後のグループは残り全て
                        end_idx = n_neurons
                    else:
                        end_idx = start_idx + neurons_per_group
                    
                    # 各グループに異なる乱数シードを適用して初期化
                    seed = 42 + group_idx  # ベースシード42から始める
                    np.random.seed(seed)
                    n_inputs = self.w_hidden[layer_idx].shape[1]
                    self.w_hidden[layer_idx][start_idx:end_idx, :] = np.random.randn(end_idx - start_idx, n_inputs) * np.sqrt(2.0 / n_inputs)
                
                print(f"  Layer {layer_idx}: {n_neurons}ニューロンを{self.cluster_init_groups}グループに分割（各{neurons_per_group}ニューロン）")
            
            # 乱数シードをリセット
            np.random.seed()
        
        # ★新機能★ コラムメンバーごとの重み多様化（Top-K学習の効果を最大化）
        # 目的: 同じコラム内のニューロンが異なる初期重みを持つことで、
        #      入力に対する応答パターンが多様化し、異なるTop-Kが選ばれるようになる
        # 効果: Gini係数の改善（0.9→0.3）、活性ニューロン率の向上（8%→40%+）
        if self.column_weight_diversity > 0 and hasattr(self, 'column_membership_all_layers'):
            print(f"\n[★新機能★ コラムメンバー重み多様化]")
            for layer_idx in range(self.n_layers):
                membership = self.column_membership_all_layers[layer_idx]  # [n_classes, n_hidden]
                n_neurons = self.n_hidden[layer_idx]
                
                # 各クラスのメンバーニューロンに異なるスケールを適用
                for class_idx in range(self.n_output):
                    member_neurons = np.where(membership[class_idx])[0]
                    n_members = len(member_neurons)
                    
                    if n_members == 0:
                        continue
                    
                    # 各メンバーに異なるスケール係数を適用
                    # 範囲: [1.0 - diversity, 1.0 + diversity]
                    # 例: diversity=0.3 → [0.7, 1.3]
                    scale_factors = np.random.uniform(
                        1.0 - self.column_weight_diversity,
                        1.0 + self.column_weight_diversity,
                        n_members
                    )
                    
                    # 重みに適用
                    for i, neuron_idx in enumerate(member_neurons):
                        self.w_hidden[layer_idx][neuron_idx, :] *= scale_factors[i]
                
                print(f"  Layer {layer_idx}: 各クラスのメンバーに多様性={self.column_weight_diversity:.2f} 適用")
                print(f"    スケール範囲=[{1.0-self.column_weight_diversity:.2f}, {1.0+self.column_weight_diversity:.2f}]")
        
        # バイアスの初期化（隠れ層のみ）
        self.b_hidden = []
        for layer_idx, layer_size in enumerate(self.n_hidden):
            if self.bias_diversity > 0:
                # バイアス多様化: [-bias_diversity, +bias_diversity]の一様分布
                bias = np.random.uniform(-self.bias_diversity, self.bias_diversity, layer_size)
                print(f"  [バイアス初期化] Layer {layer_idx}: 範囲=[{-self.bias_diversity:.1f}, {self.bias_diversity:.1f}], 平均={np.mean(bias):.3f}, 標準偏差={np.std(bias):.3f}")
            else:
                # バイアスなし（従来通り）
                bias = np.zeros(layer_size)
            self.b_hidden.append(bias)
        
        # Dale's Principleの初期化（必須要素1）- 第1層のみ
        sign_matrix = np.outer(self.ei_flags_hidden[0], self.ei_flags_input)
        self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
        
        # アミン濃度の記憶領域（必須要素3）- 各層ごと
        self.amine_concentrations = []
        for layer_size in self.n_hidden:
            amine = np.zeros((n_output, layer_size, 2))
            self.amine_concentrations.append(amine)
        
        # ★勝者独り勝ち防止★ ニューロン疲労メカニズム
        # 各層・各クラスのニューロンごとの疲労度を追跡
        self.neuron_fatigue = []
        for layer_size in self.n_hidden:
            # [n_classes, layer_size]の形状
            fatigue = np.zeros((n_output, layer_size))
            self.neuron_fatigue.append(fatigue)
        
        # 疲労メカニズムのハイパーパラメータ
        self.fatigue_accumulation = 0.2   # 勝利時の疲労蓄積量（調整：0.1→0.2）
        self.fatigue_recovery = 0.90      # 疲労回復率（毎サンプル、調整：0.95→0.90）
        self.fatigue_enabled = False      # 疲労メカニズムの無効化（自然競争テスト）
        
        # ★ED法アミン拡散★ 活性度ベース関係性マップの初期化
        self.activity_association_maps = initialize_activity_association(
            n_classes=n_output,
            hidden_sizes=self.n_hidden
        )
        print(f"\n[アミン拡散機構初期化]")
        print(f"  - 活性度ベース関係性マップ作成完了")
        print(f"  - 層数: {len(self.activity_association_maps)}")
        
        if self.fatigue_enabled:
            print(f"\n[勝者独り勝ち防止機能]")
            print(f"  - 疲労メカニズム: 有効")
            print(f"  - 疲労蓄積率: {self.fatigue_accumulation}")
            print(f"  - 疲労回復率: {self.fatigue_recovery}")
        
        # ★ニューロンレベルトラッキング★（コラム競合分析用）
        self.enable_neuron_tracking = False  # デフォルトでは無効
        self.neuron_stats = []
        for layer_idx, layer_size in enumerate(self.n_hidden):
            layer_stats = {
                'winner_count': np.zeros((self.n_output, layer_size)),  # [クラス, ニューロン]
                'activation_sum': np.zeros((self.n_output, layer_size)),
                'weight_update_count': np.zeros((self.n_output, layer_size)),
                'last_epoch_stats': None  # 前エポックの統計を保存
            }
            self.neuron_stats.append(layer_stats)
    
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
            a_hidden = np.dot(self.w_hidden[layer_idx], z_current) + self.b_hidden[layer_idx]
            
            # 活性化関数: tanh（固定）
            z_hidden = tanh_activation(a_hidden)
            
            z_hiddens.append(z_hidden)
            z_current = z_hidden
            
            # ★対策5★ 層間正規化（アミン伝播の安定化）
            if self.use_layer_norm:
                mean = np.mean(z_current)
                std = np.std(z_current) + 1e-8
                z_current = (z_current - mean) / std
        
        # 出力層の計算（★SoftMax活性化★）
        a_output = np.dot(self.w_output, z_hiddens[-1])
        z_output = softmax(a_output)  # ★変更: sigmoid → softmax★
        
        # 注意: 側方抑制は順伝播では適用せず、学習時の確率ベース競合で実現
        
        return z_hiddens, z_output, x_paired
    
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
        
        # 勾配適用
        self.w_output += gradients['w_output']
        self.lateral_weights += gradients['lateral_weights']
        
        for layer_idx in range(self.n_layers):
            self.w_hidden[layer_idx] += gradients['w_hidden'][layer_idx]
            self.b_hidden[layer_idx] += gradients['b_hidden'][layer_idx]
            
            # 第1層のDale's Principle強制
            if layer_idx == 0:
                sign_matrix = np.outer(self.ei_flags_hidden[0], self.ei_flags_input)
                self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
        
        # 出力重みの正則化
        weight_penalty = 0.00001 * self.w_output
        self.w_output -= weight_penalty
        
        # ★ED法アミン拡散★ アミン濃度の更新
        # 注: 2層以上の場合は形状の問題があるため一時的にコメントアウト
        # self._update_amine_concentrations(z_hiddens, z_output, y_true)
    
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
    
    def _update_amine_concentrations(self, z_hiddens, z_output, y_true):
        """
        アミン濃度の更新（ED法準拠）
        
        Args:
            z_hiddens: 各隠れ層の出力のリスト
            z_output: 出力層の確率分布
            y_true: 正解クラス
        """
        # 出力誤差の計算
        error_output = np.zeros(self.n_output)
        error_output[y_true] = 1.0 - z_output[y_true]  # 正解クラスの誤差
        for c in range(self.n_output):
            if c != y_true:
                error_output[c] = -z_output[c]  # 不正解クラスの誤差
        
        # 各層のアミン濃度を更新
        for layer_idx in range(self.n_layers):
            # 活性度ベース関係性の更新
            self.activity_association_maps[layer_idx] = update_activity_association(
                association_map=self.activity_association_maps[layer_idx],
                class_idx=y_true,
                hidden_activations=z_hiddens[layer_idx]
            )
            
            # 出力重みベースのアミン配分
            exc_amine, inh_amine = distribute_amine_by_output_weights(
                w_output=self.w_output,
                error_output=error_output,
                z_hidden=z_hiddens[layer_idx]
            )
            
            # アミン濃度を保存（各クラス×各ニューロン×[興奮性, 抑制性]）
            for c in range(self.n_output):
                # 関係性マップで重み付け
                association_weight = self.activity_association_maps[layer_idx][c]
                self.amine_concentrations[layer_idx][c, :, 0] = exc_amine * association_weight
                self.amine_concentrations[layer_idx][c, :, 1] = inh_amine * association_weight
    
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
        
        # ◆変更◆ Cross-Entropy損失計算
        loss = cross_entropy_loss(z_output, y_true)
        
        # 重みの更新（アミン濃度更新を含む）
        self.update_weights(x_paired, z_hiddens, z_output, y_true)
        
        # ★勝者独り勝ち防止★ 疲労の回復処理
        if self.fatigue_enabled:
            for layer_idx in range(self.n_layers):
                # 全ニューロンの疲労を徐々に回復
                self.neuron_fatigue[layer_idx] *= self.fatigue_recovery
        
        return loss, correct
    
    def train_readout_ridge(self, x_train, y_train):
        """
        Ridge回帰による出力層の一括学習（ELM/RC標準手法）
        
        利点:
        - 解析的解法（1回で最適解）
        - 学習率不要
        - 過学習抑制（正則化パラメータλ）
        - 学習時間大幅短縮
        
        Args:
            x_train: 訓練データ [n_samples, n_input]
            y_train: 正解ラベル [n_samples]
        
        Returns:
            loss: 平均損失
            accuracy: 正解率
        """
        n_samples = len(x_train)
        
        # 1. リザーバ活性化を収集（順伝播）
        H = []  # リザーバ活性化行列 [n_samples, n_reservoir]
        for i in range(n_samples):
            z_hiddens, _, _ = self.forward(x_train[i])
            H.append(z_hiddens[-1])  # 最終層の活性化
        H = np.array(H)
        
        # 2. 目標出力をone-hot化
        T = np.zeros((n_samples, self.n_output))
        for i, y in enumerate(y_train):
            T[i, y] = 1.0
        
        # 3. Ridge回帰の解析解
        # W_out = (H^T H + λI)^{-1} H^T T
        n_reservoir = H.shape[1]
        I = np.eye(n_reservoir)
        
        try:
            # 数値安定性を考慮した解法
            HTH = H.T @ H
            HTT = H.T @ T
            w_output_ridge = np.linalg.solve(HTH + self.ridge_lambda * I, HTT)
            # forwardメソッドとの互換性のため転置して保存
            self.w_output = w_output_ridge.T
        except np.linalg.LinAlgError:
            # 特異行列の場合は擬似逆行列を使用
            print("  [警告] Ridge回帰で特異行列を検出、擬似逆行列を使用")
            w_output_ridge = np.linalg.pinv(H) @ T
            self.w_output = w_output_ridge.T
        
        # 4. 精度評価
        predictions = H @ w_output_ridge
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == y_train)
        
        # 5. 損失計算（二乗誤差）
        errors = T - predictions
        loss = np.mean(np.sum(errors**2, axis=1))
        
        return loss, accuracy
    
    def train_epoch(self, x_train, y_train, return_true_accuracy=True):
        """
        1エポックの学習
        
        Args:
            x_train: 訓練データ
            y_train: 訓練ラベル
            return_true_accuracy: Trueの場合、学習完了後に全データを再評価して真の訓練精度を返す
                                 Falseの場合、学習中の平均精度を返す（従来の動作、非推奨）
        
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
        n_samples = len(x_train)
        total_loss = 0.0
        n_correct = 0
        
        # 学習ループ
        for i in range(n_samples):
            loss, correct = self.train_one_sample(x_train[i], y_train[i])
            total_loss += loss
            if correct:
                n_correct += 1
        
        # 学習中の平均精度と損失
        training_accuracy = n_correct / n_samples
        avg_loss = total_loss / n_samples
        
        if return_true_accuracy:
            # 学習完了後、全訓練データを最終重みで再評価（推奨）
            # これにより、Test精度と同じ条件で評価される
            true_accuracy, true_loss = self.evaluate(x_train, y_train)
            return true_accuracy, true_loss
        else:
            # 学習中の平均精度を返す（従来動作、非推奨）
            return training_accuracy, avg_loss
    
    def evaluate(self, x_test, y_test):
        """
        テストデータでの評価
        
        Returns:
            accuracy: テスト精度
            loss: 平均損失
        """
        n_samples = len(x_test)
        total_loss = 0.0
        n_correct = 0
        
        for i in range(n_samples):
            # 順伝播のみ
            z_hiddens, z_output, _ = self.forward(x_test[i])
            
            # 予測
            y_pred = np.argmax(z_output)
            if y_pred == y_test[i]:
                n_correct += 1
            
            # ◆変更◆ Cross-Entropy損失計算
            loss = cross_entropy_loss(z_output, y_test[i])
            total_loss += loss
        
        accuracy = n_correct / n_samples
        avg_loss = total_loss / n_samples
        
        return accuracy, avg_loss
    
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
            
            # 活性化関数: tanh（固定）
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
    
    def train_epoch_minibatch_tf(self, train_dataset, x_train=None, y_train=None, return_true_accuracy=True):
        """
        ミニバッチ学習版エポック（TensorFlow Dataset API使用、勾配平均化方式）
        
        この関数はTensorFlow Data API (tf.data.Dataset) を使用します。
        これは業界標準の手法であり、以下の利点があります：
          1. データ処理パイプラインの信頼性が国際的に認知されている
          2. シャッフル機能が最適化され、再現性が保証される
          3. バッチ処理が効率的に実行される
        
        **v030修正（2025-12-28）**:
          標準的なミニバッチ学習の実装（勾配平均化方式 - PyTorch方式）
          
          実装方針:
          - バッチ内の全サンプルの勾配を蓄積
          - 勾配を平均化（÷ batch_size）
          - 平均化された勾配で一度だけ重み更新
          - **学習率スケーリングなし**（batch_sizeに依存しない実効lr）
          
          理論的背景:
          - PyTorch方式（勾配平均化）: 
            grad_avg = Σ(∂L/∂W) / batch_size
            W += lr × grad_avg
            → 実効学習率 = lr （batch_sizeに依存しない）
          
          - TensorFlow方式（勾配合計）:
            grad_sum = Σ(∂L/∂W)
            W += lr × grad_sum
            → 実効学習率 = lr × batch_size（batch_size依存）
          
          - 本実装の選択理由:
            1. batch_sizeを変更しても学習率を調整する必要がない
            2. PyTorchでも広く使われている標準的な方式
            3. 既存の学習率パラメータがそのまま使える
            4. 安定した学習が可能
          
          - v031の問題点（修正済み）:
            平均化後にbatch_size倍していたため、実効lr = lr × batch_size
            → batch_size=64で実効lr=6.4となり学習不可能だった
            → v030に戻すことで解決
        
        Args:
            train_dataset: tf.data.Dataset（バッチ化・シャッフル済み）
            x_train: 訓練データ全体（return_true_accuracy=Trueの場合に必要）
            y_train: 訓練ラベル全体（return_true_accuracy=Trueの場合に必要）
            return_true_accuracy: Trueの場合、学習完了後に全データを再評価して真の訓練精度を返す
                                 Falseの場合、学習中の平均精度を返す（従来の動作、非推奨）
        
        Returns:
            accuracy: 訓練精度
            avg_loss: 平均損失
        
        Notes:
            - TensorFlow Dataset APIで前処理（シャッフル・バッチ化）済み
            - ED法準拠: 勾配計算に_compute_gradients()使用（微分の連鎖律不使用）
            - ミニバッチ学習: 勾配を平均化（PyTorch方式）
            - batch_sizeに依存しない安定した学習
            
            return_true_accuracy=True（デフォルト、推奨）の場合:
                学習完了後の最終重みで全訓練データを再評価し、真の訓練精度を返します。
                x_train と y_train を必ず渡してください。
            
            使用例:
            >>> from modules.data_loader import create_tf_dataset
            >>> train_dataset = create_tf_dataset(
            ...     x_train, y_train, batch_size=128, shuffle=True, seed=42
            ... )
            >>> train_acc, train_loss = network.train_epoch_minibatch_tf(
            ...     train_dataset, x_train, y_train
            ... )
        """
        total_loss = 0.0
        n_correct = 0
        n_samples = 0
        
        # TensorFlow Datasetからバッチを取得
        for x_batch_tf, y_batch_tf in train_dataset:
            # TensorをNumPyに変換（既存コードとの互換性）
            x_batch = x_batch_tf.numpy()
            y_batch = y_batch_tf.numpy()
            
            batch_size = len(x_batch)
            
            # ステップ1: バッチ内の全サンプルの勾配を蓄積
            accumulated_gradients = None
            
            for i in range(batch_size):
                x_sample = x_batch[i]
                y_sample = y_batch[i]
                
                # 順伝播
                z_hiddens, z_output, x_paired = self.forward(x_sample)
                
                # 予測と損失
                y_pred = np.argmax(z_output)
                n_correct += (y_pred == y_sample)
                total_loss += cross_entropy_loss(z_output, y_sample)
                
                # 勾配計算（重みは更新しない）
                gradients = self._compute_gradients(x_paired, z_hiddens, z_output, y_sample)
                
                # 勾配の蓄積
                if accumulated_gradients is None:
                    # 初回: 構造をコピー
                    accumulated_gradients = {
                        'w_output': gradients['w_output'].copy(),
                        'lateral_weights': gradients['lateral_weights'].copy(),
                        'w_hidden': [g.copy() for g in gradients['w_hidden']]
                    }
                else:
                    # 2回目以降: 加算
                    accumulated_gradients['w_output'] += gradients['w_output']
                    accumulated_gradients['lateral_weights'] += gradients['lateral_weights']
                    for layer_idx in range(self.n_layers):
                        accumulated_gradients['w_hidden'][layer_idx] += gradients['w_hidden'][layer_idx]
                
                n_samples += 1
            
            # ステップ2: 勾配を平均化
            accumulated_gradients['w_output'] /= batch_size
            accumulated_gradients['lateral_weights'] /= batch_size
            for layer_idx in range(self.n_layers):
                accumulated_gradients['w_hidden'][layer_idx] /= batch_size
            
            # ステップ3: 平均化された勾配で一度だけ重みを更新（PyTorch方式）
            # 学習率スケーリングなし → 実効lr = lr （batch_sizeに依存しない）
            # これにより batch_size=8 でも 64 でも同じ学習率で安定した学習が可能
            self.w_output += accumulated_gradients['w_output']
            self.lateral_weights += accumulated_gradients['lateral_weights']
            
            for layer_idx in range(self.n_layers):
                self.w_hidden[layer_idx] += accumulated_gradients['w_hidden'][layer_idx]
                
                # 第1層のDale's Principle強制
                if layer_idx == 0:
                    sign_matrix = np.outer(self.ei_flags_hidden[0], self.ei_flags_input)
                    self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
            
            # 出力重みの正則化
            weight_penalty = 0.00001 * self.w_output
            self.w_output -= weight_penalty
        
        # 学習中の平均精度と損失
        training_accuracy = n_correct / n_samples if n_samples > 0 else 0.0
        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
        
        if return_true_accuracy:
            # 学習完了後、全訓練データを最終重みで再評価（推奨）
            if x_train is None or y_train is None:
                raise ValueError(
                    "return_true_accuracy=True の場合、x_train と y_train を渡してください。"
                )
            true_accuracy, true_loss = self.evaluate(x_train, y_train)
            return true_accuracy, true_loss
        else:
            # 学習中の平均精度を返す（従来動作、非推奨）
            return training_accuracy, avg_loss
    
    def _compute_gradients(self, x_paired, z_hiddens, z_output, y_true):
        """
        勾配を計算（v028ベース・シンプル版）
        
        経緯:
        - v028ベース（このメソッド）: シンプルなアミン拡散
        - v029改良版（上記、現在無効）: カラム/非カラム分離加算方式
        - 2024/12/30 初回: v029改良版を優先するため、このメソッドをリネーム
        - 2024/12/30 再変更: v029で学習できなくなったため、このメソッドに戻しました
        
        このメソッドの特徴:
        - シンプルな拡散係数適用（amine_diffused = amine * diffusion_coef）
        - コラム構造での重み付けのみ
        - より安定的な実装
        
        Args:
            x_paired: 入力ベクトル（興奮性・抑制性ペア）
            z_hiddens: 隠れ層出力のリスト
            z_output: 出力層出力
            y_true: 正解ラベル
        
        Returns:
            gradients: 各層の勾配を含む辞書
        """
        # ============================================
        # 1. 出力層の勾配計算
        # ============================================
        target_probs = np.zeros(self.n_output)
        target_probs[y_true] = 1.0
        error_output = target_probs - z_output
        
        saturation_output = np.abs(z_output) * (1.0 - np.abs(z_output))
        delta_w_output = self.learning_rate * np.outer(
            error_output * saturation_output,
            z_hiddens[-1]
        )
        
        # ============================================
        # 2. 出力層のアミン濃度計算と側方抑制（★修正★全クラスに配分）
        # ============================================
        winner_class = np.argmax(z_output)
        delta_lateral = np.zeros_like(self.lateral_weights)
        amine_concentration_output = np.zeros((self.n_output, 2))
        
        # ★オリジナル実装★ winner_class と y_true の2クラスのみにアミン配分
        if winner_class == y_true:
            # 正解時：正解クラスのみに興奮性アミン
            error_correct = 1.0 - z_output[y_true]
            if error_correct > 0:
                amine_concentration_output[y_true, 0] = error_correct * self.initial_amine
        else:
            # 不正解時：
            # 1) 勝者クラスに抑制性アミン
            error_winner = 0.0 - z_output[winner_class]
            if error_winner < 0:
                amine_concentration_output[winner_class, 1] = -error_winner * self.initial_amine
            
            # 2) 正解クラスに側方抑制考慮の興奮性アミン
            error_correct = 1.0 - z_output[y_true]
            if error_correct > 0:
                lateral_effect = self.lateral_weights[winner_class, y_true]
                if lateral_effect < 0:
                    enhanced_amine = self.initial_amine * (1.0 - lateral_effect)
                else:
                    enhanced_amine = self.initial_amine
                amine_concentration_output[y_true, 0] = error_correct * enhanced_amine
            
            # 注: 側方抑制の学習は無効化（検証結果で効果なしと判明）
        
        # デバッグ出力は無効化（学習速度優先）
        
        # ============================================
        # 3. 隠れ層の勾配計算（多層アミン拡散）
        # ============================================
        delta_w_hidden = []
        
        for layer_idx in range(self.n_layers - 1, -1, -1):
            # 入力の取得
            if layer_idx == 0:
                z_input = x_paired
            else:
                z_input = z_hiddens[layer_idx - 1]
            
            # 拡散係数の選択
            if layer_idx == self.n_layers - 1:
                diffusion_coef = self.u1
            else:
                diffusion_coef = self.u2
            
            # ★v028修正★ オリジナルCコード準拠のアミン拡散（legacy版も同様に修正）
            amine_mask = amine_concentration_output >= 1e-8
            
            # ステップ1: アミン濃度に拡散係数を適用（全ニューロン一律）
            amine_diffused = amine_concentration_output * diffusion_coef
            
            # ★新方式★ メンバーシップフラグ + Top-K学習
            # ステップ2a: メンバーシップでフィルタリング（boolean）
            membership = self.column_membership_all_layers[layer_idx]  # [n_classes, n_neurons] boolean
            
            # ステップ2b: 各クラスのメンバーニューロンから活性値ベースでTop-K選択
            amine_hidden_3d = np.zeros((self.n_output, 2, self.n_hidden[layer_idx]))
            z_current = z_hiddens[layer_idx]  # 現在の隠れ層活性
            
            # アミン濃度が非ゼロのクラスのみ処理（学習信号がある）
            active_classes = np.where(np.any(amine_diffused >= 1e-8, axis=1))[0]
            
            for class_idx in active_classes:
                # メンバーニューロンを取得
                member_neurons = np.where(membership[class_idx])[0]
                if len(member_neurons) == 0:
                    continue
                
                # 全メンバー内で活性値に基づいてランク付け
                # （学習が進むと強いニューロンが高活性になる）
                member_activations = z_current[member_neurons]
                sorted_indices = np.argsort(-member_activations)  # 降順でソート
                
                # ★新方式★ 全メンバーに学習信号を配分（線形減衰型協調学習 v4）
                # 1位=100%, 2位=70%, 3位=40%, 4位以降=線形減衰（5位で0到達）
                # 最強競争原理：上位4-5位に集中、Top-3独占に近い精度を狙う
                for rank, idx in enumerate(sorted_indices):
                    neuron_idx = member_neurons[idx]
                    
                    if rank == 0:
                        learning_weight = 1.0
                    elif rank == 1:
                        learning_weight = 0.7
                    elif rank == 2:
                        learning_weight = 0.4
                    else:
                        # 4位以降：線形減衰 max(0, 0.33 - 0.33 * (rank - 3))
                        # 4位=0.33, 5位=0.0（線形4位→0パターン + 疲労メカニズム）
                        learning_weight = max(0.0, 0.33 - 0.33 * (rank - 3))
                    
                    # ★勝者独り勝ち防止★ 疲労度による学習係数の調整
                    if self.fatigue_enabled:
                        fatigue = self.neuron_fatigue[layer_idx][class_idx, neuron_idx]
                        # 疲労度が高いほど学習係数を減少（0 ≤ fatigue < 1）
                        learning_weight = learning_weight * (1.0 - fatigue)
                    
                    # 興奮性・抑制性アミンを配分
                    amine_hidden_3d[class_idx, :, neuron_idx] = (
                        amine_diffused[class_idx, :] * learning_weight
                    )
                    
                    # ★勝者独り勝ち防止★ 上位ニューロン（学習参加者）の疲労蓄積
                    if self.fatigue_enabled and learning_weight > 0:
                        # 学習した分だけ疲労が蓄積
                        self.neuron_fatigue[layer_idx][class_idx, neuron_idx] += (
                            self.fatigue_accumulation * learning_weight
                        )
                        # 疲労度の上限を設定（飽和を防ぐ）
                        self.neuron_fatigue[layer_idx][class_idx, neuron_idx] = min(
                            self.neuron_fatigue[layer_idx][class_idx, neuron_idx], 
                            0.9  # 最大90%まで疲労
                        )
            amine_hidden_3d = amine_hidden_3d * amine_mask[:, :, np.newaxis]
            
            # ★ニューロントラッキング★ 各アクティブクラスの勝者と活性を記録
            if self.enable_neuron_tracking:
                for track_class_idx in active_classes:
                    self._track_neuron_activity(layer_idx, track_class_idx, z_hiddens[layer_idx], amine_hidden_3d)
            
            # デバッグ出力は無効化（学習速度優先）
            
            # 活性ニューロンの特定
            neuron_mask = np.any(amine_hidden_3d >= 1e-8, axis=(0, 1))
            active_neurons = np.where(neuron_mask)[0]
            
            if len(active_neurons) == 0:
                delta_w_hidden.insert(0, np.zeros_like(self.w_hidden[layer_idx]))
                continue
            
            # 活性化関数の勾配: tanh（固定）
            z_active = z_hiddens[layer_idx][active_neurons]
            saturation_term_raw = np.abs(z_active) * (1.0 - np.abs(z_active))
            saturation_term = np.maximum(saturation_term_raw, 1e-3)
            
            # 学習信号強度の計算
            layer_lr = self.layer_specific_lr[layer_idx]
            learning_signals_3d = (
                layer_lr * 
                amine_hidden_3d[:, :, active_neurons] * 
                saturation_term[np.newaxis, np.newaxis, :]
            )
            
            # 勾配の計算
            n_combinations = self.n_output * 2
            learning_signals_2d = learning_signals_3d.reshape(n_combinations, -1)
            z_input_tile = np.tile(z_input, (len(active_neurons), 1))
            
            learning_signals_expanded = learning_signals_2d.T[:, :, np.newaxis]
            z_input_expanded = z_input_tile[:, np.newaxis, :]
            delta_w_active = np.sum(
                learning_signals_expanded * z_input_expanded,
                axis=1
            )
            
            # フルサイズの勾配行列を構築
            dw = np.zeros_like(self.w_hidden[layer_idx])
            dw[active_neurons, :] = delta_w_active
            delta_w_hidden.insert(0, dw)
        
        # バイアスの勾配計算（隠れ層のみ）
        delta_b_hidden = []
        for layer_idx in range(self.n_layers):
            # バイアスの勾配は重みの勾配の行和
            delta_b = np.sum(delta_w_hidden[layer_idx], axis=1)
            delta_b_hidden.append(delta_b)
        
        return {
            'w_output': delta_w_output,
            'lateral_weights': delta_lateral,
            'w_hidden': delta_w_hidden,
            'b_hidden': delta_b_hidden
        }
    
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

    def extract_detailed_metrics(self, epoch, x_samples=None, y_samples=None):
        """
        詳細デバッグ用メトリクス抽出（1-5エポック目用）
        
        Args:
            epoch: 現在のエポック番号
            x_samples: サンプルデータ（活性化統計用、オプション）
            y_samples: サンプルラベル（オプション）
        
        Returns:
            metrics: 詳細メトリクスの辞書
        """
        metrics = {
            'epoch': epoch,
            'timestamp': str(np.datetime64('now')),
        }
        
        # 1. 重み統計（隠れ層 + 出力層）
        weight_stats = {}
        # 隠れ層の重み
        for i, w in enumerate(self.w_hidden):
            weight_stats[f'layer_{i}'] = {
                'mean': float(np.mean(w)),
                'std': float(np.std(w)),
                'min': float(np.min(w)),
                'max': float(np.max(w)),
                'l2_norm': float(np.linalg.norm(w)),
                'zeros': int(np.sum(np.abs(w) < 1e-8)),
                'total': int(w.size),
                'zero_ratio': float(np.sum(np.abs(w) < 1e-8) / w.size),
            }
        # 出力層の重み
        weight_stats['output_layer'] = {
            'mean': float(np.mean(self.w_output)),
            'std': float(np.std(self.w_output)),
            'min': float(np.min(self.w_output)),
            'max': float(np.max(self.w_output)),
            'l2_norm': float(np.linalg.norm(self.w_output)),
            'zeros': int(np.sum(np.abs(self.w_output) < 1e-8)),
            'total': int(self.w_output.size),
            'zero_ratio': float(np.sum(np.abs(self.w_output) < 1e-8) / self.w_output.size),
        }
        metrics['weight_stats'] = weight_stats
        
        # 2. アミン濃度統計
        amine_stats = {}
        for i, amine in enumerate(self.amine_concentrations):
            # アミン濃度の合計（クラスごと）
            amine_sum_per_class = np.sum(amine, axis=(1, 2))  # shape: (n_classes,)
            
            amine_stats[f'layer_{i}'] = {
                'mean': float(np.mean(amine_sum_per_class)),
                'std': float(np.std(amine_sum_per_class)),
                'min': float(np.min(amine_sum_per_class)),
                'max': float(np.max(amine_sum_per_class)),
                'range': float(np.max(amine_sum_per_class) - np.min(amine_sum_per_class)),
                'class_balance': {
                    int(c): float(amine_sum_per_class[c]) for c in range(len(amine_sum_per_class))
                },
                'entropy': float(-np.sum((amine_sum_per_class / (np.sum(amine_sum_per_class) + 1e-10)) * 
                                         np.log(amine_sum_per_class / (np.sum(amine_sum_per_class) + 1e-10) + 1e-10))),
            }
        metrics['amine_stats'] = amine_stats
        
        # 3. 側方抑制重み統計
        lateral_stats = {
            'output_layer': {
                'mean': float(np.mean(self.lateral_weights)),
                'std': float(np.std(self.lateral_weights)),
                'min': float(np.min(self.lateral_weights)),
                'max': float(np.max(self.lateral_weights)),
                'l2_norm': float(np.linalg.norm(self.lateral_weights)),
            }
        }
        metrics['lateral_stats'] = lateral_stats
        
        # 4. サンプルデータがあれば活性化統計を計算
        if x_samples is not None and len(x_samples) > 0:
            # ランダムに100サンプル選択（または全サンプル）
            n_samples = min(100, len(x_samples))
            indices = np.random.choice(len(x_samples), n_samples, replace=False)
            sample_x = x_samples[indices]
            
            activation_stats = {}
            for i in range(len(self.n_hidden)):
                layer_activations = []
                for x in sample_x:
                    # Forward pass to this layer
                    z_hiddens, _, _ = self.forward(x)
                    layer_activations.append(z_hiddens[i])
                
                layer_activations = np.array(layer_activations)
                activation_stats[f'layer_{i}'] = {
                    'mean': float(np.mean(layer_activations)),
                    'std': float(np.std(layer_activations)),
                    'min': float(np.min(layer_activations)),
                    'max': float(np.max(layer_activations)),
                    'dead_neurons': int(np.sum(np.max(layer_activations, axis=0) < 1e-4)),
                    'active_neurons': int(np.sum(np.max(layer_activations, axis=0) >= 1e-4)),
                    'total_neurons': int(self.n_hidden[i]),
                    'sparsity': float(np.mean(layer_activations < 1e-4)),
                }
            metrics['activation_stats'] = activation_stats
        
        # 5. コラム帰属統計
        column_usage_stats = {}
        for i, affinity in enumerate(self.column_affinity_all_layers):
            n_classes = affinity.shape[0]
            n_neurons = affinity.shape[1]
            
            # 各クラスに帰属するニューロン数
            class_neuron_counts = {}
            for c in range(n_classes):
                count = np.sum(affinity[c] > 1e-8)
                class_neuron_counts[int(c)] = int(count)
            
            # 複数クラスに帰属するニューロン数
            participation_per_neuron = np.sum(affinity > 1e-8, axis=0)
            multi_class_neurons = np.sum(participation_per_neuron > 1)
            
            column_usage_stats[f'layer_{i}'] = {
                'class_neuron_counts': class_neuron_counts,
                'multi_class_neurons': int(multi_class_neurons),
                'single_class_neurons': int(np.sum(participation_per_neuron == 1)),
                'zero_class_neurons': int(np.sum(participation_per_neuron == 0)),
                'avg_affinity': float(np.mean(affinity[affinity > 1e-8])) if np.any(affinity > 1e-8) else 0.0,
                'max_affinity': float(np.max(affinity)),
            }
        metrics['column_usage_stats'] = column_usage_stats
        
        return metrics
    
    def save_detailed_metrics(self, metrics, filepath):
        """詳細メトリクスをJSON形式で保存"""
        import json
        from datetime import datetime
        
        # numpy型をPython標準型に変換
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.datetime64):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_metrics = convert_to_serializable(metrics)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
    
    def set_neuron_tracking(self, enable=True):
        """
        ニューロンレベルトラッキングを有効/無効化
        
        Args:
            enable: True=有効化, False=無効化
        """
        self.enable_neuron_tracking = enable
        if enable:
            print(f"\n[ニューロントラッキング有効化]")
        else:
            print(f"\n[ニューロントラッキング無効化]")
    
    def reset_neuron_stats(self):
        """
        ニューロン統計をリセット（エポック開始時に呼び出す）
        """
        for layer_idx in range(self.n_layers):
            self.neuron_stats[layer_idx]['winner_count'] = np.zeros((self.n_output, self.n_hidden[layer_idx]))
            self.neuron_stats[layer_idx]['activation_sum'] = np.zeros((self.n_output, self.n_hidden[layer_idx]))
            self.neuron_stats[layer_idx]['weight_update_count'] = np.zeros((self.n_output, self.n_hidden[layer_idx]))
    
    def _track_neuron_activity(self, layer_idx, class_idx, z_hidden, amine_hidden_3d):
        """
        ニューロン活動を記録（内部メソッド）
        
        Args:
            layer_idx: 層インデックス
            class_idx: クラスインデックス
            z_hidden: 隠れ層の出力
            amine_hidden_3d: アミン濃度 [n_output, 2, n_hidden]
        """
        if not self.enable_neuron_tracking:
            return
        
        # 活性化の記録
        self.neuron_stats[layer_idx]['activation_sum'][class_idx] += np.abs(z_hidden)
        
        # アミン濃度から勝者を特定（興奮性アミンが配分された全ニューロン）
        exc_amine = amine_hidden_3d[class_idx, 0, :]  # [n_hidden]
        # Top-K学習対応: アミンが配分された全ニューロンを勝者としてカウント
        winners = np.where(exc_amine > 1e-8)[0]
        for winner_idx in winners:
            self.neuron_stats[layer_idx]['winner_count'][class_idx, winner_idx] += 1
        
        # 重み更新されたニューロンをカウント（アミン濃度>0のニューロン）
        updated_neurons = np.any(amine_hidden_3d[class_idx] > 1e-8, axis=0)
        self.neuron_stats[layer_idx]['weight_update_count'][class_idx] += updated_neurons.astype(int)
    
    def get_neuron_stats(self):
        """
        現在のニューロン統計を取得
        
        Returns:
            neuron_stats: 各層のニューロン統計のリスト
        """
        return self.neuron_stats
    
    def diagnose_column_structure(self):
        """
        コラム構造の詳細診断（v032: membership方式対応）
        各層のコラム参加率、重複度、membership統計を分析
        """
        print("\n" + "="*80)
        print("コラム構造診断（membership方式）")
        print("="*80)
        
        for layer_idx in range(self.n_layers):
            membership = self.column_membership_all_layers[layer_idx]  # [n_classes, n_neurons] boolean
            n_neurons = self.n_hidden[layer_idx]
            n_classes = self.n_output
            
            print(f"\n【Layer {layer_idx} - {n_neurons}ニューロン】")
            
            # 1. 各クラスのコラム参加率
            print("\n1. 各クラスのコラムメンバーニューロン数:")
            class_neuron_counts = []
            for class_idx in range(n_classes):
                # membershipフラグがTrueのニューロン数
                participating = int(np.sum(membership[class_idx]))
                class_neuron_counts.append(participating)
                participation_rate = participating / n_neurons * 100
                print(f"  Class {class_idx}: {participating:3d}個 ({participation_rate:5.1f}%)")
            
            print(f"  平均: {np.mean(class_neuron_counts):.1f}個")
            print(f"  標準偏差: {np.std(class_neuron_counts):.1f}個")
            
            # 2. ニューロンごとの重複度（何個のクラスのメンバーか）
            print("\n2. ニューロンの重複度分析:")
            overlap_counts = np.zeros(n_neurons, dtype=int)
            for neuron_idx in range(n_neurons):
                # このニューロンがメンバーになっているクラス数
                n_classes_for_neuron = int(np.sum(membership[:, neuron_idx]))
                overlap_counts[neuron_idx] = n_classes_for_neuron
            
            # ヒストグラム
            unique_overlaps, counts = np.unique(overlap_counts, return_counts=True)
            print("  重複度分布:")
            for overlap, count in zip(unique_overlaps, counts):
                pct = count / n_neurons * 100
                print(f"    {overlap}クラスのメンバー: {count:3d}個 ({pct:5.1f}%)")
            
            print(f"  平均重複度: {np.mean(overlap_counts):.2f}クラス/ニューロン")
            print(f"  非メンバーニューロン数: {np.count_nonzero(overlap_counts == 0)}個")
            
            # 3. membership統計
            print("\n3. membershipフラグ統計:")
            total_flags = n_classes * n_neurons
            true_flags = int(np.sum(membership))
            print(f"  総フラグ数: {total_flags}")
            print(f"  Trueフラグ数: {true_flags} ({true_flags/total_flags*100:.1f}%)")
            print(f"  Falseフラグ数: {total_flags - true_flags} ({(total_flags-true_flags)/total_flags*100:.1f}%)")
            
            # 4. クラス間の重複ニューロン分析
            print("\n4. クラス間重複メンバーニューロン数（上位5ペア）:")
            overlap_matrix = np.zeros((n_classes, n_classes), dtype=int)
            for i in range(n_classes):
                for j in range(i+1, n_classes):
                    # クラスiとjの両方のメンバーになっているニューロン数
                    both_member = int(np.sum(membership[i] & membership[j]))
                    overlap_matrix[i, j] = both_member
            
            # 上位5ペアを表示
            overlap_pairs = []
            for i in range(n_classes):
                for j in range(i+1, n_classes):
                    overlap_pairs.append((i, j, overlap_matrix[i, j]))
            overlap_pairs.sort(key=lambda x: x[2], reverse=True)
            
            for i, j, count in overlap_pairs[:5]:
                print(f"  Class {i} ⇔ Class {j}: {count}個共有")
        
        print("\n" + "="*80)
