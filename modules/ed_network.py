#!/usr/bin/env python3
"""
コラムED法ネットワーク（教育用シンプル版）

コラムED法の核心的アルゴリズムのみを実装。
4369行のフル版 (modules/ed_network.py) から不要な機能を全て除去し、
コアロジックの理解を容易にしている。

■ 除去した機能:
  CuPy/GPU, affinity方式, NC学習, 競合抑制, 側方協調, 層正規化,
  出力空間バイアス, コラム脱相関, 刈り込み, 受容野, ミニバッチTF,
  leaky_relu, uniform/xavier/flat初期化, 重み正規化/バランス調整

■ コラムED法の学習フロー:
  1. 順伝播: 入力 → E/Iペア → 隠れ層(tanh) → 出力(softmax)
  2. 出力誤差: target(one-hot) - prediction
  3. アミン信号: 正解クラスのみ正のアミン
  4. アミン拡散: コラムmembership + ランクLUTで各ニューロンの学習量を決定
  5. 重み更新: lr × amine × saturation_suppression × input（外積）
  6. Dale's Principle: 第1層の重み符号を強制（興奮性/抑制性の維持）
"""

import numpy as np

from modules.activation_functions import (
    tanh_activation, softmax, softmax_batch, cross_entropy_loss
)
from modules.column_structure import create_column_membership


class SimpleColumnEDNetwork:
    """
    コラムED法ネットワーク

    微分の連鎖律を用いた誤差逆伝播法を一切使用しない。
    代わりに、大脳皮質のアミン拡散機構とコラム構造により学習を実現。
    """

    def __init__(self, n_input=784, n_hidden=None, n_output=10,
                 learning_rate=0.15, u1=0.5, u2=0.8,
                 base_column_radius=0.4, column_neurons=None,
                 participation_rate=0.1, use_hexagonal=True,
                 gradient_clip=0.0, hidden_sparsity=None,
                 column_lr_factors=None, init_scales=None,
                 layer_learning_rates=None,
                 output_weight_decay=0.0,
                 output_gradient_clip=0.0,
                 layer_gradient_clips=None,
                 lut_base_rate=0.0,
                 uncertainty_modulation=0.0,
                 hc_strength=0.0,
                 pv_nc_gain=0.0,
                pv_pool_mode='nc',
                pv_gain_mode='multiplicative',
                 homeostatic_rate=0.0,
                 vip_modulation=0.0,
                 sst_rate=0.0,
                 sst_target=0.3,
                 skip_connections=None,
                 li_strength=0.0,
                 li_soft_temp=0.0,
                 hebb_strength=0.0,
                 nc_hebb_lr=0.0,
                 prediction_error_strength=0.0,
                 input_gate_strength=0.0,
                 attention_boost_strength=0.0,
                 seed=None):
        """
        ネットワーク初期化

        Args:
            n_input: 入力次元数（MNISTなら784）
            n_hidden: 隠れ層ニューロン数のリスト（例: [2048] or [2048, 1024]）
            n_output: 出力クラス数
            learning_rate: 基準学習率（出力層＝非コラムニューロンの学習率）
            u1: アミン拡散係数（出力層→最終隠れ層）
            u2: アミン拡散係数（隠れ層間、多層時に使用）
            base_column_radius: コラム基本半径
            column_neurons: 各クラスのコラムニューロン数（int または層別リスト）
                0を指定すると全ニューロンをコラムに帰属させる
            participation_rate: コラム参加率
            use_hexagonal: ハニカム配置を使用するか
            gradient_clip: 勾配クリッピング閾値
            hidden_sparsity: 非コラムニューロンの重みスパース率（リスト）
            column_lr_factors: コラムニューロンの学習率倍率（リスト）
            init_scales: 層別重み初期化スケール（リスト、長さ=層数+1）
            layer_learning_rates: 層ごとの学習率（リスト、長さ=層数+1）
            seed: 乱数シード
        """
        if n_hidden is None:
            n_hidden = [2048]
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]

        # 乱数シード設定
        if seed is not None:
            np.random.seed(seed)

        # ============================================
        # パラメータ保存
        # ============================================
        self.n_input = n_input
        self.n_hidden = list(n_hidden)
        self.n_output = n_output
        self.n_layers = len(n_hidden)
        self.learning_rate = learning_rate
        self.u1 = u1
        self.u2 = u2
        self.gradient_clip = gradient_clip
        self.output_gradient_clip = output_gradient_clip
        # 層別勾配クリッピング（指定時はgradient_clipより優先）
        if layer_gradient_clips is not None:
            self.layer_gradient_clips = list(layer_gradient_clips)
        else:
            self.layer_gradient_clips = None
        self.lut_base_rate = lut_base_rate

        # column_neuronsを層別リストに正規化
        # 0 = 全ニューロンをコラムに帰属（n_hidden // n_output 個/クラス）
        if column_neurons is None:
            self.column_neurons_per_layer = [None] * self.n_layers
        elif isinstance(column_neurons, (list, tuple)):
            self.column_neurons_per_layer = list(column_neurons)
        else:
            self.column_neurons_per_layer = [column_neurons] * self.n_layers
        self.column_neurons = column_neurons  # 後方互換

        self.initial_amine = 1.0
        self.output_weight_decay = output_weight_decay
        self.uncertainty_modulation = uncertainty_modulation
        self.hc_strength = hc_strength
        self.pv_nc_gain = pv_nc_gain
        self.pv_pool_mode = pv_pool_mode
        self.pv_gain_mode = pv_gain_mode
        self.homeostatic_rate = homeostatic_rate
        self.vip_modulation = vip_modulation
        self.sst_rate = sst_rate
        self.sst_target = sst_target

        # D7-4: スキップ接続 [(src_layer, dst_layer, alpha), ...]
        if skip_connections is not None:
            self.skip_connections = list(skip_connections)
        else:
            self.skip_connections = []

        # D6: 隠れ層内側抑制
        self.li_strength = li_strength      # D6-1: ハード側抑制強度
        self.li_soft_temp = li_soft_temp    # D6-2: ソフト側抑制温度

        # D8: ヘブ則
        self.hebb_strength = hebb_strength  # D8-1: コラム内ヘブ強化
        self.nc_hebb_lr = nc_hebb_lr        # D8-3: NCヘブ自己組織化

        # P1: 層間予測エラー伝播（上位層からの逆投影でアミン変調）
        self.prediction_error_strength = prediction_error_strength
        # P2: L6フィードバック（深層活性による入力ゲート制御）
        self.input_gate_strength = input_gate_strength
        # P3: L1注意ブースト（出力確信度による浅層活性増幅）
        self.attention_boost_strength = attention_boost_strength

        # 層別学習率
        if layer_learning_rates is not None:
            self.layer_lrs = list(layer_learning_rates)
        else:
            self.layer_lrs = [learning_rate] * (self.n_layers + 1)

        # 層別スパース率
        if hidden_sparsity is None:
            self.hidden_sparsity = [0.0] * self.n_layers
        elif isinstance(hidden_sparsity, (int, float)):
            self.hidden_sparsity = [float(hidden_sparsity)] * self.n_layers
        else:
            self.hidden_sparsity = list(hidden_sparsity)

        # コラムニューロンの勾配抑制係数
        if column_lr_factors is None:
            self.column_lr_factors = [1.0] * self.n_layers
        else:
            self.column_lr_factors = list(column_lr_factors)

        # 層別初期化スケール
        if init_scales is None:
            if self.n_layers == 1:
                init_scales = [0.4, 1.0]
            elif self.n_layers == 2:
                init_scales = [0.7, 1.8, 0.8]
            else:
                init_scales = [0.7] + [1.8] * (self.n_layers - 1) + [0.8]
        self.init_scales = init_scales

        # ============================================
        # コラム構造の初期化
        # ============================================
        self.column_membership_all_layers = []
        self.neuron_positions_all_layers = []
        self.class_coords_all_layers = []

        for layer_idx, layer_size in enumerate(self.n_hidden):
            cn_layer = self.column_neurons_per_layer[layer_idx]
            # cn=0: 全ニューロンをコラムに帰属
            if cn_layer == 0:
                cn_layer = layer_size // n_output
            membership, positions, coords = create_column_membership(
                n_hidden=layer_size,
                n_classes=n_output,
                participation_rate=participation_rate,
                use_hexagonal=use_hexagonal,
                column_radius=base_column_radius,
                column_neurons=cn_layer
            )
            self.column_membership_all_layers.append(membership)
            self.neuron_positions_all_layers.append(positions)
            self.class_coords_all_layers.append(coords)

        # PV型NCゲイン変調用: 事前計算
        self.nc_mask_all_layers = []
        self.pv_membership_f = []
        self.pv_col_sizes = []
        self.pv_n_memberships = []
        for layer_idx in range(self.n_layers):
            membership = self.column_membership_all_layers[layer_idx]
            is_column = np.any(membership, axis=0)
            self.nc_mask_all_layers.append(~is_column)
            membership_f = membership.astype(np.float64)
            self.pv_membership_f.append(membership_f)
            self.pv_col_sizes.append(np.sum(membership_f, axis=1))  # (n_output,)
            self.pv_n_memberships.append(np.sum(membership, axis=0).astype(np.float64))  # (n_hidden,)

        # コラム構造の表示
        for layer_idx in range(self.n_layers):
            membership = self.column_membership_all_layers[layer_idx]
            counts = [int(np.sum(membership[c])) for c in range(n_output)]
            print(f"  コラム構造 Layer {layer_idx}: 各クラス{counts[0]}ニューロン")

        # ============================================
        # ランク依存学習率のルックアップテーブル（LUT）
        # ============================================
        # 層別に作成: 各層のcn値に応じた線形減衰LUT
        max_rank = 256
        self._learning_weight_luts = []
        for layer_idx in range(self.n_layers):
            cn_layer = self.column_neurons_per_layer[layer_idx]
            if cn_layer == 0:
                cn_layer = self.n_hidden[layer_idx] // n_output
            cn = cn_layer if cn_layer is not None else 1
            lut = np.zeros(max_rank, dtype=np.float32)
            for i in range(max_rank):
                if i < cn:
                    lut[i] = (cn - i) / cn
                else:
                    lut[i] = self.lut_base_rate
            self._learning_weight_luts.append(lut)
        # 後方互換: 第1層のLUTをデフォルトとして保持
        self._learning_weight_lut = self._learning_weight_luts[0]

        # ============================================
        # E/Iフラグ（Dale's Principle用）
        # ============================================
        # 入力ペア: 前半=興奮性(+1), 後半=抑制性(-1)
        n_input_paired = n_input * 2
        self.ei_flags_input = np.array(
            [1 if i < n_input else -1 for i in range(n_input_paired)]
        )
        # 隠れ層: 全て興奮性
        self.ei_flags_hidden = [np.ones(size) for size in self.n_hidden]

        # ============================================
        # 隠れ層の重み初期化（He初期化）
        # ============================================
        self.w_hidden = []
        for layer_idx in range(self.n_layers):
            n_in = n_input_paired if layer_idx == 0 else self.n_hidden[layer_idx - 1]
            n_out = self.n_hidden[layer_idx]

            # He初期化: std = sqrt(2/fan_in) × layer_scale
            base_scale = np.sqrt(2.0 / n_in)
            layer_scale = self.init_scales[layer_idx]
            w = np.random.randn(n_out, n_in) * (base_scale * layer_scale)

            # 非コラムニューロンのスパース化
            layer_sparsity = self.hidden_sparsity[layer_idx]
            if layer_sparsity > 0.0:
                membership = self.column_membership_all_layers[layer_idx]
                is_column = np.any(membership, axis=0)
                non_column_indices = np.where(~is_column)[0]
                for neuron_idx in non_column_indices:
                    mask = np.random.rand(n_in) < layer_sparsity
                    w[neuron_idx, mask] = 0.0

            self.w_hidden.append(w)

        # ============================================
        # 出力層の重み初期化（He初期化）
        # ============================================
        fan_in = self.n_hidden[-1]
        output_scale = self.init_scales[-1]
        std_he = np.sqrt(2.0 / fan_in) * output_scale
        self.w_output = np.random.randn(n_output, fan_in) * std_he

        # ============================================
        # Dale's Principle: 第1層の重み符号を強制
        # ============================================
        # sign_matrix: 興奮性入力への重みは正、抑制性入力への重みは負
        self._sign_matrix_layer0 = np.outer(
            self.ei_flags_hidden[0], self.ei_flags_input
        )
        self.w_hidden[0] = np.abs(self.w_hidden[0]) * self._sign_matrix_layer0

        # ============================================
        # P1: 逆投影重み（予測エラー伝播用）
        # ============================================
        # 上位層→下位層への逆投影重みを初期化（固定、学習しない）
        # 脳のL5/6→L2/3予測信号に対応
        if self.prediction_error_strength > 0 and self.n_layers > 1:
            self._feedback_weights = []
            for layer_idx in range(self.n_layers - 1):
                n_lower = self.n_hidden[layer_idx]
                n_upper = self.n_hidden[layer_idx + 1]
                # 正規化された逆投影（出力次元で正規化）
                w_fb = np.random.randn(n_lower, n_upper) * np.sqrt(1.0 / n_upper)
                self._feedback_weights.append(w_fb)

        # P2: 入力ゲート重み（L6フィードバック用）
        if self.input_gate_strength > 0 and self.n_layers > 1:
            n_input_paired = n_input * 2
            n_deep = self.n_hidden[-1]
            self._input_gate_weights = np.random.randn(n_input_paired, n_deep) * np.sqrt(1.0 / n_deep)

        # ============================================
        # 統計カウンタ
        # ============================================
        self.winner_selection_counts = np.zeros(n_output, dtype=int)
        self.total_training_samples = 0
        self.class_training_counts = np.zeros(n_output, dtype=int)

        # NCゲートマスク（互換性用、simple版では未使用）
        self._nc_gate_mask = None

        # Phase 2: ホメオスタティック調整用累積バッファ
        # エポック内でNCニューロンの平均絶対活性を集計し、外れ値を正規化
        self._nc_act_accum = [np.zeros(self.n_hidden[l]) for l in range(self.n_layers)]
        self._nc_act_count = 0

        # Phase 4: SST型動的バイアス補正バッファ
        self._sst_bias = [np.zeros(self.n_hidden[l]) for l in range(self.n_layers)]
        self._sst_act_accum = [np.zeros(self.n_hidden[l]) for l in range(self.n_layers)]
        self._sst_act_count = 0

        # ============================================
        # 統計カウンタ（続き）
        # ============================================

    def _compute_pv_gains(self, relative_strength):
        """PV型ゲイン変調のゲインを計算する。"""
        if self.pv_gain_mode == 'divisive':
            gains = (
                (1.0 + self.pv_nc_gain) * relative_strength /
                (relative_strength + self.pv_nc_gain + 1e-8)
            )
        else:
            gains = 1.0 + self.pv_nc_gain * (relative_strength - 1.0)
        return np.clip(gains, 1.0 - self.pv_nc_gain, 1.0 + self.pv_nc_gain)

    # ================================================================
    # 順伝播
    # ================================================================

    def forward(self, x):
        """
        順伝播（単一サンプル）

        入力 → E/Iペア → 隠れ層(tanh) → 出力(softmax)

        Args:
            x: 入力ベクトル [n_input]

        Returns:
            z_hiddens: 各隠れ層の出力リスト
            z_output: 出力確率分布 [n_output]
            x_paired: E/Iペア化入力 [n_input * 2]
        """
        # E/Iペア: 入力を複製して興奮性/抑制性の対を作る
        x_paired = np.concatenate([x, x])

        # P2: L6フィードバック（入力ゲート制御）
        # 前回の最終隠れ層活性に基づいて入力を選択的にゲーティング
        # 脳のL6→視床フィードバック「今はその情報は要らない/もっと詳しく送れ」に対応
        if self.input_gate_strength > 0 and self.n_layers > 1 and hasattr(self, '_prev_z_deep'):
            gate_raw = np.dot(self._input_gate_weights, self._prev_z_deep)
            gate = 1.0 + self.input_gate_strength * np.tanh(gate_raw)
            x_paired = x_paired * gate

        z_hiddens = []
        z_current = x_paired

        for layer_idx in range(self.n_layers):
            a_hidden = np.dot(self.w_hidden[layer_idx], z_current)
            z_hidden = tanh_activation(a_hidden)

            # PV型NCゲイン変調: NCの集団活性を参照信号としてコラムニューロンをゲイン変調
            if self.pv_nc_gain > 0.0:
                nc_mask = self.nc_mask_all_layers[layer_idx]
                membership_f = self.pv_membership_f[layer_idx]
                col_sizes = self.pv_col_sizes[layer_idx]
                n_memberships = self.pv_n_memberships[layer_idx]
                abs_z = np.abs(z_hidden)
                if self.pv_pool_mode == 'all':
                    pool_mean = np.mean(abs_z) + 1e-8
                else:
                    pool_mean = np.mean(abs_z[nc_mask]) + 1e-8
                col_means = (membership_f @ abs_z) / (col_sizes + 1e-8)  # (n_output,)
                relative_strength = col_means / pool_mean
                gains = self._compute_pv_gains(relative_strength)
                gain_map = membership_f.T @ gains  # (n_hidden,)
                multi = n_memberships > 1
                if np.any(multi):
                    gain_map[multi] /= n_memberships[multi]
                gain_map[nc_mask] = 1.0
                z_hidden *= gain_map

            # P1: 層間予測エラー（上位層からの逆投影による変調）
            # 前回の上位層活性との予測エラーを用いて現在の活性を変調
            # 脳のL2/3エラーユニット「予測と現実のズレ」に対応
            if (self.prediction_error_strength > 0 and
                layer_idx < self.n_layers - 1 and
                hasattr(self, '_prev_z_hiddens') and
                self._prev_z_hiddens is not None):
                z_upper_prev = self._prev_z_hiddens[layer_idx + 1]
                predicted = np.dot(self._feedback_weights[layer_idx], z_upper_prev)
                prediction_error = z_hidden - np.tanh(predicted)
                # 予測エラーが大きい部分を強化（=注目すべき新しい情報）
                error_magnitude = np.abs(prediction_error)
                boost = 1.0 + self.prediction_error_strength * error_magnitude
                z_hidden = z_hidden * boost
                z_hidden = np.clip(z_hidden, -1.0, 1.0)

            # P3: L1注意ブースト（出力確信度による浅層活性増幅）
            # 前回の出力確信度が低いとき、浅層の感度を上げて情報取得を強化
            # 脳のL1注意信号「今はここを重視せよ」に対応
            if (self.attention_boost_strength > 0 and
                layer_idx < self.n_layers // 2 and
                hasattr(self, '_prev_confidence')):
                # 確信度が低い(=不確実)ほど強くブースト
                uncertainty = 1.0 - self._prev_confidence
                boost = 1.0 + self.attention_boost_strength * uncertainty
                z_hidden = z_hidden * boost
                z_hidden = np.clip(z_hidden, -1.0, 1.0)

            # D7-4: スキップ接続（加算型残差）
            for src, dst, alpha in self.skip_connections:
                if dst == layer_idx and src < layer_idx:
                    src_z = z_hiddens[src]
                    if src_z.shape[0] == z_hidden.shape[0]:
                        z_hidden = np.tanh(z_hidden + alpha * src_z)

            # D6: 隠れ層内側抑制（コラム間コントラスト強化）
            if self.li_strength > 0.0 or self.li_soft_temp > 0.0:
                z_hidden = self._apply_lateral_inhibition(z_hidden, layer_idx)

            # Phase 4: SST型動的バイアス補正（高活性ニューロンを選択的に抑制）
            # 生物学的背景: SSTニューロンが主細胞の過剰発火を動的フィードバック抑制
            # バイアスは end_of_epoch_regularize() でエポック終了後に更新
            if self.sst_rate > 0.0:
                z_hidden = np.clip(z_hidden - self._sst_bias[layer_idx], -1.0, 1.0)

            z_hiddens.append(z_hidden)
            z_current = z_hidden

        # D4: 水平結合（最終隠れ層の同クラスコラムニューロン間ゲイン変調）
        if self.hc_strength > 0.0:
            z_current = self._apply_horizontal_connection(z_current, self.n_layers - 1)
            z_hiddens[-1] = z_current

        # 出力層: 重み積 → softmax
        a_output = np.dot(self.w_output, z_current)
        z_output = softmax(a_output)

        # P1/P2/P3用：次回の順伝播で使う状態を保存
        if self.prediction_error_strength > 0:
            self._prev_z_hiddens = [z.copy() for z in z_hiddens]
        if self.input_gate_strength > 0 and self.n_layers > 1:
            self._prev_z_deep = z_hiddens[-1].copy()
        if self.attention_boost_strength > 0:
            self._prev_confidence = float(np.max(z_output))

        return z_hiddens, z_output, x_paired

    def _apply_horizontal_connection(self, z_hidden, layer_idx):
        """
        D4: 同クラスコラムニューロン間の水平結合（乗算型ゲイン変調）

        脳のL2/3水平結合を模倣: 同種機能コラム間で情報統合する。
        v051のHCが「正帰還ループ問題」で失敗した教訓を踏まえ、
        乗算型ゲイン変調（加算ではなく）を採用し、発散を防止。

        各クラスのコラムニューロンの平均活性を計算し、
        個々のニューロンの活性をクラス平均の方向にゲイン変調する。
        """
        membership = self.column_membership_all_layers[layer_idx]
        z_mod = z_hidden.copy()

        for class_idx in range(self.n_output):
            col_neurons = np.where(membership[class_idx])[0]
            if len(col_neurons) < 2:
                continue

            # 同クラスコラムニューロンの平均活性
            col_acts = z_hidden[col_neurons]
            col_mean = np.mean(col_acts)

            # 乗算型ゲイン変調: 平均が正なら正のニューロンを強化、負なら負を強化
            # ゲイン = 1.0 + hc_strength * sign(mean) * sign(neuron)
            # → 同符号なら強化(>1.0)、異符号なら抑制(<1.0)
            if abs(col_mean) > 1e-6:
                gain = 1.0 + self.hc_strength * np.sign(col_mean) * np.sign(col_acts)
                gain = np.clip(gain, 1.0 - self.hc_strength, 1.0 + self.hc_strength)
                z_mod[col_neurons] = col_acts * gain

        return z_mod

    def _apply_lateral_inhibition(self, z_hidden, layer_idx):
        """
        D6: 隠れ層内のコラム間側抑制

        D6-1 (li_strength): 勝者コラム以外を固定比率で減衰（ハード側抑制）
        D6-2 (li_soft_temp): 活性比例のソフト側抑制（温度パラメータで強度調節）
        v051の失敗教訓: 正帰還ループ防止のため乗算型ゲイン変調を採用
        """
        membership = self.column_membership_all_layers[layer_idx]
        z_mod = z_hidden.copy()

        # 各クラスのコラム平均|活性|を計算
        col_means = np.zeros(self.n_output)
        for c in range(self.n_output):
            col_neurons = np.where(membership[c])[0]
            if len(col_neurons) > 0:
                col_means[c] = np.mean(np.abs(z_hidden[col_neurons]))

        max_mean = np.max(col_means)
        if max_mean < 1e-8:
            return z_mod

        if self.li_soft_temp > 0.0:
            # D6-2: ソフト側抑制（温度付きsoftmax的ゲイン）
            scaled = col_means / (self.li_soft_temp * max_mean + 1e-8)
            gains = scaled / (np.max(scaled) + 1e-8)  # 0～1に正規化
            for c in range(self.n_output):
                col_neurons = np.where(membership[c])[0]
                if len(col_neurons) > 0:
                    z_mod[col_neurons] *= gains[c]
        elif self.li_strength > 0.0:
            # D6-1: ハード側抑制（勝者以外を減衰）
            winner = np.argmax(col_means)
            for c in range(self.n_output):
                if c == winner:
                    continue
                col_neurons = np.where(membership[c])[0]
                if len(col_neurons) > 0:
                    z_mod[col_neurons] *= (1.0 - self.li_strength)

        return z_mod

    def forward_batch(self, x_batch):
        """
        バッチ順伝播（評価用）

        Args:
            x_batch: [batch_size, n_input]

        Returns:
            z_hiddens_batch, z_output_batch, x_paired_batch
        """
        # E/Iペア（バッチ版）
        x_paired_batch = np.concatenate([x_batch, x_batch], axis=1)

        z_hiddens_batch = []
        z_current = x_paired_batch

        for layer_idx in range(self.n_layers):
            a_hidden = np.dot(z_current, self.w_hidden[layer_idx].T)
            z_hidden = tanh_activation(a_hidden)

            # PV型NCゲイン変調（バッチ版・ベクトル化）
            if self.pv_nc_gain > 0.0:
                nc_mask = self.nc_mask_all_layers[layer_idx]
                membership_f = self.pv_membership_f[layer_idx]
                col_sizes = self.pv_col_sizes[layer_idx]
                n_memberships = self.pv_n_memberships[layer_idx]
                abs_z = np.abs(z_hidden)  # (batch, n_hidden)
                if self.pv_pool_mode == 'all':
                    pool_mean = np.mean(abs_z, axis=1, keepdims=True) + 1e-8
                else:
                    pool_mean = np.mean(abs_z[:, nc_mask], axis=1, keepdims=True) + 1e-8
                col_means = (abs_z @ membership_f.T) / (col_sizes[np.newaxis, :] + 1e-8)  # (batch, n_output)
                relative_strength = col_means / pool_mean
                gains = self._compute_pv_gains(relative_strength)
                gain_map = gains @ membership_f  # (batch, n_hidden)
                multi = n_memberships > 1
                if np.any(multi):
                    gain_map[:, multi] /= n_memberships[multi]
                gain_map[:, nc_mask] = 1.0
                z_hidden *= gain_map

            # D7-4: スキップ接続（バッチ版）
            for src, dst, alpha in self.skip_connections:
                if dst == layer_idx and src < layer_idx:
                    src_z = z_hiddens_batch[src]
                    if src_z.shape[1] == z_hidden.shape[1]:
                        z_hidden = np.tanh(z_hidden + alpha * src_z)

            # D6: 隠れ層内側抑制（バッチ版）
            if self.li_strength > 0.0 or self.li_soft_temp > 0.0:
                for i in range(z_hidden.shape[0]):
                    z_hidden[i] = self._apply_lateral_inhibition(z_hidden[i], layer_idx)

            # Phase 4: SST型動的バイアス補正（バッチ版）
            if self.sst_rate > 0.0:
                z_hidden = np.clip(z_hidden - self._sst_bias[layer_idx], -1.0, 1.0)

            z_hiddens_batch.append(z_hidden)
            z_current = z_hidden

        # D4: 水平結合（バッチ版 — 各サンプルに個別適用）
        if self.hc_strength > 0.0:
            for i in range(z_current.shape[0]):
                z_current[i] = self._apply_horizontal_connection(z_current[i], self.n_layers - 1)
            z_hiddens_batch[-1] = z_current

        a_output = np.dot(z_current, self.w_output.T)
        z_output_batch = softmax_batch(a_output)

        return z_hiddens_batch, z_output_batch, x_paired_batch

    # ================================================================
    # 勾配計算（コラムED法の核心）
    # ================================================================

    def _compute_gradients(self, x_paired, z_hiddens, z_output, y_true, y_pred=None):
        """
        ED法による勾配計算（微分の連鎖律を使用しない）

        ■ アルゴリズム:
          1. 出力誤差 = target(one-hot) - prediction(softmax)
          2. 出力層勾配 = lr × error × saturation × last_hidden
          3. アミン濃度 = 正解クラスのみ正のアミン信号を生成
          4. 各隠れ層について（出力側から入力側へ）:
             a. アミン拡散: 拡散係数(u1/u2)でアミンを減衰
             b. コラムmembership: 正解クラスのコラムニューロンを特定
             c. ランクLUT: 活性値の高い順にランクを付け、学習率を決定
             d. 非コラムニューロン: 学習率=0（リザバーとして固定）
             e. 飽和抑制項: |z| × (1-|z|) で飽和域のニューロンの学習を抑制
             f. 勾配計算: 学習信号 × 入力の外積
             g. 勾配クリッピング
             h. コラムニューロンの勾配をcolumn_lr_factorsで抑制

        ★重要★ 逆伝播との違い:
          - 微分の連鎖律 dL/dw = dL/dz × dz/da × da/dw は使用しない
          - 代わりにアミン拡散機構で各層が独立に学習量を決定
          - 飽和抑制項 |z|*(1-|z|) はtanh微分 (1-z²) とは異なる

        Args:
            x_paired: E/Iペア化入力
            z_hiddens: 各隠れ層の出力リスト
            z_output: 出力確率分布
            y_true: 正解クラスインデックス

        Returns:
            gradients: {'w_output': ..., 'w_hidden': [...]}
        """
        gradients = {
            'w_output': None,
            'w_hidden': [None] * self.n_layers,
        }

        # --------------------------------------------------
        # 1. 出力層の勾配
        # --------------------------------------------------
        target_probs = np.zeros(self.n_output)
        target_probs[y_true] = 1.0
        error_output = target_probs - z_output

        # 飽和抑制項（出力層）
        saturation_output = np.abs(z_output) * (1.0 - np.abs(z_output))

        output_lr = self.layer_lrs[-1] if len(self.layer_lrs) > self.n_layers else self.learning_rate
        gradients['w_output'] = output_lr * np.outer(
            error_output * saturation_output,
            z_hiddens[-1]
        )

        # --------------------------------------------------
        # 2. アミン濃度の計算（正解クラスのみ）
        # --------------------------------------------------
        # ★純粋ED法★ 正解クラスのみ学習信号を生成
        amine_concentration = np.zeros((self.n_output, 2))
        error_correct = 1.0 - z_output[y_true]
        if error_correct > 0:
            amine_signal = error_correct * self.initial_amine
            # D5-3: 不確実性変調（予測符号化インスパイア）
            # 出力エントロピーが高い=予測が不確実なとき学習信号を増強
            if self.uncertainty_modulation > 0:
                entropy = -np.sum(z_output * np.log(z_output + 1e-10))
                max_entropy = np.log(self.n_output)
                normalized_entropy = entropy / max_entropy  # 0(確信)～1(最大不確実)
                amine_signal *= (1.0 + self.uncertainty_modulation * normalized_entropy)
            # Phase 3: VIP型学習率変調（コラム-NC整合度による脱抑制）
            # 生物学的背景: VIPニューロン→SSTを抑制→主細胞の脱抑制（学習強化）
            # 最終隠れ層で正解クラスのコラム活性がNCより強いほどアミンを増強（脱抑制）
            # 弱いほど抑制（SSTが主細胞を抑制している状態をモデル化）
            # NOTE: vip_modulation > 1.0 は非推奨。下限クリップが負値になり
            # そのサンプルの隠れ層学習が停止する可能性がある。推奨範囲: 0.1〜0.5
            if self.vip_modulation > 0.0:
                z_last = z_hiddens[-1]
                col_mask = self.column_membership_all_layers[-1][y_true]
                nc_mask = self.nc_mask_all_layers[-1]
                if np.any(col_mask) and np.any(nc_mask):
                    col_act = np.mean(np.abs(z_last[col_mask]))
                    nc_act = np.mean(np.abs(z_last[nc_mask])) + 1e-8
                    coherence = col_act / nc_act  # 1.0が等活性基準
                    vip_factor = np.clip(
                        1.0 + self.vip_modulation * (coherence - 1.0),
                        1.0 - self.vip_modulation,
                        1.0 + self.vip_modulation
                    )
                    amine_signal *= vip_factor
            amine_concentration[y_true, 0] = amine_signal

        # --------------------------------------------------
        # 3. 各隠れ層のアミン拡散と勾配計算（出力側→入力側）
        # --------------------------------------------------
        for layer_idx in range(self.n_layers - 1, -1, -1):
            # この層への入力を取得
            z_input = x_paired if layer_idx == 0 else z_hiddens[layer_idx - 1]

            # 拡散係数: 最終層はu1、それ以外はu2
            diffusion_coef = self.u1 if layer_idx == self.n_layers - 1 else self.u2

            # アミン拡散
            amine_mask = amine_concentration >= 1e-8
            amine_diffused = amine_concentration * diffusion_coef

            # P1: 予測エラーによるアミン変調（層ごと）
            # 上位層からの逆投影と現在層の差が大きい=新しい情報が多い層で学習を増強
            if (self.prediction_error_strength > 0 and
                layer_idx < self.n_layers - 1 and
                hasattr(self, '_feedback_weights')):
                z_upper = z_hiddens[layer_idx + 1]
                predicted = np.dot(self._feedback_weights[layer_idx], z_upper)
                pred_error = np.mean(np.abs(z_hiddens[layer_idx] - np.tanh(predicted)))
                # 予測エラーが大きい層でアミンを増強（最大2倍）
                amine_boost = 1.0 + self.prediction_error_strength * pred_error
                amine_diffused = amine_diffused * amine_boost

            # コラムmembership方式でのアミン分配
            membership = self.column_membership_all_layers[layer_idx]
            z_current = z_hiddens[layer_idx]
            n_neurons = self.n_hidden[layer_idx]

            # アミン濃度が非ゼロのクラスを特定
            active_classes = np.where(np.any(amine_diffused >= 1e-8, axis=1))[0]

            if len(active_classes) == 0:
                continue

            # ベクトル化されたアミン分配
            active_membership = membership[active_classes]  # [n_active, n_neurons]

            # メンバーニューロンの活性値ランクを計算（降順）
            masked_activations = np.where(active_membership, z_current, -np.inf)
            sorted_indices = np.argsort(-masked_activations, axis=1)
            ranks = np.argsort(sorted_indices, axis=1)

            # ランクから学習率を取得（層別LUT参照）
            layer_lut = self._learning_weight_luts[layer_idx]
            clamped_ranks = np.minimum(ranks, len(layer_lut) - 1)
            learning_weights = layer_lut[clamped_ranks]

            # 非コラムニューロンの学習率は0（リザバーとして固定）
            learning_weights = np.where(active_membership, learning_weights, 0.0)


            # アミン拡散値に学習率を適用
            amine_hidden_3d = np.zeros((self.n_output, 2, n_neurons))
            amine_hidden_3d[active_classes] = (
                amine_diffused[active_classes, :, np.newaxis] *
                learning_weights[:, np.newaxis, :]
            )
            amine_hidden_3d = amine_hidden_3d * amine_mask[:, :, np.newaxis]

            # 活性ニューロンの特定
            neuron_mask = np.any(amine_hidden_3d >= 1e-8, axis=(0, 1))
            active_neurons = np.where(neuron_mask)[0]

            if len(active_neurons) == 0:
                continue

            # 飽和抑制項: |z| × (1 - |z|)
            # ★重要★ tanh微分 (1-z²) ではない。ED法固有の飽和抑制
            z_active = z_hiddens[layer_idx][active_neurons]
            saturation_term = np.abs(z_active) * (1.0 - np.abs(z_active))
            saturation_term = np.maximum(saturation_term, 1e-3)

            # 学習信号を計算
            layer_lr = self.layer_lrs[layer_idx]
            learning_signals_3d = (
                layer_lr *
                amine_hidden_3d[:, :, active_neurons] *
                saturation_term[np.newaxis, np.newaxis, :]
            )

            # 勾配計算: 学習信号の合計 × 入力（外積）
            n_combinations = self.n_output * 2
            learning_signals_flat = learning_signals_3d.reshape(n_combinations, -1).T
            signal_sum = learning_signals_flat.sum(axis=1)
            delta_w_batch = signal_sum[:, np.newaxis] * z_input[np.newaxis, :]

            # 層ごとの符号制約（第2層以降）
            if layer_idx > 0:
                w_sign = np.sign(self.w_hidden[layer_idx][active_neurons, :])
                w_sign[w_sign == 0] = 1
                delta_w_batch *= w_sign

            # 勾配クリッピング（層別gc優先、なければグローバルgc）
            gc_value = self.gradient_clip
            if self.layer_gradient_clips is not None and layer_idx < len(self.layer_gradient_clips):
                gc_value = self.layer_gradient_clips[layer_idx]
            if gc_value > 0:
                delta_w_norms = np.linalg.norm(delta_w_batch, axis=1, keepdims=True)
                clip_mask = delta_w_norms > gc_value
                delta_w_batch = np.where(
                    clip_mask,
                    delta_w_batch * (gc_value / delta_w_norms),
                    delta_w_batch
                )

            # コラムニューロンの勾配抑制
            lr_factor = self.column_lr_factors[layer_idx]
            if lr_factor < 1.0:
                is_column = np.any(membership, axis=0)
                active_is_column = is_column[active_neurons]
                if np.any(active_is_column):
                    delta_w_batch[active_is_column, :] *= lr_factor

            # スパース形式で保存
            gradients['w_hidden'][layer_idx] = (active_neurons, delta_w_batch)

        return gradients

    # ================================================================
    # 重み更新
    # ================================================================

    def update_weights(self, x_paired, z_hiddens, z_output, y_true, y_pred=None):
        """
        重みの更新（ED法準拠、微分の連鎖律不使用）
        """
        gradients = self._compute_gradients(x_paired, z_hiddens, z_output, y_true, y_pred=y_pred)

        # 出力層の更新
        output_grad = gradients['w_output']
        if self.output_gradient_clip > 0:
            g_norm = np.linalg.norm(output_grad)
            if g_norm > self.output_gradient_clip:
                output_grad = output_grad * (self.output_gradient_clip / g_norm)
        self.w_output += output_grad

        # サンプル単位の出力層weight decay
        if self.output_weight_decay > 0:
            self.w_output *= (1.0 - self.output_weight_decay / self._n_train_samples)

        # 隠れ層の更新
        for layer_idx in range(self.n_layers):
            sparse_grad = gradients['w_hidden'][layer_idx]
            if sparse_grad is not None:
                active_neurons, delta_w_batch = sparse_grad
                self.w_hidden[layer_idx][active_neurons] += delta_w_batch

                # 第1層のDale's Principle強制
                if layer_idx == 0:
                    self.w_hidden[0][active_neurons] = (
                        np.abs(self.w_hidden[0][active_neurons]) *
                        self._sign_matrix_layer0[active_neurons]
                    )

        # D8-1: コラム内ヘブ強化（正解クラスのコラムニューロンの重みを微小強化）
        if self.hebb_strength > 0.0:
            self._apply_hebbian_column(z_hiddens, y_true)

        # D8-3: NCヘブ自己組織化（非コラムニューロンの重みをヘブ則で更新）
        if self.nc_hebb_lr > 0.0:
            self._apply_nc_hebbian(x_paired, z_hiddens)

        # 出力重みの正則化はエポック単位で適用（end_of_epoch_regularize()）

    def _apply_hebbian_column(self, z_hiddens, y_true):
        """
        D8-1: コラム内ヘブ強化

        正解クラスのコラムニューロンのうち強く発火したものの入力重みを微小強化。
        「共に発火するニューロンは結合が強まる」原理に基づく。
        重みクリッピングで発散を防止（v051のdecay方式との違い）。
        """
        for layer_idx in range(self.n_layers):
            membership = self.column_membership_all_layers[layer_idx]
            col_neurons = np.where(membership[y_true])[0]
            if len(col_neurons) == 0:
                continue

            z_col = z_hiddens[layer_idx][col_neurons]
            # 活性が正のニューロンのみ（発火しているもの）
            active_mask = z_col > 0.1
            if not np.any(active_mask):
                continue

            active_idx = col_neurons[active_mask]
            z_active = z_col[active_mask]

            # 入力を取得
            if layer_idx == 0:
                # 第1層はDale's Principleが適用されるため、ヘブ則は適用しない
                continue
            z_input = z_hiddens[layer_idx - 1]

            # ヘブ則: Δw = η × z_out × z_in（外積の簡略版）
            for i, ni in enumerate(active_idx):
                delta = self.hebb_strength * z_active[i] * z_input
                self.w_hidden[layer_idx][ni] += delta
                # 重みクリッピングで発散防止
                np.clip(self.w_hidden[layer_idx][ni], -2.0, 2.0,
                        out=self.w_hidden[layer_idx][ni])

    def _apply_nc_hebbian(self, x_paired, z_hiddens):
        """
        D8-3: NCヘブ自己組織化

        非コラムニューロンの重みをヘブ則で更新。
        ED法の学習とは独立した自己組織化パスで、リザバーの質を改善。
        """
        for layer_idx in range(self.n_layers):
            membership = self.column_membership_all_layers[layer_idx]
            is_column = np.any(membership, axis=0)
            nc_indices = np.where(~is_column)[0]
            if len(nc_indices) == 0:
                continue

            z_nc = z_hiddens[layer_idx][nc_indices]
            # 活性が十分なNCニューロンのみ
            active_mask = np.abs(z_nc) > 0.1
            if not np.any(active_mask):
                continue

            active_nc = nc_indices[active_mask]
            z_active = z_nc[active_mask]

            # 入力
            if layer_idx == 0:
                z_input = x_paired
            else:
                z_input = z_hiddens[layer_idx - 1]

            # ヘブ則更新
            delta = self.nc_hebb_lr * np.outer(z_active, z_input)
            self.w_hidden[layer_idx][active_nc] += delta
            # 重みクリッピング
            np.clip(self.w_hidden[layer_idx][active_nc], -2.0, 2.0,
                    out=self.w_hidden[layer_idx][active_nc])

    # ================================================================
    # 学習率調整
    # ================================================================

    def scale_learning_rates(self, factor):
        """全層の学習率を指定倍率でスケーリング"""
        self.layer_lrs = [lr * factor for lr in self.layer_lrs]

    def set_learning_rates(self, lrs):
        """学習率を直接設定"""
        self.layer_lrs = list(lrs)

    def end_of_epoch_regularize(self):
        """エポック終了後の正則化処理（output weight decay + ホメオスタティック調整）
        サンプル単位OWDが有効な場合はOWDをスキップ（重複適用防止）
        """
        if self.output_weight_decay > 0 and not hasattr(self, '_n_train_samples'):
            self.w_output *= (1.0 - self.output_weight_decay)

        # Phase 2: ホメオスタティック調整
        # エポック全体の平均絶対活性を基準に、外れ値NCニューロンの重みをスケーリング
        # D8-3崩壊との違い: エポック単位 × 外れ値のみ × 双方向スケール（増強も抑制も）
        if self.homeostatic_rate > 0.0 and self._nc_act_count > 0:
            for l in range(self.n_layers):
                avg_act = self._nc_act_accum[l] / self._nc_act_count
                nc_mask = self.nc_mask_all_layers[l]
                if not np.any(nc_mask):
                    continue
                nc_avg = avg_act[nc_mask]
                global_mean = np.mean(nc_avg) + 1e-8
                ratio = nc_avg / global_mean  # 1.0が基準
                # 外れ値のみスケール（全体平均の2倍超 or 0.1倍未満）
                outlier = (ratio > 2.0) | (ratio < 0.1)
                if not np.any(outlier):
                    continue
                scale = np.ones_like(nc_avg)
                # 1/ratio でスケール（高活性なら縮小、低活性なら拡大）
                # homeostatic_rate でクリップして急激な変化を防止
                scale[outlier] = np.clip(
                    1.0 / ratio[outlier],
                    1.0 - self.homeostatic_rate,
                    1.0 + self.homeostatic_rate
                )
                nc_indices = np.where(nc_mask)[0]
                outlier_indices = nc_indices[outlier]
                self.w_hidden[l][outlier_indices] *= scale[outlier, np.newaxis]
            # バッファリセット
            self._nc_act_accum = [np.zeros(self.n_hidden[l]) for l in range(self.n_layers)]
            self._nc_act_count = 0

        # Phase 4: SST型動的バイアス更新
        # 平均活性が目標(sst_target)より高いニューロンのバイアスを増やし次エポックで抑制
        if self.sst_rate > 0.0 and self._sst_act_count > 0:
            for l in range(self.n_layers):
                mean_act = self._sst_act_accum[l] / self._sst_act_count
                delta = self.sst_rate * (mean_act - self.sst_target)
                # ±sst_rateでクリップ（1エポックの最大変化幅を制限）
                delta = np.clip(delta, -self.sst_rate, self.sst_rate)
                self._sst_bias[l] = np.clip(
                    self._sst_bias[l] + delta,
                    -0.5, 0.5  # バイアス上限（tanh出力域の半分）
                )
            self._sst_act_accum = [np.zeros(self.n_hidden[l]) for l in range(self.n_layers)]
            self._sst_act_count = 0

    def collect_epoch_diagnostics(self, x_sample, y_sample, n_diag=500):
        """エポック終了時に診断情報を収集する（学習には影響しない読み取り専用）

        Args:
            x_sample: 診断用入力サンプル配列
            y_sample: 診断用ラベル配列
            n_diag: 診断に使うサンプル数
        Returns:
            dict: 各種診断統計
        """
        diag = {}
        n = min(n_diag, len(x_sample))
        indices = np.random.choice(len(x_sample), n, replace=False)

        # --- 層別活性化統計 ---
        layer_act_stats = []
        layer_amine_stats = []
        for layer_idx in range(self.n_layers):
            layer_act_stats.append({'abs_mean': [], 'saturation_rate': [], 'std': []})
            layer_amine_stats.append({'amine_true': [], 'amine_total': []})

        output_scores = {'true_score': [], 'pred_score': [], 'max_score': []}
        class_correct = np.zeros(self.n_output, dtype=int)
        class_total = np.zeros(self.n_output, dtype=int)
        confusion = np.zeros((self.n_output, self.n_output), dtype=int)

        for i in indices:
            x = x_sample[i]
            y = int(y_sample[i])
            z_hiddens, z_output, x_paired = self.forward(x)
            y_pred = np.argmax(z_output)

            # クラス別精度
            class_total[y] += 1
            if y_pred == y:
                class_correct[y] += 1
            confusion[y, y_pred] += 1

            # 出力スコア
            output_scores['true_score'].append(float(z_output[y]))
            output_scores['pred_score'].append(float(z_output[y_pred]))
            output_scores['max_score'].append(float(np.max(z_output)))

            # 層別活性化統計
            for layer_idx, z in enumerate(z_hiddens):
                abs_z = np.abs(z)
                layer_act_stats[layer_idx]['abs_mean'].append(float(np.mean(abs_z)))
                layer_act_stats[layer_idx]['saturation_rate'].append(
                    float(np.mean(abs_z > 0.95)))
                layer_act_stats[layer_idx]['std'].append(float(np.std(z)))

            # アミン濃度推定（正解クラスのerror_correct）
            error_correct = 1.0 - z_output[y]
            for layer_idx in range(self.n_layers):
                diffusion = self.u1 if layer_idx == self.n_layers - 1 else self.u2
                amine_at_layer = error_correct * (diffusion ** (self.n_layers - layer_idx))
                layer_amine_stats[layer_idx]['amine_true'].append(float(amine_at_layer))

        # 集約
        diag['layer_activation'] = []
        for layer_idx in range(self.n_layers):
            diag['layer_activation'].append({
                'abs_mean': float(np.mean(layer_act_stats[layer_idx]['abs_mean'])),
                'saturation_rate': float(np.mean(layer_act_stats[layer_idx]['saturation_rate'])),
                'std': float(np.mean(layer_act_stats[layer_idx]['std'])),
            })

        diag['layer_amine'] = []
        for layer_idx in range(self.n_layers):
            diag['layer_amine'].append({
                'mean': float(np.mean(layer_amine_stats[layer_idx]['amine_true'])),
            })

        # 重み統計
        diag['weight_stats'] = []
        for layer_idx in range(self.n_layers):
            w = self.w_hidden[layer_idx]
            diag['weight_stats'].append({
                'norm': float(np.linalg.norm(w)),
                'abs_mean': float(np.mean(np.abs(w))),
                'max': float(np.max(np.abs(w))),
            })
        w_out = self.w_output
        diag['output_weight'] = {
            'norm': float(np.linalg.norm(w_out)),
            'abs_mean': float(np.mean(np.abs(w_out))),
            'max': float(np.max(np.abs(w_out))),
        }

        # 出力スコア
        diag['output_scores'] = {
            'true_score_mean': float(np.mean(output_scores['true_score'])),
            'max_score_mean': float(np.mean(output_scores['max_score'])),
        }

        # クラス別精度
        diag['class_accuracy'] = {}
        for c in range(self.n_output):
            if class_total[c] > 0:
                diag['class_accuracy'][c] = float(class_correct[c] / class_total[c])
        diag['confusion'] = confusion

        return diag

    # ================================================================
    # 学習
    # ================================================================

    def train_one_sample(self, x, y_true):
        """1サンプルの学習"""
        z_hiddens, z_output, x_paired = self.forward(x)

        y_pred = np.argmax(z_output)
        correct = (y_pred == y_true)

        self.winner_selection_counts[y_pred] += 1
        self.total_training_samples += 1

        loss = cross_entropy_loss(z_output, y_true)

        self.update_weights(x_paired, z_hiddens, z_output, y_true, y_pred=y_pred)
        self.class_training_counts[y_true] += 1

        # Phase 2: ホメオスタティック調整用NCニューロン活性累積
        if self.homeostatic_rate > 0.0:
            for l, z in enumerate(z_hiddens):
                self._nc_act_accum[l] += np.abs(z)
            self._nc_act_count += 1

        # Phase 4: SST型動的バイアス補正用 全ニューロン活性累積
        if self.sst_rate > 0.0:
            for l, z in enumerate(z_hiddens):
                self._sst_act_accum[l] += np.abs(z)
            self._sst_act_count += 1

        return loss, correct

    def train_epoch(self, x_train, y_train, return_true_accuracy=True,
                    progress_callback=None):
        """
        1エポックの学習

        Args:
            x_train: 訓練データ
            y_train: 訓練ラベル
            return_true_accuracy: Trueなら学習後に全データを再評価（推奨）
            progress_callback: 進捗コールバック callback(network, i, n_samples)

        Returns:
            accuracy, loss
        """
        import time as _time
        n_samples = len(x_train)
        self._n_train_samples = n_samples
        total_loss = 0.0
        n_correct = 0
        _last_callback_time = _time.time()

        # エポック開始時に視床ゲート状態をリセット（shuffle後に前エポックの文脈を持ち込まない）
        self._prev_error = 0.0

        # Phase 2: ホメオスタティック累積バッファをエポック開始時にリセット
        if self.homeostatic_rate > 0.0:
            self._nc_act_accum = [np.zeros(self.n_hidden[l]) for l in range(self.n_layers)]
            self._nc_act_count = 0

        # Phase 4: SST累積バッファをエポック開始時にリセット
        if self.sst_rate > 0.0:
            self._sst_act_accum = [np.zeros(self.n_hidden[l]) for l in range(self.n_layers)]
            self._sst_act_count = 0

        for i in range(n_samples):
            loss, correct = self.train_one_sample(x_train[i], y_train[i])
            total_loss += loss
            if correct:
                n_correct += 1

            # 約5秒ごとにコールバック
            if progress_callback is not None:
                now = _time.time()
                if now - _last_callback_time >= 5.0:
                    _last_callback_time = now
                    progress_callback(self, i, n_samples)

        # エポック終了後の正則化（weight decay等）
        self.end_of_epoch_regularize()

        if return_true_accuracy:
            true_accuracy, true_loss = self.evaluate_parallel(x_train, y_train)
            return true_accuracy, true_loss
        else:
            return n_correct / n_samples, total_loss / n_samples

    # ================================================================
    # 評価
    # ================================================================

    def evaluate_parallel(self, x_test, y_test, batch_size=256,
                          return_per_class=False):
        """
        バッチ評価

        Args:
            x_test: テストデータ
            y_test: テストラベル
            batch_size: バッチサイズ
            return_per_class: クラス別精度も返すか

        Returns:
            accuracy, avg_loss [, class_accuracies]
        """
        n_samples = len(x_test)
        n_batches = (n_samples + batch_size - 1) // batch_size

        all_predictions = []
        all_losses = []

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)

            x_batch = x_test[start:end]
            y_batch = y_test[start:end]

            _, z_output_batch, _ = self.forward_batch(x_batch)

            y_pred_batch = np.argmax(z_output_batch, axis=1)
            all_predictions.extend(y_pred_batch)

            batch_losses = -np.log(
                z_output_batch[np.arange(len(y_batch)), y_batch] + 1e-10
            )
            all_losses.extend(batch_losses)

        all_predictions = np.array(all_predictions)
        n_correct = np.sum(all_predictions == y_test)
        accuracy = n_correct / n_samples
        avg_loss = np.mean(all_losses)

        if return_per_class:
            class_correct = np.zeros(self.n_output, dtype=int)
            class_total = np.zeros(self.n_output, dtype=int)
            for i in range(n_samples):
                class_total[y_test[i]] += 1
                if all_predictions[i] == y_test[i]:
                    class_correct[y_test[i]] += 1
            class_accuracies = []
            for c in range(self.n_output):
                class_accuracies.append(
                    class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
                )
            return accuracy, avg_loss, class_accuracies

        return accuracy, avg_loss

    def evaluate_with_errors(self, x_data, y_data, batch_size=256):
        """
        バッチ評価（不正解サンプル収集付き）

        Returns:
            accuracy, avg_loss, error_list[(sample_idx, true_label, pred_label), ...]
        """
        n_samples = len(x_data)
        n_batches = (n_samples + batch_size - 1) // batch_size

        all_predictions = []
        all_losses = []

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)

            x_batch = x_data[start:end]
            y_batch = y_data[start:end]

            _, z_output_batch, _ = self.forward_batch(x_batch)
            y_pred_batch = np.argmax(z_output_batch, axis=1)
            all_predictions.extend(y_pred_batch)

            batch_losses = -np.log(
                z_output_batch[np.arange(len(y_batch)), y_batch] + 1e-10
            )
            all_losses.extend(batch_losses)

        all_predictions = np.array(all_predictions)
        n_correct = np.sum(all_predictions == y_data)
        accuracy = n_correct / n_samples
        avg_loss = np.mean(all_losses)

        error_list = [
            (int(i), int(y_data[i]), int(all_predictions[i]))
            for i in range(n_samples)
            if all_predictions[i] != y_data[i]
        ]

        return accuracy, avg_loss, error_list

    # ================================================================
    # 統計
    # ================================================================

    def reset_winner_selection_stats(self):
        """勝者選択統計をリセット"""
        self.winner_selection_counts = np.zeros(self.n_output, dtype=int)
        self.total_training_samples = 0

    def reset_class_training_stats(self):
        """クラス学習回数統計をリセット"""
        self.class_training_counts = np.zeros(self.n_output, dtype=int)
