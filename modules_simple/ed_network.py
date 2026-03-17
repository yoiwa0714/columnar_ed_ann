#!/usr/bin/env python3
"""
ED法ネットワークモジュール（公開版・簡易構成）

★Pure ED + Column Structure Network★
  - RefinedDistributionEDNetworkクラス
  - 多層多クラス分類対応
  - ED法準拠（微分の連鎖律不使用）
  - HyperParams統合対応

使用例:
    from modules_simple.ed_network import RefinedDistributionEDNetwork
    from modules_simple.hyperparameters import HyperParams

    hp = HyperParams()
    config = hp.get_config(n_layers=1)

    network = RefinedDistributionEDNetwork(
        n_input=784,
        n_hidden=config['hidden'],
        n_output=10,
        learning_rate=config['learning_rate'],
        base_column_radius=config['base_column_radius']
    )

    network.train_one_sample(x, y_true)
    accuracy, loss = network.evaluate(x_test, y_test)
"""

import numpy as np

from .activation_functions import tanh_activation, softmax, cross_entropy_loss
from .neuron_structure import create_ei_pairs
from .column_structure import create_column_membership


def softmax_batch(a):
    """バッチ対応softmax（数値安定版）"""
    a_shifted = a - np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a_shifted)
    return exp_a / np.sum(exp_a, axis=1, keepdims=True)


def create_ei_pairs_batch(x_batch):
    """バッチ対応の興奮性・抑制性入力ペア生成"""
    return np.concatenate([x_batch, x_batch], axis=1)


class RefinedDistributionEDNetwork:
    """
    Pure ED + Column Structure Network（公開版）

    金子勇氏考案のError Diffusion (ED)法に大脳皮質のコラム構造を導入。
    微分の連鎖律を用いず、アミン拡散機構による学習を行う。
    """

    def __init__(self, n_input=784, n_hidden=[250], n_output=10,
                 output_lr=0.15, non_column_lr=None, column_lr=None,
                 u1=0.5, u2=0.8,
                 base_column_radius=1.0,
                 column_neurons=None, participation_rate=1.0,
                 use_hexagonal=True,
                 gradient_clip=0.0,
                 hidden_sparsity=None,
                 init_scales=None,
                 seed=None,
                 verbose=False):
        """
        Args:
            n_input: 入力次元数（784 for MNIST）
            n_hidden: 隠れ層ニューロン数のリスト（例: [2048] or [2048, 1024]）
            n_output: 出力クラス数
            output_lr: 出力層の学習率（デフォルト: 0.15）
            non_column_lr: 非コラムニューロンの学習率（層別リスト、Noneならoutput_lrと同値）
            column_lr: コラムニューロンの学習率（層別リスト、Noneならnon_column_lrと同値）
            u1: アミン拡散係数（出力層→最終隠れ層）
            u2: アミン拡散係数（隠れ層間）
            base_column_radius: 基準コラム半径（256ニューロン層での値）
            column_neurons: 各クラスのコラムに割り当てるニューロン数
            participation_rate: コラム参加率（0.0-1.0）
            use_hexagonal: Trueならハニカム構造
            gradient_clip: 勾配クリッピング値（0で無効）
            hidden_sparsity: 隠れ層スパース率リスト
            init_scales: 層別初期化スケール係数
            seed: 乱数シード（未使用、互換性のため）
            verbose: Trueなら詳細な初期化情報を表示
        """
        self.verbose = verbose
        # パラメータ保存
        self.n_input = n_input
        self.n_hidden = n_hidden if isinstance(n_hidden, list) else [n_hidden]
        self.n_layers = len(self.n_hidden)
        self.n_output = n_output

        # 3系統学習率 (output_lr / non_column_lr / column_lr)
        self._output_lr = output_lr
        if non_column_lr is not None:
            if isinstance(non_column_lr, list):
                self._non_column_lr = non_column_lr
            else:
                self._non_column_lr = [non_column_lr] * self.n_layers
        else:
            self._non_column_lr = [output_lr] * self.n_layers
        if column_lr is not None:
            if isinstance(column_lr, list):
                self._column_lr = column_lr
            else:
                self._column_lr = [column_lr] * self.n_layers
        else:
            self._column_lr = list(self._non_column_lr)

        # 内部パラメータの計算（勾配計算で使用）
        # layer_lrs: non_column_lrを基準に設定（隠れ層部分 + 出力層）
        self.layer_lrs = list(self._non_column_lr) + [self._output_lr]
        # column_lr_factors: column_lr / non_column_lr の比率として計算
        self.column_lr_factors = []
        for i in range(self.n_layers):
            if self._non_column_lr[i] > 0:
                self.column_lr_factors.append(self._column_lr[i] / self._non_column_lr[i])
            else:
                self.column_lr_factors.append(0.0)

        if self.verbose:
            print(f"\n[3系統学習率]")
            print(f"  output_lr: {self._output_lr:.6f}")
            for i in range(self.n_layers):
                print(f"  Layer {i}: non_column_lr={self._non_column_lr[i]:.6f}, column_lr={self._column_lr[i]:.6f}")

        self.u1 = u1
        self.u2 = u2
        self.initial_amine = 1.0

        # 層依存のcolumn_radius（sqrtスケーリング）
        self.base_column_radius = base_column_radius
        self.column_radius_per_layer = [
            base_column_radius * np.sqrt(n / 256.0) for n in self.n_hidden
        ]
        if self.verbose:
            print(f"\n[層依存column_radius自動計算]")
            for i, (n, r) in enumerate(zip(self.n_hidden, self.column_radius_per_layer)):
                print(f"  Layer {i}: {n}ニューロン → radius={r:.2f}")

        self.column_neurons = column_neurons
        self.participation_rate = participation_rate
        self.use_hexagonal = use_hexagonal
        self.gradient_clip = gradient_clip

        # 隠れ層スパース率の正規化
        if hidden_sparsity is None:
            self.hidden_sparsity = [0.0] * len(self.n_hidden)
        else:
            self.hidden_sparsity = list(hidden_sparsity)
            if len(self.hidden_sparsity) != len(self.n_hidden):
                raise ValueError(
                    f"hidden_sparsityの数({len(self.hidden_sparsity)})が"
                    f"隠れ層数({len(self.n_hidden)})と一致しません。"
                )

        # 層別初期化スケール係数
        if init_scales is None:
            n_layers = len(self.n_hidden)
            if n_layers == 1:
                init_scales = [0.4, 1.0]
            elif n_layers == 2:
                init_scales = [0.3, 0.5, 1.0]
            elif n_layers == 3:
                init_scales = [0.3, 0.5, 0.7, 1.0]
            elif n_layers == 4:
                init_scales = [0.3, 0.5, 0.7, 0.9, 1.0]
            else:
                init_scales = [0.3] + [0.3 + (0.7 * i / (n_layers - 1)) for i in range(1, n_layers)] + [1.0]
        self.init_scales = init_scales

        # ランク依存学習率LUT（コラムニューロンの活性値ランクに応じた学習率）
        # column_neuronsに応じた線形減衰: 1位が最大、下位ほど弱い学習信号
        max_rank_in_lut = 256
        self._learning_weight_lut = np.zeros(max_rank_in_lut, dtype=np.float32)
        cn = self.column_neurons if self.column_neurons is not None else 1
        for i in range(max_rank_in_lut):
            if i < cn:
                self._learning_weight_lut[i] = (cn - i) / cn
            else:
                self._learning_weight_lut[i] = 0.0

        # コラム構造の初期化（Membership方式）
        self.column_membership_all_layers = []
        self.neuron_positions_all_layers = []
        self.class_coords_all_layers = []

        for layer_idx, layer_size in enumerate(self.n_hidden):
            layer_radius = self.column_radius_per_layer[layer_idx]

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

        if self.verbose:
            print(f"\n[コラム構造初期化]")
            print(f"  - Membership方式（ブールフラグ、勝者=活性値ランクベース）")
            if use_hexagonal:
                print(f"  - 方式: ハニカム構造(2-3-3-2配置)")
            if column_neurons is not None:
                print(f"  - モード: 完全コラム化（各クラス{column_neurons}ニューロン）")
                print(f"  - 参加率: {column_neurons * n_output / self.n_hidden[0] * 100:.1f}%")
            elif participation_rate is not None:
                print(f"  - モード: 参加率指定（{participation_rate * 100:.0f}%）")

            for layer_idx in range(len(self.column_membership_all_layers)):
                membership = self.column_membership_all_layers[layer_idx]
                non_zero_counts = [int(np.count_nonzero(membership[c])) for c in range(n_output)]
                print(f"  - 層{layer_idx+1}: 各クラスの帰属ニューロン数={non_zero_counts}")

        # 興奮性・抑制性フラグ
        n_input_paired = n_input * 2
        self.ei_flags_input = np.array([1 if i < n_input else -1
                                        for i in range(n_input_paired)])

        # 各隠れ層のユニット変換ベクトル（Dale's Principle用、全て+1=興奮性）
        self._ones_hidden = []
        for layer_size in self.n_hidden:
            self._ones_hidden.append(np.ones(layer_size))

        # ================================================================
        # 重みの初期化
        # ================================================================
        self.w_hidden = []
        for layer_idx in range(self.n_layers):
            if layer_idx == 0:
                n_in = n_input_paired
                n_out = self.n_hidden[0]
            else:
                n_in = self.n_hidden[layer_idx - 1]
                n_out = self.n_hidden[layer_idx]

            # He初期化
            base_scale = np.sqrt(2.0 / n_in)
            layer_scale = self.init_scales[layer_idx]
            scale = base_scale * layer_scale
            w = np.random.randn(n_out, n_in) * scale
            if self.verbose:
                print(f"  [重み初期化] Layer {layer_idx}: He初期化, "
                      f"scale={scale:.4f} (base={base_scale:.4f}×{layer_scale})")

            # 非コラムニューロンのスパース化（コラム内は密結合維持）
            layer_sparsity = self.hidden_sparsity[layer_idx] if layer_idx < len(self.hidden_sparsity) else 0.0
            if layer_sparsity > 0.0 and layer_idx < len(self.column_membership_all_layers):
                membership = self.column_membership_all_layers[layer_idx]
                is_column_neuron = np.any(membership, axis=0)
                non_column_mask = ~is_column_neuron
                n_non_column = np.sum(non_column_mask)
                n_zeros_hidden = 0
                if n_non_column > 0:
                    non_column_indices = np.where(non_column_mask)[0]
                    for neuron_idx in non_column_indices:
                        sparsity_mask = np.random.rand(n_in) < layer_sparsity
                        w[neuron_idx, sparsity_mask] = 0.0
                        n_zeros_hidden += np.sum(sparsity_mask)
                    total_weights = n_non_column * n_in
                    n_column = n_out - n_non_column
                    if self.verbose:
                        print(f"  [隠れ層スパース化] Layer {layer_idx}: "
                              f"コラム{n_column}個(密結合維持), 非コラム{n_non_column}個, "
                              f"{n_zeros_hidden}/{total_weights}重みを0に設定 "
                              f"(実効率{n_zeros_hidden/total_weights*100:.1f}%, 指定率{layer_sparsity*100:.1f}%)")

            self.w_hidden.append(w)

        # 出力層の重み初期化（He初期化）
        fan_in = self.n_hidden[-1]
        output_scale = self.init_scales[-1]
        std_he = np.sqrt(2.0 / fan_in) * output_scale
        self.w_output = np.random.randn(n_output, fan_in) * std_he
        if self.verbose:
            print(f"  [重み初期化] Output Layer: He初期化, std={std_he:.4f}")
            actual_abs_means = [np.abs(self.w_output[c]).mean() for c in range(n_output)]
            print(f"  [重み初期化] クラス別絶対値平均: min={min(actual_abs_means):.6f}, "
                  f"max={max(actual_abs_means):.6f}, mean={np.mean(actual_abs_means):.6f}")

        # Dale's Principleの初期化 ─ 第1層のみ
        # 生物学的制約: 各ニューロンは興奮性または抑制性のいずれか一方のみ
        self._sign_matrix_layer0 = np.outer(self._ones_hidden[0], self.ei_flags_input)
        self.w_hidden[0] = np.abs(self.w_hidden[0]) * self._sign_matrix_layer0

        # 非verbose時のサマリー表示
        if not self.verbose:
            cn_str = f"cn={column_neurons}" if column_neurons is not None else f"pr={participation_rate}"
            print(f"ネットワーク初期化: {self.n_layers}層 {self.n_hidden}, "
                  f"{cn_str}, He初期化, is={self.init_scales}")

    # ================================================================
    # 順伝播
    # ================================================================

    def forward(self, x):
        """
        順伝播（多クラス分類版）

        隠れ層: tanh活性化（双方向性、飽和特性）
        出力層: SoftMax（確率分布化）

        Args:
            x: 入力データ（shape: [n_input]）

        Returns:
            z_hiddens: 各隠れ層の出力のリスト
            z_output: 出力層の確率分布（SoftMax、合計=1.0）
            x_paired: 入力ペア（興奮性+抑制性）
        """
        # 入力ペア構造: x → [x, x]（Dale's Principleの符号行列で興奮性・抑制性を実現）
        x_paired = create_ei_pairs(x)

        z_hiddens = []
        z_current = x_paired

        for layer_idx in range(self.n_layers):
            a_hidden = np.dot(self.w_hidden[layer_idx], z_current)
            z_hidden = tanh_activation(a_hidden)
            z_hiddens.append(z_hidden)
            z_current = z_hidden

        # 出力層: SoftMax活性化
        a_output = np.dot(self.w_output, z_hiddens[-1])
        z_output = softmax(a_output)

        return z_hiddens, z_output, x_paired

    # ================================================================
    # 勾配計算（ED法準拠、微分の連鎖律不使用）
    # ================================================================

    def _compute_gradients(self, x_paired, z_hiddens, z_output, y_true):
        """
        ED法による勾配計算

        ★核心★ 微分の連鎖律を使わない。代わりに:
        1. 出力層の誤差からアミン濃度を生成
        2. アミンがコラム構造に沿って隠れ層に拡散
        3. 各層は独立に飽和抑制項 abs(z)*(1-abs(z)) で学習

        Args:
            x_paired: 入力ペア
            z_hiddens: 各隠れ層の出力
            z_output: 出力層の確率分布
            y_true: 正解クラス

        Returns:
            gradients: 各層の勾配辞書
        """
        gradients = {
            'w_output': None,
            'w_hidden': [None] * self.n_layers,
        }

        # ============================================
        # 1. 出力層の勾配計算
        # ============================================
        target_probs = np.zeros(self.n_output)
        target_probs[y_true] = 1.0
        error_output = target_probs - z_output

        # ★飽和抑制項★ ED法の核心 ─ シグモイド微分ではない
        saturation_output = np.abs(z_output) * (1.0 - np.abs(z_output))

        output_lr = self.layer_lrs[-1]
        gradients['w_output'] = output_lr * np.outer(
            error_output * saturation_output,
            z_hiddens[-1]
        )

        # ============================================
        # 2. 出力層のアミン濃度計算
        # ============================================
        # 純粋ED法: 正解クラスのみ学習
        amine_concentration = np.zeros(self.n_output)
        error_correct = 1.0 - z_output[y_true]
        if error_correct > 0:
            amine_concentration[y_true] = error_correct * self.initial_amine

        # ============================================
        # 3. 多層アミン拡散と勾配計算（逆順、微分の連鎖律不使用）
        # ============================================
        for layer_idx in range(self.n_layers - 1, -1, -1):
            if layer_idx == 0:
                z_input = x_paired
            else:
                z_input = z_hiddens[layer_idx - 1]

            # 拡散係数の選択（最終隠れ層=u1、それ以外=u2）
            if layer_idx == self.n_layers - 1:
                diffusion_coef = self.u1
            else:
                diffusion_coef = self.u2

            # アミン拡散（コラム構造に沿って選択的に拡散）
            amine_mask = amine_concentration >= 1e-8
            amine_diffused = amine_concentration * diffusion_coef

            # Membership方式: 活性値ランクベースTop-K学習
            membership = self.column_membership_all_layers[layer_idx]
            z_current = z_hiddens[layer_idx]
            n_neurons = self.n_hidden[layer_idx]

            active_classes = np.where(amine_diffused >= 1e-8)[0]
            n_active = len(active_classes)

            if n_active == 0:
                amine_hidden = np.zeros((self.n_output, n_neurons))
            else:
                # コラムメンバーの活性値でランク計算
                active_membership = membership[active_classes]
                masked_activations = np.where(active_membership, z_current, -np.inf)
                sorted_indices = np.argsort(-masked_activations, axis=1)
                ranks = np.argsort(sorted_indices, axis=1)

                # ランクからLUT参照で学習率を取得
                clamped_ranks = np.minimum(ranks, len(self._learning_weight_lut) - 1)
                learning_weights = self._learning_weight_lut[clamped_ranks]

                # 非コラムニューロンはamine=0（学習しない）
                learning_weights = np.where(active_membership, learning_weights, 0.0)

                # アミン拡散値に学習率を適用
                amine_hidden = np.zeros((self.n_output, n_neurons))
                amine_hidden[active_classes] = (
                    amine_diffused[active_classes, np.newaxis] *
                    learning_weights
                )

            amine_hidden = amine_hidden * amine_mask[:, np.newaxis]

            # 活性ニューロンの特定（アミン非ゼロの行のみ更新）
            neuron_mask = np.any(amine_hidden >= 1e-8, axis=0)
            active_neurons = np.where(neuron_mask)[0]

            if len(active_neurons) == 0:
                gradients['w_hidden'][layer_idx] = None
                continue

            # ★飽和抑制項★ abs(z)*(1-abs(z)) ─ 微分の連鎖律ではない
            z_active = z_hiddens[layer_idx][active_neurons]
            saturation_term_raw = np.abs(z_active) * (1.0 - np.abs(z_active))
            saturation_term = np.maximum(saturation_term_raw, 1e-3)

            # 学習信号 = 学習率 × アミン拡散量 × 飽和抑制項
            layer_lr = self.layer_lrs[layer_idx]
            learning_signals = (
                layer_lr *
                amine_hidden[:, active_neurons] *
                saturation_term[np.newaxis, :]
            )

            # 勾配計算（全クラスの信号を合計 × 入力でouter product）
            signal_sum = learning_signals.sum(axis=0)
            delta_w_batch = signal_sum[:, np.newaxis] * z_input[np.newaxis, :]

            # 第2層以降の符号制約
            if layer_idx > 0:
                w_sign = np.sign(self.w_hidden[layer_idx][active_neurons, :])
                w_sign[w_sign == 0] = 1
                delta_w_batch *= w_sign

            # 勾配クリッピング
            if self.gradient_clip > 0:
                delta_w_norms = np.linalg.norm(delta_w_batch, axis=1, keepdims=True)
                clip_mask = delta_w_norms > self.gradient_clip
                delta_w_batch = np.where(
                    clip_mask,
                    delta_w_batch * (self.gradient_clip / delta_w_norms),
                    delta_w_batch
                )

            # コラムニューロンの学習率抑制（重み飽和防止）
            layer_lr_factor = self.column_lr_factors[layer_idx]
            if layer_lr_factor < 1.0 and layer_idx < len(self.column_membership_all_layers):
                membership = self.column_membership_all_layers[layer_idx]
                is_column_neuron = np.any(membership, axis=0)
                active_is_column = is_column_neuron[active_neurons]
                if np.any(active_is_column):
                    delta_w_batch[active_is_column, :] *= layer_lr_factor

            # スパース形式で保存（active行のみ更新）
            gradients['w_hidden'][layer_idx] = (active_neurons, delta_w_batch)

        return gradients

    # ================================================================
    # 重み更新
    # ================================================================

    def update_weights(self, x_paired, z_hiddens, z_output, y_true):
        """
        重みの更新（ED法準拠、微分の連鎖律不使用）

        各層が独立にアミン拡散信号に基づいて重み更新を行う。
        第1層にはDale's Principleの符号制約を適用。
        """
        gradients = self._compute_gradients(x_paired, z_hiddens, z_output, y_true)

        # 勾配適用
        self.w_output += gradients['w_output']

        for layer_idx in range(self.n_layers):
            sparse_grad = gradients['w_hidden'][layer_idx]
            if sparse_grad is not None:
                active_neurons, delta_w_batch = sparse_grad
                self.w_hidden[layer_idx][active_neurons] += delta_w_batch

                # 第1層のDale's Principle強制（active行のみ）
                if layer_idx == 0:
                    self.w_hidden[0][active_neurons] = (
                        np.abs(self.w_hidden[0][active_neurons]) *
                        self._sign_matrix_layer0[active_neurons]
                    )

        # 出力重みの微弱正則化
        self.w_output *= (1.0 - 0.00001)

    # ================================================================
    # 学習
    # ================================================================

    def train_one_sample(self, x, y_true):
        """
        1サンプルの学習（オンライン学習）

        Args:
            x: 入力データ
            y_true: 正解クラス

        Returns:
            loss: Cross-Entropy損失
            correct: 正解か否か
        """
        z_hiddens, z_output, x_paired = self.forward(x)

        y_pred = np.argmax(z_output)
        correct = (y_pred == y_true)

        loss = cross_entropy_loss(z_output, y_true)

        # 純粋ED法: 正解クラスのみ学習
        self.update_weights(x_paired, z_hiddens, z_output, y_true)

        return loss, correct

    def train_epoch(self, x_train, y_train, return_true_accuracy=True,
                    collect_errors=False, progress_callback=None):
        """
        1エポックの学習

        Args:
            x_train: 訓練データ
            y_train: 訓練ラベル
            return_true_accuracy: True=学習後再評価で真の訓練精度を返す（推奨）
            collect_errors: True=不正解サンプル情報を収集して返す
            progress_callback: callback(network, i, n_samples) 形式

        Returns:
            accuracy: 訓練精度
            loss: 平均損失
            errors: collect_errors=Trueの場合のみ、(index, true_label, pred_label)のリスト
        """
        import time as _time
        n_samples = len(x_train)
        total_loss = 0.0
        n_correct = 0
        _last_callback_time = _time.time()

        for i in range(n_samples):
            loss, correct = self.train_one_sample(x_train[i], y_train[i])
            total_loss += loss
            if correct:
                n_correct += 1

            # 約5秒ごとにコールバック（ヒートマップ更新等）
            if progress_callback is not None:
                now = _time.time()
                if now - _last_callback_time >= 5.0:
                    _last_callback_time = now
                    progress_callback(self, i, n_samples)

        training_accuracy = n_correct / n_samples
        avg_loss = total_loss / n_samples

        if return_true_accuracy:
            # 学習後の最終重みで再評価（テスト精度と公平な比較が可能）
            if collect_errors:
                true_accuracy, true_loss, errors = self.evaluate_with_errors(x_train, y_train)
                return true_accuracy, true_loss, errors
            else:
                true_accuracy, true_loss = self.evaluate_parallel(x_train, y_train)
                return true_accuracy, true_loss
        else:
            return training_accuracy, avg_loss

    # ================================================================
    # 評価
    # ================================================================

    def evaluate_parallel(self, x_test, y_test, batch_size=256, return_per_class=False):
        """
        並列バッチ評価（順伝播をバッチ処理で高速化）

        学習には影響しないため、精度は従来版と完全一致。

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

        all_predictions = []
        all_losses = []

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            x_batch = x_test[start_idx:end_idx]
            y_batch = y_test[start_idx:end_idx]

            # バッチ順伝播
            _, z_output_batch, _ = self.forward_batch(x_batch)

            y_pred_batch = np.argmax(z_output_batch, axis=1)
            all_predictions.extend(y_pred_batch)

            # Cross-Entropy損失
            batch_losses = -np.log(
                z_output_batch[np.arange(len(y_batch)), y_batch] + 1e-10
            )
            all_losses.extend(batch_losses)

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

    def forward_batch(self, x_batch):
        """
        バッチ順伝播（evaluate_parallel用）

        Args:
            x_batch: 入力データバッチ shape: [batch_size, n_input]

        Returns:
            z_hiddens_batch: 各隠れ層の出力リスト
            z_output_batch: 出力層の確率分布
            x_paired_batch: 入力ペアバッチ
        """
        x_paired_batch = create_ei_pairs_batch(x_batch)

        z_hiddens_batch = []
        z_current = x_paired_batch

        for layer_idx in range(self.n_layers):
            a_hidden_batch = np.dot(z_current, self.w_hidden[layer_idx].T)
            z_hidden_batch = tanh_activation(a_hidden_batch)
            z_hiddens_batch.append(z_hidden_batch)
            z_current = z_hidden_batch

        a_output_batch = np.dot(z_current, self.w_output.T)
        z_output_batch = softmax_batch(a_output_batch)

        return z_hiddens_batch, z_output_batch, x_paired_batch

    def evaluate_with_errors(self, x_data, y_data, batch_size=256):
        """
        評価と不正解サンプル情報の収集

        Args:
            x_data: 入力データ
            y_data: ラベル
            batch_size: バッチサイズ

        Returns:
            accuracy: 精度
            avg_loss: 平均損失
            errors: (index, true_label, pred_label)のリスト
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
        n_correct = np.sum(all_predictions == y_data)
        accuracy = n_correct / n_samples
        avg_loss = np.mean(all_losses)

        errors = []
        for i in range(n_samples):
            if all_predictions[i] != y_data[i]:
                errors.append((i, int(y_data[i]), int(all_predictions[i])))

        return accuracy, avg_loss, errors
