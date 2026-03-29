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
                 layer_learning_rates=None, seed=None):
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
            column_neurons: 各クラスのコラムニューロン数
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
        self.column_neurons = column_neurons
        self.gradient_clip = gradient_clip
        self.initial_amine = 1.0

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
            membership, positions, coords = create_column_membership(
                n_hidden=layer_size,
                n_classes=n_output,
                participation_rate=participation_rate,
                use_hexagonal=use_hexagonal,
                column_radius=base_column_radius,
                column_neurons=column_neurons
            )
            self.column_membership_all_layers.append(membership)
            self.neuron_positions_all_layers.append(positions)
            self.class_coords_all_layers.append(coords)

        # コラム構造の表示
        for layer_idx in range(self.n_layers):
            membership = self.column_membership_all_layers[layer_idx]
            counts = [int(np.sum(membership[c])) for c in range(n_output)]
            print(f"  コラム構造 Layer {layer_idx}: 各クラス{counts[0]}ニューロン")

        # ============================================
        # ランク依存学習率のルックアップテーブル（LUT）
        # ============================================
        # コラム内でのニューロンの活性値ランクに応じた学習率倍率
        # cn依存型線形減衰: rank=0(1位)で1.0、rank=cn-1で1/cn
        max_rank = 256
        cn = column_neurons if column_neurons is not None else 1
        self._learning_weight_lut = np.zeros(max_rank, dtype=np.float32)
        for i in range(max_rank):
            if i < cn:
                self._learning_weight_lut[i] = (cn - i) / cn
            # rank >= cn は 0.0（非コラムニューロンの学習には不参加）

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
        # 統計カウンタ
        # ============================================
        self.winner_selection_counts = np.zeros(n_output, dtype=int)
        self.total_training_samples = 0
        self.class_training_counts = np.zeros(n_output, dtype=int)

        # NCゲートマスク（互換性用、simple版では未使用）
        self._nc_gate_mask = None

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

        z_hiddens = []
        z_current = x_paired

        for layer_idx in range(self.n_layers):
            a_hidden = np.dot(self.w_hidden[layer_idx], z_current)
            z_hidden = tanh_activation(a_hidden)
            z_hiddens.append(z_hidden)
            z_current = z_hidden

        # 出力層: 重み積 → softmax
        a_output = np.dot(self.w_output, z_current)
        z_output = softmax(a_output)

        return z_hiddens, z_output, x_paired

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
            z_hiddens_batch.append(z_hidden)
            z_current = z_hidden

        a_output = np.dot(z_current, self.w_output.T)
        z_output_batch = softmax_batch(a_output)

        return z_hiddens_batch, z_output_batch, x_paired_batch

    # ================================================================
    # 勾配計算（コラムED法の核心）
    # ================================================================

    def _compute_gradients(self, x_paired, z_hiddens, z_output, y_true):
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
            amine_concentration[y_true, 0] = error_correct * self.initial_amine

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

            # ランクから学習率を取得（LUT参照）
            clamped_ranks = np.minimum(ranks, len(self._learning_weight_lut) - 1)
            learning_weights = self._learning_weight_lut[clamped_ranks]

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

            # 勾配クリッピング
            if self.gradient_clip > 0:
                delta_w_norms = np.linalg.norm(delta_w_batch, axis=1, keepdims=True)
                clip_mask = delta_w_norms > self.gradient_clip
                delta_w_batch = np.where(
                    clip_mask,
                    delta_w_batch * (self.gradient_clip / delta_w_norms),
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

    def update_weights(self, x_paired, z_hiddens, z_output, y_true):
        """
        重みの更新（ED法準拠、微分の連鎖律不使用）
        """
        gradients = self._compute_gradients(x_paired, z_hiddens, z_output, y_true)

        # 出力層の更新
        self.w_output += gradients['w_output']

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

        # 出力重みの正則化（微小な重み減衰）
        self.w_output *= (1.0 - 0.00001)

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

        self.update_weights(x_paired, z_hiddens, z_output, y_true)
        self.class_training_counts[y_true] += 1

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
        total_loss = 0.0
        n_correct = 0
        _last_callback_time = _time.time()

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
