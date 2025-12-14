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

import numpy as np
from .activation_functions import sigmoid, tanh_activation, softmax, cross_entropy_loss
from .neuron_structure import create_ei_pairs, create_ei_flags
from .amine_diffusion import (
    initialize_activity_association,
    update_activity_association,
    distribute_amine_by_output_weights
)
from .column_structure import (
    hex_distance,
    create_hexagonal_column_affinity,
    create_column_affinity,
    create_lateral_weights
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
                 use_layer_norm=False, gradient_clip=0.0, hyperparams=None):
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
            hyperparams: HyperParamsインスタンス（Noneなら個別パラメータ使用）
        
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
        
        # ★新機能★ 層ごとの適応的学習率（z_inputとsaturation_termのスケールに対応）
        # 分析結果：
        #   - Layer 1の平均|Δw|: 0.120520
        #   - Layer 0の平均|Δw|: 0.407937（3.54倍）
        #   - 原因1: 最大|z_input|の差（Layer 0: 1.0, Layer 1: 0.5-0.8）
        #   - 原因2: saturation_termの差（Layer 0が極端に小さい: 1/17）
        # 修正不要：saturation_termとz_inputの自然な違いを保持
        # → 層ごとの学習率は同じに戻す
        self.layer_specific_lr = [learning_rate] * self.n_layers
        
        self.lateral_lr = lateral_lr  # 側方抑制の学習率
        self.u1 = u1  # アミン拡散係数（出力層→最終隠れ層）
        self.u2 = u2  # アミン拡散係数（隠れ層間）
        self.initial_amine = 1.0  # 基準アミン濃度
        
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
        
        # 側方抑制（必須要素6）- ゼロ初期化、学習中に動的更新
        self.lateral_weights = create_lateral_weights(n_output)
        
        # ★重要★ コラム帰属度マップの初期化（ハニカム構造版）
        self.column_affinity_all_layers = []
        for layer_idx, layer_size in enumerate(self.n_hidden):
            # 各層に対応するradiusを取得
            layer_radius = self.column_radius_per_layer[layer_idx]
            
            if use_hexagonal:
                # ハニカム構造（2-3-3-2配置）
                affinity = create_hexagonal_column_affinity(
                    n_hidden=layer_size,
                    n_classes=n_output,
                    column_radius=layer_radius,
                    column_neurons=column_neurons,
                    participation_rate=participation_rate
                )
            else:
                # 円環構造（v027更新: participation_rate対応）
                affinity = create_column_affinity(
                    n_hidden=layer_size,
                    n_classes=n_output,
                    column_size=int(layer_radius * 10),  # 層ごとのradiusをsizeに変換
                    overlap=overlap,
                    use_gaussian=True,
                    column_neurons=column_neurons,
                    participation_rate=participation_rate
                )
            self.column_affinity_all_layers.append(affinity)
        
        print(f"\n[コラム構造初期化]")
        print(f"  - コラム帰属度マップ作成完了")
        if use_hexagonal:
            print(f"  - 方式: ハニカム構造(2-3-3-2配置)")
            if column_neurons is not None:
                print(f"  - モード: 完全コラム化（各クラス{column_neurons}ニューロン）")
                print(f"  - 参加率: {column_neurons * n_output / self.n_hidden[0] * 100:.1f}%")
            elif participation_rate is not None:
                print(f"  - モード: 参加率指定（{participation_rate * 100:.0f}%）")
                print(f"  - 各クラス約{int(self.n_hidden[0] * participation_rate / n_output)}ニューロン")
            else:
                print(f"  - モード: 半径ベース（radius={self.column_radius_per_layer[0]:.2f}）")
        else:
            print(f"  - 方式: 円環構造（v027更新: 中心化配置+participation_rate対応）")
            if column_neurons is not None:
                print(f"  - モード: 完全コラム化（各クラス{column_neurons}ニューロン）")
                print(f"  - 参加率: {column_neurons * n_output / self.n_hidden[0] * 100:.1f}%")
            elif participation_rate is not None:
                print(f"  - モード: 参加率指定（{participation_rate * 100:.0f}%）")
                print(f"  - 各クラス約{int(self.n_hidden[0] * participation_rate / n_output)}ニューロン")
            else:
                print(f"  - モード: 従来方式（コラムサイズ: {int(self.column_radius_per_layer[0] * 10)}）")
        
        for layer_idx, affinity in enumerate(self.column_affinity_all_layers):
            non_zero_counts = [np.count_nonzero(affinity[c] > 1e-8) for c in range(n_output)]
            print(f"  - 層{layer_idx+1}: 各クラスの帰属ニューロン数={non_zero_counts}")
        
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
            if layer_idx == 0:
                # Layer 0: より小さい初期値（元のXavierの0.3倍）で飽和を防ぐ
                # 理由: 入力次元が大きい(1568) → Wx の絶対値が大きくなる → tanh飽和
                # 0.1倍では小さすぎたため、0.3倍に調整
                scale = np.sqrt(1.0 / n_in) * 0.3
                print(f"  [重み初期化] Layer {layer_idx}: scale={scale:.4f} (飽和防止, 0.3x)")
            else:
                # Layer 1+: 少し小さめのXavier初期化（0.5倍）
                # Layer 0とのバランスを取るため
                scale = np.sqrt(1.0 / n_in) * 0.5
                print(f"  [重み初期化] Layer {layer_idx}: scale={scale:.4f} (調整, 0.5x)")
            
            w = np.random.randn(n_out, n_in) * scale
            self.w_hidden.append(w)
        
        # 出力層の重み
        self.w_output = np.random.randn(n_output, self.n_hidden[-1]) * np.sqrt(1.0 / self.n_hidden[-1])
        print(f"  [重み初期化] 出力層: scale={np.sqrt(1.0 / self.n_hidden[-1]):.4f}")
        
        # Dale's Principleの初期化（必須要素1）- 第1層のみ
        sign_matrix = np.outer(self.ei_flags_hidden[0], self.ei_flags_input)
        self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
        
        # アミン濃度の記憶領域（必須要素3）- 各層ごと
        self.amine_concentrations = []
        for layer_size in self.n_hidden:
            amine = np.zeros((n_output, layer_size, 2))
            self.amine_concentrations.append(amine)
    
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
        # ============================================
        # 1. 出力層の重み更新
        # ============================================
        target_probs = np.zeros(self.n_output)
        target_probs[y_true] = 1.0
        error_output = target_probs - z_output
        
        saturation_output = np.abs(z_output) * (1.0 - np.abs(z_output))
        delta_w_output = self.learning_rate * np.outer(
            error_output * saturation_output,
            z_hiddens[-1]
        )
        self.w_output += delta_w_output
        
        # ============================================
        # 2. 出力層のアミン濃度計算
        # ============================================
        winner_class = np.argmax(z_output)
        amine_concentration_output = np.zeros((self.n_output, 2))
        
        if winner_class == y_true:
            error_correct = 1.0 - z_output[y_true]
            if error_correct > 0:
                amine_concentration_output[y_true, 0] = error_correct * self.initial_amine
        else:
            error_winner = 0.0 - z_output[winner_class]
            if error_winner < 0:
                amine_concentration_output[winner_class, 1] = -error_winner * self.initial_amine
            
            error_correct = 1.0 - z_output[y_true]
            if error_correct > 0:
                lateral_effect = self.lateral_weights[winner_class, y_true]
                if lateral_effect < 0:
                    enhanced_amine = self.initial_amine * (1.0 - lateral_effect)
                else:
                    enhanced_amine = self.initial_amine
                amine_concentration_output[y_true, 0] = error_correct * enhanced_amine
            
            self.lateral_weights[winner_class, y_true] -= self.lateral_lr * (1.0 + self.lateral_weights[winner_class, y_true])
        
        # ============================================
        # 3. 多層アミン拡散と重み更新（逆順、微分の連鎖律不使用）
        # ============================================
        # ★重要★ ED法の原理：出力層のアミン濃度を全ての隠れ層で使用
        # 微分の連鎖律を使わず、誤差信号（アミン濃度）を直接各層に拡散
        
        # 出力層から第1層へ逆順にアミン拡散
        for layer_idx in range(self.n_layers - 1, -1, -1):
            # 入力の取得
            if layer_idx == 0:
                z_input = x_paired
            else:
                z_input = z_hiddens[layer_idx - 1]
            
            # 拡散係数の選択（最終層はu1、それ以外はu2）
            if layer_idx == self.n_layers - 1:
                diffusion_coef = self.u1
            else:
                diffusion_coef = self.u2
            
            # ========== ベクトル化された重み更新（4重ループを解消）==========
            # 元のループ構造:
            #   for class_idx (10クラス):
            #     for amine_type (興奮性/抑制性):
            #       for neuron_idx (512ニューロン):
            #         重み更新 (同じニューロンに複数回加算)
            # 
            # ベクトル化戦略:
            #   - [class, amine_type, neuron]の3次元配列で一括計算
            #   - 全組み合わせ(10×2=20)の更新を一度に実行
            
            # 1. 有意なアミンのマスク (閾値1e-8以上)
            amine_mask = amine_concentration_output >= 1e-8  # [n_output, 2]
            
            # 2. コラム帰属度による重み付け拡散 (ブロードキャスト)
            # amine_concentration_output[:, :, np.newaxis]: [n_output, 2, 1]
            # column_affinity_all_layers[layer_idx][:, np.newaxis, :]: [n_output, 1, n_hidden]
            # 結果: [n_output, 2, n_hidden] (各クラス×各アミンタイプ×各ニューロン)
            amine_hidden_3d = (
                amine_concentration_output[:, :, np.newaxis] * 
                diffusion_coef * 
                self.column_affinity_all_layers[layer_idx][:, np.newaxis, :]
            )
            
            # 3. マスク適用（有意なアミンのみ処理）
            amine_hidden_3d = amine_hidden_3d * amine_mask[:, :, np.newaxis]
            
            # 4. ニューロンマスク：少なくとも1つのクラス/アミンで閾値超え
            neuron_mask = np.any(amine_hidden_3d >= 1e-8, axis=(0, 1))  # [n_hidden]
            active_neurons = np.where(neuron_mask)[0]
            
            if len(active_neurons) == 0:
                continue  # 有意なニューロンがなければスキップ
            
            # 5. 活性化関数の勾配（saturation_term）をベクトル化
            z_active = z_hiddens[layer_idx][active_neurons]  # [n_active]
            
            if self.activation == 'leaky_relu':
                saturation_term_raw = np.where(z_active > 0, 1.0, self.leaky_alpha)
            else:  # sigmoid
                saturation_term_raw = np.abs(z_active) * (1.0 - np.abs(z_active))
            
            saturation_term = np.maximum(saturation_term_raw, 1e-3)  # [n_active]
            
            # 6. 学習信号強度の計算（3次元配列のまま）
            layer_lr = self.layer_specific_lr[layer_idx]
            # amine_hidden_3d[:, :, active_neurons]: [n_output, 2, n_active]
            # saturation_term[np.newaxis, np.newaxis, :]: [1, 1, n_active]
            # 結果: [n_output, 2, n_active]
            learning_signals_3d = (
                layer_lr * 
                amine_hidden_3d[:, :, active_neurons] * 
                saturation_term[np.newaxis, np.newaxis, :]
            )
            
            # 7. 重み更新の計算（バッチ処理）
            # learning_signals_3d: [n_output, 2, n_active]
            # z_input: [n_input]
            # 
            # 戦略：各（class, amine）ペアごとの更新を計算し、最後に加算
            # 
            # learning_signals_3d.reshape(-1, n_active): [n_output*2, n_active]
            # これを転置: [n_active, n_output*2]
            # 外積: [n_active, n_output*2] × [n_input] → [n_active, n_output*2, n_input]
            # 最後にn_output*2次元を合算
            
            n_combinations = self.n_output * 2  # 10クラス × 2アミンタイプ = 20
            learning_signals_flat = learning_signals_3d.reshape(n_combinations, -1).T  # [n_active, 20]
            
            # 各組み合わせの重み更新を計算
            # learning_signals_flat[:, :, np.newaxis]: [n_active, 20, 1]
            # z_input[np.newaxis, np.newaxis, :]: [1, 1, n_input]
            # 結果: [n_active, 20, n_input]
            delta_w_3d = learning_signals_flat[:, :, np.newaxis] * z_input[np.newaxis, np.newaxis, :]
            
            # 8. 全組み合わせの更新を合算（元のコードの複数回加算を再現）
            delta_w_batch = np.sum(delta_w_3d, axis=1)  # [n_active, n_input]
            
            # 9. 層ごとの重み更新ルールを適用
            if layer_idx == 0:
                # 第1層: Dale's Principle適用（符号は後で強制）
                pass  # delta_w_batchはそのまま
            else:
                # 第2層以降: 符号保持
                w_sign = np.sign(self.w_hidden[layer_idx][active_neurons, :])
                w_sign[w_sign == 0] = 1
                delta_w_batch *= w_sign
            
            # 10. gradient clipping（行ごとにクリッピング）
            if self.gradient_clip > 0:
                delta_w_norms = np.linalg.norm(delta_w_batch, axis=1, keepdims=True)  # [n_active, 1]
                clip_mask = delta_w_norms > self.gradient_clip
                delta_w_batch = np.where(
                    clip_mask,
                    delta_w_batch * (self.gradient_clip / delta_w_norms),
                    delta_w_batch
                )
            
            # 11. 重み更新の適用
            self.w_hidden[layer_idx][active_neurons, :] += delta_w_batch
            
            # 第1層の場合、Dale's Principleで符号強制
            if layer_idx == 0:
                sign_matrix = np.outer(self.ei_flags_hidden[0], self.ei_flags_input)
                self.w_hidden[0] = np.abs(self.w_hidden[0]) * sign_matrix
        
        # 出力重みの軽度な正則化
        weight_penalty = 0.00001 * self.w_output
        self.w_output -= weight_penalty
    
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
        
        # 重みの更新
        self.update_weights(x_paired, z_hiddens, z_output, y_true)
        
        return loss, correct
    
    def train_epoch(self, x_train, y_train):
        """
        1エポックの学習
        
        Returns:
            accuracy: 訓練精度
            loss: 平均損失
        """
        n_samples = len(x_train)
        total_loss = 0.0
        n_correct = 0
        
        for i in range(n_samples):
            loss, correct = self.train_one_sample(x_train[i], y_train[i])
            total_loss += loss
            if correct:
                n_correct += 1
        
        accuracy = n_correct / n_samples
        avg_loss = total_loss / n_samples
        
        return accuracy, avg_loss
    
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
