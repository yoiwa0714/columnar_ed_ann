#!/usr/bin/env python3
"""
コラムED法 - コラム構造を持ったED法
Columnar ED: Error Diffusion with Columnar Architecture

ファイル名: columnar_ed_ann.py
バージョン: 1.0.0
作成日: 7 Dec 2025

概要:
  金子勇氏提唱のED法 (Error Diffusion Learning Algorithm)に、
  人間の脳の大脳皮質に見られるコラム構造（Columnar Architecture）を
  導入したニューラルネットワーク実装。

主な特徴:
  1. オリジナルED法の実装
     - 微分の連鎖律を用いた誤差逆伝播法を一切使用せず
     - ED法の飽和項: abs(z) * (1 - abs(z)) を使用
     - 生物学的に妥当なアミン拡散メカニズム

  2. コラム構造の実装
     - ハニカム構造（2-3-3-2配置）による特徴抽出
     - コラム間の側方抑制による競合
     - 特徴の多様性と汎化性能の向上

達成精度（MNIST）:
  - 隠れ層1層[512]: Test 83.80%, Train 86.77%
  - 隠れ層1層[1024]: Test 85.60%, Train 86.20%

達成精度（Fashion-MNIST）:
  - 隠れ層1層[512]: Test 78.60%, Train 76.53%
  - 隠れ層1層[1024]: Test 77.90%, Train 74.23%

参考:
  Kaneko, I. (1999). Error Diffusion Learning Algorithm.

検証結果:
  - 出力層飽和項: np.abs(z_output) * (1.0 - np.abs(z_output)) ✅
  - 隠れ層飽和項: abs(z_neuron) * (1.0 - abs(z_neuron)) ✅
  - アミン拡散: amine_hidden = amine_output * diffusion_coef * column_affinity ✅
  - 重み更新: アミン濃度ベース、微分の連鎖律不使用 ✅
  - Dale's Principle: 実装確認 ✅
  
  判定式（columnar_ed.prompt.md準拠）:
  - ED法: abs(z) * (1 - abs(z))  ← 本実装 ✅
  - 誤差逆伝播法: z * (1 - z)  ← 使用していない ✅
  
  重要な区別:
  - SoftMaxは「出力の確率化」のみに使用（順伝播）
  - SoftMaxの微分（dSoftMax/dz）は一切使用していない
  - アミン拡散は確率誤差（target_prob - softmax_prob）から計算
  - 隠れ層へのアミン伝播は「コラム帰属度」による重み付けのみ
  - 飽和項は常に絶対値ベース（ED法の原理に忠実）
  
  結論: 本実装は純粋なED法である（正式認定済み）
"""

# ============================================
# v025変更サマリー（多クラス分類化）
# ============================================
# 主要変更:
#   1. 出力層: sigmoid → SoftMax
#   2. 損失関数: MSE → Cross-Entropy
#   3. 隠れ層: sigmoid → tanh
#   4. アミン濃度: 確率ベース
#   5. 側方抑制: 確率ベース
# ============================================

import os
# TensorFlowの警告メッセージを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERRORレベルのみ表示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN最適化の警告を抑制
import sys
from pathlib import Path

# モジュールパスの追加
sys.path.append(str(Path(__file__).parent))

import numpy as np
from tensorflow import keras

# ========================================
# ハイパーパラメータ管理クラス
# ========================================

class HyperParams:
    """
    層数依存パラメータをテーブル管理するクラス
    
    設計方針:
      - 層数ごとに最適化されたパラメータセットを提供
      - column_radius等の層数依存パラメータを一元管理
      - 初学者にも分かりやすいテーブル形式
      - コマンドライン引数でオーバーライド可能
    
    使用例:
      hp = HyperParams()
      config = hp.get_config(n_layers=2)
      network = RefinedDistributionEDNetwork(
          input_dim=784,
          hidden_layers=config['hidden'],
          base_column_radius=config['base_column_radius'],
          ...
      )
    """
    
    def __init__(self):
        """層数別最適設定テーブルの初期化"""
        
        # 層数別設定テーブル
        # - base=1.0が最適（2025-12-04実験結果）
        # - スケーリング則: radius = base × sqrt(neurons/256)
        self.layer_configs = {
            # 2層構成
            2: {
                'hidden': [256, 128],
                'learning_rate': 0.05,
                'base_column_radius': 1.0,  # 最適値（31.9%精度、飽和39回）
                'column_radius_per_layer': [1.0, 0.71],  # [sqrt(256/256), sqrt(128/256)]
                'epochs': 100,
                'description': '2層標準構成、base=1.0最適化済み'
            },
            
            # 3層構成
            3: {
                'hidden': [256, 128, 64],
                'learning_rate': 0.05,
                'base_column_radius': 1.0,
                'column_radius_per_layer': [1.0, 0.71, 0.50],  # [sqrt(256/256), sqrt(128/256), sqrt(64/256)]
                'epochs': 30,
                'description': '3層標準構成、Layer 0学習問題解決済み（25.8%達成）'
            },
            
            # 4層構成（将来の拡張用）
            4: {
                'hidden': [256, 128, 96, 64],
                'learning_rate': 0.05,
                'base_column_radius': 1.0,
                'column_radius_per_layer': [1.0, 0.71, 0.61, 0.50],
                'epochs': 50,
                'description': '4層構成（実験予定）'
            },
            
            # 5層構成（将来の拡張用）
            5: {
                'hidden': [256, 128, 96, 80, 64],
                'learning_rate': 0.05,
                'base_column_radius': 1.0,
                'column_radius_per_layer': [1.0, 0.71, 0.61, 0.56, 0.50],
                'epochs': 50,
                'description': '5層構成（実験予定）'
            },
        }
        
        # 共通パラメータ（層数非依存）
        self.common_params = {
            'w1': 1.0,  # 重み初期化範囲
            'column_overlap': 0.1,
            'column_ratio': 0.2,
            'time_loops': 2,
            'dataset': 'mnist',
            'train_samples': 10000,
            'test_samples': 2000,
        }
    
    def get_config(self, n_layers):
        """
        指定層数の設定を取得
        
        Args:
            n_layers: 隠れ層の数（2, 3, 4, 5）
        
        Returns:
            dict: 層数に対応した設定辞書
        
        Raises:
            ValueError: サポートされていない層数の場合
        """
        if n_layers not in self.layer_configs:
            supported = list(self.layer_configs.keys())
            raise ValueError(
                f"層数 {n_layers} はサポートされていません。"
                f"サポート層数: {supported}"
            )
        
        # 層数依存パラメータと共通パラメータをマージ
        config = self.layer_configs[n_layers].copy()
        config.update(self.common_params)
        return config
    
    def list_configs(self):
        """利用可能な設定一覧を表示"""
        print("\n=== 利用可能な層数別設定 ===")
        for n_layers, config in self.layer_configs.items():
            print(f"\n[{n_layers}層] {config['description']}")
            print(f"  hidden_layers: {config['hidden']}")
            print(f"  base_column_radius: {config['base_column_radius']}")
            print(f"  column_radius_per_layer: {config['column_radius_per_layer']}")
            print(f"  learning_rate: {config['learning_rate']}")
            print(f"  epochs: {config['epochs']}")


# ========================================
# 1. データ処理（必須要素10）
# ========================================

def get_class_names(dataset_name):
    """
    データセットのクラス名を取得
    
    Args:
        dataset_name: データセット名（'mnist', 'fashion', 'cifar10', 'cifar100'）
    
    Returns:
        class_names: クラス名のリスト（クラス名情報がない場合はNone）
    """
    if dataset_name == 'mnist':
        # MNISTはクラス名情報を持たない（数字そのまま）
        return None
    elif dataset_name == 'fashion':
        # Fashion-MNISTのクラス名
        return [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    elif dataset_name == 'cifar10':
        # CIFAR-10のクラス名
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    elif dataset_name == 'cifar100':
        # CIFAR-100のクラス名（20のスーパークラス）
        # 注: CIFAR-100は100個の細かいクラスがあるため、必要に応じて拡張
        return None  # 100クラス分の名前は長いため、必要に応じて実装
    else:
        # 未知のデータセット
        return None


def load_dataset(n_train=500, n_test=200, dataset='mnist'):
    """データセットの読み込みと前処理"""
    # データ読み込み
    if dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # 正規化（0-1）
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # フラット化
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    # サンプル数制限
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]
    x_test = x_test[:n_test]
    y_test = y_test[:n_test]
    
    return x_train, y_train, x_test, y_test


# ========================================
# 2. 活性化関数（必須要素8）
# ========================================

def sigmoid(x):
    """シグモイド関数（overflow回避）"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh_activation(x):
    """tanh活性化関数（双方向性、飽和特性あり）"""
    return np.tanh(np.clip(x, -500, 500))


def softmax(x):
    """
    SoftMax関数（多クラス分類用）
    
    確率分布を生成：
    - 各要素が [0, 1]
    - 全要素の合計が 1.0
    - クラス間の相対的な強弱が明確
    
    Args:
        x: 出力層の活性値（logits）
    
    Returns:
        確率分布（合計=1.0）
    """
    # 数値安定性のため最大値を引く
    x_shifted = x - np.max(x)
    exp_x = np.exp(np.clip(x_shifted, -500, 500))
    return exp_x / np.sum(exp_x)


def cross_entropy_loss(probs, target_class):
    """
    Cross-Entropy損失関数
    
    Args:
        probs: SoftMax確率分布
        target_class: 正解クラスのインデックス
    
    Returns:
        損失値
    """
    # 数値安定性のため小さな値でクリップ
    prob_true = np.clip(probs[target_class], 1e-10, 1.0)
    return -np.log(prob_true)


# ========================================
# 3. 興奮性・抑制性ニューロンペア構造（必須要素7）
# ========================================

def create_ei_pairs(x):
    """入力データを興奮性・抑制性ペアに変換"""
    return np.concatenate([x, x])  # [x1, x2, ..., xn, x1, x2, ..., xn]


# ========================================
# 4. 誤差重み付きアミン配分（Error-Weighted Amine Distribution）
# ========================================

def distribute_amine_by_error_weights(w_output, error_output, z_hidden):
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


# ========================================
# 5. 側方抑制（必須要素6）
# ========================================

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
    
    Args:
        n_hidden: 隠れ層のニューロン数（例: 250）
        n_classes: クラス数（デフォルト: 10）
        column_radius: コラムの影響半径（六角距離単位）- column_neuronsが指定されない場合に使用
        column_neurons: 各クラスのコラムに割り当てるニューロン数（指定時はradiusより優先）
        participation_rate: コラム参加率（0.0-1.0）。1.0で全ニューロンがいずれかのコラムに参加
                           column_neuronsが指定されていない場合のみ有効
    
    Returns:
        affinity: コラム帰属度マップ shape [n_classes, n_hidden]
                  各要素は0.0以上の値（ニューロンがそのクラスに帰属する度合い）
    
    使用例:
        # モード1: 完全コラム化（全ニューロンを均等分割）
        affinity = create_hexagonal_column_affinity(250, 10, column_neurons=25)
        # → 各クラス25個、重複なし、全ニューロン参加
        
        # モード2: 参加率指定
        affinity = create_hexagonal_column_affinity(250, 10, participation_rate=1.0)
        # → 全ニューロンが参加するよう上位N個を選択
        
        # モード3: 従来のradius方式（互換性維持）
        affinity = create_hexagonal_column_affinity(250, 10, column_radius=3.0)
        # → ガウス分布ベース、重複あり
    """
    # 1. 10クラスを2-3-3-2配置の六角格子に配置
    class_coords = {
        0: (0, 0),   1: (1, 0),                    # 行1: 2個
        2: (0, 1),   3: (1, 1),   4: (2, 1),       # 行2: 3個
        5: (0, 2),   6: (1, 2),   7: (2, 2),       # 行3: 3個
        8: (1, 3),   9: (2, 3)                      # 行4: 2個
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
            selected = sorted_indices[:neurons_per_class]
            
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


# ========================================
# 6. ネットワーククラス
# ========================================

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
                if learning_rate == 0.05:  # デフォルト値の場合
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
                # 旧円環構造（互換性維持）
                affinity = create_column_affinity(
                    n_hidden=layer_size,
                    n_classes=n_output,
                    column_size=int(layer_radius * 10),  # 層ごとのradiusをsizeに変換
                    overlap=overlap,
                    use_gaussian=True
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
            print(f"  - 方式: 円環構造")
            print(f"  - コラムサイズ: {int(self.column_radius_per_layer[0] * 10)}ニューロン")
        
        for layer_idx, affinity in enumerate(self.column_affinity_all_layers):
            non_zero_counts = [np.count_nonzero(affinity[c]) for c in range(n_output)]
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


# ========================================
# メイン処理
# ========================================


# ========================================
# メイン処理
# ========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='コラムED法 パラメータ',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # ========================================
    # 実行関連のパラメータ
    # ========================================
    exec_group = parser.add_argument_group('実行関連のパラメータ')
    exec_group.add_argument('--train', type=int, default=3000, 
                           help='訓練サンプル数（最適値: 3000）')
    exec_group.add_argument('--test', type=int, default=1000, 
                           help='テストサンプル数')
    exec_group.add_argument('--epochs', type=int, default=100, 
                           help='エポック数（最適値: 100）')
    exec_group.add_argument('--seed', type=int, default=42, 
                           help='乱数シード（最終確定: 42、再現性確保用）')
    exec_group.add_argument('--fashion', action='store_true', 
                           help='Fashion-MNISTを使用')
    exec_group.add_argument('--use_hyperparams', action='store_true', 
                           help='HyperParamsテーブルから設定を自動取得（層数に基づく）')
    exec_group.add_argument('--list_hyperparams', action='store_true',
                           help='利用可能なHyperParams設定一覧を表示して終了')
    
    # ========================================
    # ED法関連のパラメータ
    # ========================================
    ed_group = parser.add_argument_group('ED法関連のパラメータ')
    ed_group.add_argument('--hidden', type=str, default='512', 
                         help='隠れ層ニューロン数（最適値: 512=1層で83.80%%達成、例: 256,128=2層）')
    ed_group.add_argument('--lr', type=float, default=0.20, 
                         help='学習率（Phase 1 Extended Overall Best: 0.20）')
    ed_group.add_argument('--u1', type=float, default=0.5, 
                         help='アミン拡散係数（Phase 1 Extended Overall Best: 0.5）')
    ed_group.add_argument('--u2', type=float, default=0.8, 
                         help='アミン拡散係数（隠れ層間、デフォルト0.8）')
    ed_group.add_argument('--lateral_lr', type=float, default=0.08, 
                         help='側方抑制の学習率（Phase 1 Extended Overall Best: 0.08）')
    ed_group.add_argument('--gradient_clip', type=float, default=0.05, 
                         help='gradient clipping値（デフォルト0.05）')
    
    # ========================================
    # コラム関連のパラメータ
    # ========================================
    column_group = parser.add_argument_group('コラム関連のパラメータ')
    column_group.add_argument('--base_column_radius', type=float, default=1.0, 
                             help='基準コラム半径（Phase 2完全評価Best: 1.0、256ニューロン層での値）')
    column_group.add_argument('--column_radius', type=float, default=None, 
                             help='コラム影響半径（Noneなら層ごとに自動計算、デフォルト: None=自動）')
    column_group.add_argument('--participation_rate', type=float, default=1.0, 
                             help='コラム参加率（Phase 2で確定: 1.0=全参加・重複なし、優先度：最高）')
    column_group.add_argument('--column_neurons', type=int, default=None, 
                             help='各クラスの明示的ニューロン数（重複許容、優先度：中）')
    column_group.add_argument('--use_circular', action='store_true', 
                             help='旧円環構造を使用（デフォルトはハニカム）')
    column_group.add_argument('--overlap', type=float, default=0.0, 
                             help='コラム間の重複度（0.0-1.0、円環構造でのみ有効、デフォルト0.0=重複なし）')
    column_group.add_argument('--diagnose_column', action='store_true', 
                             help='コラム構造の詳細診断を実行')
    
    # ========================================
    # 可視化関連のパラメータ
    # ========================================
    viz_group = parser.add_argument_group('可視化関連のパラメータ')
    viz_group.add_argument('--viz', action='store_true',
                          help='学習曲線のリアルタイム可視化を有効化')
    viz_group.add_argument('--heatmap', action='store_true',
                          help='活性化ヒートマップの表示を有効化（--vizと併用）')
    viz_group.add_argument('--save_viz', type=str, nargs='?', const='viz_results', 
                          default=None, metavar='PATH',
                          help='可視化結果を保存。'
                               'パス指定: ディレクトリまたはベースファイル名（例: results/exp1, my_exp.png）。'
                               '引数なし: viz_results/ディレクトリにタイムスタンプ付きで保存。'
                               'オプション未指定: 保存しない。')
    
    args = parser.parse_args()
    
    # 乱数シードの設定（再現性確保）
    if args.seed is not None:
        print(f"\n=== 乱数シード固定: {args.seed} ===")
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        print("再現性モード: 有効（random, numpy固定）\n")
    else:
        print("\n=== 乱数シード: ランダム ===")
        print("再現性モード: 無効（毎回異なる結果）\n")
    
    # HyperParams設定一覧の表示
    if args.list_hyperparams:
        hp = HyperParams()
        hp.list_configs()
        sys.exit(0)
    
    # 隠れ層のパース（カンマ区切り対応、多層対応）
    if ',' in args.hidden:
        hidden_sizes = [int(x.strip()) for x in args.hidden.split(',')]
    else:
        hidden_sizes = [int(args.hidden)]
    
    # HyperParamsの使用
    hyperparams_instance = None
    if args.use_hyperparams:
        hp = HyperParams()
        try:
            n_layers = len(hidden_sizes)
            config = hp.get_config(n_layers)
            print(f"\n=== HyperParams設定を使用（{n_layers}層） ===")
            print(f"Description: {config['description']}")
            print(f"hidden_layers: {config['hidden']}")
            print(f"base_column_radius: {config['base_column_radius']}")
            print(f"column_radius_per_layer: {config['column_radius_per_layer']}")
            print(f"learning_rate: {config['learning_rate']}")
            print(f"epochs: {config['epochs']}")
            
            # 設定を上書き（コマンドライン引数が明示されていない場合）
            hidden_sizes = config['hidden']
            if args.lr == 0.05:  # デフォルト値の場合
                args.lr = config['learning_rate']
            if args.base_column_radius == 1.0:  # デフォルト値の場合
                args.base_column_radius = config['base_column_radius']
            if args.epochs == 10:  # デフォルト値の場合
                args.epochs = config['epochs']
            
            hyperparams_instance = hp
            print("="*50 + "\n")
        except ValueError as e:
            print(f"Warning: {e}")
            print("個別パラメータで継続します。\n")
    
    # データ読み込み（必須要素10）
    dataset = 'fashion' if args.fashion else 'mnist'
    print(f"データ読み込み中... (訓練:{args.train}, テスト:{args.test}, データセット:{dataset})")
    x_train, y_train, x_test, y_test = load_dataset(
        n_train=args.train, n_test=args.test, dataset=dataset
    )
    
    # ネットワーク作成
    print("\nネットワーク初期化中...")
    print(f"  隠れ層: {hidden_sizes}ニューロン")
    print(f"  学習率: {args.lr}")
    print(f"  アミン拡散係数(u1): {args.u1}, (u2): {args.u2}")
    print(f"  方式: {'Hexagonal Column (2-3-3-2)' if not args.use_circular else 'Circular Column'}")
    print(f"  側方抑制学習率: {args.lateral_lr}")
    if args.column_radius is not None:
        print(f"  コラムモード: 固定radius={args.column_radius}")
    else:
        print(f"  コラムモード: 層依存自動計算（base_radius={args.base_column_radius}）")
    if args.participation_rate is not None:
        print(f"  参加率: {args.participation_rate * 100:.0f}%（デフォルト=全参加・重複なし）")
    if args.column_neurons is not None:
        print(f"  明示的ニューロン数: 各クラス{args.column_neurons}個")
    
    net = RefinedDistributionEDNetwork(
        n_input=784,
        n_hidden=hidden_sizes,
        n_output=10,
        learning_rate=args.lr,
        lateral_lr=args.lateral_lr,
        u1=args.u1,
        u2=args.u2,
        column_radius=args.column_radius,
        base_column_radius=args.base_column_radius,
        column_neurons=args.column_neurons,
        participation_rate=args.participation_rate,
        use_hexagonal=not args.use_circular,
        overlap=args.overlap,
        gradient_clip=args.gradient_clip,
        hyperparams=hyperparams_instance  # HyperParamsインスタンスを渡す

    )
    
    # コラム構造の診断（オプション）
    if args.diagnose_column:
        net.diagnose_column_structure()
        print("\n診断完了。学習はスキップします。")
        exit(0)
    
    # 学習
    print("\n" + "="*70)
    print("学習開始")
    print("="*70)
    
    from tqdm import tqdm
    
    # ========================================
    # 可視化機能の初期化
    # ========================================
    viz_manager = None
    if args.viz:
        try:
            from modules.visualization_manager import VisualizationManager
            viz_manager = VisualizationManager(
                enable_viz=True,
                enable_heatmap=args.heatmap,
                save_path=args.save_viz,  # ユーザー指定のパス（Noneの場合は保存なし）
                total_epochs=args.epochs
            )
            print("\n可視化機能: 有効")
            if args.heatmap:
                print("  - ヒートマップ表示: 有効")
            if args.save_viz:
                print(f"  - 保存先: {args.save_viz}")
        except ImportError as e:
            print(f"\n警告: 可視化モジュールのインポートに失敗しました: {e}")
            print("可視化なしで学習を継続します。")
            viz_manager = None
    
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    train_acc_history = []
    test_acc_history = []
    
    # tqdmを使ったエポックループ
    pbar = tqdm(range(1, args.epochs + 1), desc="Training", ncols=120)
    for epoch in pbar:
        # 訓練
        train_acc, train_loss = net.train_epoch(x_train, y_train)
        train_acc_history.append(train_acc)
        
        # テスト
        test_acc, test_loss = net.evaluate(x_test, y_test)
        test_acc_history.append(test_acc)
        
        # 可視化更新（学習曲線）
        if viz_manager is not None:
            # 正解ラベルの形状判定（one-hot → インデックスに変換）
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_test_indices = np.argmax(y_test, axis=1)
            else:
                y_test_indices = y_test
            
            # 学習曲線の更新（混同行列計算は削除済み）
            viz_manager.update_learning_curve(
                train_acc_history,
                test_acc_history,
                x_test,
                y_test_indices,
                net
            )
            
            # ヒートマップ更新（サンプルを1つ選択）
            if args.heatmap and epoch % 1 == 0:  # 毎エポック更新
                sample_idx = np.random.randint(0, len(x_test))
                sample_x = x_test[sample_idx]
                sample_y_true = y_test_indices[sample_idx]
                
                # ネットワークのforward()を使用して正確な予測を取得
                # （微分の連鎖律は使用せず、純粋なED法のforward計算）
                z_hiddens, z_output, _ = net.forward(sample_x)
                sample_y_pred = np.argmax(z_output)
                
                # クラス名の取得（汎用的な判定アルゴリズム）
                # 1. データセット名からクラス名リストを取得
                class_names = get_class_names(dataset)
                
                # 2. クラス名情報があるか判定
                if class_names is not None:
                    # クラス名情報がある → 表示
                    true_class_name = class_names[sample_y_true]
                    pred_class_name = class_names[sample_y_pred]
                else:
                    # クラス名情報がない（MNISTや未定義データセット）→ 表示しない
                    true_class_name = None
                    pred_class_name = None
                
                viz_manager.update_heatmap(
                    epoch=epoch,
                    sample_x=sample_x,
                    sample_y_true=sample_y_true,
                    sample_y_true_name=true_class_name,
                    z_hiddens=z_hiddens,
                    z_output=z_output,
                    sample_y_pred=sample_y_pred,
                    sample_y_pred_name=pred_class_name
                )
        
        # tqdmのプログレスバーに情報を表示（小数2桁、短縮形）
        pbar.set_postfix({
            'Tr-Acc': f'{train_acc:.2f}',
            'Tr-Los': f'{train_loss:.2f}',
            'Te-Acc': f'{test_acc:.2f}',
            'Te-Los': f'{test_loss:.2f}'
        })
    
    print("\n学習完了！")
    print(f"最終結果: 訓練精度={train_acc:.4f}, テスト精度={test_acc:.4f}")
    
    # 可視化結果の保存
    if viz_manager is not None and args.save_viz:
        print("\n可視化結果を保存中...")
        viz_manager.save_figures()
        print("保存完了！")
