#!/usr/bin/env python3
"""
高速化ED学習コア - 純粋ED法実装（高速版）

NumPy最適化による高速化ED学習の実装
誤差逆伝播法を使わない純粋なError-Diffusion手法

作成者: ED-SNN開発チーム  
作成日: 2025年9月28日
バージョン: v2_fast
"""

import numpy as np
import time
from typing import List, Optional

class EDCoreFast:
    """高速化Error-Diffusion学習コア"""
    
    def __init__(self, 
                 n_input: int, 
                 n_hidden: int, 
                 n_output: int,
                 max_units: Optional[int] = None):
        """
        高速化ED学習システム初期化
        
        Parameters:
        -----------
        n_input : int
            入力ユニット数
        n_hidden : int  
            隠れユニット数
        n_output : int
            出力ユニット数
        max_units : int, optional
            最大ユニット数（メモリ効率化）
        """
        
        print("高速ED法ネットワーク初期化:")
        print(f"  入力: {n_input}, 隠れ: {n_hidden}, 出力: {n_output}")
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.output_units = n_output
        
        # メモリ効率化
        if max_units is None:
            self.max_units = n_input + n_hidden + n_output + 10
        else:
            self.max_units = max_units
            
        print(f"  最大ユニット: {self.max_units}")
        print(f"  メモリ最適化: 有効")
        
        # 初期化
        self._initialize_matrices()
        self._initialize_statistics()
        
    def _initialize_matrices(self):
        """行列初期化（NumPy最適化）"""
        
        # 重み行列（3次元 -> 2次元で効率化）
        self.weights = np.random.normal(0, 0.1, 
                                      size=(self.n_output, self.max_units))
        
        # 出力行列（簡素化）
        self.outputs = np.zeros((self.n_output, self.max_units))
        
        # 入力バッファ
        self.input_buffer = np.zeros(self.max_units)
        
        # アミン濃度（ベクトル化）
        self.amine_positive = np.zeros((self.n_output, self.max_units))
        self.amine_negative = np.zeros((self.n_output, self.max_units))
        
        # ED法パラメータ
        self.initial_amine = 1.0
        self.amine_decay = 0.95
        self.weight_lr = 0.01
        self.error = 0.0
        
    def _initialize_statistics(self):
        """統計情報初期化"""
        self.computation_stats = {
            'forward_time': 0.0,
            'backward_time': 0.0,
            'weight_update_time': 0.0,
            'total_operations': 0
        }
        
    def sigmoid_vectorized(self, x: np.ndarray) -> np.ndarray:
        """ベクトル化シグモイド関数"""
        # クリッピングで数値安定性向上
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))
        
    def neuro_output_calc(self, inputs):
        """ニューロン出力計算（高速化版）"""
        start_time = time.time()
        
        # 入力サイズ制限
        input_size = min(len(inputs), self.n_input)
        
        # 入力バッファ更新
        self.input_buffer[:input_size] = inputs[:input_size]
        if len(self.input_buffer) > input_size:
            self.input_buffer[input_size:] = 0
        
        outputs_list = []
        
        # 各出力について計算（簡略化）
        for ot in range(self.n_output):
            # 重みと入力の内積
            weights_slice = self.weights[ot, :input_size]
            inputs_slice = self.input_buffer[:input_size]
            
            # 行列積計算
            weighted_sum = np.dot(weights_slice, inputs_slice)
            
            # シグモイド活性化
            output_value = self.sigmoid_vectorized(np.array([weighted_sum]))[0]
            
            # 出力保存
            self.outputs[ot, 0] = output_value
            outputs_list.append(float(output_value))
        
        # 統計更新
        self.computation_stats['forward_time'] += time.time() - start_time
        self.computation_stats['total_operations'] += 1
        
        return outputs_list
        
    def neuro_teach_calc(self, target_pattern: List[float]):
        """高速化教師信号・アミン濃度計算"""
        start_time = time.time()
        
        self.error = 0.0
        
        # ベクトル化誤差計算
        targets = np.array(target_pattern[:self.n_output])
        current_outputs = np.array([self.outputs[i, 0] for i in range(self.n_output)])
        
        errors = targets - current_outputs
        self.error = np.sum(np.abs(errors))
        
        # アミン濃度設定（ベクトル化）
        for ot in range(self.n_output):
            if errors[ot] > 0:  # 正誤差
                self.amine_positive[ot, :] = self.initial_amine
                self.amine_negative[ot, :] = 0.0
            else:  # 負誤差
                self.amine_positive[ot, :] = 0.0
                self.amine_negative[ot, :] = self.initial_amine
                
        self.computation_stats['backward_time'] += time.time() - start_time
        
    def neuro_weight_calc(self):
        """高速化重み更新計算"""
        start_time = time.time()
        
        # ベクトル化重み更新
        for ot in range(self.n_output):
            # アミン濃度による重み更新
            positive_update = self.amine_positive[ot, :] * self.weight_lr
            negative_update = self.amine_negative[ot, :] * self.weight_lr
            
            # 重み更新（ベクトル演算）
            self.weights[ot, :] += positive_update - negative_update
            
            # アミン減衰
            self.amine_positive[ot, :] *= self.amine_decay
            self.amine_negative[ot, :] *= self.amine_decay
        
        self.computation_stats['weight_update_time'] += time.time() - start_time
        
    def get_performance_stats(self) -> dict:
        """性能統計取得"""
        total_time = (self.computation_stats['forward_time'] + 
                     self.computation_stats['backward_time'] + 
                     self.computation_stats['weight_update_time'])
        
        ops = max(1, self.computation_stats['total_operations'])
        
        return {
            'total_time': total_time,
            'average_time_per_operation': total_time / ops,
            'forward_time_ratio': self.computation_stats['forward_time'] / max(total_time, 1e-6),
            'backward_time_ratio': self.computation_stats['backward_time'] / max(total_time, 1e-6),
            'weight_update_ratio': self.computation_stats['weight_update_time'] / max(total_time, 1e-6),
            'total_operations': ops
        }
        
    def reset_stats(self):
        """統計リセット"""
        self._initialize_statistics()
        
    def get_weights(self) -> np.ndarray:
        """重み行列取得"""
        return self.weights.copy()
        
    def set_weights(self, weights: np.ndarray):
        """重み行列設定"""
        if weights.shape == self.weights.shape:
            self.weights = weights.copy()
        else:
            raise ValueError(f"重み行列のサイズが不正: {weights.shape} != {self.weights.shape}")
            
    def get_current_error(self) -> float:
        """現在の誤差取得"""
        return self.error
        
    def __repr__(self) -> str:
        stats = self.get_performance_stats()
        return (f"EDCoreFast(input={self.n_input}, hidden={self.n_hidden}, "
                f"output={self.n_output}, ops={stats['total_operations']}, "
                f"avg_time={stats['average_time_per_operation']:.4f}s)")

if __name__ == "__main__":
    # 高速化テスト
    print("🚀 高速化EDコアテスト")
    
    # 小規模ネットワークでテスト
    ed_core = EDCoreFast(n_input=10, n_hidden=5, n_output=2)
    
    # テストデータ
    test_input = np.random.rand(10)
    test_target = [0.8, 0.2]
    
    print(f"\n📊 テスト実行:")
    print(f"入力: {test_input[:3]}...")
    print(f"目標: {test_target}")
    
    # 計算実行
    start_time = time.time()
    outputs = ed_core.neuro_output_calc(test_input)
    ed_core.neuro_teach_calc(test_target)
    ed_core.neuro_weight_calc()
    total_time = time.time() - start_time
    
    print(f"出力: {outputs}")
    print(f"誤差: {ed_core.get_current_error():.4f}")
    print(f"実行時間: {total_time:.4f}秒")
    
    # 性能統計
    stats = ed_core.get_performance_stats()
    print(f"\n📈 性能統計:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n✅ 高速化EDコアテスト完了")