"""
ED学習コア最適化版 - 高速化実装

ed_multi_snn.prompt.md準拠の最適化機能:
- NumPy行列演算による高速化
- メモリ効率改善
- ベクトル化計算

作成者: ED-SNN開発チーム
作成日: 2025年9月28日
バージョン: v002 - 高速化版
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import time

class EDCoreFast:
    """
    ED法核心アルゴリズム - 高速化版
    
    🎯 ed_multi_snn.prompt.md準拠の最適化機能:
    - NumPy行列演算によるベクトル化計算
    - メモリ効率改善
    - 3重ループの最適化
    
    理論準拠: 金子勇氏 Error Diffusion Learning Algorithm (1999)
    """
    
    def __init__(self, hyperparams=None):
        """ED法コア初期化（高速化版）"""
        # デフォルトパラメータ
        if hyperparams is None:
            class DefaultParams:
                learning_rate = 0.3
                initial_amine = 0.7
                sigmoid_threshold = 0.7
                diffusion_rate = 1.0
                initial_weight_1 = 1.0
                initial_weight_2 = 1.0
            hyperparams = DefaultParams()
            
        self.hyperparams = hyperparams
        self.learning_rate = hyperparams.learning_rate
        self.initial_amine = hyperparams.initial_amine
        self.sigmoid_threshold = hyperparams.sigmoid_threshold
        
        # ネットワーク構造
        self.input_units = 0
        self.hidden_units = 0
        self.output_units = 0
        self.total_units = 0
        self.max_units = 0
        
        # 高速化用データ構造（NumPy配列）
        self.weights = None              # 重み行列（最適化済み）
        self.outputs = None              # 出力値配列
        self.amine_positive = None       # 正アミン濃度
        self.amine_negative = None       # 負アミン濃度
        self.input_buffer = None         # 入力バッファ
        
        # 統計情報
        self.error = 0.0
        self.computation_stats = {
            'forward_time': 0.0,
            'backward_time': 0.0,
            'weight_update_time': 0.0,
            'total_operations': 0
        }
        
    def initialize_network(self, input_size: int, hidden_size: int, output_size: int):
        """
        高速化ネットワーク初期化
        
        Parameters:
        -----------
        input_size : int
            入力サイズ
        hidden_size : int
            隠れ層サイズ  
        output_size : int
            出力サイズ
        """
        self.input_units = input_size
        self.hidden_units = hidden_size
        self.output_units = output_size
        self.total_units = input_size + hidden_size + output_size
        
        # 安全なmax_units計算（メモリ効率改善）
        self.max_units = min(self.total_units + 100, 2000)  # メモリ使用量制限
        
        # NumPy配列による高速化データ構造
        self._initialize_fast_arrays()
        
        print(f"高速ED法ネットワーク初期化:")
        print(f"  入力: {input_size}, 隠れ: {hidden_size}, 出力: {output_size}")
        print(f"  最大ユニット: {self.max_units}")
        print(f"  メモリ最適化: 有効")
        
    def _initialize_fast_arrays(self):
        """高速化配列の初期化"""
        # 重み配列（出力×出力×ユニット：独立出力ニューロンアーキテクチャ）
        self.weights = np.random.uniform(
            -self.hyperparams.initial_weight_1,
            self.hyperparams.initial_weight_1,
            (self.output_units, self.max_units + 1, self.max_units + 1)
        )
        
        # 出力配列
        self.outputs = np.zeros((self.output_units, self.max_units + 1))
        
        # アミン濃度配列（正・負分離でメモリ効率改善）
        self.amine_positive = np.zeros((self.output_units, self.max_units + 1))
        self.amine_negative = np.zeros((self.output_units, self.max_units + 1))
        
        # 入力バッファ
        self.input_buffer = np.zeros(self.max_units + 1)
        
        # バイアス設定
        self.input_buffer[0] = 1.0
        self.input_buffer[1] = 1.0
        
    def sigmoid_vectorized(self, x_array: np.ndarray) -> np.ndarray:
        """
        ベクトル化シグモイド関数（高速化）
        
        Parameters:
        -----------
        x_array : np.ndarray
            入力配列
            
        Returns:
        --------
        np.ndarray
            シグモイド出力配列
        """
        # オーバーフロー防止
        scaled_x = -2.0 * x_array / self.sigmoid_threshold
        safe_x = np.clip(scaled_x, -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(safe_x))
        
    def neuro_output_calc(self, inputs):
        """ニューロン出力計算（高速化版）"""
        # 入力バッファ更新（サイズ制限）
        max_input_size = min(len(inputs), self.n_input)
        
        self.input_buffer[:max_input_size] = inputs[:max_input_size]
        if len(self.input_buffer) > max_input_size:
            self.input_buffer[max_input_size:] = 0
        
        outputs_list = []
        
        # 簡単な行列計算版
        for ot in range(self.output_units):
            # 重みと入力の行列積
            output_weights = self.weights[ot, :self.n_input, :self.n_input]  # [n_input, n_input]
            input_data = self.input_buffer[:self.n_input]  # [n_input]
            
            # 行列乗算
            weighted_sum = np.sum(output_weights * input_data[np.newaxis, :], axis=1)
            
            # シグモイド活性化
            output_values = self.sigmoid_vectorized(weighted_sum)
            
            # 出力設定
            self.outputs[ot, :len(output_values)] = output_values                # 隠れ層計算（もし存在する場合）
                if self.hidden_units > 0:
                    hidden_start = self.input_units + 3
                    hidden_end = hidden_start + self.hidden_units
                    
                    # 隠れ層の重み行列演算
                    hidden_weights = self.weights[ot, :hidden_end, hidden_start:hidden_end]
                    hidden_inputs = self.outputs[ot, :hidden_end]
                    
                    hidden_sums = np.dot(hidden_weights, hidden_inputs)
                    self.outputs[ot, hidden_start:hidden_end] = self.sigmoid_vectorized(hidden_sums)
                
                # 出力ニューロン計算
                output_pos = self.input_units + 2
                output_weights = self.weights[ot, :, output_pos]
                output_sum = np.dot(output_weights, self.outputs[ot, :])
                
                self.outputs[ot, output_pos] = self.sigmoid_vectorized(np.array([output_sum]))[0]
            
            outputs_list.append(float(self.outputs[ot, output_pos]))
        
        # 統計更新
        self.computation_stats['forward_time'] += time.time() - start_time
        self.computation_stats['total_operations'] += 1
        
        return outputs_list
        
    def neuro_teach_calc(self, target_pattern: List[float]):
        """
        高速化教師信号・アミン濃度計算
        
        Parameters:
        -----------
        target_pattern : List[float]
            目標パターン
        """
        start_time = time.time()
        
        # ベクトル化アミン濃度計算
        for ot in range(self.output_units):
            output_pos = self.input_units + 2
            error = target_pattern[ot] - self.outputs[ot, output_pos]
            self.error += abs(error)
            
            # アミン濃度設定（ベクトル化）
            if error > 0:  # 正誤差
                self.amine_positive[ot, :] = self.initial_amine
                self.amine_negative[ot, :] = 0.0
            else:  # 負誤差
                self.amine_positive[ot, :] = 0.0
                self.amine_negative[ot, :] = self.initial_amine
                
        self.computation_stats['backward_time'] += time.time() - start_time
        
    def neuro_weight_calc(self):
        """
        高速化重み更新（NumPy最適化）
        
        ed_multi_snn.prompt.md準拠の高速化実装
        """
        start_time = time.time()
        
        # 各出力ニューロンについて並列更新
        for ot in range(self.output_units):
            # アミン効果計算（ベクトル化）
            amine_effect = self.amine_positive[ot, :] - self.amine_negative[ot, :]
            
            # 出力値による学習率調整（ベクトル化）
            output_factor = self.outputs[ot, :] * (1.0 - np.abs(self.outputs[ot, :]))
            
            # 重み更新（行列演算による高速化）
            for j in range(self.max_units + 1):
                if j < len(self.outputs[ot]) and self.outputs[ot, j] != 0:
                    # 学習率・出力・アミン効果の積（ベクトル化）
                    delta = self.learning_rate * self.outputs[ot, j] * amine_effect
                    
                    # マスク作成（j!=kの条件）
                    mask = np.ones(self.max_units + 1, dtype=bool)
                    mask[j] = False
                    
                    # 重み更新（ベクトル化）
                    self.weights[ot, j, mask] += delta[mask]
        
        self.computation_stats['weight_update_time'] += time.time() - start_time
        
    def predict(self, input_pattern: List[float]) -> int:
        """高速化予測"""
        outputs = self.neuro_output_calc(input_pattern)
        return int(np.argmax(outputs))
        
    def get_output_values(self, input_pattern: List[float]) -> List[float]:
        """高速化出力値取得"""
        return self.neuro_output_calc(input_pattern)
        
    def reset_error(self):
        """エラーカウンターリセット"""
        self.error = 0.0
        
    def get_network_info(self) -> Dict[str, Any]:
        """ネットワーク情報取得"""
        avg_forward = (self.computation_stats['forward_time'] / 
                      max(self.computation_stats['total_operations'], 1))
        avg_backward = (self.computation_stats['backward_time'] / 
                       max(self.computation_stats['total_operations'], 1))
        avg_weight_update = (self.computation_stats['weight_update_time'] / 
                            max(self.computation_stats['total_operations'], 1))
        
        return {
            'input_units': self.input_units,
            'hidden_units': self.hidden_units,
            'output_units': self.output_units,
            'total_units': self.total_units,
            'learning_rate': self.learning_rate,
            'sigmoid_threshold': self.sigmoid_threshold,
            'current_error': self.error,
            'weight_shape': self.weights.shape if self.weights is not None else None,
            'performance_stats': {
                'avg_forward_time': avg_forward,
                'avg_backward_time': avg_backward,
                'avg_weight_update_time': avg_weight_update,
                'total_operations': self.computation_stats['total_operations']
            }
        }
        
    def get_performance_report(self) -> str:
        """性能レポート生成"""
        info = self.get_network_info()
        stats = info['performance_stats']
        
        return f"""
🚀 ED法高速化版性能レポート:
  平均フォワードパス時間: {stats['avg_forward_time']:.6f}秒
  平均教師信号計算時間: {stats['avg_backward_time']:.6f}秒  
  平均重み更新時間: {stats['avg_weight_update_time']:.6f}秒
  総操作数: {stats['total_operations']}
  重み配列形状: {info['weight_shape']}
"""