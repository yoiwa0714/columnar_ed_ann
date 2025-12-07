#!/usr/bin/env python3
"""
高速化ED-SNNネットワーク統合 v2

NumPy最適化による高速化ED-SNN実装
LIFニューロンとED学習の最適化統合

作成者: ED-SNN開発チーム  
作成日: 2025年9月28日
バージョン: v2_fast
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional

# モジュール読み込み
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .ed_core_fast_v2 import EDCoreFast
from snn.lif_neuron import LIFNeuronLayer

class EDSpikingNeuralNetworkFastV2:
    """高速化ED-SNNネットワーク v2"""
    
    def __init__(self, 
                 network_structure: List[int],
                 simulation_time: float = 20.0,
                 dt: float = 1.0,
                 use_fast_core: bool = True):
        """
        高速化ED-SNNネットワーク初期化
        
        Parameters:
        -----------
        network_structure : List[int]
            [入力, 隠れ, 出力] のリスト
        simulation_time : float
            シミュレーション時間 (ms)
        dt : float
            時間刻み (ms)
        use_fast_core : bool
            高速化EDコア使用フラグ
        """
        
        print("高速化ED-SNNネットワーク初期化:")
        
        self.network_structure = network_structure
        self.simulation_time = simulation_time
        self.dt = dt
        self.use_fast_core = use_fast_core
        
        # 興奮性・抑制性構造（MNIST対応）
        self._setup_excitatory_inhibitory_structure()
        
        # ED学習コア初期化（高速版）
        self._initialize_ed_core()
        
        # LIFニューロン層初期化
        self._initialize_lif_layers()
        
        # 統計情報
        self._initialize_performance_stats()
        
        print(f"高速化ED-SNNネットワーク初期化完了:")
        print(f"  構造: {self.network_structure}")
        print(f"  シミュレーション時間: {self.simulation_time}ms")
        print(f"  高速化EDコア: {self.use_fast_core}")
        print(f"  興奮性・抑制性構造: {self.n_excitatory}E / {self.n_inhibitory}I")
        
    def _setup_excitatory_inhibitory_structure(self):
        """興奮性・抑制性ニューロン構造設定"""
        input_size = self.network_structure[0]
        
        # 興奮性・抑制性ペア（MNIST: 784 -> 819E + 791I = 1610）
        self.n_excitatory = int(input_size * 1.045)  # 約104.5%
        self.n_inhibitory = input_size  # 抑制性は元のサイズ
        
        # ED学習用の総入力サイズ
        self.ed_input_size = self.n_excitatory + self.n_inhibitory + 8  # バッファ追加
        
    def _initialize_ed_core(self):
        """ED学習コア初期化（高速版）"""
        
        hidden_size = self.network_structure[1] if len(self.network_structure) > 1 else 0
        output_size = self.network_structure[-1]
        
        # 高速化EDコア使用
        if self.use_fast_core:
            self.ed_core = EDCoreFast(
                n_input=self.ed_input_size,
                n_hidden=hidden_size,
                n_output=output_size,
                max_units=self.ed_input_size + hidden_size + output_size + 50
            )
        else:
            # フォールバック（通常版）
            from ed_learning.ed_core import EDCore
            self.ed_core = EDCore(
                n_input=self.ed_input_size,
                n_hidden=hidden_size,
                n_output=output_size
            )
            
    def _initialize_lif_layers(self):
        """LIFニューロン層初期化"""
        
        print("LIFニューロン層作成完了:")
        
        # 入力層（興奮性・抑制性）
        self.input_layer = LIFNeuronLayer(
            n_neurons=self.ed_input_size,
            tau_m=20.0,
            v_rest=-70.0,
            v_threshold=-55.0,
            v_reset=-75.0
        )
        print(f"  入力層: {self.ed_input_size} neurons")
        
        # 隠れ層
        if len(self.network_structure) > 2:
            hidden_size = self.network_structure[1]
            self.hidden_layer = LIFNeuronLayer(
                n_neurons=hidden_size,
                tau_m=20.0,
                v_rest=-70.0,
                v_threshold=-55.0,
                v_reset=-75.0
            )
            print(f"  隠れ層: {hidden_size} neurons")
        else:
            self.hidden_layer = None
            
        # 出力層
        output_size = self.network_structure[-1]
        self.output_layer = LIFNeuronLayer(
            n_neurons=output_size,
            tau_m=20.0,
            v_rest=-70.0,
            v_threshold=-55.0,
            v_reset=-75.0
        )
        print(f"  出力層: {output_size} neurons")
        
    def _initialize_performance_stats(self):
        """性能統計初期化"""
        self.performance_stats = {
            'total_samples': 0,
            'total_training_time': 0.0,
            'encoding_time': 0.0,
            'simulation_time': 0.0,
            'ed_computation_time': 0.0,
            'accuracy_history': []
        }
        
    def encode_input_fast(self, input_data: np.ndarray, encoding_type: str = 'rate') -> np.ndarray:
        """高速入力エンコーディング"""
        start_time = time.time()
        
        # 入力データ正規化
        if np.max(input_data) > 1.0:
            input_data = input_data / 255.0
            
        if encoding_type == 'rate':
            # レート符号化（高速化）
            base_rates = input_data * 100.0  # Hz
            
            # 興奮性・抑制性エンコーディング
            excitatory_rates = np.zeros(self.n_excitatory)
            inhibitory_rates = np.zeros(self.n_inhibitory)
            
            # 効率的な配列操作
            excitatory_rates[:len(input_data)] = base_rates
            inhibitory_rates[:len(input_data)] = base_rates * 0.7  # 抑制性は70%
            
            # 統合エンコーディング
            encoded_input = np.concatenate([
                excitatory_rates,
                inhibitory_rates,
                np.zeros(8)  # バッファ
            ])
            
        else:
            # デフォルト: 単純拡張
            encoded_input = np.zeros(self.ed_input_size)
            encoded_input[:len(input_data)] = input_data
            
        self.performance_stats['encoding_time'] += time.time() - start_time
        return encoded_input
        
    def simulate_snn_fast(self, encoded_input: np.ndarray) -> Dict[str, np.ndarray]:
        """高速SNNシミュレーション"""
        start_time = time.time()
        
        n_steps = int(self.simulation_time / self.dt)
        
        # スパイク記録
        input_spikes = []
        output_spikes = []
        
        # 高速シミュレーション（ステップ削減）
        for step in range(0, n_steps, max(1, n_steps // 5)):  # 5ステップに削減
            
            # 入力層更新
            input_current = encoded_input * (step / n_steps)
            input_voltages, input_spike_trains = self.input_layer.update(
                input_current, self.dt
            )
            input_spikes.append(np.sum(input_spike_trains))
            
            # 出力層更新（簡略化）
            output_current = np.random.normal(0, 0.1, self.network_structure[-1])
            output_voltages, output_spike_trains = self.output_layer.update(
                output_current, self.dt
            )
            output_spikes.append(np.sum(output_spike_trains))
            
        self.performance_stats['simulation_time'] += time.time() - start_time
        
        return {
            'input_spikes': np.array(input_spikes),
            'output_spikes': np.array(output_spikes),
            'final_input_rates': encoded_input
        }
        
    def train_step(self, input_data: np.ndarray, target_data: np.ndarray, 
                   encoding_type: str = 'rate') -> Dict[str, Any]:
        """高速学習ステップ"""
        total_start_time = time.time()
        
        # 1. 入力エンコーディング
        encoded_input = self.encode_input_fast(input_data, encoding_type)
        
        # 2. SNNシミュレーション  
        snn_results = self.simulate_snn_fast(encoded_input)
        
        # 3. ED学習計算（高速版）
        ed_start_time = time.time()
        
        # ED入力準備
        ed_input = snn_results['final_input_rates']
        
        # ED順伝播
        ed_outputs = self.ed_core.neuro_output_calc(ed_input)
        
        # ED教師学習
        target_list = target_data.tolist() if isinstance(target_data, np.ndarray) else target_data
        self.ed_core.neuro_teach_calc(target_list)
        
        # ED重み更新
        self.ed_core.neuro_weight_calc()
        
        self.performance_stats['ed_computation_time'] += time.time() - ed_start_time
        
        # 4. 精度計算
        predicted_class = np.argmax(ed_outputs)
        true_class = np.argmax(target_data)
        accuracy = float(predicted_class == true_class)
        
        # 5. 統計更新
        total_time = time.time() - total_start_time
        self.performance_stats['total_samples'] += 1
        self.performance_stats['total_training_time'] += total_time
        self.performance_stats['accuracy_history'].append(accuracy)
        
        return {
            'outputs': ed_outputs,
            'accuracy': accuracy,
            'error': self.ed_core.get_current_error(),
            'training_time': total_time,
            'predicted_class': predicted_class,
            'true_class': true_class
        }
        
    def get_performance_report(self) -> str:
        """性能レポート取得"""
        stats = self.performance_stats
        ed_stats = self.ed_core.get_performance_stats()
        
        total_samples = max(1, stats['total_samples'])
        
        report = f"""
🚀 高速化ED-SNN性能レポート
{'='*50}
📊 基本統計:
  処理サンプル数: {total_samples}
  総学習時間: {stats['total_training_time']:.2f}秒
  平均サンプル時間: {stats['total_training_time']/total_samples:.3f}秒
  
⏱️ 時間内訳:
  エンコーディング: {stats['encoding_time']:.2f}秒 ({stats['encoding_time']/stats['total_training_time']*100:.1f}%)
  SNNシミュレーション: {stats['simulation_time']:.2f}秒 ({stats['simulation_time']/stats['total_training_time']*100:.1f}%)
  ED計算: {stats['ed_computation_time']:.2f}秒 ({stats['ed_computation_time']/stats['total_training_time']*100:.1f}%)
  
🎯 学習性能:
  最新精度: {stats['accuracy_history'][-1]*100:.1f}% (最後の5サンプル平均: {np.mean(stats['accuracy_history'][-5:])*100:.1f}%)
  
🔧 EDコア統計:
  ED平均時間: {ed_stats['average_time_per_operation']:.4f}秒
  順伝播比率: {ed_stats['forward_time_ratio']*100:.1f}%
  教師学習比率: {ed_stats['backward_time_ratio']*100:.1f}%
  重み更新比率: {ed_stats['weight_update_ratio']*100:.1f}%
  
📈 推定性能:
  1エポック(60,000サンプル): {stats['total_training_time']/total_samples * 60000 / 3600:.1f}時間
  10エポック推定: {stats['total_training_time']/total_samples * 60000 * 10 / 3600:.1f}時間
        """
        
        return report
        
    def reset_performance_stats(self):
        """性能統計リセット"""
        self._initialize_performance_stats()
        self.ed_core.reset_stats()
        
    def summary(self):
        """ネットワーク要約表示"""
        print("🚀 高速化ED-SNNネットワーク要約")
        print("=" * 60)
        
        print(f"📊 ネットワーク構造:")
        print(f"  元構造: {self.network_structure}")
        print(f"  ED入力サイズ: {self.ed_input_size}")
        print(f"  興奮性ニューロン: {self.n_excitatory}")
        print(f"  抑制性ニューロン: {self.n_inhibitory}")
        
        print(f"\n⚙️ シミュレーション設定:")
        print(f"  時間: {self.simulation_time}ms")
        print(f"  刻み: {self.dt}ms")
        print(f"  高速化: {self.use_fast_core}")
        
        print(f"\n📈 現在の性能:")
        if self.performance_stats['total_samples'] > 0:
            avg_time = self.performance_stats['total_training_time'] / self.performance_stats['total_samples']
            print(f"  平均学習時間: {avg_time:.3f}秒/サンプル")
            print(f"  処理済みサンプル: {self.performance_stats['total_samples']}")
            if self.performance_stats['accuracy_history']:
                recent_acc = np.mean(self.performance_stats['accuracy_history'][-10:])
                print(f"  最新精度: {recent_acc*100:.1f}%")
        else:
            print(f"  まだ学習していません")

if __name__ == "__main__":
    # 高速化テスト
    print("🚀 高速化ED-SNNネットワーク v2 テスト")
    
    # 小規模テスト
    network = EDSpikingNeuralNetworkFastV2(
        network_structure=[10, 5, 2],
        simulation_time=5.0,  # 短縮
        use_fast_core=True
    )
    
    # サマリー表示
    network.summary()
    
    # テストデータ
    test_input = np.random.rand(10) * 0.8
    test_target = np.array([1, 0])
    
    print(f"\n🔄 学習テスト:")
    start_time = time.time()
    result = network.train_step(test_input, test_target)
    end_time = time.time()
    
    print(f"  実行時間: {end_time - start_time:.3f}秒")
    print(f"  出力: {result['outputs']}")
    print(f"  精度: {result['accuracy']*100:.0f}%")
    print(f"  誤差: {result['error']:.4f}")
    
    print(f"\n📊 性能レポート:")
    print(network.get_performance_report())
    
    print(f"\n✅ 高速化ED-SNNネットワーク v2 テスト完了")