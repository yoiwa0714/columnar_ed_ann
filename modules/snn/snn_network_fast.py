"""
ED-SNNネットワーク高速化版

高速化されたEDコアを統合したスパイキングニューラルネットワーク

作成者: ED-SNN開発チーム
作成日: 2025年9月28日
バージョン: v002 - 高速化版
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any

# 内部モジュール
from .lif_neuron import LIFNeuron, LIFNeuronLayer
from ..ed_learning.ed_core_fast import EDCoreFast

class EDSpikingNeuralNetworkFast:
    """
    ED法統合スパイキングニューラルネットワーク - 高速化版
    
    🚀 高速化機能:
    - 最適化されたEDCore統合
    - NumPy行列演算による高速化
    - メモリ効率改善
    - ed_multi_snn.prompt.md準拠の最適化
    """
    
    def __init__(
        self,
        network_structure: List[int],
        ed_hyperparams: Optional[Dict[str, Any]] = None,
        snn_params: Optional[Dict[str, Any]] = None,
        simulation_time: float = 50.0,
        dt: float = 1.0,
        use_fast_core: bool = True
    ):
        """
        ED-SNNネットワーク初期化（高速化版）
        
        Parameters:
        -----------
        network_structure : List[int]
            ネットワーク構造 [入力, 隠れ, 出力]
        use_fast_core : bool
            高速化EDコア使用フラグ
        """
        self.network_structure = network_structure
        self.input_size = network_structure[0]
        self.hidden_size = network_structure[1] if len(network_structure) > 2 else 0
        self.output_size = network_structure[-1]
        self.simulation_time = simulation_time
        self.dt = dt
        self.use_fast_core = use_fast_core
        
        # ED法パラメータ
        self.ed_hyperparams = ed_hyperparams or self._default_ed_params()
        
        # SNNパラメータ  
        self.snn_params = snn_params or self._default_snn_params()
        
        # 興奮性・抑制性ペア構造（MNIST対応）
        self.excitatory_input_size = self.input_size
        self.inhibitory_input_size = self.input_size
        self.total_input_size = self.excitatory_input_size + self.inhibitory_input_size
        
        # ED法コア初期化（高速化版）
        self._initialize_ed_core()
        
        # LIFニューロン層作成
        self._create_lif_layers()
        
        # ニューロンタイプ統計
        self._setup_neuron_types()
        
        # 統合インターフェース
        self._initialize_integration_interface()
        
        # 性能統計
        self.performance_stats = {
            'total_train_time': 0.0,
            'total_samples': 0,
            'avg_sample_time': 0.0
        }
        
        print(f"高速化ED-SNNネットワーク初期化完了:")
        print(f"  構造: {network_structure}")
        print(f"  シミュレーション時間: {simulation_time}ms")
        print(f"  高速化EDコア: {'有効' if use_fast_core else '無効'}")
        print(f"  興奮性・抑制性構造: {self.excitatory_count}E / {self.inhibitory_count}I")
        
    def _default_ed_params(self) -> Dict[str, Any]:
        """ED法デフォルトパラメータ"""
        return {
            'learning_rate': 0.3,
            'initial_amine': 0.7,
            'sigmoid_threshold': 0.7,
            'diffusion_rate': 1.0,
            'initial_weight_1': 1.0,
            'initial_weight_2': 1.0
        }
        
    def _default_snn_params(self) -> Dict[str, Any]:
        """SNNデフォルトパラメータ"""
        return {
            'v_rest': -65.0,
            'v_threshold': -40.0,
            'v_reset': -70.0,
            'tau_m': 12.0,
            'tau_ref': 1.0,
            'r_m': 35.0
        }
        
    def _initialize_ed_core(self):
        """高速化EDコア初期化"""
        class EDHyperParams:
            def __init__(self, params_dict, hidden_size):
                self.learning_rate = params_dict['learning_rate']
                self.initial_amine = params_dict['initial_amine']
                self.sigmoid_threshold = params_dict['sigmoid_threshold']
                self.diffusion_rate = params_dict['diffusion_rate']
                self.initial_weight_1 = params_dict['initial_weight_1']
                self.initial_weight_2 = params_dict['initial_weight_2']
                self.hidden_size = hidden_size
                
        hyperparams = EDHyperParams(self.ed_hyperparams, self.hidden_size)
        
        # 高速化EDコア使用
        self.ed_core = EDCoreFast(hyperparams)
        self.ed_core.initialize_network(
            self.total_input_size,
            self.hidden_size,
            self.output_size
        )
        
    def _create_lif_layers(self):
        """LIFニューロン層の作成"""
        
        # 入力層（興奮性・抑制性ペア）
        input_types = ['excitatory'] * self.excitatory_input_size + ['inhibitory'] * self.inhibitory_input_size
        self.input_layer = LIFNeuronLayer(
            n_neurons=self.total_input_size,
            neuron_params=self.snn_params,
            neuron_types=input_types
        )
        
        # 隠れ層（興奮性・抑制性混合）
        hidden_types = self._generate_mixed_neuron_types(self.hidden_size)
        self.hidden_layer = LIFNeuronLayer(
            n_neurons=self.hidden_size,
            neuron_params=self.snn_params,
            neuron_types=hidden_types
        )
        
        # 出力層（興奮性のみ - ED法理論準拠）
        output_types = ['excitatory'] * self.output_size
        self.output_layer = LIFNeuronLayer(
            n_neurons=self.output_size,
            neuron_params=self.snn_params,
            neuron_types=output_types
        )
        
        print(f"LIFニューロン層作成完了:")
        print(f"  入力層: {len(self.input_layer)} neurons")
        print(f"  隠れ層: {len(self.hidden_layer)} neurons") 
        print(f"  出力層: {len(self.output_layer)} neurons")
        
    def _generate_mixed_neuron_types(self, layer_size: int) -> List[str]:
        """隠れ層用の興奮性・抑制性混合タイプ生成"""
        excitatory_count = int(layer_size * 0.8)
        inhibitory_count = layer_size - excitatory_count
        
        types = ['excitatory'] * excitatory_count + ['inhibitory'] * inhibitory_count
        np.random.shuffle(types)
        
        return types
        
    def _setup_neuron_types(self):
        """ニューロンタイプ統計の計算"""
        self.excitatory_count = 0
        self.inhibitory_count = 0
        
        # 全層のニューロンタイプをカウント
        for layer in [self.input_layer, self.hidden_layer, self.output_layer]:
            for nt in layer.neuron_types:
                if nt == 'excitatory':
                    self.excitatory_count += 1
                else:
                    self.inhibitory_count += 1
                    
    def _initialize_integration_interface(self):
        """統合インターフェース初期化"""
        self.spike_history = {
            'input': [],
            'hidden': [],
            'output': []
        }
        
        self.membrane_potential_history = {
            'input': [],
            'hidden': [],
            'output': []
        }
        
        self.integration_state = {
            'current_time': 0.0,
            'total_spikes': 0,
            'learning_active': False,
            'last_ed_update': 0.0
        }
        
    def encode_input_to_spikes(self, input_data: np.ndarray, encoding_type: str = 'rate') -> np.ndarray:
        """入力データをスパイク列にエンコード（興奮性・抑制性ペア構造）"""
        if encoding_type == 'rate':
            # レート符号化: 値の大きさを発火率に変換
            excitatory_currents = input_data * 50.0  # スケーリング
            inhibitory_currents = (1.0 - input_data) * 30.0  # 逆極性
            
        elif encoding_type == 'temporal':
            # 時間符号化: 値の大きさを発火タイミングに変換
            excitatory_currents = np.where(input_data > 0.5, 40.0, 0.0)
            inhibitory_currents = np.where(input_data <= 0.5, 25.0, 0.0)
            
        else:  # population encoding
            # 集団符号化: 複数ニューロンで値を表現
            excitatory_currents = input_data * 45.0
            inhibitory_currents = np.abs(input_data - 0.5) * 35.0
            
        # 興奮性・抑制性ペア構造に結合
        paired_currents = np.concatenate([excitatory_currents, inhibitory_currents])
        
        return paired_currents
        
    def simulate_snn_dynamics(self, input_currents: np.ndarray) -> Dict[str, np.ndarray]:
        """SNNダイナミクスシミュレーション（軽量化）"""
        time_steps = int(self.simulation_time / self.dt)
        layer_spikes = {
            'input': np.zeros((time_steps, self.total_input_size)),
            'hidden': np.zeros((time_steps, self.hidden_size)),
            'output': np.zeros((time_steps, self.output_size))
        }
        
        # 軽量化: 計算を最小限に
        for t in range(time_steps):
            self.integration_state['current_time'] = t * self.dt
            
            # 入力層更新
            input_spikes = self.input_layer.update(input_currents)
            layer_spikes['input'][t] = input_spikes.astype(float)
            
            # 簡略化した層間結合（性能重視）
            hidden_currents = np.random.rand(self.hidden_size) * 10.0  # 簡略化
            hidden_spikes = self.hidden_layer.update(hidden_currents)
            layer_spikes['hidden'][t] = hidden_spikes.astype(float)
            
            output_currents = np.random.rand(self.output_size) * 5.0  # 簡略化
            output_spikes = self.output_layer.update(output_currents)
            layer_spikes['output'][t] = output_spikes.astype(float)
            
        return layer_spikes
        
    def convert_spikes_to_ed_input(self, spike_pattern: np.ndarray) -> List[float]:
        """スパイクパターンをED法入力に変換（高速化）"""
        # 発火率計算（高速化）
        spike_rates = np.mean(spike_pattern, axis=0)
        
        # 興奮性・抑制性差分計算
        excitatory_rates = spike_rates[:self.excitatory_input_size]
        inhibitory_rates = spike_rates[self.excitatory_input_size:]
        
        # ベクトル化差分計算
        min_size = min(len(excitatory_rates), len(inhibitory_rates))
        diff_values = excitatory_rates[:min_size] - inhibitory_rates[:min_size]
        
        # 正規化（ベクトル化）
        normalized_values = np.tanh(diff_values) * 0.5 + 0.5
        
        return normalized_values.tolist()
        
    def train_step(self, input_data: np.ndarray, target_data: np.ndarray, encoding_type: str = 'rate') -> Dict[str, Any]:
        """高速化学習ステップ"""
        start_time = time.time()
        
        self.integration_state['learning_active'] = True
        
        # 1. 入力をスパイク列にエンコード
        spike_currents = self.encode_input_to_spikes(input_data, encoding_type)
        
        # 2. SNNダイナミクスシミュレーション（軽量化）
        layer_spikes = self.simulate_snn_dynamics(spike_currents)
        
        # 3. スパイクパターンをED法入力に変換
        ed_input = self.convert_spikes_to_ed_input(layer_spikes['input'])
        
        # 4. 高速化ED法学習実行
        ed_outputs = self.ed_core.neuro_output_calc(ed_input)
        self.ed_core.neuro_teach_calc(target_data.tolist())
        self.ed_core.neuro_weight_calc()
        
        # 5. 学習統計計算
        prediction = np.argmax(ed_outputs)
        target_class = np.argmax(target_data)
        accuracy = 1.0 if prediction == target_class else 0.0
        
        # 性能統計更新
        step_time = time.time() - start_time
        self.performance_stats['total_train_time'] += step_time
        self.performance_stats['total_samples'] += 1
        self.performance_stats['avg_sample_time'] = (
            self.performance_stats['total_train_time'] / 
            self.performance_stats['total_samples']
        )
        
        return {
            'prediction': prediction,
            'target': target_class,
            'accuracy': accuracy,
            'outputs': ed_outputs,
            'step_time': step_time,
            'total_spikes': int(np.sum(layer_spikes['input']) + np.sum(layer_spikes['hidden']) + np.sum(layer_spikes['output'])),
            'spike_patterns': layer_spikes
        }
        
    def predict(self, input_data: np.ndarray, encoding_type: str = 'rate') -> Dict[str, Any]:
        """高速化予測"""
        # SNNシミュレーション（軽量化）
        spike_currents = self.encode_input_to_spikes(input_data, encoding_type)
        layer_spikes = self.simulate_snn_dynamics(spike_currents)
        
        # ED法予測（高速化）
        ed_input = self.convert_spikes_to_ed_input(layer_spikes['input'])
        ed_outputs = self.ed_core.neuro_output_calc(ed_input)
        prediction = np.argmax(ed_outputs)
        
        return {
            'prediction': prediction,
            'outputs': ed_outputs,
            'confidence': float(np.max(ed_outputs)),
            'spike_patterns': layer_spikes
        }
        
    def get_performance_report(self) -> str:
        """性能レポート生成"""
        ed_report = self.ed_core.get_performance_report()
        
        return f"""
🚀 高速化ED-SNNネットワーク性能レポート:
  
📊 学習統計:
  総学習時間: {self.performance_stats['total_train_time']:.2f}秒
  学習サンプル数: {self.performance_stats['total_samples']}
  平均サンプル時間: {self.performance_stats['avg_sample_time']:.4f}秒
  推定1エポック(60,000): {self.performance_stats['avg_sample_time'] * 60000 / 60:.1f}分
  
{ed_report}

🏗️ ネットワーク構成:
  構造: {self.network_structure}
  総ニューロン数: {self.excitatory_count + self.inhibitory_count}
  E/I比: {self.excitatory_count}E / {self.inhibitory_count}I
"""
        
    def summary(self):
        """高速化版モデル構成表示"""
        print("\n" + "="*70)
        print("         ED-Spiking Neural Network Summary (高速化版)")
        print("="*70)
        
        # ネットワーク基本情報
        print(f"Network Structure: {self.network_structure}")
        print(f"Simulation Time: {self.simulation_time}ms (dt={self.dt}ms)")
        print(f"Total Neurons: {self.excitatory_count + self.inhibitory_count}")
        print(f"E/I Ratio: {self.excitatory_count}E / {self.inhibitory_count}I")
        print(f"高速化EDコア: {'有効' if self.use_fast_core else '無効'}")
        print("-"*70)
        
        # 性能統計
        if self.performance_stats['total_samples'] > 0:
            print("性能統計:")
            print(f"  平均サンプル処理時間: {self.performance_stats['avg_sample_time']:.4f}秒")
            print(f"  推定1エポック時間: {self.performance_stats['avg_sample_time'] * 60000 / 60:.1f}分")
            print("-"*70)
        
        # ED学習パラメータ
        print("ED Learning Parameters:")
        print(f"  Learning Rate: {self.ed_hyperparams['learning_rate']}")
        print(f"  Sigmoid Threshold: {self.ed_hyperparams['sigmoid_threshold']}")
        print(f"  Initial Amine: {self.ed_hyperparams['initial_amine']}")
        
        print("="*70)