"""
SNNネットワーククラス - ED法統合版

LIFニューロンと機能するマルチクラス分類ED法を統合
ed_multi_snn.prompt.md仕様準拠

作成者: ED-SNN開発チーム
作成日: 2025年9月28日  
バージョン: v001 - 基本統合版
理論準拠: 金子勇氏 Error Diffusion Learning Algorithm + SNN拡張
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any

# 内部モジュール
from .lif_neuron import LIFNeuron, LIFNeuronLayer
from ..ed_learning.ed_core import EDCore
from ..utils.profiler import profile_function, TimingContext, profiler


class EDSpikingNeuralNetwork:
    """
    ED法統合スパイキングニューラルネットワーク
    
    🎯 核心機能:
    - LIFニューロンによるスパイクダイナミクス
    - 機能するマルチクラス分類ED法学習
    - 興奮性・抑制性ニューロンペア構造
    - 独立出力ニューロンアーキテクチャ
    
    仕様準拠:
    - ed_multi_snn.prompt.md 100%準拠
    - 金子勇氏ED法理論完全保持
    - SNN拡張機能統合
    """
    
    def __init__(
        self,
        network_structure: List[int],
        ed_hyperparams: Optional[Dict[str, Any]] = None,
        snn_params: Optional[Dict[str, Any]] = None,
        simulation_time: float = 50.0,
        dt: float = 1.0
    ):
        """
        ED-SNNネットワーク初期化
        
        Parameters:
        -----------
        network_structure : List[int]
            ネットワーク構造 [input_size, hidden_size, output_size]
            例: [784, 32, 10] for MNIST
        ed_hyperparams : dict, optional
            ED法ハイパーパラメータ
        snn_params : dict, optional
            SNNパラメータ（LIFニューロン設定）
        simulation_time : float
            シミュレーション時間 [ms]
        dt : float
            時間ステップ [ms]
        """
        self.network_structure = network_structure
        self.input_size = network_structure[0]
        self.hidden_size = network_structure[1] if len(network_structure) > 1 else 32
        self.output_size = network_structure[2] if len(network_structure) > 2 else 10
        
        self.simulation_time = simulation_time
        self.dt = dt
        
        # デフォルトパラメータ設定
        self.ed_hyperparams = ed_hyperparams or self._default_ed_params()
        self.snn_params = snn_params or self._default_snn_params()
        
        # ED法コア初期化
        self.ed_core = EDCore(self._create_ed_hyperparams())
        
        # SNN構造初期化（ed_multi_snn.prompt.md準拠）
        self._initialize_snn_structure()
        
        # 統合インターフェース
        self._initialize_integration_interface()
        
        print(f"ED-SNNネットワーク初期化完了:")
        print(f"  構造: {network_structure}")
        print(f"  シミュレーション時間: {simulation_time}ms")
        print(f"  興奮性・抑制性構造: {self.excitatory_count}E / {self.inhibitory_count}I")
        
    def _default_ed_params(self) -> Dict[str, Any]:
        """ED法デフォルトパラメータ（ed_v032準拠）"""
        return {
            'learning_rate': 0.3,      # Phase 2最適値
            'initial_amine': 0.7,      # Phase 2最適値
            'sigmoid_threshold': 0.7,  # Phase 1最適値
            'diffusion_rate': 1.0,
            'initial_weight_1': 1.0,
            'initial_weight_2': 1.0
        }
        
    def _default_snn_params(self) -> Dict[str, Any]:
        """SNNデフォルトパラメータ（LIF標準値）"""
        return {
            'v_rest': -65.0,        # 静止膜電位 [mV]
            'v_threshold': -40.0,   # 発火閾値 [mV]
            'v_reset': -70.0,       # リセット電位 [mV]
            'tau_m': 12.0,          # 膜時定数 [ms]
            'tau_ref': 1.0,         # 不応期 [ms]
            'r_m': 35.0             # 膜抵抗 [MΩ]
        }
        
    def _create_ed_hyperparams(self):
        """ED法用ハイパーパラメータオブジェクト作成"""
        class EDHyperParams:
            def __init__(self, params, hidden_size):
                for key, value in params.items():
                    setattr(self, key, value)
                # 隠れ層サイズ追加
                self.hidden_neurons = params.get('hidden_neurons', hidden_size)
                    
        return EDHyperParams(self.ed_hyperparams, self.hidden_size)
        
    def _initialize_snn_structure(self):
        """SNN構造初期化（ed_multi_snn.prompt.md準拠）"""
        
        # 🎯 興奮性・抑制性ニューロンペア構造（ED法理論準拠）
        # 入力層: 興奮性・抑制性ペアで構成（MNIST: 784 → 1568）
        self.excitatory_input_size = self.input_size
        self.inhibitory_input_size = self.input_size  
        self.total_input_size = self.excitatory_input_size + self.inhibitory_input_size
        
        print(f"興奮性・抑制性ペア構造:")
        print(f"  興奮性入力: {self.excitatory_input_size}")
        print(f"  抑制性入力: {self.inhibitory_input_size}")
        print(f"  総入力サイズ: {self.total_input_size}")
        
        # ED法コアにネットワーク構造を設定
        self.ed_core.initialize_network(
            self.total_input_size,  # 興奮性・抑制性ペア後サイズ
            self.hidden_size,
            self.output_size
        )
        
        # 🧠 LIFニューロン層の作成
        self._create_lif_layers()
        
        # 🔗 ニューロンタイプの設定
        self._setup_neuron_types()
        
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
        # 80% 興奮性, 20% 抑制性 (生物学的比率)
        excitatory_count = int(layer_size * 0.8)
        inhibitory_count = layer_size - excitatory_count
        
        types = ['excitatory'] * excitatory_count + ['inhibitory'] * inhibitory_count
        
        # ランダムシャッフル
        np.random.shuffle(types)
        
        return types
        
    def _setup_neuron_types(self):
        """ニューロンタイプ統計の計算"""
        self.excitatory_count = 0
        self.inhibitory_count = 0
        
        # 全層のニューロンタイプをカウント
        for layer in [self.input_layer, self.hidden_layer, self.output_layer]:
            for neuron in layer.neurons:
                if neuron.neuron_type == 'excitatory':
                    self.excitatory_count += 1
                else:
                    self.inhibitory_count += 1
                    
    def _initialize_integration_interface(self):
        """SNN-ED統合インターフェース初期化"""
        
        # スパイク履歴管理
        self.spike_history = {
            'input': [],
            'hidden': [],
            'output': []
        }
        
        # 膜電位履歴
        self.membrane_potential_history = {
            'input': [],
            'hidden': [],
            'output': []
        }
        
        # ED-SNN統合状態
        self.integration_state = {
            'current_time': 0.0,
            'total_spikes': 0,
            'learning_active': False,
            'last_ed_update': 0.0
        }
        
    @profile_function("encode_input_to_spikes")
    def encode_input_to_spikes(self, input_data: np.ndarray, encoding_type: str = 'rate') -> np.ndarray:
        """
        入力データをスパイク列にエンコード（興奮性・抑制性ペア構造）
        
        Parameters:
        -----------
        input_data : np.ndarray
            入力データ (例: MNIST 784次元)
        encoding_type : str
            エンコーディングタイプ ('rate', 'temporal', 'population')
            
        Returns:
        --------
        np.ndarray
            興奮性・抑制性ペア構造のスパイク電流
        """
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
        
    @profile_function("simulate_snn_dynamics")
    def simulate_snn_dynamics(self, input_currents: np.ndarray) -> Dict[str, np.ndarray]:
        """
        SNNダイナミクスシミュレーション
        
        Parameters:
        -----------
        input_currents : np.ndarray
            入力電流パターン
            
        Returns:
        --------
        Dict[str, np.ndarray]
            各層のスパイクパターン
        """
        time_steps = int(self.simulation_time / self.dt)
        layer_spikes = {
            'input': np.zeros((time_steps, self.total_input_size)),
            'hidden': np.zeros((time_steps, self.hidden_size)),
            'output': np.zeros((time_steps, self.output_size))
        }
        
        # 時間発展シミュレーション
        for t in range(time_steps):
            self.integration_state['current_time'] = t * self.dt
            
            # 入力層更新
            input_spikes = self.input_layer.update(input_currents)
            layer_spikes['input'][t] = input_spikes.astype(float)
            
            # 隠れ層への結合重みに基づく電流計算
            hidden_currents = self._calculate_layer_currents(
                input_spikes, 'input_to_hidden', t
            )
            
            # 隠れ層更新
            hidden_spikes = self.hidden_layer.update(hidden_currents)
            layer_spikes['hidden'][t] = hidden_spikes.astype(float)
            
            # 出力層への結合重みに基づく電流計算
            output_currents = self._calculate_layer_currents(
                hidden_spikes, 'hidden_to_output', t
            )
            
            # 出力層更新
            output_spikes = self.output_layer.update(output_currents)
            layer_spikes['output'][t] = output_spikes.astype(float)
            
        # スパイク履歴更新
        self.spike_history['input'].append(layer_spikes['input'])
        self.spike_history['hidden'].append(layer_spikes['hidden'])
        self.spike_history['output'].append(layer_spikes['output'])
        
        # 統計更新
        total_spikes = (layer_spikes['input'].sum() + 
                       layer_spikes['hidden'].sum() + 
                       layer_spikes['output'].sum())
        self.integration_state['total_spikes'] += total_spikes
        
        return layer_spikes
        
    def _calculate_layer_currents(self, source_spikes: np.ndarray, connection_type: str, time_step: int) -> np.ndarray:
        """
        層間結合重みに基づく電流計算
        
        Parameters:
        -----------
        source_spikes : np.ndarray
            送信側スパイクパターン
        connection_type : str
            結合タイプ ('input_to_hidden', 'hidden_to_output')
        time_step : int
            現在の時間ステップ
            
        Returns:
        --------
        np.ndarray
            目標層への入力電流
        """
        if connection_type == 'input_to_hidden':
            # 入力層→隠れ層の電流計算
            # ED法の重み配列から対応する重みを取得
            currents = np.zeros(self.hidden_size)
            
            # 簡易的な結合重み計算（後でED法重みと統合）
            for i in range(self.hidden_size):
                weighted_sum = 0.0
                for j, spike in enumerate(source_spikes):
                    if spike > 0:  # スパイクがある場合
                        # ED法重み配列から重みを取得
                        weight = self._get_ed_weight(0, j + 2, i + self.total_input_size + 3)
                        weighted_sum += weight * 30.0  # スパイクの電流強度
                currents[i] = weighted_sum
                
        else:  # hidden_to_output
            # 隠れ層→出力層の電流計算
            currents = np.zeros(self.output_size)
            
            for i in range(self.output_size):
                weighted_sum = 0.0
                for j, spike in enumerate(source_spikes):
                    if spike > 0:
                        # ED法重み配列から重みを取得
                        weight = self._get_ed_weight(i, self.total_input_size + 3 + j, self.total_input_size + 2)
                        weighted_sum += weight * 25.0
                currents[i] = weighted_sum
                
        return currents
        
    def _get_ed_weight(self, output_neuron: int, from_unit: int, to_unit: int) -> float:
        """ED法重み配列から重み値を取得"""
        try:
            if (output_neuron < self.ed_core.output_weights.shape[0] and
                from_unit < self.ed_core.output_weights.shape[1] and
                to_unit < self.ed_core.output_weights.shape[2]):
                return self.ed_core.output_weights[output_neuron, from_unit, to_unit]
            else:
                return 0.0
        except (AttributeError, IndexError):
            return np.random.normal(0, 0.1)  # フォールバック
            
    @profile_function("convert_spikes_to_ed_input")
    def convert_spikes_to_ed_input(self, spike_pattern: np.ndarray) -> List[float]:
        """
        スパイクパターンをED法入力形式に変換
        
        Parameters:
        -----------
        spike_pattern : np.ndarray
            スパイクパターン (time_steps x neurons)
            
        Returns:
        --------
        List[float]
            ED法用入力パターン
        """
        # スパイク発火率を計算
        spike_rates = np.mean(spike_pattern, axis=0)
        
        # 興奮性・抑制性ペアの処理
        excitatory_rates = spike_rates[:self.excitatory_input_size]
        inhibitory_rates = spike_rates[self.excitatory_input_size:]
        
        # 差分計算によるED法入力生成
        ed_input = []
        for i in range(min(len(excitatory_rates), len(inhibitory_rates))):
            # 興奮性 - 抑制性の差分
            diff_value = excitatory_rates[i] - inhibitory_rates[i]
            # 正規化してシグモイド範囲に調整
            normalized_value = np.tanh(diff_value) * 0.5 + 0.5
            ed_input.append(float(normalized_value))
            
        return ed_input
        
    @profile_function("train_step")
    def train_step(self, input_data: np.ndarray, target_data: np.ndarray, encoding_type: str = 'rate') -> Dict[str, Any]:
        """
        ED-SNN統合学習ステップ
        
        Parameters:
        -----------
        input_data : np.ndarray
            入力データ
        target_data : np.ndarray  
            目標データ
        encoding_type : str
            エンコーディングタイプ
            
        Returns:
        --------
        Dict[str, Any]
            学習結果情報
        """
        self.integration_state['learning_active'] = True
        
        # 1. 入力をスパイク列にエンコード
        spike_currents = self.encode_input_to_spikes(input_data, encoding_type)
        
        # 2. SNNダイナミクスシミュレーション
        layer_spikes = self.simulate_snn_dynamics(spike_currents)
        
        # 3. スパイクパターンをED法入力に変換
        ed_input = self.convert_spikes_to_ed_input(layer_spikes['input'])
        
        # 4. ED法学習実行
        ed_outputs = self.ed_core.neuro_output_calc(ed_input)
        self.ed_core.neuro_teach_calc(target_data.tolist())
        self.ed_core.neuro_weight_calc()
        
        # 5. 学習統計計算
        prediction = np.argmax(ed_outputs)
        target_class = np.argmax(target_data)
        accuracy = 1.0 if prediction == target_class else 0.0
        
        self.integration_state['last_ed_update'] = self.integration_state['current_time']
        
        return {
            'prediction': prediction,
            'target': target_class,
            'accuracy': accuracy,
            'outputs': ed_outputs,
            'total_spikes': self.integration_state['total_spikes'],
            'simulation_time': self.integration_state['current_time'],
            'spike_patterns': layer_spikes
        }
        
    def predict(self, input_data: np.ndarray, encoding_type: str = 'rate') -> Dict[str, Any]:
        """
        ED-SNN統合予測
        
        Parameters:
        -----------
        input_data : np.ndarray
            入力データ
        encoding_type : str
            エンコーディングタイプ
            
        Returns:
        --------
        Dict[str, Any]
            予測結果
        """
        # SNNシミュレーション
        spike_currents = self.encode_input_to_spikes(input_data, encoding_type)
        layer_spikes = self.simulate_snn_dynamics(spike_currents)
        
        # ED法予測
        ed_input = self.convert_spikes_to_ed_input(layer_spikes['input'])
        ed_outputs = self.ed_core.neuro_output_calc(ed_input)
        prediction = np.argmax(ed_outputs)
        
        return {
            'prediction': prediction,
            'outputs': ed_outputs,
            'confidence': float(np.max(ed_outputs)),
            'spike_patterns': layer_spikes
        }
        
    def reset_network(self):
        """ネットワーク状態リセット"""
        # LIF層リセット
        self.input_layer.reset_all()
        self.hidden_layer.reset_all()
        self.output_layer.reset_all()
        
        # ED法リセット
        self.ed_core.reset_error()
        
        # 履歴クリア
        for key in self.spike_history:
            self.spike_history[key].clear()
        for key in self.membrane_potential_history:
            self.membrane_potential_history[key].clear()
            
        # 統合状態リセット
        self.integration_state['current_time'] = 0.0
        self.integration_state['total_spikes'] = 0
        self.integration_state['learning_active'] = False
        
    def get_network_info(self) -> Dict[str, Any]:
        """ネットワーク詳細情報取得"""
        return {
            'network_structure': self.network_structure,
            'total_neurons': (len(self.input_layer) + len(self.hidden_layer) + len(self.output_layer)),
            'excitatory_count': self.excitatory_count,
            'inhibitory_count': self.inhibitory_count,
            'simulation_time': self.simulation_time,
            'ed_info': self.ed_core.get_network_info(),
            'integration_state': self.integration_state.copy(),
            'snn_params': self.snn_params,
            'ed_params': self.ed_hyperparams
        }
        
    def summary(self):
        """
        TensorFlowスタイルのモデル構成表示
        """
        print("\n" + "="*70)
        print("              ED-Spiking Neural Network Summary")
        print("="*70)
        
        # ネットワーク基本情報
        print(f"Network Structure: {self.network_structure}")
        print(f"Simulation Time: {self.simulation_time}ms (dt={self.dt}ms)")
        print(f"Total Neurons: {self.excitatory_count + self.inhibitory_count}")
        print(f"E/I Ratio: {self.excitatory_count}E / {self.inhibitory_count}I")
        print("-"*70)
        
        # 層別詳細情報
        print(f"{'Layer Type':<15} {'Neurons':<10} {'E/I Composition':<20} {'Parameters':<15}")
        print("-"*70)
        
        # 入力層
        input_e = sum(1 for nt in self.input_layer.neuron_types if nt == 'excitatory')
        input_i = len(self.input_layer) - input_e
        print(f"{'Input Layer':<15} {len(self.input_layer):<10} "
              f"{input_e}E + {input_i}I{'':<8} {'Paired E/I':<15}")
        
        # 隠れ層
        hidden_e = sum(1 for nt in self.hidden_layer.neuron_types if nt == 'excitatory')
        hidden_i = len(self.hidden_layer) - hidden_e
        print(f"{'Hidden Layer':<15} {len(self.hidden_layer):<10} "
              f"{hidden_e}E + {hidden_i}I{'':<8} {'Mixed 80/20':<15}")
        
        # 出力層
        output_e = sum(1 for nt in self.output_layer.neuron_types if nt == 'excitatory')
        output_i = len(self.output_layer) - output_e
        print(f"{'Output Layer':<15} {len(self.output_layer):<10} "
              f"{output_e}E + {output_i}I{'':<8} {'Excitatory':<15}")
        
        print("-"*70)
        
        # ED法情報
        ed_info = self.ed_core.get_network_info()
        print("ED Learning Parameters:")
        print(f"  Learning Rate: {self.ed_hyperparams['learning_rate']}")
        print(f"  Sigmoid Threshold: {self.ed_hyperparams['sigmoid_threshold']}")
        print(f"  Initial Amine: {self.ed_hyperparams['initial_amine']}")
        print(f"  Weight Array Shape: {ed_info['weight_shape']}")
        
        # SNN パラメータ情報
        print("\nLIF Neuron Parameters:")
        print(f"  Membrane Time Constant: {self.snn_params['tau_m']}ms")
        print(f"  Threshold Voltage: {self.snn_params['v_threshold']}mV")
        print(f"  Reset Voltage: {self.snn_params['v_reset']}mV")
        print(f"  Refractory Period: {self.snn_params['tau_ref']}ms")
        
        print("="*70)
        
        # 接続統計
        total_possible_connections = (
            len(self.input_layer) * len(self.hidden_layer) +
            len(self.hidden_layer) * len(self.output_layer)
        )
        
        print(f"\nConnection Statistics:")
        print(f"  Input → Hidden: {len(self.input_layer)} × {len(self.hidden_layer)} = "
              f"{len(self.input_layer) * len(self.hidden_layer):,} connections")
        print(f"  Hidden → Output: {len(self.hidden_layer)} × {len(self.output_layer)} = "
              f"{len(self.hidden_layer) * len(self.output_layer):,} connections")
        print(f"  Total Connections: {total_possible_connections:,}")
        
        # メモリ使用量推定
        weight_memory = ed_info['weight_shape'][0] * ed_info['weight_shape'][1] * ed_info['weight_shape'][2] * 8  # float64
        neuron_memory = (len(self.input_layer) + len(self.hidden_layer) + len(self.output_layer)) * 64  # 推定
        
        print(f"\nMemory Estimation:")
        print(f"  Weight Arrays: {weight_memory / 1024 / 1024:.2f} MB")
        print(f"  Neuron States: {neuron_memory / 1024:.2f} KB")
        print(f"  Total Estimated: {(weight_memory + neuron_memory) / 1024 / 1024:.2f} MB")
        
        print("="*70)

    def __repr__(self) -> str:
        return (f"EDSpikingNeuralNetwork("
                f"structure={self.network_structure}, "
                f"neurons={self.excitatory_count}E+{self.inhibitory_count}I, "
                f"sim_time={self.simulation_time}ms)")