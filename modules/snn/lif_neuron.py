"""
LIF (Leaky Integrate-and-Fire) ニューロンモジュール

エラー拡散法（ED）準拠のスパイキングニューロン実装
既存実装から移植・簡素化

作成者: ED-SNN開発チーム  
作成日: 2025年1月XX日
バージョン: v001 - 基本実装
参考実装: /home/yoichi/develop/ai/edm/src/relational_ed/snn/modules/neurons/lif_neuron.py
"""

import numpy as np
from typing import Optional, Union


class LIFNeuron:
    """
    LIF (Leaky Integrate-and-Fire) ニューロン
    
    生物学的に妥当なスパイキングニューロンモデル
    ED法での学習に対応した実装
    """
    
    def __init__(
        self,
        v_rest: float = -65.0,      # 静止膜電位 [mV]
        v_threshold: float = -40.0,  # 発火閾値 [mV]  
        v_reset: float = -70.0,     # リセット電位 [mV]
        tau_m: float = 12.0,        # 膜時定数 [ms]
        tau_ref: float = 1.0,       # 不応期 [ms]
        dt: float = 1.0,            # 時間ステップ [ms]
        r_m: float = 35.0,          # 膜抵抗 [MΩ]
        neuron_type: str = 'excitatory'  # ニューロンタイプ
    ):
        """
        LIFニューロンの初期化
        
        Parameters:
        -----------
        v_rest : float
            静止膜電位 [mV]
        v_threshold : float
            発火閾値 [mV]
        v_reset : float
            リセット電位 [mV]
        tau_m : float
            膜時定数 [ms]
        tau_ref : float
            不応期 [ms]
        dt : float
            時間ステップ [ms]
        r_m : float
            膜抵抗 [MΩ]
        neuron_type : str
            ニューロンタイプ ('excitatory' or 'inhibitory')
        """
        # 基本パラメータ
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.tau_m = tau_m
        self.tau_ref = tau_ref
        self.dt = dt
        self.r_m = r_m
        self.neuron_type = neuron_type
        
        # パラメータ検証
        self._validate_parameters()
        
        # 状態変数
        self.v_membrane = v_rest     # 現在の膜電位
        self.last_spike_time = -np.inf  # 最後のスパイク時刻
        self.current_time = 0.0      # 現在時刻
        
        # スパイク記録
        self.spike_count = 0
        self.spike_times = []
        
        # ED法関連
        self.amine_concentration = 0.0  # アミン濃度（ED法用）
        
    def _validate_parameters(self):
        """パラメータの妥当性確認"""
        if self.v_threshold <= self.v_rest:
            raise ValueError(
                f"閾値({self.v_threshold})は静止電位({self.v_rest})より大きい必要があります"
            )
        if self.tau_m <= 0:
            raise ValueError(f"膜時定数({self.tau_m})は正の値である必要があります")
        if self.tau_ref < 0:
            raise ValueError(f"不応期({self.tau_ref})は非負である必要があります")
        if self.dt <= 0:
            raise ValueError(f"時間ステップ({self.dt})は正の値である必要があります")
            
    def update(self, input_current: float) -> bool:
        """
        ニューロン状態を1時間ステップ更新
        
        Parameters:
        -----------
        input_current : float
            入力電流 [pA]
            
        Returns:
        --------
        bool
            スパイクが発生した場合True
        """
        # 時間更新
        self.current_time += self.dt
        
        # 不応期チェック
        if self._is_in_refractory_period():
            return False
            
        # 膜電位更新 (LIF方程式)
        self._update_membrane_potential(input_current)
        
        # スパイク検出
        if self._check_spike():
            self._fire_spike()
            return True
            
        return False
        
    def _is_in_refractory_period(self) -> bool:
        """不応期中かどうかチェック"""
        return (self.current_time - self.last_spike_time) < self.tau_ref
        
    def _update_membrane_potential(self, input_current: float):
        """膜電位の更新（LIF方程式）"""
        # LIF方程式: dv/dt = (v_rest - v + R*I) / tau_m
        dv_dt = ((self.v_rest - self.v_membrane) + self.r_m * input_current) / self.tau_m
        self.v_membrane += dv_dt * self.dt
        
    def _check_spike(self) -> bool:
        """スパイク発生の判定"""
        return self.v_membrane >= self.v_threshold
        
    def _fire_spike(self):
        """スパイク発生処理"""
        self.v_membrane = self.v_reset
        self.last_spike_time = self.current_time
        self.spike_count += 1
        self.spike_times.append(self.current_time)
        
    def reset_state(self):
        """ニューロン状態をリセット"""
        self.v_membrane = self.v_rest
        self.last_spike_time = -np.inf
        self.current_time = 0.0
        self.spike_count = 0
        self.spike_times = []
        self.amine_concentration = 0.0
        
    def get_membrane_potential(self) -> float:
        """現在の膜電位を取得"""
        return self.v_membrane
        
    def get_spike_rate(self, time_window: Optional[float] = None) -> float:
        """
        スパイク発火率を計算
        
        Parameters:
        -----------
        time_window : float, optional
            計算する時間窓 [ms]。Noneの場合は全期間
            
        Returns:
        --------
        float
            発火率 [Hz]
        """
        if time_window is None:
            time_window = self.current_time
            
        if time_window <= 0:
            return 0.0
            
        # 指定時間窓内のスパイク数をカウント
        recent_spikes = [
            t for t in self.spike_times 
            if t >= (self.current_time - time_window)
        ]
        
        return len(recent_spikes) / (time_window / 1000.0)  # Hz変換
        
    def set_amine_concentration(self, concentration: float):
        """ED法用アミン濃度設定"""
        self.amine_concentration = concentration
        
    def get_amine_concentration(self) -> float:
        """ED法用アミン濃度取得"""
        return self.amine_concentration
        
    def get_neuron_info(self) -> dict:
        """ニューロンの詳細情報取得"""
        return {
            'neuron_type': self.neuron_type,
            'v_membrane': self.v_membrane,
            'v_rest': self.v_rest,
            'v_threshold': self.v_threshold,
            'spike_count': self.spike_count,
            'current_time': self.current_time,
            'spike_rate': self.get_spike_rate(),
            'amine_concentration': self.amine_concentration
        }
        
    def __repr__(self) -> str:
        return (f"LIFNeuron(type={self.neuron_type}, "
                f"v={self.v_membrane:.2f}mV, "
                f"spikes={self.spike_count}, "
                f"rate={self.get_spike_rate():.2f}Hz)")


class LIFNeuronLayer:
    """
    LIFニューロンの層実装
    
    複数のLIFニューロンを効率的に管理
    """
    
    def __init__(
        self,
        n_neurons: int,
        neuron_params: Optional[dict] = None,
        neuron_types: Optional[list] = None
    ):
        """
        LIFニューロン層の初期化
        
        Parameters:
        -----------
        n_neurons : int
            ニューロン数
        neuron_params : dict, optional
            LIFニューロンのパラメータ辞書
        neuron_types : list, optional
            各ニューロンのタイプリスト
        """
        self.n_neurons = n_neurons
        
        # デフォルトパラメータ
        if neuron_params is None:
            neuron_params = {}
            
        # デフォルトニューロンタイプ
        if neuron_types is None:
            neuron_types = ['excitatory'] * n_neurons
            
        # ニューロンタイプを保存
        self.neuron_types = neuron_types
            
        # ニューロン作成
        self.neurons = []
        for i in range(n_neurons):
            neuron = LIFNeuron(
                neuron_type=neuron_types[i],
                **neuron_params
            )
            self.neurons.append(neuron)
            
    def update(self, input_currents: np.ndarray) -> np.ndarray:
        """
        全ニューロンを更新
        
        Parameters:
        -----------
        input_currents : np.ndarray
            各ニューロンへの入力電流
            
        Returns:
        --------
        np.ndarray
            各ニューロンのスパイク状態 (bool array)
        """
        if len(input_currents) != self.n_neurons:
            raise ValueError(
                f"入力電流数({len(input_currents)})がニューロン数({self.n_neurons})と一致しません"
            )
            
        spikes = np.zeros(self.n_neurons, dtype=bool)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.update(input_currents[i])
            
        return spikes
        
    def reset_all(self):
        """全ニューロンをリセット"""
        for neuron in self.neurons:
            neuron.reset_state()
            
    def get_membrane_potentials(self) -> np.ndarray:
        """全ニューロンの膜電位取得"""
        return np.array([neuron.get_membrane_potential() for neuron in self.neurons])
        
    def get_spike_counts(self) -> np.ndarray:
        """全ニューロンのスパイク数取得"""
        return np.array([neuron.spike_count for neuron in self.neurons])
        
    def get_spike_rates(self, time_window: Optional[float] = None) -> np.ndarray:
        """全ニューロンの発火率取得"""
        return np.array([neuron.get_spike_rate(time_window) for neuron in self.neurons])
        
    def __len__(self) -> int:
        return self.n_neurons
        
    def __getitem__(self, index) -> LIFNeuron:
        return self.neurons[index]