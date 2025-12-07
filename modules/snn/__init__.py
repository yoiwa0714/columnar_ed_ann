"""
SNN (Spiking Neural Network) モジュール

スパイキングニューラルネットワークの実装:
- LIF (Leaky Integrate-and-Fire) ニューロン
- 興奮性・抑制性ニューロンペア構造
- スパイク信号処理
- ED法との統合インターフェース
"""

from .lif_neuron import LIFNeuron, LIFNeuronLayer
from .snn_network import EDSpikingNeuralNetwork

__all__ = ['LIFNeuron', 'LIFNeuronLayer', 'EDSpikingNeuralNetwork']