#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ED-SNN ヒートマップ統合クラス v1.0
ed_multi_snn.prompt.md準拠・スパイキングニューラルネットワーク統合版

移植元: ed_v032_simple.py EDHeatmapIntegration
対応: Milestone 3最適化済みED法準拠SNNアーキテクチャ

機能:
1. SNNネットワークとヒートマップ可視化の統合
2. スパイク活動データの収集・変換
3. リアルタイム可視化制御
4. 学習進捗統合表示
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple


class EDSNNHeatmapIntegration:
    """
    ED-SNN学習システムとヒートマップ可視化の統合クラス
    
    既存のSNN学習機能を変更せず、補助的にヒートマップ機能を提供
    """
    
    def __init__(self, args, snn_network, class_names: Optional[Dict[int, str]] = None, 
                 image_shape: Optional[Tuple[int, ...]] = None):
        """
        初期化
        
        Args:
            args: HyperParamsオブジェクト (v015) またはコマンドライン引数
            snn_network: SNNネットワークインスタンス
            class_names: クラス名マッピング {0: "T-shirt/top", 1: "Trouser", ...}
                        指定しない場合は数値のみ表示（MNIST等）
            image_shape: 入力画像の形状 ((28, 28), (32, 32, 3)など)
                        指定しない場合は(28, 28)をデフォルトとする
        """
        self.args = args
        self.snn_network = snn_network
        self.visualizer = None
        self.update_counter = 0
        self.update_interval = 1  # 毎回更新でリアルタイム表示
        self.current_epoch = 0
        self._heatmap_ready = False
        self.class_names = class_names or {}  # クラス名マッピング
        self.image_shape = image_shape or (28, 28)  # デフォルトはMNIST形状
        
        # v015対応: enable_heatmap属性をチェック（後方互換性維持）
        enable_heatmap = getattr(args, 'enable_heatmap', False) or getattr(args, 'heatmap', False)
        if enable_heatmap:
            self._initialize_snn_heatmap_visualizer()
            self._setup_snn_heatmap_callback()
    
    def _initialize_snn_heatmap_visualizer(self):
        """SNNヒートマップ可視化システムを初期化"""
        try:
            from modules.snn_heatmap_visualizer import SNNHeatmapRealtimeVisualizer
            
            # SNN構造に合わせた層形状設定
            layer_shapes = self._calculate_snn_layer_shapes()
            
            # SNNパラメータ準備
            snn_params = self._prepare_snn_parameters()
            
            # 実行パラメータ準備
            exec_params = self._prepare_execution_parameters()
            
            # ヒートマップ可視化システム初期化
            self.visualizer = SNNHeatmapRealtimeVisualizer(
                layer_shapes=layer_shapes,
                show_parameters=True,
                update_interval=0.8,  # 0.8秒間隔で更新
                colormap='rainbow',
                snn_params=snn_params,
                exec_params=exec_params,
                class_names=self.class_names  # クラス名マッピングを渡す
            )
            
            print("🎯 SNNヒートマップ可視化システム初期化完了")
            print("🎯 ヒートマップウィンドウ表示は学習開始まで待機...")
            self._heatmap_ready = False
            
        except ImportError as e:
            print(f"❌ SNNヒートマップ可視化モジュールの読み込みに失敗しました: {e}")
            self.visualizer = None
        except Exception as e:
            print(f"❌ SNNヒートマップ可視化システムの初期化に失敗しました: {e}")
            self.visualizer = None
    
    def _calculate_snn_layer_shapes(self) -> List[Tuple[int, ...]]:
        """SNN層構造に基づいて表示形状を計算（ed_v032_simple.py準拠）"""
        layer_shapes = []
        
        # 入力層: image_shapeをそのまま使用
        # グレースケール: (28, 28)
        # カラー: (32, 32, 3)
        layer_shapes.append(self.image_shape)
        
        # 隠れ層構造を取得（正方形に近い形状で表示）
        # v015対応: hidden_layersリストまたはhidden文字列を処理
        hidden_sizes = None
        if hasattr(self.args, 'hidden_layers') and self.args.hidden_layers:
            # v015形式: 既にリストに解析済み
            hidden_sizes = self.args.hidden_layers
        elif hasattr(self.args, 'hidden') and self.args.hidden:
            # 旧形式: 文字列から解析
            if isinstance(self.args.hidden, str):
                hidden_sizes = [int(x.strip()) for x in self.args.hidden.split(',')]
            else:
                hidden_sizes = self.args.hidden
        
        if hidden_sizes:
            for hidden_size in hidden_sizes:
                grid_shape = self._calculate_square_grid_shape(hidden_size)
                layer_shapes.append(grid_shape)
        else:
            # デフォルト隠れ層: [128] → 12x12正方形（v015準拠）
            layer_shapes.append((12, 12))  # 128 → 12x12 (144ニューロン表示)
        
        # 出力層: クラス数に基づいて動的に設定
        output_size = getattr(self.args, 'output_size', 10)
        output_shape = self._calculate_square_grid_shape(output_size)
        layer_shapes.append(output_shape)
        
        return layer_shapes
    
    def _calculate_square_grid_shape(self, neuron_count: int) -> Tuple[int, int]:
        """ニューロン数から正方形に近いグリッド形状を計算（ed_v032_simple.py準拠）"""
        if neuron_count <= 0:
            return (1, 1)
        
        # ed_v032_simple.py準拠の正方形形状計算
        sqrt_count = int(np.sqrt(neuron_count))
        
        # 完全正方形またはそれに近い形状を優先
        if sqrt_count * sqrt_count == neuron_count:
            return (sqrt_count, sqrt_count)
        
        # 正方形に近い形状を探索
        best_diff = float('inf')
        best_shape = (sqrt_count, sqrt_count + 1)
        
        for height in range(max(1, sqrt_count - 2), sqrt_count + 3):
            width = (neuron_count + height - 1) // height
            diff = abs(height - width)
            if diff < best_diff:
                best_diff = diff
                best_shape = (height, width)
        
        return best_shape
    
    def _prepare_snn_parameters(self) -> Dict[str, Any]:
        """SNNパラメータを準備（ed_v032_simple.py準拠形式）"""
        snn_params = {}
        
        # ED法アルゴリズムパラメータ（args準拠、ed_snn_v024.py対応）
        # 学習率 (--lr, --learning_rate)
        if hasattr(self.args, 'learning_rate'):
            snn_params['学習率'] = f"{self.args.learning_rate:.3f}"
        else:
            snn_params['学習率'] = "0.050"
        
        # アミン濃度 (--ami, --amine)
        if hasattr(self.args, 'initial_amine'):
            snn_params['amine'] = f"{self.args.initial_amine:.3f}"
        else:
            snn_params['amine'] = "0.250"
        
        # 拡散係数 (--dif, --diffusion)
        if hasattr(self.args, 'diffusion_rate'):
            snn_params['diffusion'] = f"{self.args.diffusion_rate:.3f}"
        else:
            snn_params['diffusion'] = "0.300"
        
        # シグモイド閾値 (--sig, --sigmoid)
        if hasattr(self.args, 'sigmoid_threshold'):
            snn_params['sigmoid'] = f"{self.args.sigmoid_threshold:.3f}"
        else:
            snn_params['sigmoid'] = "0.700"
        
        # 重み初期値1 (--w1, --weight1)
        if hasattr(self.args, 'initial_weight_1'):
            snn_params['weight1'] = f"{self.args.initial_weight_1:.3f}"
        else:
            snn_params['weight1'] = "0.300"
        
        # 重み初期値2 (--w2, --weight2)
        if hasattr(self.args, 'initial_weight_2'):
            snn_params['weight2'] = f"{self.args.initial_weight_2:.3f}"
        else:
            snn_params['weight2'] = "0.500"
        
        return snn_params
    
    def _prepare_execution_parameters(self) -> Dict[str, Any]:
        """実行パラメータを準備（ed_v032_simple.py準拠形式）"""
        exec_params = {}
        
        # データセット情報（v015対応: fashion_mnistフラグ）
        if hasattr(self.args, 'fashion_mnist'):
            exec_params['データセット'] = 'FASHION_MNIST' if self.args.fashion_mnist else 'MNIST'
        elif hasattr(self.args, 'dataset'):
            exec_params['データセット'] = self.args.dataset.upper()
        else:
            exec_params['データセット'] = 'FASHION_MNIST'
        
        # 学習設定
        if hasattr(self.args, 'epochs'):
            exec_params['エポック数'] = str(self.args.epochs)
        else:
            exec_params['エポック数'] = '10'
            
        if hasattr(self.args, 'train_samples'):
            exec_params['学習サンプル'] = str(self.args.train_samples)
        else:
            exec_params['学習サンプル'] = '512'
            
        if hasattr(self.args, 'test_samples'):
            exec_params['テストサンプル'] = str(self.args.test_samples)
        else:
            exec_params['テストサンプル'] = '512'
        
        # 隠れ層構造（v015対応: hidden_layersリスト）
        if hasattr(self.args, 'hidden_layers') and self.args.hidden_layers:
            hidden_str = ','.join(map(str, self.args.hidden_layers))
            exec_params['隠れ層構造'] = f"[{hidden_str}]"
        elif hasattr(self.args, 'hidden') and self.args.hidden:
            exec_params['隠れ層構造'] = f"[{self.args.hidden}]"
        else:
            exec_params['隠れ層構造'] = '[128]'
        
        # その他のパラメータ
        if hasattr(self.args, 'batch_size'):
            exec_params['batch_size'] = str(self.args.batch_size)
        else:
            exec_params['batch_size'] = '128'
        exec_params['seed'] = str(getattr(self.args, 'seed', 42)) if hasattr(self.args, 'seed') and self.args.seed else 'None'
        exec_params['詳細表示'] = 'ON' if getattr(self.args, 'verbose', False) else 'OFF'
        
        return exec_params
    
    def _setup_snn_heatmap_callback(self):
        """SNNネットワークにヒートマップコールバックを設定"""
        if hasattr(self.snn_network, 'set_heatmap_callback'):
            self.snn_network.set_heatmap_callback(self._snn_heatmap_callback)
            print("✅ SNNヒートマップコールバック設定完了")
        else:
            print("⚠️ SNNネットワークはヒートマップコールバックをサポートしていません")
    
    def _snn_heatmap_callback(self, spike_data: Dict[str, Any]):
        """SNNネットワークからのヒートマップコールバック"""
        if self.visualizer and self._heatmap_ready:
            try:
                # スパイクデータを変換
                spike_activities = self._convert_snn_spike_data(spike_data)
                
                # ヒートマップ更新
                self.update_snn_heatmap(spike_activities, 
                                        spike_data.get('epoch', 0),
                                        spike_data.get('sample_idx', 0),
                                        spike_data.get('true_label', -1),
                                        spike_data.get('predicted_label', -1),
                                        spike_data.get('spike_stats', {}))
            except Exception as e:
                print(f"⚠️ SNNヒートマップコールバックエラー: {e}")
    
    def _convert_snn_spike_data(self, spike_data: Dict[str, Any]) -> List[np.ndarray]:
        """SNNスパイクデータをヒートマップ用に変換"""
        spike_activities = []
        
        # 層別スパイク活動を取得
        layer_activities = spike_data.get('layer_activities', [])
        
        for layer_activity in layer_activities:
            if isinstance(layer_activity, (list, tuple)):
                # リスト/タプルの場合はnumpy配列に変換
                activity_array = np.array(layer_activity)
            elif isinstance(layer_activity, np.ndarray):
                # 既にnumpy配列の場合はそのまま
                activity_array = layer_activity
            else:
                # その他の場合は0配列で初期化
                activity_array = np.zeros(100)  # デフォルトサイズ
            
            # 正規化（0-1範囲に調整）
            if activity_array.max() > 0:
                activity_array = activity_array / activity_array.max()
            
            spike_activities.append(activity_array)
        
        return spike_activities
    
    def update_snn_heatmap_if_enabled(self):
        """ヒートマップが有効な場合のみ更新"""
        if self.visualizer and self._heatmap_ready:
            self.update_counter += 1
            
            # 更新間隔チェック
            if self.update_counter % self.update_interval == 0:
                try:
                    # 現在のSNN状態を取得してヒートマップ更新
                    current_state = self._get_current_snn_state()
                    if current_state:
                        spike_activities = current_state['spike_activities']
                        self.update_snn_heatmap(spike_activities,
                                                current_state.get('epoch', 0),
                                                current_state.get('sample_idx', 0),
                                                current_state.get('true_label', -1),
                                                current_state.get('predicted_label', -1))
                except Exception as e:
                    print(f"⚠️ SNNヒートマップ更新エラー: {e}")
    
    def _get_current_snn_state(self) -> Optional[Dict[str, Any]]:
        """現在のSNN状態を取得"""
        if not hasattr(self.snn_network, 'get_current_state'):
            return None
        
        try:
            return self.snn_network.get_current_state()
        except Exception as e:
            print(f"⚠️ SNN状態取得エラー: {e}")
            return None
    
    def update_snn_heatmap(self, spike_activities: List[np.ndarray], 
                           epoch: int, sample_idx: int,
                           true_label: int = -1, predicted_label: int = -1,
                           spike_stats: Optional[Dict] = None):
        """SNNヒートマップを更新"""
        if not self.visualizer:
            return
        
        try:
            # 初回表示の場合は準備フラグを設定
            if not self._heatmap_ready:
                self._heatmap_ready = True
                print("🎯 SNNヒートマップ表示開始")
            
            # 現在のエポックを更新
            self.current_epoch = epoch
            
            # ヒートマップ表示更新
            self.visualizer.update_snn_display(
                spike_activities=spike_activities,
                epoch=epoch,
                sample_idx=sample_idx,
                true_label=true_label,
                predicted_label=predicted_label,
                spike_stats=spike_stats
            )
            
        except Exception as e:
            print(f"⚠️ SNNヒートマップ更新で例外が発生しました: {e}")
    
    def start_snn_heatmap_display(self):
        """SNNヒートマップ表示を開始"""
        if self.visualizer:
            self._heatmap_ready = True
            print("🎯 SNNヒートマップ表示開始")
    
    def stop_snn_heatmap_display(self):
        """SNNヒートマップ表示を停止"""
        if self.visualizer:
            self.visualizer.close_snn_visualization()
            self._heatmap_ready = False
            print("🎯 SNNヒートマップ表示停止")
    
    def update_snn_learning_progress(self, epoch: int, train_accuracy: float, 
                                     test_accuracy: float, spike_stats: Dict):
        """SNN学習進捗を更新"""
        if self.visualizer:
            # 学習結果データを準備
            results_data = {
                'epoch': epoch,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'spike_propagation_rate': spike_stats.get('propagation_rate', 0.0),
                'total_spikes': spike_stats.get('total_spikes', 0),
                'layer_spike_counts': spike_stats.get('layer_spike_counts', [])
            }
            
            # 可視化システムに更新を通知
            self.visualizer.update_snn_learning_results(results_data)
    
    def is_heatmap_enabled(self) -> bool:
        """ヒートマップが有効かどうかを確認"""
        return self.visualizer is not None and self._heatmap_ready
    
    def get_heatmap_status(self) -> Dict[str, Any]:
        """ヒートマップシステムの状態を取得"""
        return {
            'enabled': self.visualizer is not None,
            'ready': self._heatmap_ready,
            'current_epoch': self.current_epoch,
            'update_counter': self.update_counter,
            'update_interval': self.update_interval
        }