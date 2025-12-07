#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ED-SNN ヒートマップリアルタイム表示クラス v1.0 (SNN対応版)
ed_multi_snn.prompt.md準拠・スパイキングニューラルネットワーク特化版

移植元: ed_v032_simple/modules/heatmap_realtime_visualizer_v4.py
対応: Milestone 3最適化済みED法準拠SNNアーキテクチャ

新仕様:
1. スパイク活動データ対応: 発火率・スパイク数・時間統計
2. SNN層構造対応: 入力ペア・交互隠れ層・興奮性出力
3. ED法準拠表示: アミン濃度・重み変化・学習進捗
4. リアルタイム更新: スパイク                     else:
                # 2段表示: 1段目は必ず4分割（パラメータボックス重複解消版）
                if i < 4:
                    # 1段目（4分割）- パラメータボックス重複解消のため大幅下方移動
                    ax = self.fig.add_subplot(gs[2:3, i])  # gs[2:3]でy≈0.350に配置
                    self.axes[layer_idx] = ax
                elif i < 8:
                    # 2段目 - 位置維持
                    col = i - 4
                    ax = self.fig.add_subplot(gs[3:4, col])  # 下段は維持
                    self.axes[layer_idx] = ax              # 2段表示: 1段目は必ず4分割（上段のみ15%追加下方移動）
                if i < 4:
                    # 1段目（4分割）- パラメータボックス重複解消版
                    ax = self.fig.add_subplot(gs[2:3, i])  # 15%追加下方移動
                    self.axes[layer_idx] = ax
                elif i < 8:
                    # 2段目 - 位置は変更せず維持
                    col = i - 4
                    ax = self.fig.add_subplot(gs[3:4, col])  # 下段は維持
                    self.axes[layer_idx] = ax"
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import warnings
import time
import threading

# 日本語フォント設定
def setup_japanese_font():
    """標準でインストールされている日本語フォントを設定"""
    try:
        japanese_fonts = [
            'Noto Sans CJK JP',
            'DejaVu Sans', 
            'Liberation Sans',
            'TakaoPGothic',
            'IPAexGothic',
            'sans-serif'
        ]
        
        for font_name in japanese_fonts:
            try:
                plt.rcParams['font.family'] = font_name
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, 'テスト', fontsize=10)
                plt.close(fig)
                print(f"✅ 日本語フォント設定: {font_name}")
                return True
            except:
                continue
        
        plt.rcParams['font.family'] = 'sans-serif'
        print("⚠️ 日本語フォント設定: デフォルト使用")
        return False
        
    except Exception as e:
        print(f"❌ 日本語フォント設定エラー: {e}")
        return False

# 初期化時にフォント設定を実行
setup_japanese_font()


class SNNLearningResultsTracker:
    """SNN学習結果追跡システム - ed_multi_snn.prompt.md準拠"""
    
    def __init__(self):
        """SNN学習結果データの初期化"""
        self.latest_results = {
            'epoch': 0,
            'sample_idx': 0,
            'true_label': -1,
            'predicted_label': -1,
            'train_accuracy': 0.0,
            'test_accuracy': 0.0,
            'spike_propagation_rate': 0.0,
            'total_spikes': 0,
            'layer_spike_counts': [],
            'learning_time': 0.0,
            'timestamp': time.time()
        }
    
    def update_snn_learning_results(self, results_data):
        """SNNネットワークから学習結果を更新"""
        if results_data:
            self.latest_results.update(results_data)
            self.latest_results['timestamp'] = time.time()
    
    def get_snn_subtitle_text(self):
        """SNN用サブタイトルテキストを生成"""
        epoch = self.latest_results['epoch']
        true_label = self.latest_results['true_label']
        predicted_label = self.latest_results['predicted_label']
        spike_rate = self.latest_results['spike_propagation_rate']
        total_spikes = self.latest_results['total_spikes']
        
        true_text = str(true_label) if true_label >= 0 else '-'
        pred_text = str(predicted_label) if predicted_label >= 0 else '-'
        
        return f"エポック: {epoch} | 正解: {true_text} | 予測: {pred_text} | スパイク伝播率: {spike_rate:.1f}% | 総スパイク: {total_spikes}"


class SNNDisplayTimingController:
    """SNN用表示タイミング制御システム"""
    
    def __init__(self, interval: float = 0.8):
        """
        初期化
        
        Args:
            interval: 更新間隔(秒)
        """
        self.interval = interval
        self.last_update = 0.0
        self.update_count = 0
    
    def should_update(self) -> bool:
        """更新すべきタイミングかどうかを判定"""
        current_time = time.time()
        
        if current_time - self.last_update >= self.interval:
            self.last_update = current_time
            self.update_count += 1
            return True
        
        return False
    
    def get_update_info(self) -> Dict[str, Any]:
        """更新情報を取得"""
        return {
            'update_count': self.update_count,
            'last_update': self.last_update,
            'interval': self.interval
        }


class SNNIntervalDisplaySystem:
    """SNN用インターバル表示システム"""
    
    def __init__(self, visualizer, interval: float = 0.3):
        """
        初期化
        
        Args:
            visualizer: ヒートマップ可視化クラス
            interval: インターバル更新間隔(秒)
        """
        self.visualizer = visualizer
        self.interval = interval
        self.last_activity_data = None
        self.background_thread = None
        self.running = False
    
    def set_spike_activity_data(self, spike_activities: List[np.ndarray]):
        """スパイク活動データを設定"""
        self.last_activity_data = spike_activities
    
    def start_interval_updates(self):
        """インターバル更新を開始"""
        if not self.running:
            self.running = True
            self.background_thread = threading.Thread(target=self._interval_update_loop, daemon=True)
            self.background_thread.start()
            print("🎯 SNNインターバル表示システム開始")
    
    def stop_interval_updates(self):
        """インターバル更新を停止"""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=1.0)
        print("🎯 SNNインターバル表示システム停止")
    
    def _interval_update_loop(self):
        """インターバル更新ループ - matplotlibスレッド問題回避のため無効化"""
        # matplotlibの「main thread is not in main loop」エラー回避のため
        # バックグラウンド更新を無効化し、同期更新のみ使用
        print("🎯 SNNインターバル表示システム開始（同期モード）")
        
        # 無効化された状態でループを維持（スレッド終了まで待機）
        while self.running:
            time.sleep(0.1)  # 軽量な待機
        
        print("🎯 SNNインターバル更新ループ終了")


class SNNHeatmapRealtimeVisualizer:
    """ED-SNN ヒートマップリアルタイム表示クラス v1.0 (SNN対応版)"""
    
    def __init__(self, 
                 layer_shapes: List[Tuple[int, int]], 
                 show_parameters: bool = True,
                 update_interval: float = 0.8,
                 colormap: str = 'viridis',
                 snn_params: Optional[Dict] = None,
                 exec_params: Optional[Dict] = None,
                 class_names: Optional[Dict[int, str]] = None):
        """
        初期化
        
        Args:
            layer_shapes: 各層の形状 [(height, width), ...]
            show_parameters: パラメータ表示するかどうか
            update_interval: 更新間隔(秒)
            colormap: カラーマップ
            snn_params: SNN特有パラメータ
            exec_params: 実行時設定パラメータ
            class_names: クラス名マッピング {0: "T-shirt/top", 1: "Trouser", ...}
        """
        self.layer_shapes = layer_shapes
        self.show_parameters = show_parameters
        self.update_interval = update_interval
        self.colormap = colormap
        
        # 状態管理
        self.fig = None
        self.axes = {}  # {layer_index: ax}
        self.title_ax = None
        self.param_ax_snn = None
        self.param_ax_lif = None  # LIFニューロンパラメータ用（新規追加）
        self.param_ax_exec = None
        self.heatmap_objects = {}
        self.colorbar_objects = {}
        self.is_initialized = False
        
        # パラメータ保存
        self.snn_params = snn_params or {}
        self.exec_params = exec_params or {}
        
        # クラス名マッピング（MNIST以外でクラス名情報がある場合に使用）
        self.class_names = class_names or {}
        
        # matplotlib インタラクティブモード設定
        plt.ion()  # インタラクティブモードを有効化
        
        # SNN学習結果追跡システム
        self.learning_results_tracker = SNNLearningResultsTracker()
        
        # 表示タイミング制御システム
        self.timing_controller = SNNDisplayTimingController(interval=update_interval)
        
        # インターバル表示システム
        self.interval_system = SNNIntervalDisplaySystem(self, interval=0.3)
        self.training_info = {}
    
    def _calculate_snn_layout(self, num_layers: int) -> Dict[str, Any]:
        """
        SNN用レイアウト計算: 最大2行4列, 8層超過時は省略アルゴリズム適用
        
        Args:
            num_layers: 総層数
            
        Returns:
            レイアウト情報の辞書
        """
        max_heatmaps = 8
        
        if num_layers <= max_heatmaps:
            selected_layers = list(range(num_layers))
            layout_type = f"snn_full_{num_layers}_layers"
            
            if num_layers <= 4:
                # 1段表示: 層数に応じた動的分割
                actual_rows = 1
                if num_layers == 3:
                    actual_cols = 3  # 3分割
                else:
                    actual_cols = 4   # 4分割
            else:
                # 2段表示: 1段目は必ず4分割
                actual_rows = 2
                actual_cols = 4
        else:
            # 8層超過時の省略アルゴリズム（SNN特化）
            selected_layers = []
            # 上段: 入力層 + 隠れ層1-3
            selected_layers.extend([0, 1, 2, 3])
            # 下段: 出力層の3つ前の隠れ層 + 出力層
            output_idx = num_layers - 1
            selected_layers.extend([output_idx-3, output_idx-2, output_idx-1, output_idx])
            
            actual_rows = 2
            actual_cols = 4
            layout_type = f"snn_abbreviated_{num_layers}_layers"
        
        return {
            'selected_layers': selected_layers,
            'actual_rows': actual_rows,
            'actual_cols': actual_cols,
            'layout_type': layout_type,
            'total_heatmaps': len(selected_layers)
        }
    
    def setup_snn_visualization(self, initial_spike_data: List[np.ndarray]):
        """
        SNN可視化セットアップ（ed_v032_simple.py準拠レイアウト）
        
        Args:
            initial_spike_data: 初期スパイクデータ
        """
        if self.is_initialized:
            return
        
        print("🎯 SNNヒートマップ可視化システムセットアップ開始...")
        
        # レイアウト計算
        num_layers = len(initial_spike_data)
        layout = self._calculate_snn_layout(num_layers)
        
        # フィギュア作成（ウィンドウサイズ80%縮小：ed_v032_simple.py準拠 - 初期化時のみ実行）
        self.fig = plt.figure(figsize=(11.2, 6.4), facecolor='white')
        
        # メインタイトル設定（ed_v032_simple.py準拠）
        self.fig.suptitle('ED-SNN ヒートマップリアルタイム表示', fontsize=16, fontweight='bold', y=0.95)
        
        # GridSpec設定（横配置レイアウト：全幅使用・サイズ2倍拡大 - 初期化時固定）
        # ヒートマップ領域を大きく, パラメータボックスを上部に集約
        gs = gridspec.GridSpec(4, 8, figure=self.fig, hspace=0.4, wspace=0.3)
        
        # サブタイトル領域（エポック・クラス情報 - 重複完全解消版）
        self.title_ax = self.fig.add_subplot(gs[0, :2])
        self.title_ax.axis('off')
        
        # ヒートマップ領域（層数に応じた動的分割）
        selected_layers = layout['selected_layers']
        actual_rows = layout['actual_rows']
        actual_cols = layout['actual_cols']
        
        # ヒートマップ軸作成（動的分割対応）
        self.axes = {}
        
        for i, layer_idx in enumerate(selected_layers):
            if actual_rows == 1:
                # 1段表示: 3分割または4分割
                if actual_cols == 3:
                    # 3分割の場合: 中央寄せ配置
                    col_offset = 1  # 左端を1列空ける
                    if i < 3:
                        ax = self.fig.add_subplot(gs[1:3, col_offset + i])
                        self.axes[layer_idx] = ax
                else:
                    # 4分割の場合
                    if i < 4:
                        ax = self.fig.add_subplot(gs[1:3, i])
                        self.axes[layer_idx] = ax
            else:
                # 2段表示: カスタム座標で個別位置指定（1段目y=0.370, 2段目y=0.110）
                if i < 4:
                    # 1段目（4分割）- カスタム座標でy=0.370に配置
                    x_start = i / 8.0  # 8列中のi列目開始位置
                    x_width = 1.0 / 8.0  # 1列分の幅
                    y_bottom = 0.370  # 下側y座標
                    y_height = 0.148  # GridSpecベース高さ
                    
                    ax = self.fig.add_axes((x_start, y_bottom, x_width, y_height))
                    self.axes[layer_idx] = ax
                elif i < 8:
                    # 2段目（4分割）- カスタム座標でy=0.110に配置
                    col = i - 4
                    x_start = col / 8.0  # 8列中のcol列目開始位置
                    x_width = 1.0 / 8.0  # 1列分の幅
                    y_bottom = 0.110  # 下側y座標
                    y_height = 0.148  # GridSpecベース高さ
                    
                    ax = self.fig.add_axes((x_start, y_bottom, x_width, y_height))
                    self.axes[layer_idx] = ax
            
            # 層タイトル設定
            if layer_idx in self.axes:
                layer_name = self._get_snn_layer_name(layer_idx, num_layers)
                self.axes[layer_idx].set_title(layer_name, fontsize=10, fontweight='bold', pad=10)
        
        # パラメータ表示領域（3つのボックスを横並び配置）
        if self.show_parameters:
            # ED法パラメータボックス（緑色 - 左側）
            self.param_ax_snn = self.fig.add_subplot(gs[0, 2:4])
            self.param_ax_snn.axis('off')
            
            # LIFニューロンパラメータボックス（緑色 - 中央）
            self.param_ax_lif = self.fig.add_subplot(gs[0, 4:6])
            self.param_ax_lif.axis('off')
            
            # 実行パラメータボックス（薄緑色 - 右側）
            self.param_ax_exec = self.fig.add_subplot(gs[0, 6:])
            self.param_ax_exec.axis('off')
        
        self.is_initialized = True
        print(f"✅ SNNヒートマップ可視化セットアップ完了 ({layout['layout_type']})")
        
        # インターバル表示システム開始
        self.interval_system.start_interval_updates()
        
        plt.show(block=False)
        plt.pause(0.1)
    
    def _get_snn_layer_name(self, layer_idx: int, total_layers: int) -> str:
        """SNN層名を取得"""
        if layer_idx == 0:
            return "入力層 (E/Iペア)"
        elif layer_idx == total_layers - 1:
            return "出力層 (興奮性)"
        else:
            return f"隠れ層{layer_idx} (交互)"
    
    def update_snn_parameters(self, snn_params: Optional[Dict] = None, 
                              exec_params: Optional[Dict] = None):
        """SNNパラメータ表示を更新"""
        if snn_params:
            self.snn_params.update(snn_params)
        if exec_params:
            self.exec_params.update(exec_params)
    
    def update_snn_learning_results(self, results_data: Dict):
        """SNN学習結果を更新"""
        self.learning_results_tracker.update_snn_learning_results(results_data)
    
    def _update_snn_subtitle(self):
        """SNNサブタイトルを更新（ed_v032_simple.py準拠形式、3行分割表示）"""
        if self.title_ax:
            self.title_ax.clear()
            self.title_ax.axis('off')
            
            # ed_v032_simple.py準拠のサブタイトル形式
            epoch = self.learning_results_tracker.latest_results['epoch']
            true_label = self.learning_results_tracker.latest_results['true_label']
            predicted_label = self.learning_results_tracker.latest_results['predicted_label']
            
            # クラス名情報の取得（指定されている場合）
            true_text = str(true_label) if true_label >= 0 else '-'
            pred_text = str(predicted_label) if predicted_label >= 0 else '-'
            
            # クラス名が指定されている場合は追記
            if true_label >= 0 and true_label in self.class_names:
                true_text = f"{true_label} ({self.class_names[true_label]})"
            if predicted_label >= 0 and predicted_label in self.class_names:
                pred_text = f"{predicted_label} ({self.class_names[predicted_label]})"
            
            # 正解/不正解に応じた色分け表示（視認性向上）
            # 正解: 青色（blue）、不正解: 赤色（red）
            is_correct = (true_label == predicted_label) and (true_label >= 0)
            text_color = 'blue' if is_correct else 'red'
            
            # 3行に分割表示（上から: エポック、正解クラス、予測クラス）
            # 負のX座標でボックスの左端より外側に配置（画面の真の左端に近づける）
            self.title_ax.text(-0.3, 0.85, f"エポック#: {epoch}", 
                              ha='left', va='center', fontsize=14, fontweight='bold',
                              color='black')
            self.title_ax.text(-0.3, 0.50, f"正解クラス: {true_text}", 
                              ha='left', va='center', fontsize=14, fontweight='bold',
                              color='black')
            self.title_ax.text(-0.3, 0.15, f"予測クラス: {pred_text}", 
                              ha='left', va='center', fontsize=14, fontweight='bold',
                              color=text_color)
    
    def _draw_snn_parameter_boxes(self):
        """SNNパラメータボックスを描画（3つのボックスに分割表示）"""
        if not self.show_parameters:
            return
        
        # ED法パラメータボックス（左側・緑色）
        if self.param_ax_snn:
            self.param_ax_snn.clear()
            self.param_ax_snn.axis('off')
            
            # ED法パラメータのみ
            ed_text = "ED法パラメータ設定\n"
            ed_text += f"学習率(alpha): {self.snn_params.get('学習率', '0.1')}\n"
            ed_text += f"初期アミン濃度(beta): {self.snn_params.get('初期アミン濃度', '0.25')}\n"
            ed_text += f"アミン拡散係数(u1): {self.snn_params.get('アミン拡散係数', '0.5')}\n"
            ed_text += f"シグモイド閾値(u0): {self.snn_params.get('シグモイド閾値', '1.2')}\n"
            ed_text += f"重み初期値1: {self.snn_params.get('重み初期値1', '0.3')}\n"
            ed_text += f"重み初期値2: {self.snn_params.get('重み初期値2', '0.5')}"
            
            self.param_ax_snn.text(0.05, 0.95, ed_text, 
                                   ha='left', va='top', fontsize=9,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        
        # LIFニューロンパラメータボックス（中央・緑色）
        if self.param_ax_lif:
            self.param_ax_lif.clear()
            self.param_ax_lif.axis('off')
            
            # LIFニューロンパラメータ
            lif_text = "LIFニューロンパラメータ設定\n"
            lif_text += f"静止膜電位: {self.snn_params.get('静止膜電位', '-65.0')} mV\n"
            lif_text += f"発火閾値: {self.snn_params.get('発火閾値', '-60.0')} mV\n"
            lif_text += f"リセット電位: {self.snn_params.get('リセット電位', '-70.0')} mV\n"
            lif_text += f"膜時定数: {self.snn_params.get('膜時定数', '20.0')} ms\n"
            lif_text += f"不応期: {self.snn_params.get('不応期', '2.0')} ms\n"
            lif_text += f"時間ステップ: {self.snn_params.get('時間ステップ', '1.0')} ms\n"
            lif_text += f"閾値係数: {self.snn_params.get('閾値係数', '2.0')}\n"
            lif_text += f"シミュレーション時間: {self.snn_params.get('シミュレーション時間', '50.0')} ms"
            
            self.param_ax_lif.text(0.05, 0.95, lif_text, 
                                   ha='left', va='top', fontsize=9,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        
        # 実行パラメータボックス（右上下・薄緑色：ed_v032_simple.py準拠）
        if self.param_ax_exec:
            self.param_ax_exec.clear()
            self.param_ax_exec.axis('off')
            
            # ed_v032_simple.py準拠の実行パラメータ形式
            exec_text = "実行パラメータ設定\n"
            exec_text += f"データセット: {self.exec_params.get('データセット', 'FASHION_MNIST')}\n"
            exec_text += f"エポック数: {self.exec_params.get('エポック数', '3')}\n"
            exec_text += f"学習サンプル: {self.exec_params.get('学習サンプル', '50')}\n"
            exec_text += f"テストサンプル: {self.exec_params.get('テストサンプル', '50')}\n"
            exec_text += f"隠れ層構造: {self.exec_params.get('隠れ層構造', '[64]')}\n"
            exec_text += f"ミニバッチサイズ: {self.exec_params.get('batch_size', '128')}\n"
            exec_text += f"ランダムシード: {self.exec_params.get('seed', '42')}\n"
            exec_text += f"詳細表示: {self.exec_params.get('詳細表示', 'ON')}"
            
            self.param_ax_exec.text(0.05, 0.95, exec_text, 
                                    ha='left', va='top', fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.6))
    
    def _safe_clear_heatmaps_with_pause(self):
        """軸の完全再作成でサイズ縮小を根本解決（ed_multi_snn.md準拠）"""
        try:
            # 描画を一時停止
            if self.fig and hasattr(self.fig, 'canvas'):
                plt.ioff()  # インタラクティブモードを無効化
            
            # 既存のヒートマップオブジェクトを安全に削除
            for layer_idx in list(self.heatmap_objects.keys()):
                try:
                    heatmap_obj = self.heatmap_objects[layer_idx]
                    if heatmap_obj and hasattr(heatmap_obj, 'remove'):
                        heatmap_obj.remove()
                except Exception as e:
                    pass  # エラーは無視して続行
            
            # 既存のカラーバーオブジェクトを安全に削除
            for layer_idx in list(self.colorbar_objects.keys()):
                try:
                    colorbar_obj = self.colorbar_objects[layer_idx]
                    if colorbar_obj:
                        if hasattr(colorbar_obj, 'ax') and colorbar_obj.ax:
                            colorbar_obj.ax.remove()
                        elif hasattr(colorbar_obj, 'remove'):
                            colorbar_obj.remove()
                except Exception as e:
                    pass  # エラーは無視して続行
            
            # オブジェクト辞書をクリア
            self.heatmap_objects.clear()
            self.colorbar_objects.clear()
            
            # ★根本修正: 軸も完全に削除して再作成
            for layer_idx in list(self.axes.keys()):
                try:
                    ax = self.axes[layer_idx]
                    if ax and hasattr(ax, 'remove'):
                        ax.remove()
                except Exception as e:
                    pass  # エラーは無視して続行
            
            # 軸辞書をクリア
            self.axes.clear()
            
        except Exception as e:
            print(f"⚠️ 安全クリア処理エラー: {e}")
    
    def _recreate_heatmap_axes(self, num_layers: int):
        """ヒートマップ軸を全幅使用・サイズ2倍で再作成（ed_multi_snn.md準拠）"""
        try:
            # レイアウト計算（既存のロジック使用）
            layout = self._calculate_snn_layout(num_layers)
            selected_layers = layout['selected_layers']
            actual_rows = layout['actual_rows']
            actual_cols = layout['actual_cols']
            
            # GridSpec再取得（サイズ2倍拡大設定）
            gs = gridspec.GridSpec(4, 8, figure=self.fig, hspace=0.4, wspace=0.3)
            
            # 軸辞書を新規作成
            self.axes = {}
            
            for i, layer_idx in enumerate(selected_layers):
                if actual_rows == 1:
                    # 1段表示: 全幅使用で分割
                    if actual_cols == 3:
                        # 3分割: 全幅使用（左寄せ無し）
                        col_positions = [1, 3, 5]  # 8列中で1,3,5列目を使用
                        if i < 3:
                            ax = self.fig.add_subplot(gs[1:4, col_positions[i]:col_positions[i]+2])  # 2倍拡大
                            self.axes[layer_idx] = ax
                    else:
                        # 4分割: 全庅使用
                        if i < 4:
                            ax = self.fig.add_subplot(gs[1:4, i*2:(i+1)*2])  # 2倍拡大
                            self.axes[layer_idx] = ax
                else:
                    # 2段表示: カスタム座標で個別位置指定（1段目y=0.370, 2段目y=0.110, サイズ2倍拡大）
                    if i < 4:
                        # 1段目（4分割）- カスタム座標でy=0.370に配置（サイズ2倍拡大）
                        x_start = (i * 2) / 8.0  # 8列中のi*2列目開始位置
                        x_width = 2.0 / 8.0  # 2列分の幅（2倍拡大）
                        y_bottom = 0.370  # 下側y座標
                        y_height = 0.148  # GridSpecベース高さ
                        
                        ax = self.fig.add_axes((x_start, y_bottom, x_width, y_height))
                        self.axes[layer_idx] = ax
                    elif i < 8:
                        # 2段目（4分割）- カスタム座標でy=0.110に配置（サイズ2倍拡大）
                        col = (i - 4) * 2
                        x_start = col / 8.0  # 8列中のcol列目開始位置
                        x_width = 2.0 / 8.0  # 2列分の幅（2倍拡大）
                        y_bottom = 0.110  # 下側y座標
                        y_height = 0.148  # GridSpecベース高さ
                        
                        ax = self.fig.add_axes((x_start, y_bottom, x_width, y_height))
                        self.axes[layer_idx] = ax
                
                # 層タイトル設定
                if layer_idx in self.axes:
                    layer_name = self._get_snn_layer_name(layer_idx, num_layers)
                    self.axes[layer_idx].set_title(layer_name, fontsize=12, fontweight='bold', pad=15)
        
        except Exception as e:
            print(f"⚠️ 軸再作成エラー: {e}")
    
    def _convert_spike_to_heatmap_data(self, spike_data: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """スパイクデータをヒートマップ用データに変換（カラー画像対応）"""
        try:
            if spike_data is None or spike_data.size == 0:
                # target_shapeに基づいてゼロ配列を作成
                return np.zeros(target_shape)
            
            # カラー画像の場合（3次元）
            if spike_data.ndim == 3:
                # カラー画像はそのまま返す（正規化のみ）
                if spike_data.max() > spike_data.min():
                    normalized = (spike_data - spike_data.min()) / (spike_data.max() - spike_data.min())
                    return normalized
                return spike_data
            
            # グレースケールの場合（1次元または2次元）
            if spike_data.ndim == 1:
                height, width = target_shape[:2]  # 最初の2次元を使用
                target_size = height * width
                
                if len(spike_data) == target_size:
                    # サイズが一致する場合はそのまま変形
                    reshaped = spike_data.reshape(height, width)
                elif len(spike_data) > target_size:
                    # データが大きい場合は切り詰め
                    resized_data = spike_data[:target_size]
                    reshaped = resized_data.reshape(height, width)
                else:
                    # データが小さい場合はパディング
                    padded_data = np.zeros(target_size)
                    padded_data[:len(spike_data)] = spike_data
                    reshaped = padded_data.reshape(height, width)
                
                # ed_v032_simple.py準拠の正規化 (0-1範囲)
                if reshaped.max() > reshaped.min():
                    reshaped = (reshaped - reshaped.min()) / (reshaped.max() - reshaped.min())
                
                return reshaped
                
            elif spike_data.ndim == 2:
                # 既に2次元の場合はリサイズ
                current_h, current_w = spike_data.shape
                target_h, target_w = target_shape
                
                if current_h == target_h and current_w == target_w:
                    # ed_v032_simple.py準拠の正規化
                    if spike_data.max() > spike_data.min():
                        normalized = (spike_data - spike_data.min()) / (spike_data.max() - spike_data.min())
                        return normalized
                    return spike_data
                else:
                    # サイズ調整（最近傍補間）
                    try:
                        from scipy.ndimage import zoom
                        zoom_h = target_h / current_h
                        zoom_w = target_w / current_w
                        resized = zoom(spike_data, (zoom_h, zoom_w), order=0)
                        
                        # ed_v032_simple.py準拠の正規化
                        if resized.max() > resized.min():
                            resized = (resized - resized.min()) / (resized.max() - resized.min())
                        
                        return resized
                    except ImportError:
                        # scipyが利用できない場合は平坦化して再変形
                        flat_data = spike_data.flatten()
                        return self._convert_spike_to_heatmap_data(flat_data, target_shape)
            else:
                # 多次元データの場合は平坦化してから処理
                return self._convert_spike_to_heatmap_data(spike_data.flatten(), target_shape)
        
        except Exception as e:
            print(f"[エラー] スパイクデータ変換エラー: {e}")
            return np.zeros(target_shape)
    
    def update_snn_display(self, spike_activities: List[np.ndarray], 
                           epoch: int, sample_idx: int, 
                           true_label: int = -1, predicted_label: int = -1,
                           spike_stats: Optional[Dict] = None):
        """
        SNNヒートマップ表示を更新
        
        Args:
            spike_activities: 各層のスパイク活動データ
            epoch: 現在のエポック
            sample_idx: 現在のサンプルインデックス
            true_label: 正解ラベル
            predicted_label: 予測ラベル
            spike_stats: スパイク統計情報
        """
        if not self.is_initialized:
            self.setup_snn_visualization(spike_activities)
        
        # SNN学習結果データを更新
        results_data = {
            'epoch': epoch,
            'sample_idx': sample_idx,
            'true_label': true_label,
            'predicted_label': predicted_label
        }
        
        if spike_stats:
            results_data.update(spike_stats)
        
        self.update_snn_learning_results(results_data)
        
        # インターバル表示システムにスパイク活動データを設定
        if self.interval_system:
            self.interval_system.set_spike_activity_data(spike_activities)
        
        # 表示タイミング制御
        should_update = self.timing_controller.should_update()
        if not should_update:
            return
        
        # サブタイトルとヒートマップを同期更新
        self._update_snn_subtitle()
        
        try:
            # pause()とremove()で完全クリア（軸も再作成）
            self._safe_clear_heatmaps_with_pause()
            
            # パラメータボックス描画
            self._draw_snn_parameter_boxes()
            
            # ★根本修正: 軸を完全再作成（サイズ縮小防止）
            self._recreate_heatmap_axes(len(spike_activities))
            
            # 各層のヒートマップ更新（新しい軸に描画）
            for layer_idx in self.axes.keys():
                if layer_idx < len(spike_activities):
                    ax = self.axes[layer_idx]
                    spike_data = spike_activities[layer_idx]
                    
                    # スパイクデータをヒートマップ用に変換
                    heatmap_data = self._convert_spike_to_heatmap_data(
                        spike_data, self.layer_shapes[layer_idx])
                    
                    # カラー画像かグレースケールかを判定
                    is_color_image = (heatmap_data.ndim == 3 and heatmap_data.shape[2] == 3)
                    
                    if is_color_image:
                        # カラー画像の場合：cmapなしで直接表示
                        im = ax.imshow(heatmap_data, aspect='equal', interpolation='nearest')
                        
                        # カラー画像にはカラーバー不要
                        self.colorbar_objects[layer_idx] = None
                    else:
                        # グレースケールの場合：既存のヒートマップ表示
                        # 動的範囲計算（ed_v032_simple.py準拠）
                        vmin = heatmap_data.min() if heatmap_data.size > 0 else 0
                        vmax = heatmap_data.max() if heatmap_data.size > 0 else 1
                        if vmin == vmax:
                            vmax = vmin + 1e-6  # ゼロ除算回避
                        
                        # ヒートマップ描画（正方形アスペクト比固定）
                        im = ax.imshow(heatmap_data, cmap=self.colormap, 
                                       aspect='equal', interpolation='nearest',
                                       vmin=vmin, vmax=vmax)
                        
                        # カラーバー追加（Figure明示指定でWarning解消）
                        cbar = self.fig.colorbar(im, ax=ax, shrink=0.6)
                        cbar.set_label('スパイク活動', fontsize=8)
                        self.colorbar_objects[layer_idx] = cbar
                    
                    # オブジェクト保存（新規作成されたオブジェクト）
                    self.heatmap_objects[layer_idx] = im
                    
                    # 軸設定（ed_v032_simple.py準拠）
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # 層情報表示（カラー画像対応）
                    layer_name = self._get_snn_layer_name(layer_idx, len(spike_activities))
                    shape = self.layer_shapes[layer_idx]
                    if len(shape) == 3:
                        # カラー画像: (H, W, 3)
                        h, w, c = shape
                        ax.set_title(f"{layer_name}\n({h}×{w}×{c})", 
                                    fontsize=10, fontweight='bold')
                    else:
                        # グレースケール: (H, W)
                        height, width = shape[:2]
                        ax.set_title(f"{layer_name}\n({height}×{width})", 
                                    fontsize=10, fontweight='bold')
            
            # 描画更新（pause解除とキャンバス更新 - サイズ保持）
            try:
                plt.ion()  # インタラクティブモードを再有効化
                if self.fig and hasattr(self.fig, 'canvas') and self.fig.canvas:
                    # フィギュアサイズを明示的に保持（サイズ縮小防止 - 80%サイズ）
                    current_size = self.fig.get_size_inches()
                    if not np.allclose(current_size, [11.2, 6.4], atol=0.1):
                        self.fig.set_size_inches(11.2, 6.4, forward=True)
                    
                    # matplotlibスレッド問題回避: メインスレッドでのみ実行
                    try:
                        import threading
                        if threading.current_thread() == threading.main_thread():
                            self.fig.canvas.draw_idle()
                            self.fig.canvas.flush_events()
                        # 非メインスレッドからの呼び出しは無視（エラーメッセージなし）
                    except Exception:
                        # matplotlibの更新エラーは無視（静寂な失敗）
                        pass
            except Exception as e:
                # 重要でないエラーは抑制
                pass
            
        except Exception as e:
            print(f"⚠️ SNNヒートマップ更新エラー: {e}")
    
    def close_snn_visualization(self):
        """SNN可視化システムを終了"""
        if self.interval_system:
            self.interval_system.stop_interval_updates()
        
        if self.fig:
            plt.close(self.fig)
        
        self.is_initialized = False
        print("🎯 SNNヒートマップ可視化システム終了")