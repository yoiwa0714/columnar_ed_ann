#!/usr/bin/env python3
"""
可視化マネージャー - リアルタイム学習進捗・ヒートマップ表示

columnar_ed_ann.pyから可視化機能を抽出したモジュール
- 日本語フォント設定
- 保存パス決定
- リアルタイム学習曲線表示
- 混同行列表示
- 層別活性化ヒートマップ表示

実装日: 2025-12-06
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from pathlib import Path


def setup_japanese_font():
    """
    日本語フォントを設定する（Noto Sans CJK JP優先、fallback付き）
    
    優先順位:
    1. Noto Sans CJK JP
    2. Noto Sans JP
    3. IPAexGothic / IPAGothic
    4. TakaoPGothic / VL PGothic
    """
    import matplotlib.font_manager as fm
    
    # 優先フォントリスト
    preferred_fonts = [
        'Noto Sans CJK JP',
        'Noto Sans JP',
        'IPAexGothic',
        'IPAGothic',
        'TakaoPGothic',
        'VL PGothic',
    ]
    
    # システムにインストールされている日本語フォントを検索
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 優先順位に従ってフォントを選択
    selected_font = None
    for font in preferred_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # 見つからない場合はCJKを含む任意のフォントを検索
    if selected_font is None:
        for font_name in available_fonts:
            if 'CJK' in font_name or 'Japan' in font_name or 'IPA' in font_name:
                selected_font = font_name
                break
    
    # フォントを設定
    if selected_font:
        plt.rcParams['font.family'] = selected_font
        print(f"日本語フォント設定: {selected_font}")
    else:
        print("警告: 日本語フォントが見つかりませんでした。デフォルトフォントを使用します。")


def determine_save_path(save_fig_arg):
    """
    --save_figオプションから保存パスを決定
    
    Parameters:
    -----------
    save_fig_arg : str or None
        --save_figオプションの値
    
    Returns:
    --------
    str
        決定された保存パス
    
    Examples:
    ---------
    - None指定: viz_results/viz_results_YYYYMMDD_HHMMSS.png
    - ディレクトリ指定: dir/viz_results_YYYYMMDD_HHMMSS.png
    - ファイル名指定: path/to/file.png
    """
    if save_fig_arg is None:
        # 指定なし: viz_results/viz_results_YYYYMMDD_HHMMSS.png
        viz_dir = Path('viz_results')
        viz_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return str(viz_dir / f'viz_results_{timestamp}.png')
    else:
        save_fig_path = Path(save_fig_arg)
        
        # ディレクトリのみ指定の場合 (存在するディレクトリ、または拡張子なし)
        if save_fig_path.is_dir():
            # 既存ディレクトリ
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return str(save_fig_path / f'viz_results_{timestamp}.png')
        elif not save_fig_path.suffix:
            # 拡張子なし → ディレクトリとして扱う
            save_fig_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return str(save_fig_path / f'viz_results_{timestamp}.png')
        # パス付きファイル名の場合
        elif save_fig_path.parent != Path('.'):
            save_fig_path.parent.mkdir(parents=True, exist_ok=True)
            return str(save_fig_path)
        # パス無しファイル名の場合
        else:
            return str(save_fig_path)


class VisualizationManager:
    """
    リアルタイム可視化マネージャー
    
    機能:
    - 学習曲線表示（訓練/テスト精度）
    - 混同行列表示
    - 層別活性化ヒートマップ表示
    - 図の保存
    """
    
    def __init__(self, enable_viz=False, enable_heatmap=False, save_path=None, total_epochs=100):
        """
        Parameters:
        -----------
        enable_viz : bool
            学習曲線表示の有効化
        enable_heatmap : bool
            ヒートマップ表示の有効化
        save_path : str or None
            保存パス
        total_epochs : int
            総エポック数（学習曲線の横軸設定用）
        """
        self.enable_viz = enable_viz
        self.enable_heatmap = enable_heatmap
        self.save_path = save_path
        self.total_epochs = total_epochs
        
        self.fig_viz = None
        self.fig_heatmap = None
        
        if self.enable_viz:
            plt.ion()
            self.fig_viz = plt.figure(figsize=(15, 5))
            self.fig_viz.canvas.manager.set_window_title('学習曲線 + 混同行列')
        
        if self.enable_heatmap:
            plt.ion()
            self.fig_heatmap = plt.figure(figsize=(16, 8))
            self.fig_heatmap.canvas.manager.set_window_title('層別活性化ヒートマップ')
    
    def update_learning_curve(self, train_acc_history, test_acc_history, x_test, y_test, network):
        """
        学習曲線と混同行列を更新
        
        Parameters:
        -----------
        train_acc_history : list[float]
            訓練精度履歴
        test_acc_history : list[float]
            テスト精度履歴
        x_test : array
            テストデータ
        y_test : array
            テストラベル
        network : object
            ネットワークオブジェクト (compute_confusion_matrixメソッドを持つ)
        """
        if not self.enable_viz or self.fig_viz is None:
            return
        
        self.fig_viz.clear()
        ax1, ax2 = self.fig_viz.subplots(1, 2)
        
        # 学習曲線
        epochs_list = list(range(1, len(train_acc_history) + 1))
        ax1.plot(epochs_list, train_acc_history, label='Train', marker='o', markersize=3)
        ax1.plot(epochs_list, test_acc_history, label='Test', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Learning Progress')
        ax1.legend()
        
        # 縦軸設定: 0.0〜1.0
        ax1.set_ylim(0.0, 1.0)
        ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # 横軸設定: 1〜最大エポック数
        ax1.set_xlim(1, self.total_epochs)
        # 横軸の目盛り：10分割
        x_tick_interval = self.total_epochs / 10
        x_ticks = [1] + [int(1 + i * x_tick_interval) for i in range(1, 10)] + [self.total_epochs]
        ax1.set_xticks(x_ticks)
        
        # グリッド線の設定
        ax1.grid(True, alpha=0.3)
        # 縦軸のグリッド：0.1, 0.3, 0.5, 0.7, 0.9が点線、0.2, 0.4, 0.6, 0.8が実線
        for y in [0.1, 0.3, 0.5, 0.7, 0.9]:
            ax1.axhline(y=y, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        for y in [0.2, 0.4, 0.6, 0.8]:
            ax1.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        # 横軸のグリッド：10分割して点線と実線を交互に配置
        for i, x in enumerate([int(1 + j * x_tick_interval) for j in range(1, 10)]):
            if i % 2 == 0:  # 0, 2, 4, 6, 8 -> 点線
                ax1.axvline(x=x, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            else:  # 1, 3, 5, 7 -> 実線
                ax1.axvline(x=x, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # 混同行列を表示（自前で計算）
        # y_testの形状を確認: (n_samples, n_classes)の2次元配列を想定
        if y_test.ndim == 1:
            # (n_samples,)の場合 -> クラス数を推測
            n_classes = int(y_test.max()) + 1
            y_test_labels = y_test
        else:
            # (n_samples, n_classes)の場合 -> one-hot形式
            n_classes = y_test.shape[1]
            y_test_labels = np.argmax(y_test, axis=1)
        
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(len(x_test)):
            _, z_out, _ = network.forward(x_test[i])
            pred = np.argmax(z_out)
            true = y_test_labels[i] if y_test.ndim == 1 else np.argmax(y_test[i])
            conf_matrix[true, pred] += 1
        
        sns.heatmap(conf_matrix, annot=True, fmt='d',
                   cmap='Blues', ax=ax2, cbar_kws={'label': 'Count'})
        ax2.set_xlabel('Predicted Class')
        ax2.set_ylabel('True Class')
        ax2.set_title('Confusion Matrix (Test Data)')
        
        plt.figure(self.fig_viz.number)
        plt.pause(0.1)
        plt.draw()
    
    def update_heatmap(self, epoch, sample_x, sample_y_true, z_hiddens, z_output, sample_y_pred):
        """
        層別活性化ヒートマップを更新
        
        Parameters:
        -----------
        epoch : int
            現在のエポック番号
        sample_x : array
            入力サンプル
        sample_y_true : int
            正解クラス
        z_hiddens : list[array]
            各隠れ層の活性
        z_output : array
            出力層の活性
        sample_y_pred : int
            予測クラス
        """
        if not self.enable_heatmap or self.fig_heatmap is None:
            return
        
        self.fig_heatmap.clear()
        
        # GridSpec作成（2行×4列）
        gs = gridspec.GridSpec(4, 4, figure=self.fig_heatmap, hspace=0.4, wspace=0.3)
        
        # タイトル（エポック、正解、予測） - 予測クラスの色分け：正解=青、不正解=赤
        is_correct = (sample_y_pred == sample_y_true)
        pred_color = 'blue' if is_correct else 'red'
        title_text = f'エポック: {epoch} | 正解クラス: {sample_y_true} | '
        self.fig_heatmap.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98, x=0.42, ha='right')
        self.fig_heatmap.text(0.42, 0.98, f'予測クラス: {sample_y_pred}', 
                fontsize=14, fontweight='bold', color=pred_color, 
                ha='left', va='top', transform=self.fig_heatmap.transFigure)
        
        # 表示する層を選択（入力層 + 隠れ層 + 出力層）（8層超の場合は最初の4層と最後の4層）
        # -2は入力層、-1は出力層
        total_layers = 1 + len(z_hiddens) + 1  # 入力層 + 隠れ層 + 出力層
        if total_layers <= 8:
            # 全層表示
            display_layers = [-2] + list(range(len(z_hiddens))) + [-1]  # -2:入力層、-1:出力層
        else:
            # 最初の4層と最後の4層のみ表示（入力層を含む）
            if len(z_hiddens) <= 6:
                # 隠れ層が6層以下: 入力層 + 全隠れ層 + 出力層
                display_layers = [-2] + list(range(len(z_hiddens))) + [-1]
            else:
                # 隠れ層が7層以上: 入力層 + 最初の3層 + 最後の3層 + 出力層
                display_layers = [-2] + list(range(3)) + list(range(len(z_hiddens) - 3, len(z_hiddens))) + [-1]
        
        # 各層を表示
        for plot_idx, layer_idx in enumerate(display_layers[:8]):  # 最大8層
            if layer_idx == -2:
                # 入力層
                z_data = sample_x
                layer_name = f'Input Layer ({len(sample_x)})'
            elif layer_idx == -1:
                # 出力層
                z_data = z_output
                layer_name = f'Output Layer ({len(z_output)})'
            else:
                # 隠れ層
                z_data = z_hiddens[layer_idx]
                layer_name = f'Hidden {layer_idx+1} ({len(z_data)})'
            
            # グリッド位置計算（2行×4列）
            row = plot_idx // 4
            col = plot_idx % 4
            ax = self.fig_heatmap.add_subplot(gs[row+1, col])  # 1行目はタイトル用に空ける
            
            # 活性化を2D配列に整形（正方形に近い形状、row-wise配置）
            n_neurons = len(z_data)
            side = int(np.ceil(np.sqrt(n_neurons)))
            z_reshaped = np.zeros((side, side))
            z_reshaped.flat[:n_neurons] = z_data
            im = ax.imshow(z_reshaped, cmap='rainbow', aspect='equal', vmin=0, vmax=1)
            
            ax.set_title(layer_name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # カラーバーを正しいfigureに追加
            from matplotlib import pyplot
            pyplot.figure(self.fig_heatmap.number)
            pyplot.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.figure(self.fig_heatmap.number)
        plt.pause(0.1)
        plt.draw()
    
    def save_figures(self):
        """
        可視化図を保存
        
        Returns:
        --------
        tuple[str, str] or tuple[None, None]
            (学習曲線保存パス, ヒートマップ保存パス)
        """
        if not (self.enable_viz or self.enable_heatmap):
            return None, None
        
        plt.ioff()
        
        save_path_viz = None
        save_path_heatmap = None
        
        if self.enable_viz and self.fig_viz is not None:
            save_path_viz = self.save_path.replace('.png', '_viz.png') if self.enable_viz and self.enable_heatmap else self.save_path
            plt.figure(self.fig_viz.number)
            plt.savefig(save_path_viz, dpi=150, bbox_inches='tight')
            print(f"[学習曲線保存] {save_path_viz}")
        
        if self.enable_heatmap and self.fig_heatmap is not None:
            save_path_heatmap = self.save_path.replace('.png', '_heatmap.png') if self.enable_viz and self.enable_heatmap else self.save_path
            plt.figure(self.fig_heatmap.number)
            plt.savefig(save_path_heatmap, dpi=150, bbox_inches='tight')
            print(f"[ヒートマップ保存] {save_path_heatmap}")
        
        return save_path_viz, save_path_heatmap
    
    def close(self):
        """可視化ウィンドウを閉じる"""
        if self.fig_viz is not None:
            plt.close(self.fig_viz)
        if self.fig_heatmap is not None:
            plt.close(self.fig_heatmap)
