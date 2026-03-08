#!/usr/bin/env python3
"""
可視化マネージャー - リアルタイム学習進捗・ヒートマップ表示

機能:
  - 日本語フォント設定
  - リアルタイム学習曲線表示
  - 混同行列表示
  - 層別活性化ヒートマップ表示（Gabor特徴タイル対応）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from pathlib import Path


def setup_japanese_font():
    """日本語フォントを設定する（システムから自動検索、fallback付き）"""
    import matplotlib.font_manager as fm
    
    # 優先フォントリスト（上から順に検索）
    preferred_fonts = [
        # Linux系
        'Noto Sans CJK JP', 'Noto Sans JP',
        'IPAexGothic', 'IPAGothic', 'IPAexMincho', 'IPAMincho',
        'TakaoPGothic', 'TakaoGothic', 'VL PGothic', 'VL Gothic',
        # Windows / WSL環境
        'Yu Gothic', 'Yu Gothic UI', 'Meiryo', 'Meiryo UI',
        'BIZ UDGothic', 'BIZ UDPGothic', 'BIZ UDMincho', 'BIZ UDPMincho',
        'MS Gothic', 'MS PGothic', 'MS UI Gothic',
        'MS Mincho', 'MS PMincho',
        'Yu Mincho',
        'UD Digi Kyokasho N', 'UD Digi Kyokasho NP', 'UD Digi Kyokasho NK',
        # Windows HGフォント
        'HGGothicM', 'HGPGothicM', 'HGSGothicM',
        'HGGothicE', 'HGPGothicE', 'HGSGothicE',
        'HGMinchoB', 'HGPMinchoB', 'HGSMinchoB',
        'HGMinchoE', 'HGPMinchoE', 'HGSMinchoE',
        'HGMaruGothicMPRO',
        'HGSoeiKakugothicUB', 'HGPSoeiKakugothicUB', 'HGSSoeiKakugothicUB',
        'HGSoeiKakupoptai', 'HGPSoeiKakupoptai', 'HGSSoeiKakupoptai',
        'HGSoeiPresenceEB', 'HGPSoeiPresenceEB', 'HGSSoeiPresenceEB',
        'HGGyoshotai', 'HGPGyoshotai', 'HGSGyoshotai',
        'HGKyokashotai', 'HGPKyokashotai', 'HGSKyokashotai',
        'HGSeikaishotaiPRO',
        # Android / その他
        'Droid Sans Fallback',
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    selected_font = None
    for font in preferred_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # フォールバック: フォント名にCJK/日本語関連キーワードを含むものを検索
    if selected_font is None:
        for font_name in available_fonts:
            if any(keyword in font_name for keyword in
                   ['CJK', 'Japan', 'Japanese', 'JP', 'IPA',
                    'Gothic', 'Mincho', 'Meiryo']):
                selected_font = font_name
                break
    
    if selected_font:
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"[可視化] 日本語フォント設定: {selected_font}")
        return selected_font
    else:
        # 日本語フォント未検出: グリフ欠落警告を抑制し、ユーザーに案内
        import warnings
        warnings.filterwarnings('ignore', message='.*Glyph.*missing from.*font.*')
        print("[可視化] 日本語フォントを検出できなかったため、日本語が文字化けします。")
        print("         日本語フォントのインストールをお勧めします。")
        print("         例: sudo apt install fonts-noto-cjk")
        return None


def determine_save_path(save_fig_arg):
    """--save_figオプションから保存パスを決定"""
    if save_fig_arg is None:
        viz_dir = Path('viz_results')
        viz_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return str(viz_dir / f'viz_results_{timestamp}.png')
    else:
        save_fig_path = Path(save_fig_arg)
        
        if save_fig_path.is_dir():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return str(save_fig_path / f'viz_results_{timestamp}.png')
        elif not save_fig_path.suffix:
            save_fig_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return str(save_fig_path / f'viz_results_{timestamp}.png')
        elif save_fig_path.parent != Path('.'):
            save_fig_path.parent.mkdir(parents=True, exist_ok=True)
            return str(save_fig_path)
        else:
            return str(save_fig_path)


def _infer_image_shape(n_pixels):
    """フラット化された画素数から画像形状を推定する。
    
    Returns:
        (H, W) or (H, W, C): グレースケールまたはカラー画像の形状
    """
    # カラー画像の判定: 3チャネルで割り切れて、各チャネルが正方形になるか
    if n_pixels % 3 == 0:
        per_channel = n_pixels // 3
        side = int(np.sqrt(per_channel))
        if side * side == per_channel:
            return (side, side, 3)
    # グレースケール画像
    side = int(np.sqrt(n_pixels))
    if side * side == n_pixels:
        return (side, side)
    return None


class VisualizationManager:
    """リアルタイム可視化マネージャー"""
    
    def __init__(self, enable_viz=False, enable_heatmap=False, save_path=None, total_epochs=100, window_scale=1.0):
        self.enable_viz = enable_viz
        self.enable_heatmap = enable_heatmap
        self.window_scale = window_scale if window_scale > 0 else 1.0
        
        setup_japanese_font()
        
        if save_path is not None:
            if save_path.endswith('/'):
                save_path_obj = Path(save_path)
                save_path_obj.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.save_path = str(save_path_obj / f'viz_results_{timestamp}')
            else:
                save_path_obj = Path(save_path)
                if save_path_obj.parent != Path('.'):
                    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
                if save_path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']:
                    self.save_path = str(save_path_obj.with_suffix(''))
                else:
                    self.save_path = save_path
        else:
            self.save_path = None
        
        self.total_epochs = total_epochs
        self.fig_viz = None
        self.fig_heatmap = None

        # 1920x1080基準の1/4をviz=1の基準サイズとする
        base_viz_px = (540, 270)
        # heatmapは可読性のため、viz基準の約1.4倍を採用（端数調整済み）
        base_heatmap_px = (670, 380)

        viz_px = (
            int(round(base_viz_px[0] * self.window_scale)),
            int(round(base_viz_px[1] * self.window_scale)),
        )
        heatmap_px = (
            int(round(base_heatmap_px[0] * self.window_scale)),
            int(round(base_heatmap_px[1] * self.window_scale)),
        )

        # matplotlib.figureはinch指定のため、現在のDPIから換算
        default_dpi = plt.rcParams.get('figure.dpi', 100)
        viz_figsize = (viz_px[0] / default_dpi, viz_px[1] / default_dpi)
        heatmap_figsize = (heatmap_px[0] / default_dpi, heatmap_px[1] / default_dpi)

        if self.enable_viz:
            plt.ion()
            self.fig_viz = plt.figure(figsize=viz_figsize)
            self.fig_viz.canvas.manager.set_window_title('学習曲線 + 混同行列')
            self._enforce_window_size(self.fig_viz, viz_figsize)
        
        if self.enable_heatmap:
            plt.ion()
            self.fig_heatmap = plt.figure(figsize=heatmap_figsize)
            self.fig_heatmap.canvas.manager.set_window_title('層別活性化ヒートマップ')
            self._enforce_window_size(self.fig_heatmap, heatmap_figsize)
        
        self.gabor_info = None

    def _enforce_window_size(self, fig, figsize_inches):
        """バックエンド依存で保持される既存ウィンドウサイズを上書きする。"""
        try:
            dpi = fig.get_dpi()
            w_px = int(figsize_inches[0] * dpi)
            h_px = int(figsize_inches[1] * dpi)
            manager = fig.canvas.manager
            window = getattr(manager, 'window', None)
            if window is None:
                return

            if hasattr(window, 'wm_geometry'):
                window.wm_geometry(f"{w_px}x{h_px}")
                return
            if hasattr(window, 'resize'):
                window.resize(w_px, h_px)
                return
            if hasattr(window, 'SetSize'):
                window.SetSize((w_px, h_px))
        except Exception:
            pass
    
    def update_learning_curve(self, train_acc_history, test_acc_history, x_test, y_test, network):
        """学習曲線と混同行列を更新"""
        if not self.enable_viz or self.fig_viz is None:
            return
        
        self.fig_viz.clear()
        ax1, ax2 = self.fig_viz.subplots(1, 2)
        # 可視化エリアを約90%に抑えて、軸ラベル・目盛のはみ出しを防ぐ
        self.fig_viz.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.90, wspace=0.32)

        if self.window_scale <= 1.3:
            label_fs = 9
            tick_fs = 8
        else:
            label_fs = 10
            tick_fs = 9

        # 学習曲線側は先に80%へ縮小する（混同行列側は描画後に縮小する）
        
        epochs_list = list(range(0, len(train_acc_history) + 1))
        ax1.plot(epochs_list, [0.0] + list(train_acc_history), label='訓練', marker='o', markersize=3)
        ax1.plot(epochs_list, [0.0] + list(test_acc_history), label='テスト', marker='s', markersize=3)
        ax1.set_xlabel('エポック', fontsize=label_fs)
        ax1.set_ylabel('正解率', fontsize=label_fs)
        ax1.set_title('学習進捗', fontsize=label_fs)
        ax1.legend(fontsize=tick_fs)
        ax1.tick_params(axis='both', labelsize=tick_fs)
        ax1.set_ylim(0.0, 1.0)
        ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax1.set_xlim(0, self.total_epochs)
        ax1.grid(True, alpha=0.3)
        for y in [0.1, 0.3, 0.5, 0.7, 0.9]:
            ax1.axhline(y=y, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        for y in [0.2, 0.4, 0.6, 0.8]:
            ax1.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        # 学習進捗グラフのみ80%へ縮小
        pos = ax1.get_position()
        new_w = pos.width * 0.8
        new_h = pos.height * 0.8
        new_x = pos.x0 + (pos.width - new_w) / 2
        new_y = pos.y0 + (pos.height - new_h) / 2
        ax1.set_position([new_x, new_y, new_w, new_h])
        
        if y_test.ndim == 1:
            n_classes = int(y_test.max()) + 1
            y_test_labels = y_test
        else:
            n_classes = y_test.shape[1]
            y_test_labels = np.argmax(y_test, axis=1)
        
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(len(x_test)):
            _, z_out, _ = network.forward(x_test[i])
            pred = np.argmax(z_out)
            true = y_test_labels[i] if y_test.ndim == 1 else np.argmax(y_test[i])
            conf_matrix[true, pred] += 1
        
        sns.heatmap(conf_matrix, annot=True, fmt='d',
                   cmap='Blues', ax=ax2,
                   annot_kws={'size': tick_fs},
                   cbar_kws={'label': '件数', 'fraction': 0.045, 'pad': 0.03})
        # seabornの描画後に混同行列のみ80%へ縮小して確実に反映する
        pos = ax2.get_position()
        new_w = pos.width * 0.8
        new_h = pos.height * 0.8
        left_shift = 0.025
        new_x = max(0.0, pos.x0 + (pos.width - new_w) / 2 - left_shift)
        new_y = pos.y0 + (pos.height - new_h) / 2
        ax2.set_position([new_x, new_y, new_w, new_h])
        # カラーバーは別軸のため、混同行列に追従させて右端はみ出しを防ぐ
        if ax2.collections:
            cbar = ax2.collections[0].colorbar
            if cbar is not None and getattr(cbar, 'ax', None) is not None:
                cb_ax = cbar.ax
                ax2_pos = ax2.get_position()
                cb_pos = cb_ax.get_position()
                cb_w = cb_pos.width
                cb_pad = 0.008
                max_right = 0.97

                cb_x = ax2_pos.x1 + cb_pad
                overflow = (cb_x + cb_w) - max_right
                if overflow > 0:
                    shifted_x0 = max(0.0, ax2_pos.x0 - overflow)
                    ax2.set_position([shifted_x0, ax2_pos.y0, ax2_pos.width, ax2_pos.height])
                    ax2_pos = ax2.get_position()
                    cb_x = ax2_pos.x1 + cb_pad
                if cb_x + cb_w > max_right:
                    cb_x = max_right - cb_w

                cb_ax.set_position([cb_x, ax2_pos.y0, cb_w, ax2_pos.height])
        ax2.set_xlabel('予測クラス', fontsize=label_fs)
        # 左ラベルが学習曲線へ重ならないよう、少し内側へ寄せる
        ax2.set_ylabel('真のクラス', fontsize=label_fs, labelpad=2)
        ax2.set_title('混同行列（テストデータ）', fontsize=label_fs)
        ax2.tick_params(axis='both', labelsize=tick_fs)
        
        plt.figure(self.fig_viz.number)
        plt.pause(0.1)
        plt.draw()
    
    def set_gabor_info(self, gabor_info):
        """Gabor特徴抽出器の情報を設定"""
        self.gabor_info = gabor_info
    
    def set_gabor_extractor(self, extractor):
        """Gabor特徴抽出器オブジェクトへの参照を設定（元解像度可視化用）"""
        self.gabor_extractor = extractor
    
    def _tile_gabor_features(self, features):
        """Gabor特徴量を方位×周波数のタイル画像に配置"""
        gi = self.gabor_info
        n_filters = gi['n_filters']
        pool_h, pool_w = gi['pool_output_shape']
        n_orient = gi['n_orientations']
        n_freq = gi['n_frequencies']
        n_edge = gi['n_edge_filters']
        n_channels = gi.get('n_channels', 1)
        fdpc = gi.get('feature_dim_per_channel', len(features))
        
        # カラー画像の場合はチャネル平均で1チャネル分に集約
        if n_channels > 1:
            per_ch = features.reshape(n_channels, n_filters, pool_h, pool_w)
            maps = per_ch.mean(axis=0)
        else:
            maps = features.reshape(n_filters, pool_h, pool_w)
        
        n_cols = n_orient
        n_rows = n_freq + (1 if n_edge > 0 else 0)
        
        gap = 1
        tile_h = n_rows * pool_h + (n_rows - 1) * gap
        tile_w = n_cols * pool_w + (n_cols - 1) * gap
        
        tiled = np.full((tile_h, tile_w), np.nan)
        
        idx = 0
        for r in range(n_freq):
            for c in range(n_orient):
                if idx < n_filters:
                    y0 = r * (pool_h + gap)
                    x0 = c * (pool_w + gap)
                    tiled[y0:y0+pool_h, x0:x0+pool_w] = maps[idx]
                    idx += 1
        
        if n_edge > 0:
            r = n_freq
            for c in range(n_edge):
                if idx < n_filters:
                    y0 = r * (pool_h + gap)
                    x0 = c * (pool_w + gap)
                    tiled[y0:y0+pool_h, x0:x0+pool_w] = maps[idx]
                    idx += 1
        
        return tiled

    def update_heatmap(self, epoch, sample_x, sample_y_true, sample_y_true_name, 
                      z_hiddens, z_output, sample_y_pred, sample_y_pred_name,
                      sample_x_raw=None, progress=None):
        """層別活性化ヒートマップを更新"""
        if not self.enable_heatmap or self.fig_heatmap is None:
            return
        
        self.fig_heatmap.clear()
        
        is_gabor_mode = self.gabor_info is not None and sample_x_raw is not None
        
        # 元解像度の畳み込み平均を更新（入力層表示用）
        ext = getattr(self, 'gabor_extractor', None)
        if ext is not None and sample_x_raw is not None:
            ext.transform_single(sample_x_raw)
        
        if len(z_hiddens) <= 6:
            bottom_layers = [-2] + list(range(len(z_hiddens))) + [-1]
        else:
            bottom_layers = [-2] + list(range(3)) + list(range(len(z_hiddens) - 3, len(z_hiddens))) + [-1]
        n_bottom = len(bottom_layers)
        
        outer_gs = gridspec.GridSpec(2, 1, figure=self.fig_heatmap,
                                     height_ratios=[1, 1],
                                     top=0.92, bottom=0.08,
                                     left=0.06, right=0.94,
                                     hspace=0.35)
        
        if is_gabor_mode:
            top_gs = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=outer_gs[0],
                width_ratios=[1.2, 1, 2.5], wspace=0.3)
        else:
            top_gs = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=outer_gs[0],
                width_ratios=[1.2, 1, 2.5], wspace=0.3)
        bottom_gs = gridspec.GridSpecFromSubplotSpec(1, n_bottom, subplot_spec=outer_gs[1], wspace=0.3)
        
        from matplotlib import pyplot

        # カラーバー目盛りが隣接パネルへ重ならないよう、控えめな文字サイズを使用
        if self.window_scale <= 1.0:
            cbar_tick_fs = 5
        elif self.window_scale <= 1.3:
            cbar_tick_fs = 6
        else:
            cbar_tick_fs = 7

        def add_cbar(image, axis):
            cbar = pyplot.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_tick_fs)
            return cbar
        
        # テキスト情報パネル
        ax_info = self.fig_heatmap.add_subplot(top_gs[0, 0])
        ax_info.axis('off')
        
        is_correct = (sample_y_pred == sample_y_true)
        pred_color = 'blue' if is_correct else 'red'
        
        epoch_text = f'エポック: {epoch}'
        progress_text = f'学習データ: {progress}' if progress else None
        if sample_y_true_name is not None:
            true_text = f'真のクラス: {sample_y_true} ({sample_y_true_name})'
            pred_text = f'予測クラス: {sample_y_pred} ({sample_y_pred_name})'
        else:
            true_text = f'真のクラス: {sample_y_true}'
            pred_text = f'予測クラス: {sample_y_pred}'
        
        if np.isclose(self.window_scale, 1.0):
            info_font_main = 9
            info_font_progress = 9
        elif np.isclose(self.window_scale, 1.3):
            info_font_main = 10
            info_font_progress = 10
        elif np.isclose(self.window_scale, 1.6):
            info_font_main = 10
            info_font_progress = 10
        elif np.isclose(self.window_scale, 2.0):
            info_font_main = 10
            info_font_progress = 10
        else:
            info_font_main = 13
            info_font_progress = 11

        if progress_text:
            ax_info.text(0.05, 0.85, epoch_text, fontsize=info_font_main, fontweight='bold',
                    ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.05, 0.62, progress_text, fontsize=info_font_progress, fontweight='bold',
                    ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.05, 0.39, true_text, fontsize=info_font_main, fontweight='bold',
                    ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.05, 0.16, pred_text, fontsize=info_font_main, fontweight='bold', color=pred_color,
                    ha='left', va='top', transform=ax_info.transAxes)
        else:
            ax_info.text(0.05, 0.75, epoch_text, fontsize=info_font_main, fontweight='bold',
                    ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.05, 0.50, true_text, fontsize=info_font_main, fontweight='bold',
                    ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.05, 0.25, pred_text, fontsize=info_font_main, fontweight='bold', color=pred_color,
                    ha='left', va='top', transform=ax_info.transAxes)
        
        # 画像パネル
        if is_gabor_mode:
            img_shape = self.gabor_info.get('image_shape', (28, 28))
            
            ax_raw = self.fig_heatmap.add_subplot(top_gs[0, 1])
            if np.isclose(self.window_scale, 1.0):
                ax_raw_img = ax_raw.inset_axes([0.22, 0.25, 0.70, 0.70])
                ax_raw.axis('off')
            elif np.isclose(self.window_scale, 1.3):
                ax_raw_img = ax_raw.inset_axes([0.29, 0.15, 0.70, 0.70])
                ax_raw.axis('off')
            elif np.isclose(self.window_scale, 1.6):
                ax_raw_img = ax_raw.inset_axes([0.29, 0.15, 0.70, 0.70])
                ax_raw.axis('off')
            elif np.isclose(self.window_scale, 2.0):
                ax_raw_img = ax_raw.inset_axes([0.29, 0.15, 0.70, 0.70])
                ax_raw.axis('off')
            else:
                ax_raw_img = ax_raw
            # カラー画像の場合はチャネル次元付きでreshape
            raw_shape = _infer_image_shape(len(sample_x_raw))
            if raw_shape is not None and len(raw_shape) == 3:
                img = sample_x_raw.reshape(raw_shape)
                im = ax_raw_img.imshow(img, aspect='equal', vmin=0, vmax=1)
                ax_raw_img.set_title(f'元画像 ({raw_shape[0]}×{raw_shape[1]} RGB)', fontsize=10, pad=1)
            else:
                img = sample_x_raw.reshape(img_shape)
                im = ax_raw_img.imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
                ax_raw_img.set_title(f'元画像 ({img_shape[0]}×{img_shape[1]})', fontsize=10, pad=1)
            ax_raw_img.set_xticks([])
            ax_raw_img.set_yticks([])
            
            ax_gabor = self.fig_heatmap.add_subplot(top_gs[0, 2])
            tiled = self._tile_gabor_features(sample_x)
            im = ax_gabor.imshow(tiled, cmap='hot', aspect='equal', vmin=0, vmax=1)
            gi = self.gabor_info
            pool_h, pool_w = gi['pool_output_shape']
            n_orient = gi['n_orientations']
            n_freq = gi['n_frequencies']
            gap = 1
            orient_labels = [f'{int(i * 180 / n_orient)}°' for i in range(n_orient)]
            x_ticks = [(c * (pool_w + gap) + pool_w / 2) for c in range(n_orient)]
            ax_gabor.set_xticks(x_ticks)
            ax_gabor.set_xticklabels(orient_labels, fontsize=5)
            y_labels = [f'f{f+1}' for f in range(n_freq)]
            if gi['n_edge_filters'] > 0:
                y_labels.append('Edge')
            n_rows_tile = n_freq + (1 if gi['n_edge_filters'] > 0 else 0)
            y_ticks = [(r * (pool_h + gap) + pool_h / 2) for r in range(n_rows_tile)]
            ax_gabor.set_yticks(y_ticks)
            ax_gabor.set_yticklabels(y_labels, fontsize=6)
            ax_gabor.set_title(f'Gabor特徴 ({gi["n_filters"]}maps)', fontsize=10)
            pyplot.figure(self.fig_heatmap.number)
            add_cbar(im, ax_gabor)
        else:
            ax_raw = self.fig_heatmap.add_subplot(top_gs[0, 1])
            if np.isclose(self.window_scale, 1.0):
                ax_raw_img = ax_raw.inset_axes([0.22, 0.25, 0.70, 0.70])
                ax_raw.axis('off')
            elif np.isclose(self.window_scale, 1.3):
                ax_raw_img = ax_raw.inset_axes([0.29, 0.15, 0.70, 0.70])
                ax_raw.axis('off')
            elif np.isclose(self.window_scale, 1.6):
                ax_raw_img = ax_raw.inset_axes([0.29, 0.15, 0.70, 0.70])
                ax_raw.axis('off')
            elif np.isclose(self.window_scale, 2.0):
                ax_raw_img = ax_raw.inset_axes([0.29, 0.15, 0.70, 0.70])
                ax_raw.axis('off')
            else:
                ax_raw_img = ax_raw
            n_pixels = len(sample_x)
            img_shape = _infer_image_shape(n_pixels)
            if img_shape is not None and len(img_shape) == 3:
                # カラー画像 (例: CIFAR-10 32×32×3)
                img = sample_x.reshape(img_shape)
                im = ax_raw_img.imshow(img, aspect='equal', vmin=0, vmax=1)
                ax_raw_img.set_title(f'元画像 ({img_shape[0]}×{img_shape[1]} RGB)', fontsize=10, pad=1)
            elif img_shape is not None:
                # グレースケール画像
                img = sample_x.reshape(img_shape)
                im = ax_raw_img.imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
                ax_raw_img.set_title(f'元画像 ({img_shape[0]}×{img_shape[1]})', fontsize=10, pad=1)
            else:
                # 正方形にできない場合はパディング
                side = int(np.ceil(np.sqrt(n_pixels)))
                img = np.zeros(side * side)
                img[:n_pixels] = sample_x
                img = img.reshape(side, side)
                im = ax_raw_img.imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
                ax_raw_img.set_title(f'元画像 ({side}×{side})', fontsize=10, pad=1)
            ax_raw_img.set_xticks([])
            ax_raw_img.set_yticks([])
        
        # 層パネル
        for i, layer_idx in enumerate(bottom_layers):
            ax = self.fig_heatmap.add_subplot(bottom_gs[0, i])
            
            if layer_idx == -2:
                input_data = sample_x_raw if sample_x_raw is not None else sample_x
                z_data = input_data
                layer_name = f'入力層 ({len(input_data)})'
            elif layer_idx == -1:
                z_data = z_output
                layer_name = f'出力層 ({len(z_output)})'
            else:
                z_data = z_hiddens[layer_idx]
                layer_name = f'隠れ層{layer_idx+1} ({len(z_data)})'
            
            n_neurons = len(z_data)
            
            # 入力層の特別処理
            if layer_idx == -2:
                # Gabor ON時でも入力層は変換前画像を表示して、Gabor OFF時と同じ見た目の基準に揃える
                if sample_x_raw is not None:
                    inferred_raw = _infer_image_shape(len(sample_x_raw))
                    if inferred_raw is not None and len(inferred_raw) == 3:
                        img_rgb = sample_x_raw.reshape(inferred_raw)
                        luminance = (0.299 * img_rgb[:, :, 0] +
                                     0.587 * img_rgb[:, :, 1] +
                                     0.114 * img_rgb[:, :, 2])
                        im = ax.imshow(luminance, cmap='rainbow', aspect='equal', vmin=0, vmax=1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(layer_name, fontsize=10)
                        pyplot.figure(self.fig_heatmap.number)
                        add_cbar(im, ax)
                        continue
                    if inferred_raw is not None:
                        img_raw = sample_x_raw.reshape(inferred_raw)
                        im = ax.imshow(img_raw, cmap='rainbow', aspect='equal', vmin=0, vmax=1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(layer_name, fontsize=10)
                        pyplot.figure(self.fig_heatmap.number)
                        add_cbar(im, ax)
                        continue

                # Gabor OFF + カラー画像: 輝度変換してヒートマップ
                inferred = _infer_image_shape(n_neurons)
                if inferred is not None and len(inferred) == 3:
                    img_rgb = z_data.reshape(inferred)
                    luminance = 0.299 * img_rgb[:,:,0] + 0.587 * img_rgb[:,:,1] + 0.114 * img_rgb[:,:,2]
                    im = ax.imshow(luminance, cmap='rainbow', aspect='equal', vmin=0, vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(layer_name, fontsize=10)
                    pyplot.figure(self.fig_heatmap.number)
                    add_cbar(im, ax)
                    continue
            
            side = int(np.ceil(np.sqrt(n_neurons)))
            z_reshaped = np.zeros((side, side))
            for j in range(n_neurons):
                z_reshaped[j // side, j % side] = z_data[j]
            
            if layer_idx >= 0:
                vmax_dynamic = max(0.1, np.max(z_data) * 1.1)
                im = ax.imshow(z_reshaped, cmap='rainbow', aspect='equal', vmin=0, vmax=vmax_dynamic)
            else:
                im = ax.imshow(z_reshaped, cmap='rainbow', aspect='equal', vmin=0, vmax=1)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(layer_name, fontsize=10)
            pyplot.figure(self.fig_heatmap.number)
            add_cbar(im, ax)
        
        plt.figure(self.fig_heatmap.number)
        plt.pause(0.1)
        plt.draw()
    
    def save_figures(self):
        """可視化図を保存"""
        if not (self.enable_viz or self.enable_heatmap):
            return None, None
        
        if self.save_path is None:
            return None, None
        
        plt.ioff()
        
        save_path_viz = None
        save_path_heatmap = None
        
        if self.enable_viz and self.fig_viz is not None:
            if self.enable_heatmap and self.fig_heatmap is not None:
                save_path_viz = f"{self.save_path}_viz.png"
            else:
                save_path_viz = f"{self.save_path}.png"
            plt.figure(self.fig_viz.number)
            plt.savefig(save_path_viz, dpi=150, bbox_inches='tight')
            print(f"[学習曲線保存] {save_path_viz}")
        
        if self.enable_heatmap and self.fig_heatmap is not None:
            if self.enable_viz and self.fig_viz is not None:
                save_path_heatmap = f"{self.save_path}_heatmap.png"
            else:
                save_path_heatmap = f"{self.save_path}.png"
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
