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
    日本語フォントを設定する（システムから自動検索、fallback付き）
    
    検索優先順位:
    1. Noto Sans系 (Noto Sans CJK JP, Noto Sans JP)
    2. IPA系 (IPAexGothic, IPAGothic, IPAexMincho, IPAMincho)
    3. Takao系 (TakaoPGothic, TakaoGothic)
    4. VL系 (VL PGothic, VL Gothic)
    5. Windows / WSL環境フォント (Yu Gothic, Meiryo, MS Gothic等)
    6. Windows HGフォント
    7. その他CJK/日本語関連キーワードを含むフォント
    
    Returns:
    --------
    str or None
        選択されたフォント名（見つからない場合はNone）
    """
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
    
    # システムにインストールされている全フォントを取得
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 優先順位に従ってフォントを選択
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
    
    # フォントを設定
    if selected_font:
        # matplotlibのフォント設定
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        # マイナス記号の文字化け対策
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
    
    def __init__(self, enable_viz=False, enable_heatmap=False, save_path=None, total_epochs=100, verbose=False):
        """
        Parameters:
        -----------
        enable_viz : bool
            学習曲線表示の有効化
        enable_heatmap : bool
            ヒートマップ表示の有効化
        save_path : str or None
            保存パス（ディレクトリまたはベースファイル名）
            - None: 保存なし
            - 末尾が'/'のパス: ディレクトリとして扱い、タイムスタンプ付きファイル名で保存
            - 末尾が'/'以外: ベースファイル名として扱い、_viz.png, _heatmap.png を追加
        total_epochs : int
            総エポック数（学習曲線の横軸設定用）
        """
        self.enable_viz = enable_viz
        self.enable_heatmap = enable_heatmap
        
        # 日本語フォント設定（matplotlib用）
        setup_japanese_font()
        
        # save_pathの処理
        if save_path is not None:
            # 末尾が'/'の場合、ディレクトリとして扱う
            if save_path.endswith('/'):
                save_path_obj = Path(save_path)
                save_path_obj.mkdir(parents=True, exist_ok=True)
                # タイムスタンプ付きベース名を作成
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.save_path = str(save_path_obj / f'viz_results_{timestamp}')
            else:
                # ファイル名として扱う（拡張子の有無に関わらず）
                save_path_obj = Path(save_path)
                # 親ディレクトリを作成
                if save_path_obj.parent != Path('.'):
                    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
                # 画像拡張子(.png, .jpg等)のみ除去、その他はそのまま使用
                if save_path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']:
                    self.save_path = str(save_path_obj.with_suffix(''))
                else:
                    self.save_path = save_path
        else:
            self.save_path = None
        
        self.total_epochs = total_epochs
        self.verbose = verbose
        self.fig_viz = None
        self.fig_heatmap = None
        
        if self.enable_viz:
            plt.ion()
            self.fig_viz = plt.figure(figsize=(15, 5))
            # ウィンドウタイトルを日本語に変更
            self.fig_viz.canvas.manager.set_window_title('学習曲線 + 混同行列')
        
        if self.enable_heatmap:
            plt.ion()
            self.fig_heatmap = plt.figure(figsize=(12.8, 6.4))
            # ウィンドウタイトルを日本語に変更
            self.fig_heatmap.canvas.manager.set_window_title('層別活性化ヒートマップ')
        
        # Gabor特徴情報（set_gabor_info()で設定）
        self.gabor_info = None
    
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
        
        # 学習曲線（エポック0=0%を追加し、エポック番号と一致させる）
        epochs_list = list(range(0, len(train_acc_history) + 1))
        ax1.plot(epochs_list, [0.0] + list(train_acc_history), label='訓練', marker='o', markersize=3)
        ax1.plot(epochs_list, [0.0] + list(test_acc_history), label='テスト', marker='s', markersize=3)
        ax1.set_xlabel('エポック')
        ax1.set_ylabel('正解率')
        ax1.set_title('学習進捗')
        ax1.legend()
        
        # 縦軸設定: 0.0〜1.0
        ax1.set_ylim(0.0, 1.0)
        ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # 横軸設定: 0〜最大エポック数（シンプルなロジック）
        ax1.set_xlim(0, self.total_epochs)
        # matplotlib のデフォルトの目盛りロジックを使用（自動的に適度な間隔を設定）
        # 注: set_xticks()を明示的に呼ばず、matplotlibに任せる
        
        # グリッド線の設定
        ax1.grid(True, alpha=0.3)
        # 縦軸のグリッド：0.1, 0.3, 0.5, 0.7, 0.9が点線、0.2, 0.4, 0.6, 0.8が実線
        for y in [0.1, 0.3, 0.5, 0.7, 0.9]:
            ax1.axhline(y=y, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        for y in [0.2, 0.4, 0.6, 0.8]:
            ax1.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
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
                   cmap='Blues', ax=ax2, cbar_kws={'label': '件数'})
        ax2.set_xlabel('予測クラス')
        ax2.set_ylabel('真のクラス')
        ax2.set_title('混同行列（テストデータ）')
        
        plt.figure(self.fig_viz.number)
        plt.pause(0.1)
        plt.draw()
    
    def set_gabor_info(self, gabor_info):
        """Gabor特徴抽出器の情報を設定（ヒートマップ表示用）
        
        Parameters:
        -----------
        gabor_info : dict
            GaborFeatureExtractor.get_info() の戻り値
        """
        self.gabor_info = gabor_info
    
    def _tile_gabor_features(self, features):
        """Gabor特徴量を方位×周波数のタイル画像に配置
        
        Returns:
        --------
        tiled : ndarray
            タイル配置された2D配列
        """
        gi = self.gabor_info
        n_filters = gi['n_filters']
        pool_h, pool_w = gi['pool_output_shape']
        n_orient = gi['n_orientations']
        n_freq = gi['n_frequencies']
        n_edge = gi['n_edge_filters']
        
        # 特徴量を (n_filters, pool_h, pool_w) にリシェイプ
        maps = features.reshape(n_filters, pool_h, pool_w)
        
        # レイアウト: 列=方位数、行=周波数数 + エッジ行(あれば)
        n_cols = n_orient
        n_rows = n_freq + (1 if n_edge > 0 else 0)
        
        gap = 1  # マップ間のギャップ（ピクセル）
        tile_h = n_rows * pool_h + (n_rows - 1) * gap
        tile_w = n_cols * pool_w + (n_cols - 1) * gap
        
        # NaNで初期化（ギャップ部分がカラーマップに影響しないよう）
        tiled = np.full((tile_h, tile_w), np.nan)
        
        idx = 0
        for r in range(n_freq):
            for c in range(n_orient):
                if idx < n_filters:
                    y0 = r * (pool_h + gap)
                    x0 = c * (pool_w + gap)
                    tiled[y0:y0+pool_h, x0:x0+pool_w] = maps[idx]
                    idx += 1
        
        # エッジフィルタ（最終行の左端に配置）
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
        """
        層別活性化ヒートマップを更新
        
        注意: 現在の実装では、この関数はエポックごとに1回のみ呼び出されます。
        より高頻度な更新が必要な場合は、train_epoch()内のミニバッチループから
        直接呼び出す必要があります。
        
        コラム構造の可視化:
        - create_column_membership()が距離ベースでニューロンを選択
        - 近いニューロンインデックスは物理的にも近い位置
        - 結果として、コラムの塊が自然に可視化される
        
        Parameters:
        -----------
        epoch : int
            現在のエポック番号
        sample_x : array
            入力サンプル
        sample_y_true : int
            正解クラス
        sample_y_true_name : str
            正解クラス名
        z_hiddens : list[array]
            各隠れ層の活性
        z_output : array
            出力層の活性
        sample_y_pred : int
            予測クラス
        sample_y_pred_name : str
            予測クラス名
        sample_x_raw : array or None
            Gabor変換前の元画像（Gabor特徴使用時に元画像を表示するため）
        """
        if not self.enable_heatmap or self.fig_heatmap is None:
            return
        
        self.fig_heatmap.clear()
        
        # ================================================================
        # 2行レイアウト（各要素3/4サイズ）
        # 上段: テキスト情報 | 画像情報（元画像、Gabor特徴マップ）
        # 下段: 層情報（入力層、隠れ層、出力層）
        # ================================================================
        
        is_gabor_mode = self.gabor_info is not None and sample_x_raw is not None
        
        # 下段: 入力層 + 隠れ層 + 出力層（多層の場合は最初と最後の数層のみ）
        if len(z_hiddens) <= 6:
            bottom_layers = [-2] + list(range(len(z_hiddens))) + [-1]
        else:
            bottom_layers = [-2] + list(range(3)) + list(range(len(z_hiddens) - 3, len(z_hiddens))) + [-1]
        n_bottom = len(bottom_layers)
        
        # 外側GridSpec: 2行（上段=テキスト+画像、下段=層）
        # マージンを広げて各要素を3/4サイズに縮小
        outer_gs = gridspec.GridSpec(2, 1, figure=self.fig_heatmap,
                                     height_ratios=[1, 1],
                                     top=0.92, bottom=0.08,
                                     left=0.06, right=0.94,
                                     hspace=0.35)
        # 上段: テキスト列 + 画像列（Gabor ON: +Gabor特徴列）
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
        
        # ---- 上段左: テキスト情報パネル ----
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
        
        if progress_text:
            # エポック途中: 4行表示（エポック、学習データ、真のクラス、予測クラス）
            ax_info.text(0.05, 0.85, epoch_text,
                    fontsize=13, fontweight='bold',
                    ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.05, 0.62, progress_text,
                    fontsize=11, fontweight='bold',
                    ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.05, 0.39, true_text,
                    fontsize=13, fontweight='bold',
                    ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.05, 0.16, pred_text,
                    fontsize=13, fontweight='bold', color=pred_color,
                    ha='left', va='top', transform=ax_info.transAxes)
        else:
            # エポック完了時: 3行表示（エポック、真のクラス、予測クラス）
            ax_info.text(0.05, 0.75, epoch_text,
                    fontsize=13, fontweight='bold',
                    ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.05, 0.50, true_text,
                    fontsize=13, fontweight='bold',
                    ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.05, 0.25, pred_text,
                    fontsize=13, fontweight='bold', color=pred_color,
                    ha='left', va='top', transform=ax_info.transAxes)
        
        # ---- 上段: 画像パネル ----
        if is_gabor_mode:
            # Gabor ON: 元画像 + Gabor特徴マップ
            img_shape = self.gabor_info.get('image_shape', (28, 28))
            
            # 元画像
            ax_raw = self.fig_heatmap.add_subplot(top_gs[0, 1])
            img = sample_x_raw.reshape(img_shape)
            im = ax_raw.imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
            ax_raw.set_title(f'元画像 ({img_shape[0]}×{img_shape[1]})', fontsize=10)
            pyplot.figure(self.fig_heatmap.number)
            pyplot.colorbar(im, ax=ax_raw, fraction=0.046, pad=0.04)
            
            # Gabor特徴マップ
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
            pyplot.colorbar(im, ax=ax_gabor, fraction=0.046, pad=0.04)
        else:
            # Gabor OFF: 元画像のみ（sample_xを2D画像として表示）
            ax_raw = self.fig_heatmap.add_subplot(top_gs[0, 1])
            n_pixels = len(sample_x)
            side = int(np.sqrt(n_pixels))
            if side * side == n_pixels:
                img = sample_x.reshape(side, side)
            else:
                side = int(np.ceil(np.sqrt(n_pixels)))
                img = np.zeros(side * side)
                img[:n_pixels] = sample_x
                img = img.reshape(side, side)
            im = ax_raw.imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
            ax_raw.set_title(f'元画像 ({side}×{side})', fontsize=10)
            pyplot.figure(self.fig_heatmap.number)
            pyplot.colorbar(im, ax=ax_raw, fraction=0.046, pad=0.04)
        
        # ---- 下段: 層パネル ----
        for i, layer_idx in enumerate(bottom_layers):
            ax = self.fig_heatmap.add_subplot(bottom_gs[0, i])
            
            if layer_idx == -2:
                z_data = sample_x
                layer_name = f'入力層 ({len(sample_x)})'
            elif layer_idx == -1:
                z_data = z_output
                layer_name = f'出力層 ({len(z_output)})'
            else:
                z_data = z_hiddens[layer_idx]
                layer_name = f'隠れ層{layer_idx+1} ({len(z_data)})'
            
            # 2Dグリッドに整形
            n_neurons = len(z_data)
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
            
            # 活性値統計（--verbose指定時のみ表示）
            if layer_idx >= 0 and self.verbose:
                z_min, z_max, z_mean = np.min(z_data), np.max(z_data), np.mean(z_data)
                z_nonzero = z_data[z_data > 1e-8]
                if len(z_nonzero) > 0:
                    z_nonzero_mean = np.mean(z_nonzero)
                    print(f"  {layer_name} 活性値統計: min={z_min:.4f}, max={z_max:.4f}, mean={z_mean:.4f}, 非零mean={z_nonzero_mean:.4f} ({len(z_nonzero)}/{len(z_data)}個)")
                else:
                    print(f"  {layer_name} 活性値統計: min={z_min:.4f}, max={z_max:.4f}, mean={z_mean:.4f}, 非零=0個")
            
            ax.set_title(layer_name, fontsize=10)
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
        
        if self.save_path is None:
            return None, None
        
        plt.ioff()
        
        save_path_viz = None
        save_path_heatmap = None
        
        # 両方有効な場合は区別のため接尾辞を追加、片方のみの場合は.pngのみ
        if self.enable_viz and self.fig_viz is not None:
            if self.enable_heatmap and self.fig_heatmap is not None:
                # 両方有効な場合は区別のため _viz.png を追加
                save_path_viz = f"{self.save_path}_viz.png"
            else:
                # 学習曲線のみの場合は .png のみ追加
                save_path_viz = f"{self.save_path}.png"
            plt.figure(self.fig_viz.number)
            plt.savefig(save_path_viz, dpi=150, bbox_inches='tight')
            print(f"[学習曲線保存] {save_path_viz}")
        
        if self.enable_heatmap and self.fig_heatmap is not None:
            if self.enable_viz and self.fig_viz is not None:
                # 両方有効な場合は区別のため _heatmap.png を追加
                save_path_heatmap = f"{self.save_path}_heatmap.png"
            else:
                # ヒートマップのみの場合は .png のみ追加
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
