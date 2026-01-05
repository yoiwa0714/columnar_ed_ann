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
    
    Returns:
    --------
    str or None
        選択されたフォント名（見つからない場合はNone）
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
        # matplotlibのフォント設定（より強力な設定）
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        # マイナス記号の文字化け対策
        plt.rcParams['axes.unicode_minus'] = False
        print(f"日本語フォント設定: {selected_font}")
        return selected_font
    else:
        print("警告: 日本語フォントが見つかりませんでした。デフォルトフォントを使用します。")
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
    
    def __init__(self, enable_viz=False, enable_heatmap=False, save_path=None, total_epochs=100, input_shape=None, network=None):
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
        input_shape : list or None
            入力画像の形状 [height, width] or [height, width, channels]
            カスタムデータセットの矩形画像表示に使用
        network : object or None
            ネットワークオブジェクト（コラム構造情報の取得用）
        """
        self.enable_viz = enable_viz
        self.enable_heatmap = enable_heatmap
        self.input_shape = input_shape  # 入力画像形状を保存
        self.network = network  # ネットワークオブジェクトを保存
        
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
        
        self.fig_viz = None
        self.fig_heatmap = None
        
        if self.enable_viz:
            plt.ion()
            self.fig_viz = plt.figure(figsize=(15, 5))
            # Tkinterウィンドウタイトルは英語に変更（フォント問題回避）
            self.fig_viz.canvas.manager.set_window_title('Learning Curve + Confusion Matrix')
        
        if self.enable_heatmap:
            plt.ion()
            self.fig_heatmap = plt.figure(figsize=(16, 8))
            # Tkinterウィンドウタイトルは英語に変更（フォント問題回避）
            self.fig_heatmap.canvas.manager.set_window_title('Layer-wise Activation Heatmap')
    
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
                   cmap='Blues', ax=ax2, cbar_kws={'label': 'Count'})
        ax2.set_xlabel('Predicted Class')
        ax2.set_ylabel('True Class')
        ax2.set_title('Confusion Matrix (Test Data)')
        
        plt.figure(self.fig_viz.number)
        plt.pause(0.1)
        plt.draw()
    
    def update_heatmap(self, epoch, sample_x, sample_y_true, sample_y_true_name, 
                      z_hiddens, z_output, sample_y_pred, sample_y_pred_name):
        """
        層別活性化ヒートマップを更新
        
        注意: 現在の実装では、この関数はエポックごとに1回のみ呼び出されます。
        より高頻度な更新が必要な場合は、train_epoch()内のミニバッチループから
        直接呼び出す必要があります。
        
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
        """
        if not self.enable_heatmap or self.fig_heatmap is None:
            return
        
        self.fig_heatmap.clear()
        
        # GridSpec作成（2行×4列）
        gs = gridspec.GridSpec(4, 4, figure=self.fig_heatmap, hspace=0.4, wspace=0.3)
        
        # タイトル（エポック、正解、予測）を3行で表示
        # 予測クラスの色分け：正解=青、不正解=赤
        is_correct = (sample_y_pred == sample_y_true)
        pred_color = 'blue' if is_correct else 'red'
        
        # 3行表示（行間を広げて文字の重なりを防止）- 英語表記でTkinter警告回避
        y_start = 0.98
        line_height = 0.03  # 0.02 → 0.03に拡大
        self.fig_heatmap.text(0.5, y_start, f'Epoch: {epoch}', 
                fontsize=14, fontweight='bold', 
                ha='center', va='top', transform=self.fig_heatmap.transFigure)
        
        # クラス名の表示（Noneでない場合のみ表示）
        if sample_y_true_name is not None:
            # Fashion-MNIST等のクラス名あり
            self.fig_heatmap.text(0.5, y_start - line_height, 
                    f'True Class: {sample_y_true} ({sample_y_true_name})', 
                    fontsize=14, fontweight='bold', 
                    ha='center', va='top', transform=self.fig_heatmap.transFigure)
            self.fig_heatmap.text(0.5, y_start - 2*line_height, 
                    f'Predicted Class: {sample_y_pred} ({sample_y_pred_name})', 
                    fontsize=14, fontweight='bold', color=pred_color, 
                    ha='center', va='top', transform=self.fig_heatmap.transFigure)
        else:
            # MNISTの場合はクラス番号のみ
            self.fig_heatmap.text(0.5, y_start - line_height, 
                    f'True Class: {sample_y_true}', 
                    fontsize=14, fontweight='bold', 
                    ha='center', va='top', transform=self.fig_heatmap.transFigure)
            self.fig_heatmap.text(0.5, y_start - 2*line_height, 
                    f'Predicted Class: {sample_y_pred}', 
                    fontsize=14, fontweight='bold', color=pred_color, 
                    ha='center', va='top', transform=self.fig_heatmap.transFigure)
        
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
            
            # 活性化を2D配列に整形
            n_neurons = len(z_data)
            
            # 入力層の特別処理：画像として表示
            if layer_idx == -2:
                # カスタムデータセット: input_shapeを優先使用
                if self.input_shape is not None:
                    if len(self.input_shape) == 2:
                        # グレースケール画像 [height, width]
                        h, w = self.input_shape
                        if h * w == n_neurons:
                            z_reshaped = z_data.reshape(h, w)
                            im = ax.imshow(z_reshaped, cmap='rainbow', aspect='equal', vmin=0, vmax=1)
                        else:
                            # サイズ不一致の場合は警告して正方形表示
                            print(f"警告: input_shape {self.input_shape} と実際のニューロン数 {n_neurons} が一致しません")
                            side = int(np.ceil(np.sqrt(n_neurons)))
                            z_reshaped = np.zeros((side, side))
                            z_reshaped.flat[:n_neurons] = z_data
                            im = ax.imshow(z_reshaped, cmap='rainbow', aspect='equal', vmin=0, vmax=1)
                    elif len(self.input_shape) == 3:
                        # カラー画像 [height, width, channels]
                        h, w, c = self.input_shape
                        if h * w * c == n_neurons:
                            z_reshaped = z_data.reshape(h, w, c)
                            im = ax.imshow(z_reshaped, aspect='equal', vmin=0, vmax=1)
                        else:
                            # サイズ不一致の場合は警告して正方形表示
                            print(f"警告: input_shape {self.input_shape} と実際のニューロン数 {n_neurons} が一致しません")
                            side = int(np.ceil(np.sqrt(n_neurons)))
                            z_reshaped = np.zeros((side, side))
                            z_reshaped.flat[:n_neurons] = z_data
                            im = ax.imshow(z_reshaped, cmap='rainbow', aspect='equal', vmin=0, vmax=1)
                # 標準データセット: 次元数で判定
                elif n_neurons == 3072:
                    # CIFAR-10/100: 32×32×3
                    z_reshaped = z_data.reshape(32, 32, 3)
                    im = ax.imshow(z_reshaped, aspect='equal', vmin=0, vmax=1)
                elif n_neurons == 784:
                    # MNIST/Fashion-MNIST: 28×28
                    z_reshaped = z_data.reshape(28, 28)
                    im = ax.imshow(z_reshaped, cmap='rainbow', aspect='equal', vmin=0, vmax=1)
                else:
                    # その他のサイズ：正方形に近い形状で表示
                    side = int(np.ceil(np.sqrt(n_neurons)))
                    z_reshaped = np.zeros((side, side))
                    z_reshaped.flat[:n_neurons] = z_data
                    im = ax.imshow(z_reshaped, cmap='rainbow', aspect='equal', vmin=0, vmax=1)
            else:
                # 隠れ層・出力層の表示
                side = int(np.ceil(np.sqrt(n_neurons)))
                z_reshaped = np.zeros((side, side))
                
                # 配置方式の切り替え
                if self.network is not None and self.network.use_circular:
                    # 2次元円環配置: 既に2次元座標で配置済みのため、通常配置
                    z_reshaped.flat[:n_neurons] = z_data
                else:
                    # ハニカム構造: 左上から通常配置
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
