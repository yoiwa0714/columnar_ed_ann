#!/usr/bin/env python3
"""
Columnar ED-ANN 正式版（公開用）

columnar_ed_ann.py - バージョン 1.1
"""

import os
# TensorFlowのログメッセージを抑制（情報・警告を非表示）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import time
import sys
from sklearn.metrics import confusion_matrix

# コマンドライン引数のコピー（HyperParams適用時に明示指定チェック用）
_COMMAND_LINE_ARGS = sys.argv.copy()

# モジュールインポート
from modules.hyperparameters import HyperParams
from modules.data_loader import load_dataset, get_class_names, resolve_dataset_path, load_custom_dataset
from modules.ed_network import RefinedDistributionEDNetwork
from modules.visualization_manager import VisualizationManager
from modules.data_augmentation import augment_batch, create_augmented_dataset


def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description='コラムED法 多層ニューラルネットワーク実装 (モジュール化版)\n'
                    '\n'
                    '【重要】層数に基づく自動パラメータ設定:\n'
                    '  - 1層から5層までの構成には内部テーブルで保持しているパラメータが自動適用されます\n'
                    '  - 6層以上の構成には5層のパラメータがフォールバックとして適用されます\n'
                    '  - コマンドライン引数で明示的に指定した値は自動設定より優先されます\n'
                    '  - --list_hyperparams で利用可能な設定一覧を確認できます\n',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # ========================================
    # 実行関連のパラメータ
    # ========================================
    exec_group = parser.add_argument_group('実行関連のパラメータ')
    exec_group.add_argument('--train', type=int, default=3000,
                           help='訓練サンプル数（デフォルト値: 3000）')
    exec_group.add_argument('--test', type=int, default=1000,
                           help='テストサンプル数（デフォルト値: 1000）')
    exec_group.add_argument('--epochs', type=int, default=40,
                           help='エポック数（デフォルト値: 40）')
    exec_group.add_argument('--seed', type=int, default=42,
                           help='乱数シード（デフォルト値: 42、再現性確保用）')
    exec_group.add_argument('--dataset', type=str, default='mnist',
                           help='データセット名（mnist, fashion, cifar10, cifar100）または '
                                'カスタムデータセットのパス（デフォルト: mnist）。'
                                'カスタムデータは~/.keras/datasets/配下に配置するか、絶対パスで指定。')
    exec_group.add_argument('--batch_size', type=int, default=None,
                           help='ミニバッチサイズ（デフォルト: None=オンライン学習、例: 32, 128）')
    exec_group.add_argument('--use_cupy', action='store_true',
                           help='★CuPy版★ GPU最適化バッチ学習を使用（多層ネットワーク対応）')
    
    # ========================================
    # ED法関連のパラメータ
    # ========================================
    ed_group = parser.add_argument_group('ED法関連のパラメータ')
    ed_group.add_argument('--hidden', type=str, default='2048',
                         help='隠れ層ニューロン数（例: 2048=1層, 256,128=2層）（デフォルト値: 2048、v044グリッドサーチPhase 3b最適値）')
    ed_group.add_argument('--activation', type=str, default='tanh',
                         choices=['tanh', 'sigmoid', 'leaky_relu'],
                         help='活性化関数（デフォルト: tanh）※グリッドサーチ用、将来的に削除予定')
    ed_group.add_argument('--lr', type=float, default=0.15,
                         help='学習率（デフォルト値: 0.15、v044グリッドサーチ最適値）')
    ed_group.add_argument('--layer_learning_rates', type=str, default=None,
                         help='層別学習率（カンマ区切り、例: 0.05,0.1,0.15）\n'
                              '層数+1個の値が必要（隠れ層1用、...、出力層用）\n'
                              '未指定時: 全層で--lrの値を使用（v044 Phase 5で未指定が最適と確認）\n'
                              '例: 1層=0.10,0.15、2層=0.05,0.1,0.15')
    ed_group.add_argument('--epoch_lr_schedule', type=str, default=None,
                         help='エポック依存学習率スケジューリング（カンマ区切り、例: 0.7,0.9,1.0）\n'
                              'Epoch 0から順に各係数を適用、最後の値を維持\n'
                              '例: 0.7,0.9,1.0 → Epoch 0-1: 0.7倍, Epoch 2: 0.9倍, Epoch 3+: 1.0倍\n'
                              '未指定時: 全エポックで1.0倍（通常動作、v044 Phase 5で未指定が最適と確認）')
    ed_group.add_argument('--u1', type=float, default=0.5,
                         help='アミン拡散係数u1（層数により自動設定: 1層=0.5, 2層=0.5）')
    ed_group.add_argument('--u2', type=float, default=0.8,
                         help='アミン拡散係数u2（層数により自動設定: 1層=0.8, 2層=0.5）')
    ed_group.add_argument('--uniform_amine', action='store_true', default=False,
                         help='全層均一アミン拡散（青斑核モデル）: 全隠れ層でu1を使用し層間の拡散係数差を除去')
    ed_group.add_argument('--lateral_lr', type=float, default=0.08,
                         help='側方抑制の学習率（デフォルト値: 0.08）')
    ed_group.add_argument('--gradient_clip', type=float, default=0.03,
                         help='gradient clipping値（デフォルト値: 0.03、v045 2層Phase 2で0.03最適と確認、1層では影響なし）')
    
    # ========================================
    # コラム関連のパラメータ
    # ========================================
    column_group = parser.add_argument_group('コラム関連のパラメータ')
    column_group.add_argument('--list_hyperparams', type=int, nargs='?', const=0, default=None,
                             metavar='N_LAYERS',
                             help='パラメータ設定一覧を表示して終了。\n'
                                  '引数なし: 全層数(1-5層)の簡易一覧を表示\n'
                                  '層数指定: 指定層数のデフォルト値を実行時と同じ形式で詳細表示\n'
                                  '例: --list_hyperparams 1  (1層構成の詳細)\n'
                                  '    --list_hyperparams 2  (2層構成の詳細)\n'
                                  '    --list_hyperparams    (全層数の簡易一覧)')
    column_group.add_argument('--base_column_radius', type=float, default=0.4,
                             help='基準コラム半径（デフォルト値: 0.4、256ニューロン層での値）')
    column_group.add_argument('--column_radius', type=float, default=None,
                             help='コラム影響半径（デフォルト値: None、Noneなら層ごとに自動計算）')
    column_group.add_argument('--participation_rate', type=float, default=0.1,
                             help='コラム参加率（デフォルト値: 0.1、スパース表現、優先度：最高）')
    column_group.add_argument('--column_neurons', type=int, default=1,
                             help='各クラスの明示的ニューロン数（デフォルト値: 1、リザバー構成、v044グリッドサーチ最適値）')
    column_group.add_argument('--use_circular', action='store_true',
                             help='旧円環構造を使用（デフォルトはハニカム）')
    column_group.add_argument('--overlap', type=float, default=0.0,
                             help='コラム間の重複度（デフォルト値: 0.0、0.0-1.0、円環構造でのみ有効、0.0=重複なし）')
    column_group.add_argument('--diagnose_column', action='store_true',
                             help='コラム構造の詳細診断を実行')
    column_group.add_argument('--diagnose_hidden_weights', action='store_true',
                             help='隠れ層の重み状態を詳細診断（飽和度分析等）')
    column_group.add_argument('--lateral_cooperation', type=float, default=0.0,
                             help='側方協調学習の強度（0.0-1.0、デフォルト: 0.0=無効、Phase A実証: 0.3で最適）')
    column_group.add_argument('--top_k_winners', type=int, default=None,
                             help='学習参加ニューロン数の上限（デフォルト: None=全員参加、Phase C実証: 1で最適）')
    column_group.add_argument('--debug_lc', action='store_true',
                             help='lateral_cooperationデバッグモードを有効化（影響度の詳細分析）')
    column_group.add_argument('--column_lr_factors', type=str, default="0.005,0.003",
                             help='層別コラム学習率係数（デフォルト: 0.005,0.003、v045 2層Phase 2最適値、重み飽和抑制に有効）')
    column_group.add_argument('--use_affinity', action='store_true',
                             help='Affinity方式を使用（デフォルト: False=Membership方式、実験用）')
    column_group.add_argument('--affinity_max', type=float, default=1.0,
                             help='コラムニューロンのaffinity値（デフォルト: 1.0、--use_affinityと併用）')
    column_group.add_argument('--affinity_min', type=float, default=0.0,
                             help='非コラムニューロンのaffinity値（デフォルト: 0.0、--use_affinityと併用）')
    column_group.add_argument('--rf_overlap', type=float, default=0.5,
                             help='受容野オーバーラップ率（デフォルト: 0.5、column_neurons>=2時有効）')
    column_group.add_argument('--rf_mode', type=str, default='random',
                             choices=['random', 'spatial'],
                             help='受容野分割モード（デフォルト: random）')
    column_group.add_argument('--rank_lut_mode', type=str, default='default',
                             choices=['default', 'equal', 'gradual', 'classic'],
                             help='ランクベース学習率LUTモード（default: cn依存線形減衰, equal: 均等, gradual: 緩やか減衰, classic: v045以前の急減衰）')
    column_group.add_argument('--column_decorrelation', type=float, default=0.0,
                             help='コラム内重みベクトル脱相関強度 (0.0=無効, 0.001-0.1でコラム内ニューロン分化促進)')
    column_group.add_argument('--amine_base_level', type=float, default=0.0,
                             help='非コラムニューロンへの基本アミン拡散レベル (0.0=従来動作, 0.01-1.0で全ニューロン学習)')
    column_group.add_argument('--amine_diffusion_sigma', type=float, default=0.0,
                             help='空間的アミン拡散の標準偏差 (0.0=無効, >0でコラム近傍の非コラムが学習参加。amine_base_levelと併用必須)')
    column_group.add_argument('--output_spatial_bias', type=float, default=0.0,
                             help='出力層重みの空間バイアス強度 (0.0=無効, >0で非コラムニューロンの出力重みを最近傍クラス方向にバイアス)')
    column_group.add_argument('--hebbian_alignment', type=float, default=0.0,
                             help='Hebbian重み整列強度 (0.0=無効, >0で非コラムニューロンの隠れ層重みを最近傍コラム重心にドリフト。エポック毎に適用)')
    column_group.add_argument('--lateral_inhibition', type=float, default=0.0,
                             help='側方抑制強度 (0.0=無効, >0で非コラムニューロンの活性が最近傍クラス以外の出力スコアを抑制)')
    column_group.add_argument('--enable_non_column_learning', action='store_true',
                             help='非コラムニューロンの学習参加を有効化 (デフォルト: 無効=リザバー的動作)')
    column_group.add_argument('--nc_sparse_k', type=float, default=0.0,
                             help='非コラム活性化ゲート選択率 (0.0=無効, 0.02=上位2%%, 0.05=上位5%%). HTM理論の疎分散表現モデル')
    column_group.add_argument('--nc_amine_strength', type=float, default=0.5,
                             help='活性化ゲート選択された非コラムニューロンのアミン強度 (デフォルト: 0.5)')
    column_group.add_argument('--nc_nearest_learning', action='store_true',
                             help='NC最近傍クラス帰属学習: 各NCを最近傍マイクロコラムに帰属させそのクラスからのみ学習 (デフォルト: 無効)')
    column_group.add_argument('--competitive_inhibition', action='store_true',
                             help='競合コラム間選択的抑制を有効化: 最大出力の不正解クラスのコラムニューロンに微弱な抑制信号を送信 (デフォルト: 無効)')
    column_group.add_argument('--inhibition_strength', type=float, default=0.01,
                             help='競合抑制の強度 (正の学習に対する比率、デフォルト: 0.01=1/100)')
    column_group.add_argument('--inhibition_topk', type=int, default=1,
                             help='抑制対象の不正解クラス数 (デフォルト: 1=最大競合クラスのみ)')
    column_group.add_argument('--column_lateral_inhibition', action='store_true',
                             help='コラム間活性化側方抑制: 順伝播中に最大活性コラムがライバルコラムの活性化を一時的に抑制 (デフォルト: 無効)')
    column_group.add_argument('--cli_alpha', type=float, default=0.1,
                             help='コラム間活性化側方抑制の強度 (デフォルト: 0.1)')
    
    # ========================================
    # 可視化関連のパラメータ
    # ========================================
    viz_group = parser.add_argument_group('可視化関連のパラメータ')
    viz_group.add_argument('--viz', action='store_true',
                          help='学習曲線のリアルタイム可視化を有効化')
    viz_group.add_argument('--heatmap', action='store_true',
                          help='活性化ヒートマップの表示を有効化（--vizと併用）')
    viz_group.add_argument('--save_viz', type=str, nargs='?', const='viz_results',
                          default=None, metavar='PATH',
                          help='可視化結果を保存。'
                               'パス指定: ディレクトリまたはベースファイル名（例: results/exp1, my_exp.png）。'
                               '引数なし: viz_results/ディレクトリにタイムスタンプ付きで保存。'
                               'オプション未指定: 保存しない。')
    viz_group.add_argument('--save_weights', action='store_true',
                          help='エポックごとの重み統計を保存（詳細分析用）')
    
    # ========================================
    # 初期化関連のパラメータ (v036.5新規, v036.6拡張, v038.1拡張)
    # ========================================
    init_group = parser.add_argument_group('初期化関連のパラメータ')
    init_group.add_argument('--init_method', type=str, default='he',
                           choices=['uniform', 'xavier', 'he', 'flat'],
                           help='出力層重み初期化手法（デフォルト: he、v044グリッドサーチ最適値）\n'
                                'uniform: Simple Uniform (生物学的妥当性重視)\n'
                                'xavier: Xavier/Glorot初期化 (tanh/sigmoid最適)\n'
                                'he: He/Kaiming初期化 (ReLU最適、tanh/sigmoidでも有効)\n'
                                'flat: フラット初期化 (全重み同一値、デバッグ用)')
    init_group.add_argument('--init_limit', type=float, default=0.08,
                           help='出力層重み初期化の範囲 Uniform(-limit, +limit)（デフォルト: 0.08、uniformのみ有効）')
    init_group.add_argument('--flat_value', type=float, default=0.05,
                           help='Flat初期化のベース値（デフォルト: 0.05、flatのみ有効）')
    init_group.add_argument('--flat_perturbation', type=float, default=0.001,
                           help='Flat初期化の摂動幅（デフォルト: 0.001、flatのみ有効、重み=base±perturbation）')
    init_group.add_argument('--sparsity', type=float, default=0.0,
                           help='出力層の初期化時スパース率（0.0-1.0、デフォルト: 0.0、例: 0.2=20%%を0に）')
    init_group.add_argument('--hidden_sparsity', type=str, default='0.4',
                           help='★拡張★ 隠れ層（非コラムニューロン）のスパース率（デフォルト: 0.4、v044グリッドサーチ最適値、層別指定対応）\n'
                                '形式: カンマ区切りで層数分の値を指定\n'
                                '例: --hidden_sparsity 0.2,0.3 for 2-layer network (Layer0=0.2, Layer1=0.3)\n'
                                'コラム内ニューロンは密結合を維持、非コラムニューロンのみにスパース化を適用\n'
                                '生物学的背景: 脳内コラム構造は密結合、コラム外は疎結合')
    init_group.add_argument('--init_scales', type=str, default=None,
                           help='★v040新機能★ 層別初期化スケール係数（カンマ区切り、例: 0.3,0.5,1.0）\n'
                                '層数+1個の値が必要（Layer 0用、Layer 1用、...、出力層用）\n'
                                '未指定時: 層数依存デフォルト値を使用\n'
                                '  1層: [0.4, 1.0]\n'
                                '  2層: [0.7, 0.7, 0.8]\n'
                                '  3層: [0.3, 0.5, 0.7, 1.0]\n'
                                '例: --init_method he --init_scales 0.2,0.8,1.2')
    init_group.add_argument('--normalize_output_weights', action='store_true',
                           help='★v039.4新機能★ 出力層重みをクラス別に正規化（公平な初期活性化を保証）\n'
                                '各クラスの重み絶対値平均を強制的に同一に設定\n'
                                '目的: ランダム初期化による勝者選択の偏りを排除')
    init_group.add_argument('--balance_output_weights', action='store_true',
                           help='★v043.1新機能★ 出力層重みの正負バランスを揃える\n'
                                '各クラスの重みを正50%%:負50%%に調整\n'
                                '目的: 隠れ層活性化との内積を公平化し、勝者選択の偏りを防止\n'
                                'Class 5問題の解決策として有効（勝者分布の標準偏差: 10.5%%→5.8%%）')
    
    # ========================================
    # 早期停止関連のパラメータ (v038.2新規)
    # ========================================
    early_stop_group = parser.add_argument_group('早期停止関連のパラメータ')
    early_stop_group.add_argument('--early_stop_epoch', type=int, default=None,
                                 help='早期停止判定を行うエポック（デフォルト: None=無効）')
    early_stop_group.add_argument('--early_stop_threshold', type=float, default=0.15,
                                 help='早期停止の閾値。指定エポックでTest精度がこの値以下なら停止（デフォルト: 0.15）')
    
    # ========================================
    # 動的シナプス刈り込み関連のパラメータ (v042新規)
    # ========================================
    pruning_group = parser.add_argument_group('動的シナプス刈り込み（生物学的発達的プルーニング）')
    pruning_group.add_argument('--dynamic_pruning_final_sparsity', '--dynamic_pruning_fs', 
                              type=float, default=None, dest='dynamic_pruning_final_sparsity',
                              help='動的刈り込みの目標スパース率（例: 0.4=40%%刈り込み、指定で刈り込み有効化）\n'
                                   'v044 Phase 6: fs=0.4が安定的に+0.04%%改善、fs=0.5+ep5で最良(+0.06%%)\n'
                                   '生物学的妥当性: 脳の発達的プルーニングで約40-50%%が刈り込まれる')
    pruning_group.add_argument('--pruning_start_epoch', type=int, default=None,
                              help='刈り込み開始エポック（デフォルト: None=安定期検出による自動開始）\n'
                                   'v044 Phase 6: 自動検出(Epoch 2で安定期検出)が適切に機能')
    pruning_group.add_argument('--pruning_end_epoch', type=int, default=None,
                              help='刈り込み終了エポック（デフォルト: None=最終エポックまで）\n'
                                   '指定時: start_epoch〜end_epoch間で目標スパース率を100%%達成\n'
                                   '生物学的背景: 脳の刈り込みは部位により終了時期が異なる\n'
                                   '  視覚野: 約6歳で終了、前頭前野: 20代後半で終了')
    pruning_group.add_argument('--stability_threshold', type=float, default=0.01,
                              help='安定期検出の閾値（Test精度変化率、デフォルト: 0.01=1%%）\n'
                                   'v044 Phase 6: デフォルト値0.01で適切に安定期検出')
    pruning_group.add_argument('--pruning_verbose', action='store_true',
                              help='刈り込み詳細ログを出力（デフォルト: 無効）')
    
    # ========================================
    # データ拡張関連のパラメータ (v046)
    # ========================================
    aug_group = parser.add_argument_group('データ拡張関連のパラメータ',
                                         'v046: 訓練データの水増しによる汎化性能向上')
    aug_group.add_argument('--augment', action='store_true',
                          help='データ拡張を有効化（デフォルト: 無効）\n'
                               '有効時: 訓練データにシフト・回転・ノイズを適用して水増し')
    aug_group.add_argument('--aug_shift', type=int, default=2,
                          help='シフト範囲（ピクセル、デフォルト: 2）\n'
                               '画像を上下左右にランダムシフト')
    aug_group.add_argument('--aug_rotation', type=float, default=10.0,
                          help='回転範囲（度、デフォルト: 10.0）\n'
                               '画像を±指定角度でランダム回転')
    aug_group.add_argument('--aug_noise', type=float, default=0.03,
                          help='ガウスノイズ標準偏差（デフォルト: 0.03）\n'
                               '画像にガウスノイズを付加')
    aug_group.add_argument('--aug_copies', type=int, default=1,
                          help='拡張コピー数（デフォルト: 1）\n'
                               '1=元データ+1倍=2倍、2=元データ+2倍=3倍')
    aug_group.add_argument('--aug_online', action='store_true',
                          help='オンラインデータ拡張モード（デフォルト: 無効）\n'
                               '有効時: エポック毎にランダム拡張を適用（毎回異なる変換）\n'
                               '無効時: 事前拡張（一度だけ拡張してデータセットを固定）')
    
    # ========================================
    # 固定畳み込みフィルタ（Gaborフィルタ）関連のパラメータ
    # ========================================
    gabor_group = parser.add_argument_group('固定畳み込みフィルタ（V1 Gaborフィルタ）',
                                           'V1単純型細胞を模した固定フィルタで入力特徴を抽出。逆伝播不要。')
    gabor_group.add_argument('--gabor_features', action='store_true',
                            help='Gaborフィルタ特徴抽出を有効化（デフォルト: 無効）\n'
                                 '有効時: 入力画像にGaborフィルタバンクを適用し、特徴量に変換してからED法に入力')
    gabor_group.add_argument('--gabor_orientations', type=int, default=8,
                            help='Gaborフィルタの方位数（デフォルト: 8、V1の方位選択性コラムに対応）')
    gabor_group.add_argument('--gabor_frequencies', type=int, default=2,
                            help='空間周波数の数（デフォルト: 2、低〜中周波数帯域）')
    gabor_group.add_argument('--gabor_kernel_size', type=int, default=7,
                            help='フィルタカーネルサイズ（デフォルト: 7、奇数）')
    gabor_group.add_argument('--gabor_pool_size', type=int, default=4,
                            help='平均プーリングウィンドウサイズ（デフォルト: 4）')
    gabor_group.add_argument('--gabor_pool_stride', type=int, default=4,
                            help='プーリングストライド（デフォルト: 4）')
    gabor_group.add_argument('--gabor_no_edge', action='store_true',
                            help='Sobelエッジフィルタを含めない（デフォルト: 含める）')
    
    return parser.parse_args()


def _show_detailed_hyperparams(n_layers, hp, args):
    """
    指定層数のデフォルトパラメータを実行時と同じ形式で詳細表示
    
    Args:
        n_layers: 隠れ層の層数 (1-5)
        hp: HyperParamsインスタンス
        args: argparseの結果（デフォルト値取得用）
    """
    try:
        config = hp.get_config(n_layers)
    except ValueError as e:
        print(f"\nエラー: {e}")
        return
    
    hidden_sizes = config['hidden']
    
    # YAMLの値を使用（フォールバック付き）
    default_init_scales = config.get('weight_init_scales')
    if default_init_scales is None:
        default_init_scales = [0.3 + (0.7 * i / n_layers) for i in range(n_layers)] + [1.0]
    
    hs = config.get('hidden_sparsity', 0.4)
    default_hidden_sparsity = hs if isinstance(hs, list) else [hs] * n_layers
    
    clrf = config.get('column_lr_factors')
    default_column_lr_factors = clrf if clrf else ([0.005] + [0.003] * (n_layers - 1) if n_layers > 1 else [0.01])
    
    def fmt(name, value):
        return f"  {name}: {value}"
    
    print(f"\n{'='*70}")
    print(f"パラメータ設定（{n_layers}層構成）デフォルト値一覧")
    print(f"{'='*70}")
    print(f"  説明: {config['description']}")
    
    # [実行関連のパラメータ]
    print(f"\n[実行関連のパラメータ]")
    print(fmt("train", args.train))
    print(fmt("test", args.test))
    print(fmt("epochs", config['epochs']))
    print(fmt("seed", args.seed))
    print(fmt("dataset", args.dataset))
    print(fmt("batch_size", args.batch_size))
    print(fmt("use_cupy", args.use_cupy))
    
    # [ED法関連のパラメータ]
    print(f"\n[ED法関連のパラメータ]")
    print(fmt("hidden", hidden_sizes))
    print(fmt("activation", args.activation))
    print(fmt("lr", config.get('output_lr', args.lr)))
    print(fmt("layer_learning_rates", None))
    print(fmt("epoch_lr_schedule", None))
    print(fmt("u1", config.get('u1', 0.5)))
    print(fmt("u2", config.get('u2', 0.8)))
    print(fmt("lateral_lr", config.get('lateral_lr', 0.08)))
    print(fmt("gradient_clip", config.get('gradient_clip', args.gradient_clip)))
    
    # [コラム関連のパラメータ]
    print(f"\n[コラム関連のパラメータ]")
    print(fmt("base_column_radius", config.get('base_column_radius', args.base_column_radius)))
    print(fmt("column_radius", args.column_radius))
    print(fmt("column_radius_per_layer", config.get('column_radius_per_layer', 'N/A')))
    print(fmt("participation_rate", config.get('participation_rate', 0.1)))
    print(fmt("column_neurons", config.get('column_neurons', args.column_neurons)))
    print(fmt("use_circular", args.use_circular))
    print(fmt("overlap", args.overlap))
    print(fmt("lateral_cooperation", args.lateral_cooperation))
    print(fmt("top_k_winners", args.top_k_winners))
    print(fmt("column_lr_factors", default_column_lr_factors))
    print(fmt("use_affinity", args.use_affinity))
    print(fmt("affinity_max", args.affinity_max))
    print(fmt("affinity_min", args.affinity_min))
    
    # [初期化関連のパラメータ]
    print(f"\n[初期化関連のパラメータ]")
    print(fmt("init_method", args.init_method))
    print(fmt("init_limit", args.init_limit))
    print(fmt("flat_value", args.flat_value))
    print(fmt("flat_perturbation", args.flat_perturbation))
    print(fmt("sparsity", args.sparsity))
    print(fmt("hidden_sparsity", default_hidden_sparsity))
    print(fmt("init_scales", default_init_scales))
    print(fmt("normalize_output_weights", args.normalize_output_weights))
    print(fmt("balance_output_weights", args.balance_output_weights))
    
    # [早期停止関連のパラメータ]
    print(f"\n[早期停止関連のパラメータ]")
    print(fmt("early_stop_epoch", args.early_stop_epoch))
    print(fmt("early_stop_threshold", args.early_stop_threshold))
    
    # [動的シナプス刈り込み関連のパラメータ]
    print(f"\n[動的シナプス刈り込み関連のパラメータ]")
    print(fmt("dynamic_pruning_final_sparsity", args.dynamic_pruning_final_sparsity))
    print(fmt("pruning_start_epoch", args.pruning_start_epoch))
    print(fmt("stability_threshold", args.stability_threshold))
    print(fmt("pruning_verbose", False))
    
    # [固定畳み込みフィルタ関連のパラメータ]
    if 'gabor_orientations' in config:
        print(f"\n[固定畳み込みフィルタ関連のパラメータ]")
        print(fmt("gabor_orientations", config.get('gabor_orientations', 8)))
        print(fmt("gabor_frequencies", config.get('gabor_frequencies', 2)))
        print(fmt("gabor_kernel_size", config.get('gabor_kernel_size', 7)))
        print(fmt("gabor_pool_size", args.gabor_pool_size))
        print(fmt("gabor_pool_stride", args.gabor_pool_stride))
    
    print(f"\n{'='*70}")
    
    # 推奨実行コマンドの表示
    hidden_str = ','.join(str(h) for h in hidden_sizes)
    print(f"\n推奨実行コマンド:")
    print(f"  python columnar_ed_ann.py --hidden {hidden_str} \\")
    print(f"      --train 5000 --test 5000 --column_neurons 1 \\")
    if 'gabor_orientations' in config:
        print(f"      --epochs {config['epochs']} --init_method he \\")
        print(f"      --gabor_features")
    else:
        print(f"      --epochs {config['epochs']} --init_method he")
    print()


def main():
    """メイン処理"""
    args = parse_args()
    
    # 乱数シード設定（再現性確保）
    if args.seed is not None:
        print(f"\n=== 乱数シード固定: {args.seed} ===")
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        print("再現性モード: 有効（random, numpy固定）\n")
    else:
        print("\n=== 乱数シード: ランダム ===")
        print("再現性モード: 無効（毎回異なる結果）\n")
    
    print("=" * 80)
    print("Columnar ED-ANN")
    print("=" * 80)
    
    # HyperParams設定一覧の表示
    if args.list_hyperparams is not None:
        hp = HyperParams()
        if args.list_hyperparams == 0:
            # 引数なし: 全層数の簡易一覧
            hp.list_configs()
        else:
            # 層数指定: 詳細表示
            _show_detailed_hyperparams(args.list_hyperparams, hp, args)
        import sys
        sys.exit(0)
    
    # ========================================
    # 1. 隠れ層のパース（カンマ区切り対応、多層対応）
    # ========================================
    # ドット区切りの誤使用をチェック
    if '.' in args.hidden and ',' not in args.hidden:
        print(f"\nエラー: --hidden の区切り文字が不正です")
        print(f"  指定された値: {args.hidden}")
        print(f"  多層ネットワークを指定する場合は、ドット(.)ではなくカンマ(,)を使用してください")
        print(f"  例: --hidden 2048,1024")
        import sys
        sys.exit(1)
    
    try:
        if ',' in args.hidden:
            hidden_sizes = [int(x.strip()) for x in args.hidden.split(',')]
        else:
            hidden_sizes = [int(args.hidden)]
    except ValueError as e:
        print(f"\nエラー: --hidden の値を整数に変換できません")
        print(f"  指定された値: {args.hidden}")
        print(f"  各層のニューロン数は整数で指定してください")
        print(f"  例: --hidden 1024 (1層) または --hidden 1024,512 (2層)")
        import sys
        sys.exit(1)
    
    # ========================================
    # 1.1. HyperParams(YAML)からの自動パラメータ取得
    #   config/hyperparameters.yaml を読み込み、
    #   コマンドラインで未指定のパラメータをYAML値で上書き
    #   ※パースセクション(1.5-1.9)より前に実行する必要がある
    # ========================================
    hp = HyperParams()
    n_layers = len(hidden_sizes)
    
    # コマンドラインで引数が明示的に指定されたかをチェック
    def is_arg_specified(arg_names):
        """指定された引数名のいずれかがコマンドラインで明示的に指定されたかをチェック"""
        if isinstance(arg_names, str):
            arg_names = [arg_names]
        for arg_name in arg_names:
            for arg in _COMMAND_LINE_ARGS:
                if arg.startswith(f'--{arg_name}=') or arg.startswith(f'--{arg_name}') or arg == f'-{arg_name}':
                    return True
        return False
    
    try:
        config = hp.get_config(n_layers)
        
        # コマンドラインで明示されていないパラメータのみYAML値で上書き
        if not is_arg_specified('hidden'):
            hidden_sizes = config['hidden']
        
        if not is_arg_specified(['lr', 'learning_rate']):
            args.lr = config.get('output_lr', args.lr)
        if not is_arg_specified('u1') and 'u1' in config:
            args.u1 = config['u1']
        if not is_arg_specified('u2') and 'u2' in config:
            args.u2 = config['u2']
        if not is_arg_specified('lateral_lr') and 'lateral_lr' in config:
            args.lateral_lr = config['lateral_lr']
        if not is_arg_specified('base_column_radius') and 'base_column_radius' in config:
            args.base_column_radius = config['base_column_radius']
        if not is_arg_specified('participation_rate') and 'participation_rate' in config:
            args.participation_rate = config['participation_rate']
        if not is_arg_specified('epochs'):
            args.epochs = config['epochs']
        
        # Gaborフィルタパラメータの自動適用
        if not is_arg_specified('gabor_orientations') and 'gabor_orientations' in config:
            args.gabor_orientations = config['gabor_orientations']
        if not is_arg_specified('gabor_frequencies') and 'gabor_frequencies' in config:
            args.gabor_frequencies = config['gabor_frequencies']
        if not is_arg_specified('gabor_kernel_size') and 'gabor_kernel_size' in config:
            args.gabor_kernel_size = config['gabor_kernel_size']
        
        # 初期化・構造パラメータの自動適用
        if not is_arg_specified('init_scales') and 'weight_init_scales' in config and config['weight_init_scales'] is not None:
            args.init_scales = ','.join(str(v) for v in config['weight_init_scales'])
        if not is_arg_specified('hidden_sparsity') and 'hidden_sparsity' in config:
            hs = config['hidden_sparsity']
            args.hidden_sparsity = ','.join(str(v) for v in hs) if isinstance(hs, list) else str(hs)
        if not is_arg_specified('gradient_clip') and 'gradient_clip' in config:
            args.gradient_clip = config['gradient_clip']
        if not is_arg_specified('column_lr_factors') and 'column_lr_factors' in config:
            args.column_lr_factors = ','.join(str(v) for v in config['column_lr_factors'])
        if not is_arg_specified('column_neurons') and 'column_neurons' in config:
            args.column_neurons = config['column_neurons']
        
    except ValueError as e:
        print(f"Warning: YAML設定の取得に失敗: {e}")
        print("個別パラメータで継続します。\n")
    
    # ========================================
    # 1.5. 層別学習率のパース（v039新機能）
    # ========================================
    layer_lrs = None
    if args.layer_learning_rates is not None:
        try:
            layer_lrs = [float(x.strip()) for x in args.layer_learning_rates.split(',')]
            n_layers = len(hidden_sizes)
            expected_length = n_layers + 1  # 隠れ層 + 出力層
            
            if len(layer_lrs) != expected_length:
                print(f"\nエラー: --layer_learning_rates には {expected_length} 個の値が必要です")
                print(f"  （{n_layers}層ネットワーク: 入力層用×1 + 隠れ層用×{n_layers}）")
                print(f"  指定された値: {len(layer_lrs)}個 {layer_lrs}")
                import sys
                sys.exit(1)
            
            print(f"\n=== 層別学習率モード有効 ===")
            for i, lr in enumerate(layer_lrs[:-1]):
                print(f"  Layer {i}: learning_rate={lr:.4f}")
            print(f"  Output Layer: learning_rate={layer_lrs[-1]:.4f}")
            print()
        except ValueError as e:
            print(f"\nエラー: --layer_learning_rates のパースに失敗: {e}")
            print(f"  形式: カンマ区切りの数値（例: 0.05,0.1,0.15）")
            import sys
            sys.exit(1)
    
    # ========================================
    # 1.6. 層別初期化スケールのパース（v040新機能）
    # ========================================
    init_scales = None
    if args.init_scales is not None:
        try:
            init_scales = [float(x.strip()) for x in args.init_scales.split(',')]
            n_layers = len(hidden_sizes)
            expected_length = n_layers + 1  # 隠れ層 + 出力層
            
            if len(init_scales) != expected_length:
                print(f"\nエラー: --init_scales には {expected_length} 個の値が必要です")
                print(f"  （{n_layers}層ネットワーク: Layer 0用×1, ... Layer {n_layers-1}用×1, 出力層用×1）")
                print(f"  指定された値: {len(init_scales)}個 {init_scales}")
                import sys
                sys.exit(1)
            
            # 妥当性チェック（正の数であること）
            for i, scale in enumerate(init_scales):
                if scale <= 0:
                    print(f"\nエラー: --init_scales の値は正の数である必要があります: index={i}, value={scale}")
                    import sys
                    sys.exit(1)
            
            print(f"\n=== 層別初期化スケールモード有効 ===")
            for i, scale in enumerate(init_scales[:-1]):
                print(f"  Layer {i}: init_scale={scale:.4f}")
            print(f"  Output Layer: init_scale={init_scales[-1]:.4f}")
            print()
        except ValueError as e:
            print(f"\nエラー: --init_scales のパースに失敗: {e}")
            print(f"  形式: カンマ区切りの数値（例: 0.3,0.5,1.0）")
            import sys
            sys.exit(1)
    else:
        # YAMLから自動適用されなかった場合のフォールバック（段階的増加）
        n_layers = len(hidden_sizes)
        init_scales = [0.3 + (0.7 * i / n_layers) for i in range(n_layers)] + [1.0]
        
        print(f"\n=== 層別初期化スケール（フォールバック、{n_layers}層） ===")
        for i, scale in enumerate(init_scales[:-1]):
            print(f"  Layer {i}: init_scale={scale:.4f}")
        print(f"  Output Layer: init_scale={init_scales[-1]:.4f}")
        print()
    
    # ========================================
    # 1.7. 層別hidden_sparsityのパース（拡張機能）
    # ========================================
    hidden_sparsity_list = None
    if args.hidden_sparsity is not None:
        try:
            hidden_sparsity_list = [float(x.strip()) for x in args.hidden_sparsity.split(',')]
            n_layers = len(hidden_sizes)
            
            # 層数との整合性チェック（単一値なら全層に展開）
            if len(hidden_sparsity_list) == 1 and n_layers > 1:
                hidden_sparsity_list = hidden_sparsity_list * n_layers
            elif len(hidden_sparsity_list) != n_layers:
                print(f"\nエラー: --hidden_sparsity には {n_layers} 個の値が必要です")
                print(f"  （{n_layers}層ネットワーク: Layer 0用, Layer 1用, ...）")
                print(f"  指定された値: {len(hidden_sparsity_list)}個 {hidden_sparsity_list}")
                import sys
                sys.exit(1)
            
            # 値の範囲チェック（0.0-1.0）
            for i, sparsity in enumerate(hidden_sparsity_list):
                if not (0.0 <= sparsity <= 1.0):
                    print(f"\nエラー: --hidden_sparsity の値は0.0-1.0の範囲である必要があります")
                    print(f"  Layer {i}: {sparsity} (範囲外)")
                    import sys
                    sys.exit(1)
            
            print(f"\n=== 層別隠れ層スパース化モード有効 ===")
            for i, sparsity in enumerate(hidden_sparsity_list):
                print(f"  Layer {i}: hidden_sparsity={sparsity:.2f} ({sparsity*100:.0f}%)")
            print()
        except ValueError as e:
            print(f"\nエラー: --hidden_sparsity のパースに失敗: {e}")
            print(f"  形式: カンマ区切りの数値（例: 0.2,0.3）")
            import sys
            sys.exit(1)
    
    # ========================================
    # 1.8. 層別column_lr_factorsのパース（拡張機能）
    # ========================================
    column_lr_factors_list = None
    if args.column_lr_factors is not None:
        try:
            column_lr_factors_list = [float(x.strip()) for x in args.column_lr_factors.split(',')]
            n_layers = len(hidden_sizes)
            
            # 層数との整合性チェック（不足分は最後の値で埋める）
            if len(column_lr_factors_list) < n_layers:
                print(f"\n注意: --column_lr_factors の要素数({len(column_lr_factors_list)})が層数({n_layers})より少ないです")
                print(f"  不足分は最後の値({column_lr_factors_list[-1]})で埋めます")
                while len(column_lr_factors_list) < n_layers:
                    column_lr_factors_list.append(column_lr_factors_list[-1])
            elif len(column_lr_factors_list) > n_layers:
                column_lr_factors_list = column_lr_factors_list[:n_layers]
            
            # 値の範囲チェック（0.0-1.0）
            for i, lr_factor in enumerate(column_lr_factors_list):
                if not (0.0 <= lr_factor <= 1.0):
                    print(f"\nエラー: --column_lr_factors の値は0.0-1.0の範囲である必要があります")
                    print(f"  Layer {i}: {lr_factor} (範囲外)")
                    import sys
                    sys.exit(1)
            
            print(f"\n=== 層別コラム学習率係数モード有効 ===")
            for i, lr_factor in enumerate(column_lr_factors_list):
                print(f"  Layer {i}: column_lr_factor={lr_factor:.4f}")
            print()
        except ValueError as e:
            print(f"\nエラー: --column_lr_factors のパースに失敗: {e}")
            print(f"  形式: カンマ区切りの数値（例: 0.1,0.05）")
            import sys
            sys.exit(1)
    
    # ========================================
    # 1.9. エポック依存学習率スケジューリングのパース（v039.2新機能）
    # ========================================
    epoch_lr_schedule = None
    if args.epoch_lr_schedule is not None:
        try:
            epoch_lr_schedule = [float(x.strip()) for x in args.epoch_lr_schedule.split(',')]
            
            # 妥当性チェック
            if len(epoch_lr_schedule) == 0:
                print(f"\nエラー: --epoch_lr_schedule には少なくとも1個の値が必要です")
                import sys
                sys.exit(1)
            
            for scale in epoch_lr_schedule:
                if scale <= 0:
                    print(f"\nエラー: --epoch_lr_schedule の値は正の数である必要があります: {scale}")
                    import sys
                    sys.exit(1)
            
            print(f"\n=== エポック依存学習率スケジューリング有効 ===")
            print(f"  スケジュール: {epoch_lr_schedule}")
            for i, scale in enumerate(epoch_lr_schedule):
                next_epoch = i + 1 if i < len(epoch_lr_schedule) - 1 else f"{i}+"
                print(f"    Epoch {i}: {scale:.2f}倍 (~ Epoch {next_epoch})")
            print()
        except ValueError as e:
            print(f"\nエラー: --epoch_lr_schedule のパースに失敗: {e}")
            print(f"  形式: カンマ区切りの数値（例: 0.7,0.9,1.0）")
            import sys
            sys.exit(1)
    
    # ========================================
    # パラメータ設定の表示（グループ別）
    # ========================================
    def format_param(name, value, specified):
        """パラメータを表示形式にフォーマット"""
        mark = " (*)" if specified else ""
        return f"  {name}: {value}{mark}"
    
    print("\n" + "=" * 70)
    print(f"パラメータ設定（{n_layers}層構成）")
    print("=" * 70)
    
    # [実行関連のパラメータ]
    print("\n[実行関連のパラメータ]")
    print(format_param("train", args.train, is_arg_specified('train')))
    print(format_param("test", args.test, is_arg_specified('test')))
    print(format_param("epochs", args.epochs, is_arg_specified('epochs')))
    print(format_param("seed", args.seed, is_arg_specified('seed')))
    print(format_param("dataset", args.dataset, is_arg_specified('dataset')))
    print(format_param("batch_size", args.batch_size, is_arg_specified('batch_size')))
    print(format_param("use_cupy", args.use_cupy, is_arg_specified('use_cupy')))
    
    # [ED法関連のパラメータ]
    print("\n[ED法関連のパラメータ]")
    print(format_param("hidden", hidden_sizes, is_arg_specified('hidden')))
    print(format_param("activation", args.activation, is_arg_specified('activation')))
    print(format_param("lr", args.lr, is_arg_specified(['lr', 'learning_rate'])))
    print(format_param("layer_learning_rates", args.layer_learning_rates, is_arg_specified('layer_learning_rates')))
    print(format_param("epoch_lr_schedule", args.epoch_lr_schedule, is_arg_specified('epoch_lr_schedule')))
    print(format_param("u1", args.u1, is_arg_specified('u1')))
    print(format_param("u2", args.u2, is_arg_specified('u2')))
    print(format_param("uniform_amine", args.uniform_amine, is_arg_specified('uniform_amine')))
    print(format_param("amine_base_level", args.amine_base_level, is_arg_specified('amine_base_level')))
    print(format_param("amine_diffusion_sigma", args.amine_diffusion_sigma, is_arg_specified('amine_diffusion_sigma')))
    print(format_param("lateral_lr", args.lateral_lr, is_arg_specified('lateral_lr')))
    print(format_param("gradient_clip", args.gradient_clip, is_arg_specified('gradient_clip')))
    
    # [コラム関連のパラメータ]
    print("\n[コラム関連のパラメータ]")
    print(format_param("base_column_radius", args.base_column_radius, is_arg_specified('base_column_radius')))
    print(format_param("column_radius", args.column_radius, is_arg_specified('column_radius')))
    print(format_param("participation_rate", args.participation_rate, is_arg_specified('participation_rate')))
    print(format_param("column_neurons", args.column_neurons, is_arg_specified('column_neurons')))
    print(format_param("use_circular", args.use_circular, is_arg_specified('use_circular')))
    print(format_param("overlap", args.overlap, is_arg_specified('overlap')))
    print(format_param("diagnose_column", args.diagnose_column, is_arg_specified('diagnose_column')))
    print(format_param("diagnose_hidden_weights", args.diagnose_hidden_weights, is_arg_specified('diagnose_hidden_weights')))
    print(format_param("lateral_cooperation", args.lateral_cooperation, is_arg_specified('lateral_cooperation')))
    print(format_param("top_k_winners", args.top_k_winners, is_arg_specified('top_k_winners')))
    print(format_param("debug_lc", args.debug_lc, is_arg_specified('debug_lc')))
    print(format_param("column_lr_factors", column_lr_factors_list, is_arg_specified('column_lr_factors')))
    print(format_param("use_affinity", args.use_affinity, is_arg_specified('use_affinity')))
    print(format_param("affinity_max", args.affinity_max, is_arg_specified('affinity_max')))
    print(format_param("affinity_min", args.affinity_min, is_arg_specified('affinity_min')))
    print(format_param("rank_lut_mode", args.rank_lut_mode, is_arg_specified('rank_lut_mode')))
    print(format_param("column_decorrelation", args.column_decorrelation, is_arg_specified('column_decorrelation')))
    print(format_param("output_spatial_bias", args.output_spatial_bias, is_arg_specified('output_spatial_bias')))
    print(format_param("hebbian_alignment", args.hebbian_alignment, is_arg_specified('hebbian_alignment')))
    print(format_param("lateral_inhibition", args.lateral_inhibition, is_arg_specified('lateral_inhibition')))
    print(format_param("enable_non_column_learning", args.enable_non_column_learning, is_arg_specified('enable_non_column_learning')))
    print(format_param("nc_sparse_k", args.nc_sparse_k, is_arg_specified('nc_sparse_k')))
    print(format_param("nc_amine_strength", args.nc_amine_strength, is_arg_specified('nc_amine_strength')))
    print(format_param("nc_nearest_learning", args.nc_nearest_learning, is_arg_specified('nc_nearest_learning')))
    print(format_param("competitive_inhibition", args.competitive_inhibition, is_arg_specified('competitive_inhibition')))
    print(format_param("inhibition_strength", args.inhibition_strength, is_arg_specified('inhibition_strength')))
    print(format_param("inhibition_topk", args.inhibition_topk, is_arg_specified('inhibition_topk')))
    print(format_param("column_lateral_inhibition", args.column_lateral_inhibition, is_arg_specified('column_lateral_inhibition')))
    print(format_param("cli_alpha", args.cli_alpha, is_arg_specified('cli_alpha')))
    
    # [可視化関連のパラメータ]
    print("\n[可視化関連のパラメータ]")
    print(format_param("viz", args.viz, is_arg_specified('viz')))
    print(format_param("heatmap", args.heatmap, is_arg_specified('heatmap')))
    print(format_param("save_viz", args.save_viz, is_arg_specified('save_viz')))
    print(format_param("save_weights", args.save_weights, is_arg_specified('save_weights')))
    
    # [初期化関連のパラメータ]
    print("\n[初期化関連のパラメータ]")
    print(format_param("init_method", args.init_method, is_arg_specified('init_method')))
    print(format_param("init_limit", args.init_limit, is_arg_specified('init_limit')))
    print(format_param("flat_value", args.flat_value, is_arg_specified('flat_value')))
    print(format_param("flat_perturbation", args.flat_perturbation, is_arg_specified('flat_perturbation')))
    print(format_param("sparsity", args.sparsity, is_arg_specified('sparsity')))
    print(format_param("hidden_sparsity", args.hidden_sparsity, is_arg_specified('hidden_sparsity')))
    print(format_param("init_scales", init_scales, is_arg_specified('init_scales')))
    print(format_param("normalize_output_weights", args.normalize_output_weights, is_arg_specified('normalize_output_weights')))
    print(format_param("balance_output_weights", args.balance_output_weights, is_arg_specified('balance_output_weights')))
    
    # [早期停止関連のパラメータ]
    print("\n[早期停止関連のパラメータ]")
    print(format_param("early_stop_epoch", args.early_stop_epoch, is_arg_specified('early_stop_epoch')))
    print(format_param("early_stop_threshold", args.early_stop_threshold, is_arg_specified('early_stop_threshold')))
    
    # [動的シナプス刈り込み関連のパラメータ]
    # --dynamic_pruning_final_sparsity（または短縮形 --dynamic_pruning_fs）指定で有効化
    if args.dynamic_pruning_final_sparsity is not None:
        print("\n[動的シナプス刈り込み関連のパラメータ]")
        print(format_param("dynamic_pruning_final_sparsity", args.dynamic_pruning_final_sparsity, is_arg_specified('dynamic_pruning_final_sparsity')))
        print(format_param("pruning_start_epoch", args.pruning_start_epoch, is_arg_specified('pruning_start_epoch')))
        print(format_param("pruning_end_epoch", args.pruning_end_epoch, is_arg_specified('pruning_end_epoch')))
        print(format_param("stability_threshold", args.stability_threshold, is_arg_specified('stability_threshold')))
        print(format_param("pruning_verbose", args.pruning_verbose, is_arg_specified('pruning_verbose')))
    
    if args.augment:
        print("\n[データ拡張関連のパラメータ]")
        mode_str = "オンライン（エポック毎にランダム変換）" if args.aug_online else f"事前拡張（{args.aug_copies}コピー、計{args.aug_copies + 1}倍）"
        print(format_param("augment", True, is_arg_specified('augment')))
        print(format_param("aug_mode", mode_str, False))
        print(format_param("aug_shift", args.aug_shift, is_arg_specified('aug_shift')))
        print(format_param("aug_rotation", args.aug_rotation, is_arg_specified('aug_rotation')))
        print(format_param("aug_noise", args.aug_noise, is_arg_specified('aug_noise')))
        if not args.aug_online:
            print(format_param("aug_copies", args.aug_copies, is_arg_specified('aug_copies')))
    
    if args.gabor_features:
        print("\n[固定畳み込みフィルタ関連のパラメータ]")
        print(format_param("gabor_features", True, is_arg_specified('gabor_features')))
        print(format_param("gabor_orientations", args.gabor_orientations, is_arg_specified('gabor_orientations')))
        print(format_param("gabor_frequencies", args.gabor_frequencies, is_arg_specified('gabor_frequencies')))
        print(format_param("gabor_kernel_size", args.gabor_kernel_size, is_arg_specified('gabor_kernel_size')))
        print(format_param("gabor_pool_size", args.gabor_pool_size, is_arg_specified('gabor_pool_size')))
        print(format_param("gabor_pool_stride", args.gabor_pool_stride, is_arg_specified('gabor_pool_stride')))
        print(format_param("gabor_no_edge", args.gabor_no_edge, is_arg_specified('gabor_no_edge')))
    
    print("\n(*)が表示されている項目はコマンドラインオプションで指定された値が適用されています。")
    print("=" * 70 + "\n")
    
    # ========================================
    # 3. データ読み込み
    # ========================================
    # データセット名の取得（--datasetで指定、デフォルト: mnist）
    dataset = args.dataset
    
    # データセットパスの解決（標準データセット or カスタムデータ）
    dataset_path, is_custom = resolve_dataset_path(dataset)
    
    print(f"データ読み込み中... (訓練:{args.train}, テスト:{args.test}, データセット:{dataset})")
    
    # カスタムデータセットか標準データセットかで読み込み方法を切り替え
    custom_class_names = None
    custom_input_shape = None
    if is_custom:
        (x_train, y_train), (x_test, y_test), custom_class_names, custom_input_shape = load_custom_dataset(
            dataset_path=dataset_path, train_samples=args.train, test_samples=args.test
        )
    else:
        (x_train, y_train), (x_test, y_test) = load_dataset(
            dataset=dataset_path, train_samples=args.train, test_samples=args.test
        )
    
    # 入力次元とクラス数を自動検出
    n_input = x_train.shape[1]  # 自動検出: 784 (MNIST/Fashion), 3072 (CIFAR-10), etc.
    n_classes = len(np.unique(y_train))  # 自動検出: 10, 100, etc.
    
    print(f"データセット情報: 入力次元={n_input}, クラス数={n_classes}")
    
    # ========================================
    # 3.3 固定畳み込みフィルタによる特徴抽出
    # ========================================
    if args.gabor_features:
        from modules.gabor_features import GaborFeatureExtractor
        
        # 画像形状の推定
        if n_input == 784:
            img_shape = (28, 28)
        elif n_input == 3072:
            print("警告: CIFAR-10はチャンネルが3つあるため、グレースケール変換が必要です。Gabor特徴抽出をスキップします。")
            args.gabor_features = False
            img_shape = None
        else:
            side = int(np.sqrt(n_input))
            if side * side == n_input:
                img_shape = (side, side)
            else:
                print(f"警告: 入力次元{n_input}から画像形状を推定できません。Gabor特徴抽出をスキップします。")
                args.gabor_features = False
                img_shape = None
    
    if args.gabor_features:
        print(f"\nGaborフィルタ特徴抽出中... (方位:{args.gabor_orientations}, 周波数:{args.gabor_frequencies}, "
              f"カーネル:{args.gabor_kernel_size}, プール:{args.gabor_pool_size})")
        
        extractor = GaborFeatureExtractor(
            image_shape=img_shape,
            n_orientations=args.gabor_orientations,
            n_frequencies=args.gabor_frequencies,
            kernel_size=args.gabor_kernel_size,
            pool_size=args.gabor_pool_size,
            pool_stride=args.gabor_pool_stride,
            include_edge_filters=not args.gabor_no_edge
        )
        
        info = extractor.get_info()
        print(f"  フィルタ数: {info['n_filters']} (Gabor:{info['n_gabor_filters']}, エッジ:{info['n_edge_filters']})")
        print(f"  プーリング後空間サイズ: {info['pool_output_shape']}")
        print(f"  特徴次元: {info['feature_dim']} (元の入力: {n_input})")
        
        # ヒートマップ表示用にGabor変換前のテストデータを保存
        x_test_raw = x_test.copy()
        
        # 訓練データの特徴抽出
        x_train = extractor.transform(x_train)
        # テストデータの特徴抽出（訓練データの統計で正規化）
        x_test = extractor.transform_test(x_test)
        
        n_input = x_train.shape[1]  # 更新された入力次元
        print(f"  特徴抽出完了: 入力次元 {info['image_shape'][0]*info['image_shape'][1]} → {n_input}")
    
    # ========================================
    # 3.5 データ拡張（事前拡張モード）(v046)
    # ========================================
    if args.augment and not args.aug_online:
        print(f"\nデータ拡張中... (コピー数:{args.aug_copies}, シフト:{args.aug_shift}px, 回転:±{args.aug_rotation}°, ノイズ:{args.aug_noise})")
        # MNISTの画像形状を自動推定
        if n_input == 784:
            image_shape = (28, 28)
        elif n_input == 3072:
            image_shape = (32, 32, 3)  # CIFAR-10
        else:
            side = int(np.sqrt(n_input))
            image_shape = (side, side) if side * side == n_input else None
        
        if image_shape is not None:
            original_size = len(x_train)
            x_train, y_train = create_augmented_dataset(
                x_train, y_train,
                n_augmented=args.aug_copies,
                image_shape=image_shape,
                shift_range=args.aug_shift,
                rotation_range=args.aug_rotation,
                noise_std=args.aug_noise,
                seed=args.seed
            )
            print(f"データ拡張完了: {original_size} → {len(x_train)} サンプル ({len(x_train)/original_size:.1f}倍)")
        else:
            print(f"警告: 入力次元{n_input}から画像形状を推定できません。データ拡張をスキップします。")
    
    # クラス名の取得（標準データセット or カスタムデータセット）
    class_names = get_class_names(dataset, custom_class_names=custom_class_names)
    if class_names:
        print(f"クラス名: {class_names}")
    
    # ========================================
    # 4. ネットワークの構築
    # ========================================
    print("\nネットワーク初期化中...")
    
    network = RefinedDistributionEDNetwork(
        n_input=n_input,
        n_hidden=hidden_sizes,
        n_output=n_classes,
        learning_rate=args.lr,
        lateral_lr=args.lateral_lr,
        u1=args.u1,
        u2=args.u2,
        column_radius=args.column_radius,
        base_column_radius=args.base_column_radius,
        column_neurons=args.column_neurons,
        participation_rate=args.participation_rate,
        use_hexagonal=not args.use_circular,
        overlap=args.overlap,
        gradient_clip=args.gradient_clip,
        activation=args.activation,  # activationパラメータを追加
        lateral_cooperation=args.lateral_cooperation,  # v036新規: Phase A実証済み
        top_k_winners=args.top_k_winners,  # v036新規: Phase C実証済み
        hyperparams=None,  # HyperParamsの処理は既に完了しているのでNoneを渡す
        init_method=args.init_method,  # v038.1新規: 出力層重み初期化手法
        init_limit=args.init_limit,  # v036.5新規: シンプル一様分布の範囲
        sparsity=args.sparsity,  # v036.6新規: 出力層スパース率
        hidden_sparsity=hidden_sparsity_list,  # ★拡張★ 層別隠れ層（非コラム）スパース率
        column_lr_factors=column_lr_factors_list,  # ★拡張★ 層別コラム学習率係数
        use_affinity=args.use_affinity,  # v038.2新規: Affinity方式切り替え
        affinity_max=args.affinity_max,  # v038.2新規: コラムニューロンのaffinity値
        affinity_min=args.affinity_min,  # v038.2新規: 非コラムニューロンのaffinity値
        init_scales=init_scales,  # v040新規: 層別初期化スケール係数
        layer_learning_rates=layer_lrs,  # v039新規: 層別学習率
        flat_value=args.flat_value,  # v039.1新規: Flat初期化のベース値
        flat_perturbation=args.flat_perturbation,  # v039.1新規: Flat初期化の摂動幅
        normalize_output_weights=args.normalize_output_weights,  # v039.4新規: 出力層重み正規化
        balance_output_weights=args.balance_output_weights,  # v043.1新規: 出力層正負バランス調整
        uniform_amine=args.uniform_amine,  # ★提案B★ 全層均一アミン拡散（青斑核モデル）
        rf_overlap=args.rf_overlap,  # ★提案A★ 受容野オーバーラップ率
        rf_mode=args.rf_mode,  # ★提案A★ 受容野分割モード
        seed=args.seed,  # 受容野生成用シード
        rank_lut_mode=args.rank_lut_mode,  # ★提案D★ ランクベース学習率LUTモード
        amine_base_level=args.amine_base_level,  # Phase1: 全ニューロン学習復活
        column_decorrelation=args.column_decorrelation,  # ★v046★ コラム内脱相関
        amine_diffusion_sigma=args.amine_diffusion_sigma,  # ★v046★ 空間的アミン拡散
        output_spatial_bias=args.output_spatial_bias,  # ★Idea B★ 出力層空間バイアス
        hebbian_alignment=args.hebbian_alignment,  # ★Idea A★ Hebbian重み整列
        lateral_inhibition=args.lateral_inhibition,  # ★Idea D★ 側方抑制
        enable_non_column_learning=args.enable_non_column_learning,  # ★Step 1★ 非コラム学習解除
        nc_sparse_k=args.nc_sparse_k,  # ★活性化ゲート★ 非コラムの疎選択率
        nc_amine_strength=args.nc_amine_strength,  # ★活性化ゲート★ 選択NCのアミン強度
        nc_nearest_learning=args.nc_nearest_learning,  # ★NC最近傍帰属★ 最近傍クラスのみ学習
        competitive_inhibition=args.competitive_inhibition,  # ★競合抑制★ 不正解クラス選択的抑制
        inhibition_strength=args.inhibition_strength,  # ★競合抑制★ 抑制強度
        inhibition_topk=args.inhibition_topk,  # ★競合抑制★ 抑制対象クラス数
        column_lateral_inhibition=args.column_lateral_inhibition,  # ★活性化側方抑制★ コラム間活性化抑制
        cli_alpha=args.cli_alpha  # ★活性化側方抑制★ 抑制強度
    )
    
    # デバッグモード設定（v036.1で追加）
    if args.debug_lc:
        network.debug_lateral_cooperation = True
        print(f"\n[lateral_cooperation デバッグモード: 有効]")
        print(f"  - 影響度の詳細統計を収集します")
    
    # コラム構造の診断（オプション）
    if args.diagnose_column:
        network.diagnose_column_structure()
        print("\n診断完了。学習はスキップします。")
        import sys
        sys.exit(0)
    
    # 隠れ層の重み診断（オプション）- 学習前に実行して終了
    if args.diagnose_hidden_weights and args.epochs == 0:
        network.diagnose_hidden_weights()
        print("\n診断完了。学習はスキップします。")
        import sys
        sys.exit(0)
    
    # 初期重みを保存（重み診断用）
    initial_weights = None
    if args.diagnose_hidden_weights:
        initial_weights = [w.copy() for w in network.w_hidden]
        print("\n[重み診断モード: 有効]")
        print("  - 学習終了後に隠れ層の重み状態を診断します")
    
    # ========================================
    # 5. 可視化マネージャーの初期化
    # ========================================
    viz_manager = None
    if args.viz:
        try:
            viz_manager = VisualizationManager(
                enable_viz=True,
                enable_heatmap=args.heatmap,
                save_path=args.save_viz,
                total_epochs=args.epochs,
                verbose=getattr(args, 'verbose', False)
            )
            print("\n可視化機能: 有効")
            if args.heatmap:
                print("  - ヒートマップ表示: 有効")
            if args.save_viz:
                print(f"  - 保存先: {args.save_viz}")
            # Gabor特徴情報をヒートマップ表示用に設定
            if args.gabor_features:
                viz_manager.set_gabor_info(info)
        except Exception as e:
            print(f"\n警告: 可視化モジュールの初期化に失敗しました: {e}")
            print("可視化なしで学習を継続します。")
            viz_manager = None
    
    # ========================================
    # 5.5. 動的シナプス刈り込みの初期化
    # ========================================
    # --dynamic_pruning_final_sparsity（または --dynamic_pruning_fs）指定で有効化
    if args.dynamic_pruning_final_sparsity is not None:
        network.initialize_pruning(
            final_sparsity=args.dynamic_pruning_final_sparsity,
            total_epochs=args.epochs,
            start_epoch=args.pruning_start_epoch,
            end_epoch=args.pruning_end_epoch,
            verbose=args.pruning_verbose
        )
    
    # ========================================
    # 6. 学習ループ
    # ========================================
    print("\n" + "=" * 70)
    print("学習開始")
    print("=" * 70)
    
    from tqdm import tqdm
    
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    best_test_acc = 0.0
    best_epoch = 0
    train_acc_history = []
    test_acc_history = []
    
    # エポック毎のクラス別精度を記録
    class_acc_history = []  # shape: [n_epochs, n_classes]
    
    # 重み統計の記録用（--save_weightsオプション）
    weight_stats = {
        'epochs': [],
        'column_weights_mean': [],
        'non_column_weights_mean': [],
        'column_weights_std': [],
        'non_column_weights_std': [],
        'column_lr_factors': []
    }
    
    # ヒートマップ中間更新コールバック（約5秒ごとにヒートマップを更新）
    _heatmap_callback = None
    if viz_manager is not None and args.heatmap:
        def _make_heatmap_callback():
            """クロージャでviz_manager等を束縛"""
            _x_test = x_test
            _y_test_idx = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test
            _x_test_raw = x_test_raw if args.gabor_features else None
            _class_names = get_class_names(dataset)
            _vm = viz_manager
            _rng = np.random.RandomState(args.seed)
            _epoch_ref = [0]  # エポック番号を共有するための可変コンテナ
            
            def callback(net, sample_i, n_samples):
                idx = _rng.randint(0, len(_x_test))
                sx = _x_test[idx]
                sy_true = _y_test_idx[idx]
                z_h, z_o, _ = net.forward(sx)
                sy_pred = np.argmax(z_o)
                true_name = _class_names[sy_true] if _class_names else None
                pred_name = _class_names[sy_pred] if _class_names else None
                progress_str = f"{sample_i}/{n_samples}" if n_samples > 0 else f"{sample_i}"
                _vm.update_heatmap(
                    epoch=_epoch_ref[0],
                    sample_x=sx, sample_y_true=sy_true, sample_y_true_name=true_name,
                    z_hiddens=z_h, z_output=z_o,
                    sample_y_pred=sy_pred, sample_y_pred_name=pred_name,
                    sample_x_raw=_x_test_raw[idx] if _x_test_raw is not None else None,
                    progress=progress_str
                )
            
            def set_epoch(ep):
                _epoch_ref[0] = ep
            
            callback.set_epoch = set_epoch
            return callback
        _heatmap_callback = _make_heatmap_callback()
    
    # tqdmを使ったエポックループ
    pbar = tqdm(range(1, args.epochs + 1), desc="Training", ncols=120)
    for epoch in pbar:
        epoch_start = time.time()
        if _heatmap_callback is not None:
            _heatmap_callback.set_epoch(epoch)
        
        # ★v046新機能★ オンラインデータ拡張（エポック毎にランダム変換）
        if args.augment and args.aug_online:
            if n_input == 784:
                aug_image_shape = (28, 28)
            elif n_input == 3072:
                aug_image_shape = (32, 32, 3)
            else:
                side = int(np.sqrt(n_input))
                aug_image_shape = (side, side) if side * side == n_input else None
            
            if aug_image_shape is not None:
                # 元のデータを保持（初回のみ保存）
                if epoch == 1:
                    x_train_original = x_train.copy()
                    y_train_original = y_train.copy()
                
                # エポック毎に異なるseedで拡張
                x_train_aug = augment_batch(
                    x_train_original,
                    image_shape=aug_image_shape,
                    shift_range=args.aug_shift,
                    rotation_range=args.aug_rotation,
                    noise_std=args.aug_noise,
                    seed=args.seed + epoch * 1000
                )
                # 元データ + 拡張データを結合
                x_train = np.concatenate([x_train_original, x_train_aug], axis=0)
                y_train = np.concatenate([y_train_original, y_train_original], axis=0)
                
                # シャッフル
                shuffle_idx = np.random.RandomState(args.seed + epoch).permutation(len(x_train))
                x_train = x_train[shuffle_idx]
                y_train = y_train[shuffle_idx]
        
        # ★v039.3新機能★ エポック開始時に勝者選択統計をリセット
        network.reset_winner_selection_stats()
        
        # エポック開始時にクラス学習回数統計をリセット
        network.reset_class_training_stats()
        
        # エポック依存学習率スケーリングを適用（Epoch 0を0始まりに変換）
        if epoch_lr_schedule is not None:
            network.set_epoch_lr_scale(epoch - 1, epoch_lr_schedule)
        
        # 訓練（ミニバッチ or オンライン学習）
        if args.batch_size is not None:
            if args.use_cupy:
                # ★CuPy版★ GPU最適化バッチ学習（期待: 2-3倍高速化）
                train_acc, train_loss = network.train_epoch_cupy_batch(
                    x_train, y_train, batch_size=args.batch_size,
                    progress_callback=_heatmap_callback
                )
            else:
                # ミニバッチ学習（TensorFlow Dataset使用、勾配平均化方式）
                from modules.data_loader import create_tf_dataset
                train_dataset = create_tf_dataset(
                    x_train, y_train,
                    batch_size=args.batch_size,
                    shuffle=True,
                    seed=args.seed + epoch  # エポックごとに異なるシャッフル
                )
                train_acc, train_loss = network.train_epoch_minibatch_tf(train_dataset, progress_callback=_heatmap_callback)
        else:
            # オンライン学習（従来方式）
            train_acc, train_loss = network.train_epoch(x_train, y_train, progress_callback=_heatmap_callback)
        
        # ★v039.3新機能★ エポック終了後に勝者選択統計を取得
        winner_stats = network.get_winner_selection_stats()
        
        # テスト（クラス別精度も取得）★P2最適化★ 並列評価で高速化
        test_acc, test_loss, class_accs = network.evaluate_parallel(x_test, y_test, return_per_class=True)
        
        # 履歴記録
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        class_acc_history.append(class_accs)
        
        epoch_time = time.time() - epoch_start
        
        # ベストモデル更新
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
        
        # 進捗表示
        pbar.set_postfix({
            'Train': f'{train_acc:.4f}',
            'Test': f'{test_acc:.4f}',
            'Best': f'{best_test_acc:.4f}',
            'Time': f'{epoch_time:.1f}s'
        })
        
        # 詳細出力（エポックごと）
        print(f"Epoch {epoch:3d}/{args.epochs}: "
              f"Train={train_acc:.4f} (loss={train_loss:.4f}), "
              f"Test={test_acc:.4f} (loss={test_loss:.4f}), "
              f"Time={epoch_time:.2f}s")
        
        # ★v039.3新機能★ 最初の5エポックは勝者選択統計を詳細出力（デバッグ用）
        if epoch <= 5:
            print(f"\n[勝者選択頻度 - Epoch {epoch}]")
            print(f"  総学習サンプル数: {winner_stats['total_samples']}")
            print(f"  期待値（均等分布）: {winner_stats['expected_percentage']:.1f}%")
            print(f"\n  Class  Count     Percentage")
            print(f"  --------------------------------")
            for class_idx in range(network.n_output):
                count = winner_stats['counts'][class_idx]
                pct = winner_stats['percentages'][class_idx]
                # 期待値との差分を計算
                diff = pct - winner_stats['expected_percentage']
                diff_str = f"{diff:+.1f}%" if diff != 0 else "±0.0%"
                marker = "⚠️" if pct < 5.0 else "✅" if pct > 15.0 else ""
                print(f"  {class_idx:5d}  {count:5d}     {pct:6.2f}% ({diff_str}) {marker}")
            
            # クラス別学習実行回数も表示（純粋ED法: 正解クラスのみ学習）
            training_stats = network.get_class_training_stats()
            print(f"\n[クラス別学習実行回数 - Epoch {epoch}] (純粋ED法)")
            print(f"  モード: 正解クラスのみ学習")
            print(f"  総学習実行回数: {training_stats['total_samples']}")
            print(f"\n  Class  Count      Percentage")
            print(f"  --------------------------------")
            for class_idx in range(network.n_output):
                count = training_stats['counts'][class_idx]
                pct = training_stats['percentages'][class_idx]
                print(f"  {class_idx:5d}  {count:6d}     {pct:6.2f}%")
            
            # ★デバッグ追加★ Class 5の出力層重み統計
            if hasattr(network, 'w_output'):
                class5_weights = network.w_output[5, :]
                print(f"\n[Class 5 出力層重み統計 - Epoch {epoch}]")
                print(f"  min={class5_weights.min():.6f}, max={class5_weights.max():.6f}")
                print(f"  mean={class5_weights.mean():.6f}, std={class5_weights.std():.6f}")
                print(f"  abs_mean={np.abs(class5_weights).mean():.6f}")
                # 全クラスとの比較
                all_class_abs_means = [np.abs(network.w_output[c, :]).mean() for c in range(network.n_output)]
                class5_rank = sorted(all_class_abs_means, reverse=True).index(all_class_abs_means[5]) + 1
                print(f"  全クラス中の|重み|平均ランク: {class5_rank}/10")
        
        # ★v038.2新規★ 早期停止判定
        if args.early_stop_epoch is not None and epoch == args.early_stop_epoch:
            if test_acc <= args.early_stop_threshold:
                print(f"\n早期停止: Epoch {epoch}でTest精度={test_acc:.4f} <= {args.early_stop_threshold:.4f}")
                print(f"学習が成立していないため終了します。")
                break
        
        # ★v042新機能★ 動的シナプス刈り込み
        if args.dynamic_pruning_final_sparsity is not None:
            # 安定期検出をチェック（または固定開始エポック）
            if network.check_stability_for_pruning(test_acc, epoch - 1, threshold=args.stability_threshold):
                # 刈り込みを実行
                pruning_stats = network.apply_pruning(epoch - 1, verbose=args.pruning_verbose)
                if pruning_stats is not None and not args.pruning_verbose:
                    # 非詳細モードでもサマリーを1行で表示
                    total_pruned = sum(pruning_stats['total_pruned'])
                    total_conn = sum(network.pruning_stats['initial_connections'])
                    sparsity = total_pruned / total_conn if total_conn > 0 else 0
                    print(f"  [刈り込み] {total_pruned:,d}/{total_conn:,d} 刈り込み済み ({sparsity*100:.1f}%)")
        
        # ★Idea A★ Hebbian Weight Alignment（エポック後に実行）
        if args.hebbian_alignment > 0.0:
            hebb_stats = network.apply_hebbian_alignment()
            if hebb_stats:
                for lidx, st in hebb_stats.items():
                    print(f"  [Hebbian] Layer {lidx}: aligned={st['n_aligned']}, "
                          f"mean_drift={st['mean_drift']:.6f}")
        
        # ★Step 2★ 非コラムニューロン学習動態デバッグ出力
        if args.enable_non_column_learning:
            debug_info = network.get_non_column_debug_info()
            for layer_key, stats in debug_info.items():
                print(f"  [{layer_key}] "
                      f"ColNorm={stats['col_weight_norm'][0]:.4f}±{stats['col_weight_norm'][1]:.4f} "
                      f"NC_Norm={stats['nc_weight_norm'][0]:.4f}±{stats['nc_weight_norm'][1]:.4f} | "
                      f"ColΔ={stats['col_weight_delta'][0]:.6f} "
                      f"NC_Δ={stats['nc_weight_delta'][0]:.6f} | "
                      f"ColOutΔ={stats['col_out_weight_delta'][0]:.6f} "
                      f"NC_OutΔ={stats['nc_out_weight_delta'][0]:.6f} | "
                      f"ColOutContrib={stats['col_output_contrib'][0]:.4f} "
                      f"NC_OutContrib={stats['nc_output_contrib'][0]:.4f}")
        
        # 重み統計の記録
        if args.save_weights:
            # 出力層の重みからコラム/非コラムニューロンの統計を計算
            # 多層対応: 出力層に接続されている最後の隠れ層のmembershipを使用
            last_layer_idx = len(network.column_membership_all_layers) - 1
            membership = network.column_membership_all_layers[last_layer_idx]
            w_out_abs = np.abs(network.w_output)
            
            # 各クラスのコラムニューロンの平均重み
            column_weights = []
            non_column_weights = []
            for class_idx in range(network.n_output):
                member_mask = membership[class_idx]
                non_member_mask = ~member_mask
                
                # コラムニューロンの重み
                if np.any(member_mask):
                    column_weights.extend(w_out_abs[class_idx, member_mask])
                
                # 非コラムニューロンの重み
                if np.any(non_member_mask):
                    non_column_weights.extend(w_out_abs[class_idx, non_member_mask])
            
            weight_stats['epochs'].append(epoch)
            weight_stats['column_weights_mean'].append(np.mean(column_weights) if column_weights else 0)
            weight_stats['non_column_weights_mean'].append(np.mean(non_column_weights) if non_column_weights else 0)
            weight_stats['column_weights_std'].append(np.std(column_weights) if column_weights else 0)
            weight_stats['non_column_weights_std'].append(np.std(non_column_weights) if non_column_weights else 0)
            weight_stats['column_lr_factors'].append(network.column_lr_factors[-1])
        
        # 可視化更新
        if viz_manager is not None:
            # 正解ラベルの形状判定（one-hot → インデックスに変換）
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_test_indices = np.argmax(y_test, axis=1)
            else:
                y_test_indices = y_test
            
            # 学習曲線の更新
            viz_manager.update_learning_curve(
                train_acc_history,
                test_acc_history,
                x_test,
                y_test_indices,
                network
            )
            
            # ヒートマップ更新（サンプルを1つ選択）
            if args.heatmap and epoch % 1 == 0:
                sample_idx = np.random.randint(0, len(x_test))
                sample_x = x_test[sample_idx]
                sample_y_true = y_test_indices[sample_idx]
                
                # ネットワークのforward()を使用して正確な予測を取得
                z_hiddens, z_output, _ = network.forward(sample_x)
                sample_y_pred = np.argmax(z_output)
                
                # クラス名の取得
                class_names = get_class_names(dataset)
                
                if class_names is not None:
                    true_class_name = class_names[sample_y_true]
                    pred_class_name = class_names[sample_y_pred]
                else:
                    true_class_name = None
                    pred_class_name = None
                
                viz_manager.update_heatmap(
                    epoch=epoch,
                    sample_x=sample_x,
                    sample_y_true=sample_y_true,
                    sample_y_true_name=true_class_name,
                    z_hiddens=z_hiddens,
                    z_output=z_output,
                    sample_y_pred=sample_y_pred,
                    sample_y_pred_name=pred_class_name,
                    sample_x_raw=x_test_raw[sample_idx] if args.gabor_features else None
                )
    
    # ========================================
    # 7. 結果サマリー
    # ========================================
    print("\n" + "=" * 70)
    print("学習完了")
    print("=" * 70)
    print(f"最終精度: Train={train_acc:.4f}, Test={test_acc:.4f}")
    print(f"ベスト精度: Test={best_test_acc:.4f} (Epoch {best_epoch})")
    
    # ★動的プルーニングのサマリー★
    if args.dynamic_pruning_final_sparsity is not None:
        pruning_summary = network.get_pruning_summary()
        if pruning_summary is not None:
            print("\n" + "-" * 50)
            print("動的シナプス刈り込み結果")
            print("-" * 50)
            print(f"  開始エポック:     {pruning_summary['start_epoch'] if pruning_summary['start_epoch'] is not None else '未開始'}")
            print(f"  安定期検出:       {'自動検出' if pruning_summary['stable_detected'] else '固定エポック指定'}")
            print(f"  目標スパース率:   {pruning_summary['final_sparsity_target']*100:.1f}%")
            print()
            for layer_stat in pruning_summary['layer_stats']:
                print(f"  {layer_stat['layer']:>10}: {layer_stat['pruned_connections']:,d}/{layer_stat['initial_connections']:,d} "
                      f"刈り込み済み ({layer_stat['current_sparsity']*100:.1f}%)")
    
    # エポック毎のクラス別精度を表示
    print("\n" + "=" * 70)
    print("エポック毎のクラス別テスト正解率")
    print("=" * 70)
    
    # ヘッダー
    print(f"{'Epoch':>5}", end="")
    for c in range(n_classes):
        print(f"  Class{c:1d}", end="")
    print(f"  {'Average':>7}")
    print("-" * (5 + n_classes * 9 + 10))
    
    # 各エポックの結果
    for epoch_idx, class_accs in enumerate(class_acc_history):
        epoch_num = epoch_idx + 1
        print(f"{epoch_num:>5}", end="")
        for acc in class_accs:
            print(f"  {acc*100:>6.2f}%", end="")
        avg_acc = np.mean(class_accs)
        print(f"  {avg_acc*100:>6.2f}%")
    
    # 混同行列の計算と表示（v037追加）
    if args.heatmap:
        print("\n" + "=" * 70)
        print("混同行列（テストデータ）")
        print("=" * 70)
        
        # テストデータで予測
        y_pred = []
        y_true = []
        for i in range(len(x_test)):
            _, z_output, _ = network.forward(x_test[i])
            pred_class = np.argmax(z_output)
            # y_testがone-hotの場合はargmax、そうでない場合はそのまま
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                true_class = np.argmax(y_test[i])
            else:
                true_class = y_test[i]
            y_pred.append(pred_class)
            y_true.append(true_class)
        
        # 混同行列を計算
        cm = confusion_matrix(y_true, y_pred)
        
        # 混同行列を表形式で表示
        print("\n     ", end="")
        for i in range(n_classes):
            print(f"{i:>6}", end="")
        print()
        print("     " + "-" * (n_classes * 6))
        
        for i in range(n_classes):
            print(f"{i:>3} |", end="")
            for j in range(n_classes):
                print(f"{cm[i, j]:>6}", end="")
            print(f"  (正解率: {cm[i, i]/cm[i].sum()*100:.1f}%)")
        
        # クラス別正解率を分析
        print("\n" + "-" * 70)
        print("クラス別正解率分析")
        print("-" * 70)
        class_accuracies = []
        for i in range(n_classes):
            class_acc = cm[i, i] / cm[i].sum()
            class_accuracies.append(class_acc)
            print(f"クラス{i}: {cm[i, i]:>3}/{cm[i].sum():>3} = {class_acc*100:>5.1f}%")
        
        # 最も低い正解率のクラスを特定
        min_class = np.argmin(class_accuracies)
        max_class = np.argmax(class_accuracies)
        print(f"\n最低正解率: クラス{min_class} ({class_accuracies[min_class]*100:.1f}%)")
        print(f"最高正解率: クラス{max_class} ({class_accuracies[max_class]*100:.1f}%)")
        print(f"正解率範囲: {(class_accuracies[max_class] - class_accuracies[min_class])*100:.1f}ポイント")
    
    # 隠れ層の重み診断（学習後）
    if args.diagnose_hidden_weights:
        print("\n" + "=" * 70)
        print("隠れ層の重み診断（学習後）")
        print("=" * 70)
        network.diagnose_hidden_weights(initial_weights=initial_weights)
    
    # 重み統計と実際の重み配列の保存
    if args.save_weights:
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 統計値の保存
        weight_stats_file = os.path.join('logs/v038', f'weight_stats_{timestamp}.npz')
        os.makedirs(os.path.dirname(weight_stats_file), exist_ok=True)
        np.savez(weight_stats_file,
                 epochs=np.array(weight_stats['epochs']),
                 column_weights_mean=np.array(weight_stats['column_weights_mean']),
                 non_column_weights_mean=np.array(weight_stats['non_column_weights_mean']),
                 column_weights_std=np.array(weight_stats['column_weights_std']),
                 non_column_weights_std=np.array(weight_stats['non_column_weights_std']),
                 column_lr_factors=np.array(weight_stats['column_lr_factors']))
        print(f"\n✅ 重み統計を保存しました: {weight_stats_file}")
        
        # 実際の重み配列の保存（可視化用）
        weights_file = os.path.join('logs/v038', f'weights_array_{timestamp}.npz')
        weight_dict: dict[str, np.ndarray] = {'w_output': network.w_output}
        for i, w in enumerate(network.w_hidden):
            weight_dict[f'w_hidden_{i}'] = w
        np.savez(weights_file, **weight_dict)  # type: ignore[arg-type]
        print(f"✅ 重み配列を保存しました: {weights_file}")
    
    # デバッグ統計の表示（v036.1で追加）
    if args.debug_lc:
        network.print_lateral_cooperation_stats()
    
    # 可視化の最終処理
    if viz_manager is not None:
        viz_manager.save_figures()
        if args.save_viz:
            print(f"\n可視化結果を保存しました: {args.save_viz}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
