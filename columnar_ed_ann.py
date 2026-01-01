#!/usr/bin/env python3
"""
コラムED法
columnar_ed_ann.py version: 1.30.1
"""

import os
# TensorFlowのログメッセージを抑制（情報・警告を非表示）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import time

# モジュールインポート
from modules.hyperparameters import HyperParams
from modules.data_loader import load_dataset, create_tf_dataset, get_class_names
from modules.ed_network import RefinedDistributionEDNetwork
from modules.visualization_manager import VisualizationManager


def parse_weight_init_scales(wis_str):
    """
    重み初期化係数の文字列をパース
    
    記法:
        "2.25, 2.75, 12.00"           -> [2.25, 2.75, 12.00]
        "2.25, 3.0[9], 12.00"         -> [2.25, 3.0, 3.0, ..., 3.0, 12.00] (3.0が9個)
        "2.0, 3.0[5], 3.5[3], 12.00"  -> [2.0, 3.0×5, 3.5×3, 12.00]
    
    Args:
        wis_str: 重み初期化係数の文字列
    
    Returns:
        list[float]: パース済みの係数リスト
    
    Raises:
        ValueError: 不正な形式の場合
    """
    import re
    
    scales = []
    items = [item.strip() for item in wis_str.split(',')]
    
    for item in items:
        # 繰り返し記法のチェック: "3.0[10]"
        match = re.match(r'^([\d.]+)\[(\d+)\]$', item)
        if match:
            value = float(match.group(1))
            count = int(match.group(2))
            if count <= 0:
                raise ValueError(f"繰り返し回数は1以上である必要があります: {item}")
            scales.extend([value] * count)
        else:
            # 通常の値
            try:
                scales.append(float(item))
            except ValueError:
                raise ValueError(f"不正な値の形式です: '{item}'\n"
                               f"正しい形式: '2.25' または '3.0[10]' (値[繰り返し回数])")
    
    return scales


def analyze_affinity_distribution(network):
    """
    Affinity分布を解析する
    
    Args:
        network: ネットワークインスタンス
    
    Returns:
        dict: Affinity分布情報
    """
    print("\n" + "=" * 70)
    print("Affinity分布解析")
    print("=" * 70)
    
    for layer_idx, affinity in enumerate(network.column_affinity_all_layers):
        print(f"\n[Layer {layer_idx + 1}] 形状: {affinity.shape}")
        
        # 全体統計
        print(f"  全体統計:")
        print(f"    最小値: {np.min(affinity):.6e}")
        print(f"    最大値: {np.max(affinity):.6e}")
        print(f"    平均値: {np.mean(affinity):.6e}")
        print(f"    中央値: {np.median(affinity):.6e}")
        print(f"    標準偏差: {np.std(affinity):.6e}")
        
        # 非ゼロ要素の統計
        non_zero_affinity = affinity[affinity > 0]
        if len(non_zero_affinity) > 0:
            print(f"  非ゼロ要素 ({len(non_zero_affinity)}個):")
            print(f"    最小値: {np.min(non_zero_affinity):.6e}")
            print(f"    最大値: {np.max(non_zero_affinity):.6e}")
            print(f"    平均値: {np.mean(non_zero_affinity):.6e}")
        
        # 閾値別のニューロン数カウント
        thresholds = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5]
        print(f"  閾値別のニューロン数（各クラスの平均）:")
        for i in range(len(thresholds) - 1):
            low, high = thresholds[i], thresholds[i+1]
            count_per_class = []
            for class_idx in range(affinity.shape[0]):
                count = np.sum((affinity[class_idx] > low) & (affinity[class_idx] <= high))
                count_per_class.append(count)
            avg_count = np.mean(count_per_class)
            print(f"    {low:.0e} < affinity ≤ {high:.0e}: {avg_count:.1f}個/クラス")
        
        # 高affinity (> 1e-8) のニューロン数
        high_affinity_count = np.sum(np.max(affinity, axis=0) > 1e-8)
        total_neurons = affinity.shape[1]
        print(f"  高affinity（>1e-8）を持つニューロン: {high_affinity_count}/{total_neurons} ({high_affinity_count/total_neurons*100:.1f}%)")
        
        # クラス別の非ゼロニューロン数
        print(f"  クラス別の非ゼロaffinity持ちニューロン数:")
        for class_idx in range(min(affinity.shape[0], 10)):  # 最大10クラス表示
            count = np.sum(affinity[class_idx] > 0)
            print(f"    クラス{class_idx}: {count}個")
    
    print("=" * 70)


def analyze_dead_neurons(network, x_data, y_data, epoch_label=""):
    """
    デッドニューロンを解析する
    
    Args:
        network: ネットワークインスタンス
        x_data: 入力データ
        y_data: ラベルデータ
        epoch_label: エポックラベル（表示用）
    
    Returns:
        dict: デッドニューロン情報
            - 'dead_counts': 各層のデッドニューロン数のリスト
            - 'total_counts': 各層の総ニューロン数のリスト
            - 'dead_ratios': 各層のデッドニューロン率のリスト
            - 'summary': サマリー文字列
    """
    from collections import defaultdict
    
    # 各層のニューロン活性化カウント
    activation_counts = [defaultdict(int) for _ in range(len(network.n_hidden))]
    
    # 全データでforward実行
    for x, y in zip(x_data, y_data):
        z_hiddens, z_output, _ = network.forward(x)
        
        # 各層の勝者ニューロンを記録
        for layer_idx, z_hidden in enumerate(z_hiddens):
            # k-winner selection（上位k個を選択）
            if hasattr(network, 'k_winners') and network.k_winners > 0:
                k = min(network.k_winners, len(z_hidden))
                top_k_indices = np.argsort(z_hidden)[-k:]
                for idx in top_k_indices:
                    activation_counts[layer_idx][idx] += 1
            else:
                # k-winner未指定の場合は、正の活性化を持つ全ニューロン
                for idx, activation in enumerate(z_hidden):
                    if activation > 0:
                        activation_counts[layer_idx][idx] += 1
    
    # デッドニューロンのカウント
    dead_counts = []
    total_counts = []
    dead_ratios = []
    
    for layer_idx, n_neurons in enumerate(network.n_hidden):
        active_neurons = len(activation_counts[layer_idx])
        dead_neurons = n_neurons - active_neurons
        dead_ratio = dead_neurons / n_neurons if n_neurons > 0 else 0.0
        
        dead_counts.append(dead_neurons)
        total_counts.append(n_neurons)
        dead_ratios.append(dead_ratio)
    
    # サマリー文字列の生成
    summary_parts = []
    for layer_idx, (dead, total, ratio) in enumerate(zip(dead_counts, total_counts, dead_ratios)):
        summary_parts.append(f"L{layer_idx+1}:{dead}/{total}")
    
    summary = f"Dead={', '.join(summary_parts)}"
    
    return {
        'dead_counts': dead_counts,
        'total_counts': total_counts,
        'dead_ratios': dead_ratios,
        'summary': summary,
        'epoch_label': epoch_label
    }


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
    exec_group.add_argument('--epochs', type=int, default=None,
                           help='エポック数（デフォルト値: 層数に応じた自動設定）')
    exec_group.add_argument('--seed', type=int, default=42,
                           help='乱数シード（デフォルト値: 42、再現性確保用）')
    exec_group.add_argument('--fashion', action='store_true',
                           help='Fashion-MNISTを使用（後方互換性のため残存、--dataset fashionを推奨）')
    exec_group.add_argument('--dataset', type=str, default=None,
                           help='データセット名（mnist, fashion, cifar10, cifar100）または '
                                'カスタムデータセットのパス。'
                                'カスタムデータは~/.keras/datasets/配下に配置するか、絶対パスで指定。'
                                '詳細はCUSTOM_DATASET_GUIDE.mdを参照。')
    
    # ========================================
    # ED法関連のパラメータ
    # ========================================
    ed_group = parser.add_argument_group('ED法関連のパラメータ')
    ed_group.add_argument('--hidden', type=str, default='512',
                         help='隠れ層ニューロン数（例: 512=1層, 256,128=2層）（デフォルト値: 512）')
    ed_group.add_argument('--activation', type=str, default='tanh',
                         choices=['tanh', 'sigmoid', 'leaky_relu', 'clipped_leaky_relu', 'shifted_sigmoid', 'clipped_identity'],
                         help='活性化関数（デフォルト: tanh）※グリッドサーチ用、将来的に削除予定')
    ed_group.add_argument('--leaky-alpha', type=float, default=0.1,
                         help='Leaky ReLUの負勾配係数（デフォルト: 0.1、推奨値: 0.01-0.2）')
    ed_group.add_argument('--lr', type=float, default=None,
                         help='学習率（層数により自動設定: 1層=0.20, 2層=0.25、明示指定で上書き）')
    ed_group.add_argument('--u1', type=float, default=None,
                         help='アミン拡散係数u1（出力層→最終隠れ層、層数により自動設定: 1層=0.5, 2層=0.5、明示指定で上書き）')
    ed_group.add_argument('--u2', type=float, default=None,
                         help='アミン拡散係数u2（隠れ層間、層数により自動設定: 1層=0.8, 2層=0.8、明示指定で上書き）')
    ed_group.add_argument('--lateral_lr', type=float, default=None,
                         help='側方抑制の学習率（層数により自動設定: 1層=0.08, 2層=0.08、明示指定で上書き）')
    ed_group.add_argument('--gradient_clip', type=float, default=0.05,
                         help='gradient clipping値（デフォルト値: 0.05）')
    ed_group.add_argument('--batch', type=int, default=None,
                         help='ミニバッチサイズ（未指定=オンライン学習、32/128推奨）')
    ed_group.add_argument('--shuffle', action='store_true',
                         help='データをシャッフル（TensorFlow Dataset API使用、オンライン/ミニバッチ両対応）')
    
    # ========================================
    # コラム関連のパラメータ
    # ========================================
    column_group = parser.add_argument_group('コラム関連のパラメータ')
    column_group.add_argument('--list_hyperparams', action='store_true',
                             help='利用可能なHyperParams設定一覧を表示して終了')
    column_group.add_argument('--column_radius', type=float, default=None,
                             help='コラム半径（層数により自動設定、明示指定で上書き）')
    column_group.add_argument('--participation_rate', type=float, default=None,
                             help='コラム参加率（層数により自動設定、明示指定で上書き）')
    column_group.add_argument('--column_neurons', type=int, default=None,
                             help='各クラスの明示的ニューロン数（デフォルト値: None、重複許容、優先度：中）')
    column_group.add_argument('--use_circular', action='store_true',
                             help='旧円環構造を使用（デフォルトはハニカム）')
    column_group.add_argument('--overlap', type=float, default=0.0,
                             help='コラム間の重複度（デフォルト値: 0.0、0.0-1.0、円環構造でのみ有効、0.0=重複なし）')
    column_group.add_argument('--diagnose_column', action='store_true',
                             help='コラム構造の詳細診断を実行')
    column_group.add_argument('--wis', '--weight_init_scales', type=str, default=None,
                             dest='weight_init_scales',
                             help='重み初期化係数。隠れ層と出力層分を合わせて指定する。\n'
                                  'カンマ区切り、繰り返しは値[回数]で指定。\n'
                                  '例: 2.25,2.75,12.00 (隠れ層2層と出力層)\n'
                                  '    2.25,3.0[9],12.00 (隠れ層10層と出力層。Layer1-9を3.0に設定)\n'
                                  '    2.0,3.0[99],12.00 (隠れ層100層と出力層)\n'
                                  'デフォルト: HyperParamsテーブルから自動取得')
    
    # ========================================
    # 可視化関連のパラメータ
    # ========================================
    viz_group = parser.add_argument_group('可視化関連のパラメータ')
    viz_group.add_argument('--viz', action='store_true',
                          help='学習曲線のリアルタイム可視化を有効化')
    viz_group.add_argument('--heatmap', action='store_true',
                          help='活性化ヒートマップの表示を有効化（--vizと併用）')
    viz_group.add_argument('--save_viz', type=str, nargs='?', const='viz_results/',
                          default=None, metavar='PATH',
                          help='可視化結果を保存。'
                               'パス指定: 末尾"/"でディレクトリ（タイムスタンプ付き）、末尾"/"なしでベースファイル名。'
                               '例: results/ → results/viz_results_20251221_153045.png、'
                               'results/exp1 → results/exp1.png。'
                               '学習曲線とヒートマップを同時に保存する場合は、_viz.pngと_heatmap.pngが付加されます。'
                               '引数なし: viz_results/にタイムスタンプ付きで保存。'
                               'オプション未指定: 保存しない。')
    
    return parser.parse_args()


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
    
    # HyperParams設定一覧の表示
    if args.list_hyperparams:
        hp = HyperParams()
        hp.list_configs()
        import sys
        sys.exit(0)
    
    # ========================================
    # 1. 隠れ層のパース（カンマ区切り対応、多層対応）
    # ========================================
    if ',' in args.hidden:
        hidden_sizes = [int(x.strip()) for x in args.hidden.split(',')]
    else:
        hidden_sizes = [int(args.hidden)]
    
    # ========================================
    # 2. HyperParamsテーブルからの自動パラメータ取得
    # ========================================
    hp = HyperParams()
    n_layers = len(hidden_sizes)
    
    try:
        config = hp.get_config(n_layers)
        
        # コマンドラインで明示されていないパラメータ（None）のみHyperParamsテーブルの値で上書き
        # hidden_sizesは常にテーブルの値を使用（層数はユーザーが指定したものを尊重）
        if args.hidden == '512':  # デフォルト値の場合のみテーブルの構成を使用
            hidden_sizes = config['hidden']
        
        # 各パラメータ: Noneの場合のみテーブル値を適用（明示指定された値は尊重）
        if args.lr is None:
            args.lr = config['learning_rate']
        if args.u1 is None and 'u1' in config:
            args.u1 = config['u1']
        if args.u2 is None and 'u2' in config:
            args.u2 = config['u2']
        if args.lateral_lr is None and 'lateral_lr' in config:
            args.lateral_lr = config['lateral_lr']
        if args.column_radius is None:
            args.column_radius = config['column_radius']
        if args.participation_rate is None and 'participation_rate' in config:
            args.participation_rate = config['participation_rate']
        if args.epochs is None:
            args.epochs = config['epochs']
        
        # 適用後の実際の値を表示
        print(f"\n=== 層数に基づくHyperParams設定を自動適用（{n_layers}層） ===")
        print("*** コマンドライン引数で明示的に指定された値は、テーブル値より優先されます。")
        print(f"hidden_layers: {hidden_sizes}")
        print(f"learning_rate: {args.lr}")
        print(f"u1: {args.u1}")
        print(f"u2: {args.u2}")
        print(f"lateral_lr: {args.lateral_lr}")
        print(f"column_radius: {args.column_radius}")
        print(f"participation_rate: {args.participation_rate}")
        print(f"epochs: {args.epochs}")
        
        print("="*70 + "\n")
    except ValueError as e:
        print(f"Warning: HyperParamsテーブルの取得に失敗: {e}")
        print("個別パラメータで継続します。\n")
    
    # ========================================
    # 2.5 重み初期化係数の解決
    # ========================================
    weight_init_scales = None
    weight_init_source = None
    
    if args.weight_init_scales:
        # コマンドライン引数で指定された場合
        try:
            weight_init_scales = parse_weight_init_scales(args.weight_init_scales)
            
            # 層数との整合性チェック
            expected_count = n_layers + 1  # 隠れ層 + 出力層
            actual_count = len(weight_init_scales)
            if actual_count != expected_count:
                print("\n" + "="*70)
                print("エラー: --hiddenと--wisの層数が一致しません")
                print("="*70)
                print(f"--hidden指定: {args.hidden}")
                print(f"  → 隠れ層数: {n_layers}層 + 出力層 = 合計{expected_count}個の係数が必要")
                print(f"\n--wis指定: {args.weight_init_scales}")
                print(f"  → パース結果: {weight_init_scales}")
                print(f"  → 係数の個数: {actual_count}個")
                print(f"\n不足/過剰: {actual_count - expected_count:+d}個")
                print("="*70 + "\n")
                raise ValueError(
                    f"層数不一致: {expected_count}個の係数が必要ですが、{actual_count}個指定されました"
                )
            
            weight_init_source = "CLI"
            print(f"[重み初期化係数] コマンドライン引数から取得: {weight_init_scales}")
        except ValueError as e:
            print(f"エラー: --wis引数のパースに失敗しました: {e}")
            print("HyperParamsテーブルの値を使用します。\n")
            args.weight_init_scales = None  # エラー時はHyperParams使用
    
    if weight_init_scales is None:
        # HyperParamsテーブルから取得
        try:
            config = hp.get_config(n_layers)
            weight_init_scales = config.get('weight_init_scales', None)
            if weight_init_scales:
                weight_init_source = "HyperParams"
                print(f"[重み初期化係数] HyperParamsテーブルから取得: {weight_init_scales}")
            else:
                weight_init_source = "デフォルト値"
                print(f"[重み初期化係数] HyperParamsに未設定のため、デフォルト値を使用")
        except:
            weight_init_source = "デフォルト値"
            print(f"[重み初期化係数] HyperParams取得失敗、デフォルト値を使用")
    
    # ========================================
    # 3. データ読み込み
    # ========================================
    # データセット名の解決（優先順位: --dataset > --fashion > デフォルト）
    if args.dataset is not None:
        dataset = args.dataset
    elif args.fashion:
        dataset = 'fashion'
    else:
        dataset = 'mnist'
    
    # データセットパスの解決（標準データセット or カスタムデータ）
    from modules.data_loader import resolve_dataset_path, load_custom_dataset
    dataset_path, is_custom = resolve_dataset_path(dataset)
    
    print(f"データ読み込み中... (訓練:{args.train}, テスト:{args.test}, データセット:{dataset})")
    
    # カスタムデータセットか標準データセットかで読み込み方法を切り替え
    custom_class_names = None
    if is_custom:
        (x_train, y_train), (x_test, y_test), custom_class_names = load_custom_dataset(
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
    
    # クラス名の取得（標準データセット or カスタムデータセット）
    from modules.data_loader import get_class_names
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
        column_neurons=args.column_neurons,
        participation_rate=args.participation_rate,
        use_hexagonal=not args.use_circular,
        overlap=args.overlap,
        gradient_clip=args.gradient_clip,
        activation=args.activation,  # activationパラメータを追加
        leaky_alpha=args.leaky_alpha,  # Leaky ReLUの負勾配係数
        hyperparams=hp,  # HyperParamsインスタンスを渡す（重み初期化係数の取得に必要）
        weight_init_scales=weight_init_scales,  # CLI/HyperParamsから取得した値
        weight_init_source=weight_init_source  # 値の出処を記録
    )
    
    # コラム構造の診断（オプション）
    if args.diagnose_column:
        network.diagnose_column_structure()
        print("\n診断完了。学習はスキップします。")
        import sys
        sys.exit(0)
    
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
                total_epochs=args.epochs
            )
            print("\n可視化機能: 有効")
            if args.heatmap:
                print("  - ヒートマップ表示: 有効")
            if args.save_viz:
                print(f"  - 保存先: {args.save_viz}")
        except Exception as e:
            print(f"\n警告: 可視化モジュールの初期化に失敗しました: {e}")
            print("可視化なしで学習を継続します。")
            viz_manager = None
    
    # ========================================
    # 6. 学習ループ
    # ========================================
    print("\n" + "=" * 70)
    print("学習開始")
    print("=" * 70)
    
    from tqdm import tqdm
    from modules.data_loader import create_tf_dataset
    
    # TensorFlow Dataset作成（条件に応じて）
    train_dataset_tf = None
    if args.batch is not None:
        # ミニバッチ学習
        shuffle_status = "シャッフルあり" if args.shuffle else "シャッフルなし"
        print(f"ミニバッチ学習モード: batch={args.batch}, {shuffle_status}, seed={args.seed}")
        train_dataset_tf = create_tf_dataset(
            x_train, y_train,
            batch_size=args.batch,
            shuffle=args.shuffle,
            seed=args.seed
        )
    elif args.shuffle:
        # オンライン学習 + シャッフル（batch_size=1のTensorFlow Dataset）
        print(f"オンライン学習モード: シャッフルあり, seed={args.seed}")
        train_dataset_tf = create_tf_dataset(
            x_train, y_train,
            batch_size=1,
            shuffle=True,
            seed=args.seed
        )
    else:
        # オンライン学習（シャッフルなし）
        print(f"オンライン学習モード: シャッフルなし, seed={args.seed}")
    
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    best_test_acc = 0.0
    best_epoch = 0
    train_acc_history = []
    test_acc_history = []
    
    # ========================================
    # Affinity分布解析（初期化直後）
    # ========================================
    analyze_affinity_distribution(network)
    
    # ========================================
    # 初期化直後のデッドニューロン調査
    # ========================================
    print("\n" + "=" * 70)
    print("初期化直後のデッドニューロン調査")
    print("=" * 70)
    initial_dead_info = analyze_dead_neurons(network, x_test[:1000], y_test[:1000], epoch_label="初期化直後")
    print(f"{initial_dead_info['summary']}")
    for layer_idx, (dead, total, ratio) in enumerate(zip(
        initial_dead_info['dead_counts'], 
        initial_dead_info['total_counts'], 
        initial_dead_info['dead_ratios']
    )):
        print(f"  Layer {layer_idx+1}: {dead}/{total} ({ratio*100:.1f}% dead)")
    print("=" * 70)
    
    # デッドニューロン履歴（エポック毎に記録）
    dead_neuron_history = [initial_dead_info]
    
    # tqdmを使ったエポックループ
    pbar = tqdm(range(1, args.epochs + 1), desc="Training", ncols=120)
    for epoch in pbar:
        epoch_start = time.time()
        
        # 訓練
        if train_dataset_tf is not None:
            # TensorFlow Dataset API使用（ミニバッチまたはオンライン+シャッフル）
            train_acc, train_loss = network.train_epoch_minibatch_tf(train_dataset_tf)
        else:
            # オンライン学習（シャッフルなし）
            train_acc, train_loss = network.train_epoch(x_train, y_train)
        
        # テスト
        test_acc, test_loss = network.evaluate(x_test, y_test)
        
        # デッドニューロン調査（エポック毎）
        epoch_dead_info = analyze_dead_neurons(network, x_test[:1000], y_test[:1000], epoch_label=f"Epoch {epoch}")
        dead_neuron_history.append(epoch_dead_info)
        
        # 履歴記録
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        
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
              f"{epoch_dead_info['summary']}, "
              f"Time={epoch_time:.2f}s")
        
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
                    sample_y_pred_name=pred_class_name
                )
    
    # ========================================
    # 7. 結果サマリー
    # ========================================
    print("\n" + "=" * 70)
    print("学習完了")
    print("=" * 70)
    print(f"最終精度: Train={train_acc:.4f}, Test={test_acc:.4f}")
    print(f"ベスト精度: Test={best_test_acc:.4f} (Epoch {best_epoch})")
    
    # デッドニューロン解析サマリー
    print("\n" + "=" * 70)
    print("デッドニューロン解析")
    print("=" * 70)
    
    # 初期化直後と最終エポックのデッドニューロン数を表示
    if len(dead_neuron_history) > 0:
        initial_info = dead_neuron_history[0]
        final_info = dead_neuron_history[-1]
        
        print(f"\n初期化直後: {initial_info['summary']}")
        for layer_idx, (dead, total, ratio) in enumerate(zip(
            initial_info['dead_counts'], 
            initial_info['total_counts'], 
            initial_info['dead_ratios']
        )):
            print(f"  Layer {layer_idx+1}: {dead}/{total} ({ratio*100:.1f}% dead)")
        
        print(f"\n最終エポック: {final_info['summary']}")
        for layer_idx, (dead, total, ratio) in enumerate(zip(
            final_info['dead_counts'], 
            final_info['total_counts'], 
            final_info['dead_ratios']
        )):
            print(f"  Layer {layer_idx+1}: {dead}/{total} ({ratio*100:.1f}% dead)")
        
        # 改善度を計算
        print("\nデッドニューロンの変化:")
        for layer_idx in range(len(initial_info['dead_counts'])):
            initial_dead = initial_info['dead_counts'][layer_idx]
            final_dead = final_info['dead_counts'][layer_idx]
            diff = final_dead - initial_dead
            diff_sign = "+" if diff > 0 else ""
            print(f"  Layer {layer_idx+1}: {initial_dead} → {final_dead} ({diff_sign}{diff})")
    
    print("=" * 70)
    
    # 可視化の最終処理
    if viz_manager is not None:
        saved_viz, saved_heatmap = viz_manager.save_figures()
        if saved_viz or saved_heatmap:
            print("\n可視化結果を保存しました:")
            if saved_viz:
                print(f"  - 学習曲線: {saved_viz}")
            if saved_heatmap:
                print(f"  - ヒートマップ: {saved_heatmap}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
