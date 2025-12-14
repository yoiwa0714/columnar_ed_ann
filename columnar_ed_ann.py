#!/usr/bin/env python3
"""
columnar_ed_ann.py
バージョン: 1.027.1
"""

import os
# TensorFlowのログメッセージを抑制（情報・警告を非表示）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import time

# モジュールインポート
from modules.hyperparameters import HyperParams
from modules.data_loader import load_dataset, get_class_names
from modules.ed_network import RefinedDistributionEDNetwork
from modules.visualization_manager import VisualizationManager


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
    exec_group.add_argument('--fashion', action='store_true',
                           help='Fashion-MNISTを使用')
    
    # ========================================
    # ED法関連のパラメータ
    # ========================================
    ed_group = parser.add_argument_group('ED法関連のパラメータ')
    ed_group.add_argument('--hidden', type=str, default='512',
                         help='隠れ層ニューロン数（例: 512=1層, 256,128=2層）（デフォルト値: 512）')
    ed_group.add_argument('--activation', type=str, default='tanh',
                         choices=['tanh', 'sigmoid', 'leaky_relu'],
                         help='活性化関数（デフォルト: tanh）※グリッドサーチ用、将来的に削除予定')
    ed_group.add_argument('--lr', type=float, default=0.20,
                         help='学習率（層数により自動設定: 1層=0.20, 2層=0.35）')
    ed_group.add_argument('--u1', type=float, default=0.5,
                         help='アミン拡散係数u1（層数により自動設定: 1層=0.5, 2層=0.5）')
    ed_group.add_argument('--u2', type=float, default=0.8,
                         help='アミン拡散係数u2（層数により自動設定: 1層=0.8, 2層=0.5）')
    ed_group.add_argument('--lateral_lr', type=float, default=0.08,
                         help='側方抑制の学習率（デフォルト値: 0.08）')
    ed_group.add_argument('--gradient_clip', type=float, default=0.05,
                         help='gradient clipping値（デフォルト値: 0.05）')
    
    # ========================================
    # コラム関連のパラメータ
    # ========================================
    column_group = parser.add_argument_group('コラム関連のパラメータ')
    column_group.add_argument('--list_hyperparams', action='store_true',
                             help='利用可能なHyperParams設定一覧を表示して終了')
    column_group.add_argument('--base_column_radius', type=float, default=0.4,
                             help='基準コラム半径（デフォルト値: 0.4、256ニューロン層での値）')
    column_group.add_argument('--column_radius', type=float, default=None,
                             help='コラム影響半径（デフォルト値: None、Noneなら層ごとに自動計算）')
    column_group.add_argument('--participation_rate', type=float, default=0.1,
                             help='コラム参加率（デフォルト値: 0.1、スパース表現、優先度：最高）')
    column_group.add_argument('--column_neurons', type=int, default=None,
                             help='各クラスの明示的ニューロン数（デフォルト値: None、重複許容、優先度：中）')
    column_group.add_argument('--use_circular', action='store_true',
                             help='旧円環構造を使用（デフォルトはハニカム）')
    column_group.add_argument('--overlap', type=float, default=0.0,
                             help='コラム間の重複度（デフォルト値: 0.0、0.0-1.0、円環構造でのみ有効、0.0=重複なし）')
    column_group.add_argument('--diagnose_column', action='store_true',
                             help='コラム構造の詳細診断を実行')
    
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
    
    print("=" * 80)
    print("Columnar ED-ANN v027 - Modular Version")
    print("=" * 80)
    
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
        print(f"\n=== 層数に基づくHyperParams設定を自動適用（{n_layers}層） ===")
        print("*** コマンドライン引数で明示的に値を指定をされた場合は、指定された値が設定されています。")
        print(f"hidden_layers: {config['hidden']}")
        print(f"learning_rate: {config['learning_rate']}")
        print(f"u1: {config.get('u1', 'N/A')}")
        print(f"u2: {config.get('u2', 'N/A')}")
        print(f"lateral_lr: {config.get('lateral_lr', 'N/A')}")
        print(f"base_column_radius: {config['base_column_radius']}")
        print(f"participation_rate: {config.get('participation_rate', 'N/A')}")
        print(f"epochs: {config['epochs']}")
        
        # デフォルト値の定義（argparseのデフォルト値）
        DEFAULT_LR = 0.20
        DEFAULT_U1 = 0.5
        DEFAULT_U2 = 0.8
        DEFAULT_LATERAL_LR = 0.08
        DEFAULT_BASE_RADIUS = 1.0
        DEFAULT_PARTICIPATION_RATE = 1.0
        DEFAULT_EPOCHS = 100
        
        # コマンドラインで明示されていないパラメータのみHyperParamsテーブルの値で上書き
        # hidden_sizesは常にテーブルの値を使用（層数はユーザーが指定したものを尊重）
        if args.hidden == '512':  # デフォルト値の場合のみテーブルの構成を使用
            hidden_sizes = config['hidden']
        
        if args.lr == DEFAULT_LR:
            args.lr = config['learning_rate']
        if args.u1 == DEFAULT_U1 and 'u1' in config:
            args.u1 = config['u1']
        if args.u2 == DEFAULT_U2 and 'u2' in config:
            args.u2 = config['u2']
        if args.lateral_lr == DEFAULT_LATERAL_LR and 'lateral_lr' in config:
            args.lateral_lr = config['lateral_lr']
        if args.base_column_radius == DEFAULT_BASE_RADIUS:
            args.base_column_radius = config['base_column_radius']
        if args.participation_rate == DEFAULT_PARTICIPATION_RATE and 'participation_rate' in config:
            args.participation_rate = config['participation_rate']
        if args.epochs == DEFAULT_EPOCHS:
            args.epochs = config['epochs']
        
        print("="*70 + "\n")
    except ValueError as e:
        print(f"Warning: HyperParamsテーブルの取得に失敗: {e}")
        print("個別パラメータで継続します。\n")
    
    # ========================================
    # 3. データ読み込み
    # ========================================
    dataset = 'fashion' if args.fashion else 'mnist'
    print(f"データ読み込み中... (訓練:{args.train}, テスト:{args.test}, データセット:{dataset})")
    (x_train, y_train), (x_test, y_test) = load_dataset(
        dataset=dataset, train_samples=args.train, test_samples=args.test
    )
    
    n_classes = 10
    
    # ========================================
    # 4. ネットワークの構築
    # ========================================
    print("\nネットワーク初期化中...")
    
    network = RefinedDistributionEDNetwork(
        n_input=784,
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
        hyperparams=None  # HyperParamsの処理は既に完了しているのでNoneを渡す
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
    
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    best_test_acc = 0.0
    best_epoch = 0
    train_acc_history = []
    test_acc_history = []
    
    # tqdmを使ったエポックループ
    pbar = tqdm(range(1, args.epochs + 1), desc="Training", ncols=120)
    for epoch in pbar:
        epoch_start = time.time()
        
        # 訓練
        train_acc, train_loss = network.train_epoch(x_train, y_train)
        
        # テスト
        test_acc, test_loss = network.evaluate(x_test, y_test)
        
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
    
    # 可視化の最終処理
    if viz_manager is not None:
        viz_manager.save_figures()
        if args.save_viz:
            print(f"\n可視化結果を保存しました: {args.save_viz}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
