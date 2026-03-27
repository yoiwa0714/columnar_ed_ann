#!/usr/bin/env python3
"""
Columnar ED-ANN

コラムED法（Error Diffusion法 + 大脳皮質コラム構造）の実装。
微分の連鎖律を用いた誤差逆伝播法を一切使用せずに高い学習精度を実現する。

■ コラムED法の動作原理:
  1. 入力→隠れ層→出力の順伝播（活性化関数: tanh）
  2. 出力スコアが最大のクラス（勝者）に対してのみ学習
  3. 正解の場合: 勝者クラスのコラムニューロンを強化（正のアミン信号）
  4. 不正解の場合: 勝者クラスのコラムニューロンを弱化（負のアミン信号）
  5. 重み更新は勾配降下ではなく、アミン信号×入力活性の外積で行う

■ コラム構造:
  大脳皮質のコラム構造を模倣。各クラスに少数のコラムニューロンを割り当て、
  残りの大多数は固定重みのリザバーとして機能する（学習しない）。
  この構造により多クラス分類を実現。

■ 使用例:
  # 2層+Gabor特徴（デフォルト構成、約10分）
  python columnar_ed_ann.py --train 10000 --test 10000

  # 1層（約3分）
  python columnar_ed_ann.py --hidden 2048 --train 5000 --test 5000

  # 可視化付き
  python columnar_ed_ann.py --train 10000 --test 10000 --viz --heatmap

  # Gabor特徴なし
  python columnar_ed_ann.py --hidden 2048 --train 10000 --test 10000 --no_gabor
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import time
import sys

from modules.hyperparameters import HyperParams
from modules.data_loader import load_dataset
from modules.ed_network import SimpleColumnEDNetwork


def parse_args():
    """コマンドライン引数の解析（必要最小限）"""
    parser = argparse.ArgumentParser(
        prog='columnar_ed_ann.py',
        description='コラムED法\n'
                    '微分の連鎖律による誤差逆伝播法を使用せず高精度を実現\n'
                    '\n'
                    '層数に応じた最適パラメータが自動適用されます',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--hidden', type=str, default=None,
                       help='隠れ層ニューロン数（カンマ区切り）\n'
                            '例: 2048=1層, 2048,1024=2層, 2048,1024,1024=3層\n'
                            '未指定時: 2層構成 [2048,1024]')
    parser.add_argument('--train', type=int, default=10000,
                       help='訓練サンプル数（デフォルト: 10000）')
    parser.add_argument('--test', type=int, default=10000,
                       help='テストサンプル数（デフォルト: 10000）')
    parser.add_argument('--epochs', type=int, default=None,
                       help='エポック数（未指定時: 層数に応じた自動設定）')
    parser.add_argument('--seed', type=int, default=42,
                       help='乱数シード（デフォルト: 42）')
    parser.add_argument('--dataset', type=str, default='mnist',
                       help='データセット（mnist, fashion, cifar10）')
    parser.add_argument('--no_gabor', action='store_true',
                       help='Gabor特徴抽出を無効化（デフォルト: Gabor ON）')

    # 可視化
    viz_group = parser.add_argument_group('可視化')
    viz_group.add_argument('--viz', type=int, nargs='?', const=1, default=None,
                          choices=[1, 2, 3, 4], metavar='SIZE',
                          help='学習曲線のリアルタイム可視化を有効化（サイズ指定: 1-4）\n'
                               '1=基準, 2=1.3倍, 3=1.6倍, 4=2倍（ウィンドウサイズ）\n'
                               '数値省略時は1（--viz == --viz 1）')
    viz_group.add_argument('--heatmap', action='store_true',
                          help='活性化ヒートマップの表示（--vizと併用）')
    viz_group.add_argument('--save_viz', type=str, nargs='?', const='viz_results',
                          default=None, metavar='PATH',
                          help='可視化結果を保存（パス指定可）')

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_args()

    # ========================================
    # 1. 隠れ層サイズのパース
    # ========================================
    if args.hidden is not None:
        try:
            hidden_sizes = [int(x.strip()) for x in args.hidden.split(',')]
        except ValueError:
            print(f"エラー: --hidden の値が不正です: {args.hidden}")
            print("例: --hidden 2048 (1層), --hidden 2048,1024 (2層)")
            sys.exit(1)
    else:
        hidden_sizes = [2048, 1024]  # デフォルト: 2層

    n_layers = len(hidden_sizes)

    # ========================================
    # 2. YAMLから層数別パラメータ自動選択
    # ========================================
    hp = HyperParams()
    config = hp.get_config(n_layers)

    # CLI指定がない場合は層数依存の隠れ層サイズを使用
    if args.hidden is None:
        hidden_sizes = config['hidden']

    # エポック数の決定
    epochs = args.epochs if args.epochs is not None else config['epochs']

    # 学習率パラメータ
    output_lr = config['output_lr']
    non_column_lr = config['non_column_lr']
    column_lr_factors = config['column_lr_factors']
    # layer_learning_rates: non_column_lr + output_lr
    layer_lrs = list(non_column_lr) + [output_lr]

    # ========================================
    # 3. 乱数シード設定
    # ========================================
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ========================================
    # 4. データ読み込み
    # ========================================
    print(f"データ読み込み中... (データセット: {args.dataset}, 訓練: {args.train}, テスト: {args.test})")
    (x_train, y_train), (x_test, y_test) = load_dataset(
        dataset=args.dataset, train_samples=args.train, test_samples=args.test
    )

    n_input = x_train.shape[1]
    n_classes = len(np.unique(y_train))
    print(f"入力次元: {n_input}, クラス数: {n_classes}")

    # Gabor変換前のデータ保持（ヒートマップ用）
    x_train_raw = None
    x_test_raw = None

    # ========================================
    # 5. Gabor特徴抽出（V1単純型細胞モデル）
    # ========================================
    use_gabor = not args.no_gabor
    gabor_info = None
    if use_gabor:
        from modules.gabor_features import GaborFeatureExtractor

        if n_input == 784:
            img_shape = (28, 28)
        else:
            side = int(np.sqrt(n_input))
            if side * side == n_input:
                img_shape = (side, side)
            else:
                print(f"警告: 入力次元{n_input}の画像形状を推定できません。Gabor無効化。")
                use_gabor = False
                img_shape = None

    if use_gabor:
        gp = hp.gabor_params
        print(f"Gabor特徴抽出中... (方位: {gp['orientations']}, 周波数: {gp['frequencies']}, "
              f"カーネル: {gp['kernel_size']})")

        extractor = GaborFeatureExtractor(
            image_shape=img_shape,
            n_orientations=gp['orientations'],
            n_frequencies=gp['frequencies'],
            kernel_size=gp['kernel_size'],
            pool_size=gp['pool_size'],
            pool_stride=gp['pool_stride'],
            include_edge_filters=True
        )

        gabor_info = extractor.get_info()
        print(f"  フィルタ数: {gabor_info['n_filters']}, "
              f"特徴次元: {gabor_info['feature_dim']} (元: {n_input})")

        # 変換前のデータを保持（ヒートマップ用）
        x_train_raw = x_train.copy()
        x_test_raw = x_test.copy()

        x_train = extractor.transform(x_train)
        x_test = extractor.transform_test(x_test)
        n_input = x_train.shape[1]

    # ========================================
    # 6. ネットワーク構築
    # ========================================
    print(f"\nネットワーク構築中... ({n_layers}層: {hidden_sizes})")

    network = SimpleColumnEDNetwork(
        n_input=n_input,
        n_hidden=hidden_sizes,
        n_output=n_classes,
        learning_rate=output_lr,
        u1=config['u1'],
        u2=config['u2'],
        base_column_radius=config.get('base_column_radius', 0.4),
        column_neurons=config['column_neurons'],
        participation_rate=config.get('participation_rate', 0.1),
        use_hexagonal=config.get('use_hexagonal', True),
        gradient_clip=config['gradient_clip'],
        hidden_sparsity=config.get('hidden_sparsity', 0.4),
        column_lr_factors=column_lr_factors,
        init_scales=config['weight_init_scales'],
        layer_learning_rates=layer_lrs,
        seed=args.seed,
    )

    # ========================================
    # 7. 可視化マネージャーの初期化
    # ========================================
    viz_manager = None
    if args.viz is not None:
        try:
            from modules_experiment.visualization_manager import VisualizationManager
            viz_scale_map = {1: 1.00, 2: 1.30, 3: 1.60, 4: 2.00}
            viz_scale = viz_scale_map.get(args.viz, 1.00)
            viz_manager = VisualizationManager(
                enable_viz=True,
                enable_heatmap=args.heatmap,
                save_path=args.save_viz,
                total_epochs=epochs,
                verbose=False,
                window_scale=viz_scale,
            )
            if gabor_info is not None:
                viz_manager.set_gabor_info(gabor_info)
            print(f"可視化: 有効 (サイズレベル{args.viz}, 倍率x{viz_scale:.1f})")
        except Exception as e:
            print(f"警告: 可視化初期化に失敗: {e}")
            viz_manager = None

    # ========================================
    # 8. パラメータ表示
    # ========================================
    print("\n" + "=" * 60)
    print(f"パラメータ設定（{n_layers}層構成）")
    print("=" * 60)
    print(f"  hidden:            {hidden_sizes}")
    print(f"  column_neurons:    {config['column_neurons']}")
    print(f"  init_scales:       {config['weight_init_scales']}")
    print(f"  output_lr:         {output_lr}")
    print(f"  column_lr_factors: {column_lr_factors}")
    print(f"  gradient_clip:     {config['gradient_clip']}")
    print(f"  hidden_sparsity:   {config['hidden_sparsity']}")
    print(f"  u1={config['u1']}, u2={config['u2']}")
    print(f"  init_method:       {config.get('init_method', 'he')}")
    print(f"  gabor_features:    {use_gabor}"
          + (f" (kernel_size={hp.gabor_params['kernel_size']})" if use_gabor else ""))
    print(f"  epochs:            {epochs}")
    print(f"  seed:              {args.seed}")
    print("=" * 60)

    # ========================================
    # 9. 学習ループ
    # ========================================
    print("\n学習開始")
    print("-" * 60)

    from tqdm import tqdm

    best_test_acc = 0.0
    best_epoch = 0
    train_acc_history = []
    test_acc_history = []

    # ヒートマップコールバック（ヒートマップ有効時のみ）
    heatmap_callback = None
    if viz_manager is not None and args.heatmap:
        y_test_idx = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test
        rng = np.random.RandomState(args.seed)
        epoch_ref = [0]

        def heatmap_callback(net, sample_i, n_samples):
            idx = rng.randint(0, len(x_test))
            z_h, z_o, _ = net.forward(x_test[idx])
            y_true = y_test_idx[idx]
            y_pred = np.argmax(z_o)
            viz_manager.update_heatmap(
                epoch=epoch_ref[0],
                sample_x=x_test[idx],
                sample_y_true=y_true,
                sample_y_true_name=str(y_true),
                z_hiddens=z_h,
                z_output=z_o,
                sample_y_pred=y_pred,
                sample_y_pred_name=str(y_pred),
                sample_x_raw=x_test_raw[idx] if x_test_raw is not None else None,
                progress=f"{sample_i}/{n_samples}"
            )

    pbar = tqdm(range(1, epochs + 1), desc="Training", ncols=100)
    for epoch in pbar:
        epoch_start = time.time()

        if heatmap_callback is not None:
            epoch_ref[0] = epoch

        # 統計リセット
        network.reset_winner_selection_stats()
        network.reset_class_training_stats()

        # 訓練（オンライン学習: 1サンプルずつ順伝播→重み更新）
        train_acc, train_loss = network.train_epoch(
            x_train, y_train, progress_callback=heatmap_callback
        )

        # テスト評価
        test_acc, test_loss, class_accs = network.evaluate_parallel(
            x_test, y_test, return_per_class=True
        )

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        epoch_time = time.time() - epoch_start

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

        pbar.set_postfix({
            'Train': f'{train_acc:.4f}',
            'Test': f'{test_acc:.4f}',
            'Best': f'{best_test_acc:.4f}',
        })

        print(f"Epoch {epoch:3d}/{epochs}: "
              f"Train={train_acc:.4f}, Test={test_acc:.4f}, "
              f"Best={best_test_acc:.4f} (ep{best_epoch}), "
              f"Time={epoch_time:.1f}s")

        # 可視化更新
        if viz_manager is not None:
            y_test_indices = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test
            viz_manager.update_learning_curve(
                train_acc_history, test_acc_history,
                x_test, y_test_indices, network
            )

            if args.heatmap:
                sample_idx = np.random.randint(0, len(x_test))
                z_hiddens, z_output, _ = network.forward(x_test[sample_idx])
                y_true = y_test_indices[sample_idx]
                y_pred = np.argmax(z_output)
                viz_manager.update_heatmap(
                    epoch=epoch,
                    sample_x=x_test[sample_idx],
                    sample_y_true=y_true,
                    sample_y_true_name=str(y_true),
                    z_hiddens=z_hiddens,
                    z_output=z_output,
                    sample_y_pred=y_pred,
                    sample_y_pred_name=str(y_pred),
                    sample_x_raw=x_test_raw[sample_idx] if x_test_raw is not None else None,
                )

    # ========================================
    # 10. 結果サマリー
    # ========================================
    print("\n" + "=" * 60)
    print("学習完了")
    print("=" * 60)
    print(f"最終精度:   Train={train_acc:.4f}, Test={test_acc:.4f}")
    print(f"ベスト精度: Test={best_test_acc:.4f} (Epoch {best_epoch})")
    print("=" * 60)

    # 可視化の保存
    if viz_manager is not None:
        viz_manager.save_figures()
        if args.save_viz:
            print(f"\n可視化結果を保存: {args.save_viz}")

    print()


if __name__ == '__main__':
    main()
