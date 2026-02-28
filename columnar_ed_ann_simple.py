#!/usr/bin/env python3
"""
Columnar ED-ANN 簡易版（公開用）

金子勇氏考案のError Diffusion (ED)法に大脳皮質のコラム構造を導入した
ニューラルネットワーク実装。微分の連鎖律（誤差逆伝播法）を使用せず、
アミン拡散機構によって学習を行う。

★本スクリプトの特徴:
  - 最小限のコマンドライン引数（14個）
  - config/hyperparameters.yaml から最適パラメータを自動読み込み
  - Gabor特徴抽出がデフォルトON（--no_gabor で無効化）
  - modules_simple/ の簡易モジュールを使用
  - 3系統学習率（output_lr, non_column_lr, column_lr）はYAMLから自動設定

使用例:
  # 1層 + Gabor特徴 (デフォルト、約2分)
  python columnar_ed_ann_simple.py --hidden 2048 --train 5000 --test 5000

  # 2層 + Gabor特徴 (約18分)
  python columnar_ed_ann_simple.py --hidden 2048,1024 --train 20000 --test 20000 --epochs 10

  # 3層 + Gabor特徴 (約15分)
  python columnar_ed_ann_simple.py --hidden 2048,1024,1024 --train 10000 --test 10000 --epochs 10

  # コラムニューロン数・初期化スケールを指定（デフォルトはYAMLから自動設定）
  python columnar_ed_ann_simple.py --hidden 2048,1024 --column_neurons 10 --init_scales 0.7,1.8,0.8

  # Gabor無し
  python columnar_ed_ann_simple.py --hidden 2048 --train 5000 --test 5000 --no_gabor

  # リアルタイム可視化 + ヒートマップ
  python columnar_ed_ann_simple.py --hidden 2048 --train 5000 --test 5000 --viz --heatmap

  # 利用可能なハイパーパラメータ一覧を表示
  python columnar_ed_ann_simple.py --list_hyperparams

  # Fashion-MNIST
  python columnar_ed_ann_simple.py --hidden 2048 --train 5000 --test 5000 --dataset fashion
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlowの情報・警告メッセージを抑制

import argparse
import numpy as np
import sys
import time

from modules_simple.hyperparameters import HyperParams
from modules_simple.data_loader import load_dataset, get_class_names, resolve_dataset_path, load_custom_dataset
from modules_simple.ed_network import RefinedDistributionEDNetwork
from modules_simple.visualization_manager import VisualizationManager

# コマンドライン引数の原本を保持（YAML自動適用で「明示指定された引数」を判別するため）
_COMMAND_LINE_ARGS = sys.argv.copy()


def parse_args():
    """コマンドライン引数の解析（簡易版: 13引数）"""
    parser = argparse.ArgumentParser(
        description='Columnar ED-ANN 簡易版\n'
                    '\n'
                    '金子勇氏考案のED法 + 大脳皮質コラム構造によるニューラルネットワーク。\n'
                    '微分の連鎖律を使わず、アミン拡散機構で学習します。\n'
                    '\n'
                    '層数に応じた最適パラメータが config/hyperparameters.yaml から自動適用されます。\n'
                    '--list_hyperparams で設定一覧を確認できます。\n',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--hidden', type=str, default='2048',
                        help='隠れ層ニューロン数（例: 2048=1層, 2048,1024=2層）')
    parser.add_argument('--train', type=int, default=5000,
                        help='訓練サンプル数（デフォルト: 5000）')
    parser.add_argument('--test', type=int, default=5000,
                        help='テストサンプル数（デフォルト: 5000）')
    parser.add_argument('--epochs', type=int, default=None,
                        help='エポック数（未指定時はYAMLから自動設定）')
    parser.add_argument('--seed', type=int, default=42,
                        help='乱数シード（デフォルト: 42）')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='データセット名（mnist, fashion, cifar10）またはカスタムデータパス')
    parser.add_argument('--viz', action='store_true',
                        help='リアルタイム学習曲線を表示')
    parser.add_argument('--heatmap', action='store_true',
                        help='隠れ層・出力層のヒートマップを表示（--vizと併用）')
    parser.add_argument('--save_viz', type=str, default=None,
                        help='可視化結果の保存先ディレクトリ')
    parser.add_argument('--no_gabor', action='store_true',
                        help='Gabor特徴抽出を無効化（デフォルトはON）')
    parser.add_argument('--column_neurons', type=int, default=None,
                        help='コラムニューロン数（未指定時はYAMLから自動設定）')
    parser.add_argument('--init_scales', type=str, default=None,
                        help='層別初期化スケール（例: 0.7,1.8,0.8）（未指定時はYAMLから自動設定）')
    parser.add_argument('--list_hyperparams', nargs='?', type=int, const=0, default=None,
                        help='ハイパーパラメータ一覧を表示（層数を指定可能、例: --list_hyperparams 2）')
    parser.add_argument('--verbose', action='store_true',
                        help='初期化詳細（重みスケール、コラム構造、スパース率等）を表示')

    return parser.parse_args()


def is_arg_specified(arg_names):
    """指定された引数名がコマンドラインで明示的に指定されたかをチェック"""
    if isinstance(arg_names, str):
        arg_names = [arg_names]
    for arg_name in arg_names:
        for arg in _COMMAND_LINE_ARGS:
            if arg == f'--{arg_name}' or arg.startswith(f'--{arg_name}='):
                return True
    return False


def main():
    """メイン処理"""
    args = parse_args()

    # Gabor特徴はデフォルトON
    use_gabor = not args.no_gabor

    # === 乱数シード設定（再現性確保） ===
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("Columnar ED-ANN (簡易版)")
    print("=" * 70)

    # === HyperParams設定一覧の表示 ===
    if args.list_hyperparams is not None:
        hp = HyperParams()
        if args.list_hyperparams == 0:
            hp.list_configs()
        else:
            config = hp.get_config(args.list_hyperparams)
            print(f"\n{args.list_hyperparams}層構成のパラメータ:")
            for key, value in sorted(config.items()):
                print(f"  {key}: {value}")
        sys.exit(0)

    # === 隠れ層のパース ===
    try:
        if ',' in args.hidden:
            hidden_sizes = [int(x.strip()) for x in args.hidden.split(',')]
        else:
            hidden_sizes = [int(args.hidden)]
    except ValueError:
        print(f"エラー: --hidden の値が不正です: {args.hidden}")
        print(f"  例: --hidden 2048 (1層) または --hidden 2048,1024 (2層)")
        sys.exit(1)

    n_layers = len(hidden_sizes)

    # === YAMLから最適パラメータを自動取得 ===
    hp = HyperParams()
    try:
        config = hp.get_config(n_layers)
    except ValueError as e:
        print(f"Warning: YAML設定取得失敗: {e}")
        config = {}

    # コマンドラインで未指定のパラメータをYAML値で上書き
    if not is_arg_specified('hidden') and 'hidden' in config:
        hidden_sizes = config['hidden']
    if args.epochs is None:
        args.epochs = config.get('epochs', 10)

    # YAML由来のネットワークパラメータ
    u1 = config.get('u1', 0.5)
    u2 = config.get('u2', 0.8)
    base_column_radius = config.get('base_column_radius', 0.4)
    gradient_clip = config.get('gradient_clip', 0.05)

    # 3系統学習率（YAML自動設定）
    output_lr = config.get('output_lr', 0.15)
    default_non_column_lr = config.get('non_column_lr', [output_lr] * n_layers)
    if isinstance(default_non_column_lr, (int, float)):
        default_non_column_lr = [default_non_column_lr] * n_layers
    non_column_lr = list(default_non_column_lr)
    default_column_lr = config.get('column_lr', list(non_column_lr))
    if isinstance(default_column_lr, (int, float)):
        default_column_lr = [default_column_lr] * n_layers
    column_lr = list(default_column_lr)

    # column_neurons: コマンドライン指定優先、未指定時はYAML
    cn_source = 'YAML'
    column_neurons = config.get('column_neurons', 1)
    if args.column_neurons is not None:
        column_neurons = args.column_neurons
        cn_source = '指定'

    # 層別パラメータ
    # init_scales: コマンドライン指定優先、未指定時はYAML
    is_source = 'YAML'
    init_scales = config.get('weight_init_scales')
    if init_scales is None:
        init_scales = [0.3 + (0.7 * i / n_layers) for i in range(n_layers)] + [1.0]
    if args.init_scales is not None:
        try:
            init_scales = [float(x.strip()) for x in args.init_scales.split(',')]
            is_source = '指定'
        except ValueError:
            print(f"エラー: --init_scales の値が不正です: {args.init_scales}")
            print(f"  例: --init_scales 0.7,1.8,0.8")
            sys.exit(1)

    hs = config.get('hidden_sparsity', 0.4)
    hidden_sparsity = hs if isinstance(hs, list) else [hs] * n_layers

    # Gaborパラメータ
    gabor_orientations = config.get('gabor_orientations', 8)
    gabor_frequencies = config.get('gabor_frequencies', 2)
    gabor_kernel_size = config.get('gabor_kernel_size', 7)

    # === パラメータ表示 ===
    print(f"\n構成: {n_layers}層 {hidden_sizes}")
    print(f"  訓練: {args.train}サンプル, テスト: {args.test}サンプル, エポック: {args.epochs}")
    print(f"  出力層学習率: {output_lr}, 勾配クリップ: {gradient_clip}")
    print(f"  非コラム学習率: {non_column_lr}")
    print(f"  コラム学習率: {column_lr}")
    print(f"  コラムニューロン: {column_neurons} ({cn_source}), 初期化: he")
    print(f"  初期化スケール: {init_scales} ({is_source})")
    print(f"  隠れ層スパース率: {hidden_sparsity}")
    print(f"  Gabor特徴抽出: {'ON' if use_gabor else 'OFF'}")
    print(f"  シード: {args.seed}")

    # === データ読み込み ===
    dataset = args.dataset
    dataset_path, is_custom = resolve_dataset_path(dataset)

    print(f"\nデータ読み込み中... ({dataset})")

    custom_class_names = None
    if is_custom:
        (x_train, y_train), (x_test, y_test), custom_class_names = load_custom_dataset(
            dataset_path=dataset_path, train_samples=args.train, test_samples=args.test
        )
    else:
        (x_train, y_train), (x_test, y_test) = load_dataset(
            dataset=dataset_path, train_samples=args.train, test_samples=args.test
        )

    n_input = x_train.shape[1]
    n_classes = len(np.unique(y_train))
    print(f"  入力次元: {n_input}, クラス数: {n_classes}")

    # === Gabor特徴抽出 ===
    x_test_raw = None
    gabor_info = None

    n_channels = 1  # 入力チャネル数

    if use_gabor:
        if n_input == 784:
            img_shape = (28, 28)
        elif n_input % 3 == 0:
            # カラー画像（3チャネル）: 各チャネルに独立してGabor適用
            per_channel = n_input // 3
            side = int(np.sqrt(per_channel))
            if side * side == per_channel:
                n_channels = 3
                img_shape = (side, side)
            else:
                print(f"警告: 入力次元{n_input}から画像形状を推定できません。Gabor特徴抽出をスキップします。")
                use_gabor = False
        else:
            side = int(np.sqrt(n_input))
            if side * side == n_input:
                img_shape = (side, side)
            else:
                print(f"警告: 入力次元{n_input}から画像形状を推定できません。Gabor特徴抽出をスキップします。")
                use_gabor = False

    if use_gabor:
        from modules_simple.gabor_features import GaborFeatureExtractor

        ch_str = f", チャネル:{n_channels}" if n_channels > 1 else ""
        print(f"\nGabor特徴抽出中... (方位:{gabor_orientations}, 周波数:{gabor_frequencies}, カーネル:{gabor_kernel_size}{ch_str})")
        extractor = GaborFeatureExtractor(
            image_shape=img_shape,
            n_orientations=gabor_orientations,
            n_frequencies=gabor_frequencies,
            kernel_size=gabor_kernel_size,
            n_channels=n_channels,
        )

        gabor_info = extractor.get_info()
        print(f"  フィルタ数: {gabor_info['n_filters']}, 特徴次元: {gabor_info['feature_dim']} (元: {n_input})")

        # ヒートマップ用に変換前データを保存
        x_test_raw = x_test.copy()

        x_train = extractor.transform(x_train)
        x_test = extractor.transform_test(x_test)
        n_input = x_train.shape[1]
        print(f"  特徴抽出完了: 入力次元 → {n_input}")

    # === ネットワーク構築 ===
    print("\nネットワーク初期化中...")

    network = RefinedDistributionEDNetwork(
        n_input=n_input,
        n_hidden=hidden_sizes,
        n_output=n_classes,
        output_lr=output_lr,
        non_column_lr=non_column_lr,
        column_lr=column_lr,
        u1=u1,
        u2=u2,
        base_column_radius=base_column_radius,
        column_neurons=column_neurons,
        gradient_clip=gradient_clip,
        init_scales=init_scales,
        hidden_sparsity=hidden_sparsity,
        seed=args.seed,
        verbose=args.verbose,
    )

    # === 可視化の初期化 ===
    viz_manager = None
    if args.viz:
        try:
            viz_manager = VisualizationManager(
                enable_viz=True,
                enable_heatmap=args.heatmap,
                save_path=args.save_viz,
                total_epochs=args.epochs,
            )
            print("可視化: ON")
            if use_gabor and gabor_info:
                viz_manager.set_gabor_info(gabor_info)
                viz_manager.set_gabor_extractor(extractor)
        except Exception as e:
            print(f"警告: 可視化初期化失敗: {e}")
            viz_manager = None

    # === 学習ループ ===
    print("\n" + "=" * 70)
    print("学習開始")
    print("=" * 70)

    from tqdm import tqdm

    best_test_acc = 0.0
    best_epoch = 0
    train_acc_history = []
    test_acc_history = []
    class_acc_history = []

    class_names = get_class_names(dataset, custom_class_names=custom_class_names)

    # ヒートマップ中間更新コールバック
    _heatmap_callback = None
    if viz_manager is not None and args.heatmap:
        def _make_heatmap_callback():
            _x_test = x_test
            _y_test_idx = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test
            _x_test_raw = x_test_raw
            _class_names = class_names
            _vm = viz_manager
            _rng = np.random.RandomState(args.seed)
            _epoch_ref = [0]

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
                    progress=progress_str,
                )

            def set_epoch(ep):
                _epoch_ref[0] = ep

            callback.set_epoch = set_epoch
            return callback
        _heatmap_callback = _make_heatmap_callback()

    pbar = tqdm(range(1, args.epochs + 1), desc="Training", ncols=120)
    for epoch in pbar:
        epoch_start = time.time()
        if _heatmap_callback is not None:
            _heatmap_callback.set_epoch(epoch)

        # オンライン学習
        train_acc, train_loss = network.train_epoch(x_train, y_train, progress_callback=_heatmap_callback)

        # テスト評価（並列高速版）
        test_acc, test_loss, class_accs = network.evaluate_parallel(x_test, y_test, return_per_class=True)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        class_acc_history.append(class_accs)

        epoch_time = time.time() - epoch_start

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

        pbar.set_postfix({
            'Train': f'{train_acc:.4f}',
            'Test': f'{test_acc:.4f}',
            'Best': f'{best_test_acc:.4f}',
            'Time': f'{epoch_time:.1f}s',
        })

        print(f"Epoch {epoch:3d}/{args.epochs}: "
              f"Train={train_acc:.4f} (loss={train_loss:.4f}), "
              f"Test={test_acc:.4f} (loss={test_loss:.4f}), "
              f"Time={epoch_time:.2f}s")

        # 可視化更新
        if viz_manager is not None:
            y_test_indices = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test

            viz_manager.update_learning_curve(
                train_acc_history, test_acc_history,
                x_test, y_test_indices, network,
            )

            if args.heatmap:
                sample_idx = np.random.randint(0, len(x_test))
                sample_x = x_test[sample_idx]
                sample_y_true = y_test_indices[sample_idx]
                z_hiddens, z_output, _ = network.forward(sample_x)
                sample_y_pred = np.argmax(z_output)

                true_name = class_names[sample_y_true] if class_names else None
                pred_name = class_names[sample_y_pred] if class_names else None

                viz_manager.update_heatmap(
                    epoch=epoch,
                    sample_x=sample_x,
                    sample_y_true=sample_y_true,
                    sample_y_true_name=true_name,
                    z_hiddens=z_hiddens,
                    z_output=z_output,
                    sample_y_pred=sample_y_pred,
                    sample_y_pred_name=pred_name,
                    sample_x_raw=x_test_raw[sample_idx] if x_test_raw is not None else None,
                )

    # === 結果サマリー ===
    print("\n" + "=" * 70)
    print("学習完了")
    print("=" * 70)
    print(f"最終精度: Train={train_acc:.4f}, Test={test_acc:.4f}")
    print(f"ベスト精度: Test={best_test_acc:.4f} (Epoch {best_epoch})")

    # クラス別精度
    print(f"\nクラス別テスト精度 (最終エポック):")
    if class_acc_history:
        final_class_accs = class_acc_history[-1]
        for c, acc in enumerate(final_class_accs):
            name = f" ({class_names[c]})" if class_names else ""
            print(f"  Class {c}{name}: {acc * 100:.1f}%")

    # 可視化の最終処理
    if viz_manager is not None:
        viz_manager.save_figures()
        if args.save_viz:
            print(f"\n可視化結果を保存しました: {args.save_viz}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
