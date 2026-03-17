#!/usr/bin/env python3
"""
Columnar ED-ANN 簡易版（自己完結型）

modules_simple/ ディレクトリのモジュールだけを使用する自己完結型の実装。
columnar_ed_ann_simple.py と modules_simple/ だけで動作する。

方針:
- ユーザー向けインターフェースを簡潔に保つ
- modules_simple/ の教育向け簡易モジュールのみに依存
- GPU（CuPy）やミニバッチ（TF Dataset API）は使用しない
- Gabor特徴は簡易版仕様に合わせてデフォルトON
"""

from __future__ import annotations

import argparse
import io
import numpy as np
import sys
import time
from contextlib import redirect_stdout
from sklearn.metrics import confusion_matrix

# モジュールインポート（modules_simple/ のみ使用）
from modules_simple.hyperparameters import HyperParams
from modules_simple.data_loader import (
    load_dataset, get_class_names, resolve_dataset_path, load_custom_dataset
)
from modules_simple.ed_network import RefinedDistributionEDNetwork
from modules_simple.visualization_manager import VisualizationManager

# コマンドライン引数のコピー（HyperParams適用時に明示指定チェック用）
_COMMAND_LINE_ARGS = sys.argv.copy()


def parse_args() -> argparse.Namespace:
    """README_simple 準拠の最小引数を解析する。"""
    parser = argparse.ArgumentParser(
        description=(
            "Columnar ED-ANN 簡易版\n"
            "modules_simple/ を使用する自己完結型の実装。最小限のCLIで高精度を実現。"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--hidden", type=str, default="2048",
                        help="隠れ層ニューロン数（例: 2048 / 2048,1024 / 1024[4]）")
    parser.add_argument("--train", type=int, default=5000,
                        help="訓練サンプル数（デフォルト: 5000）")
    parser.add_argument("--test", type=int, default=5000,
                        help="テストサンプル数（デフォルト: 5000）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="エポック数（未指定時はYAML自動設定）")
    parser.add_argument("--seed", type=int, default=42,
                        help="乱数シード（デフォルト: 42）")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help=(
            "データセット名（mnist, fashion, cifar10, cifar100）"
            "またはカスタムデータセットのパス"
        ),
    )

    parser.add_argument("--viz", type=int, nargs="?", const=1, default=None,
                        choices=[1, 2, 3, 4], metavar="SIZE",
                        help="リアルタイム学習曲線を表示（サイズ指定: 1-4）")
    parser.add_argument("--heatmap", action="store_true",
                        help="活性化ヒートマップを表示（--vizと併用）")
    parser.add_argument("--save_viz", type=str, default=None,
                        help="可視化結果の保存先（ディレクトリ or ベースファイル名）")

    parser.add_argument("--show_train_errors", action="store_true",
                        help="最終エポックの不正解学習データを一覧表示")
    parser.add_argument("--max_errors_per_class", type=int, default=20,
                        help="不正解表示のクラスごとの上限数（デフォルト: 20）")

    parser.add_argument("--no_gabor", action="store_true",
                        help="Gabor特徴抽出を無効化（簡易版のデフォルトはON）")

    parser.add_argument("--column_neurons", type=int, default=None,
                        help="コラムニューロン数（未指定時はYAML自動設定）")
    parser.add_argument("--init_scales", type=str, default=None,
                        help="層別初期化スケール（例: 0.7,1.8,0.8）")

    parser.add_argument("--list_hyperparams", nargs="?", type=int, const=0, default=None,
                        metavar="N_LAYERS",
                        help="ハイパーパラメータ一覧を表示（引数なしで全体、数値で層別）")
    parser.add_argument("--verbose", action="store_true",
                        help="初期化詳細を表示")

    return parser.parse_args()


# ========================================
# ユーティリティ
# ========================================

def is_arg_specified(arg_names):
    """指定された引数名のいずれかがコマンドラインで明示的に指定されたかをチェック"""
    if isinstance(arg_names, str):
        arg_names = [arg_names]
    for arg_name in arg_names:
        for arg in _COMMAND_LINE_ARGS:
            if arg == f'--{arg_name}' or arg.startswith(f'--{arg_name}=') or arg == f'-{arg_name}':
                return True
    return False


def expand_repeated_values(text, name, value_parser):
    """カンマ区切り文字列を展開。token[k] 形式の繰り返し記法を許可する。"""
    if text is None:
        raise ValueError(f"--{name} が未指定です")

    expanded = []
    raw_tokens = [tok.strip() for tok in text.split(',')]
    if not raw_tokens:
        raise ValueError(f"--{name} の値が空です")

    for idx, token in enumerate(raw_tokens):
        if token == '':
            raise ValueError(f"--{name} の第{idx+1}要素が空です")

        repeat = 1
        value_text = token
        if token.endswith(']') and '[' in token:
            value_text, repeat_text = token.rsplit('[', 1)
            repeat_text = repeat_text[:-1].strip()
            value_text = value_text.strip()
            if value_text == '' or repeat_text == '':
                raise ValueError(f"--{name} の繰り返し記法が不正です: '{token}'")
            try:
                repeat = int(repeat_text)
            except ValueError as e:
                raise ValueError(f"--{name} の繰り返し回数は整数である必要があります: '{token}'") from e
            if repeat <= 0:
                raise ValueError(f"--{name} の繰り返し回数は1以上である必要があります: '{token}'")

        try:
            value = value_parser(value_text)
        except ValueError as e:
            raise ValueError(f"--{name} の値を数値に変換できません: '{value_text}'") from e

        expanded.extend([value] * repeat)

    return expanded


def main() -> None:
    args = parse_args()
    verbose = args.verbose
    use_gabor = not args.no_gabor

    def vprint(*pargs, **kwargs):
        if verbose:
            print(*pargs, **kwargs)

    def run_with_log_control(func, *fargs, **fkwargs):
        """verbose OFF時は内部ログを抑制して実行する。"""
        if verbose:
            return func(*fargs, **fkwargs)
        with redirect_stdout(io.StringIO()):
            return func(*fargs, **fkwargs)

    # ========================================
    # 1. 乱数シード設定
    # ========================================
    if args.seed is not None:
        vprint(f"\n=== 乱数シード固定: {args.seed} ===")
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)

    vprint("=" * 80)
    vprint("Columnar ED-ANN 簡易版")
    vprint("=" * 80)

    # ========================================
    # 2. HyperParams設定一覧の表示
    # ========================================
    if args.list_hyperparams is not None:
        hp = HyperParams()
        if args.list_hyperparams == 0:
            hp.list_configs()
        else:
            try:
                config = hp.get_config(args.list_hyperparams)
                n_layers = args.list_hyperparams

                print(f"\n{'='*70}")
                print(f"パラメータ設定（{n_layers}層構成）デフォルト値一覧")
                print(f"{'='*70}")
                print(f"  説明: {config['description']}")

                print(f"\n[ED法関連]")
                print(f"  hidden: {config['hidden']}")
                print(f"  output_lr: {config.get('output_lr', 'N/A')}")
                print(f"  non_column_lr: {config.get('non_column_lr', 'N/A')}")
                print(f"  column_lr: {config.get('column_lr', 'N/A')}")
                print(f"  u1: {config.get('u1', 'N/A')}")
                print(f"  u2: {config.get('u2', 'N/A')}")
                print(f"  gradient_clip: {config.get('gradient_clip', 'N/A')}")
                print(f"  epochs: {config['epochs']}")

                print(f"\n[コラム関連]")
                print(f"  base_column_radius: {config.get('base_column_radius', 'N/A')}")
                print(f"  participation_rate: {config.get('participation_rate', 'N/A')}")
                print(f"  column_neurons: {config.get('column_neurons', 'N/A')}")

                print(f"\n[初期化関連]")
                init_scales = config.get('init_scales', config.get('weight_init_scales', 'N/A'))
                print(f"  init_scales: {init_scales}")
                print(f"  hidden_sparsity: {config.get('hidden_sparsity', 'N/A')}")

                if 'gabor_orientations' in config:
                    print(f"\n[Gabor関連]")
                    print(f"  gabor_orientations: {config['gabor_orientations']}")
                    print(f"  gabor_frequencies: {config['gabor_frequencies']}")
                    print(f"  gabor_kernel_size: {config['gabor_kernel_size']}")

                print(f"\n{'='*70}")
            except ValueError as e:
                print(f"\nエラー: {e}")
        sys.exit(0)

    # ========================================
    # 3. 隠れ層のパース
    # ========================================
    if '.' in args.hidden and ',' not in args.hidden:
        print(f"\nエラー: --hidden の区切り文字が不正です")
        print(f"  指定された値: {args.hidden}")
        print(f"  多層ネットワークを指定する場合は、ドット(.)ではなくカンマ(,)を使用してください")
        print(f"  例: --hidden 2048,1024")
        sys.exit(1)

    try:
        hidden_sizes = expand_repeated_values(args.hidden, 'hidden', int)
        for i, size in enumerate(hidden_sizes):
            if size <= 0:
                raise ValueError(f"--hidden の値は正の整数である必要があります: Layer {i}={size}")
    except ValueError as e:
        print(f"\nエラー: --hidden の値を整数に変換できません")
        print(f"  指定された値: {args.hidden}")
        print(f"  詳細: {e}")
        print(f"  例: --hidden 1024 (1層), --hidden 1024,512 (2層), --hidden 1024[5] (5層)")
        sys.exit(1)

    n_layers = len(hidden_sizes)

    # ========================================
    # 4. HyperParams(YAML)からの自動パラメータ取得
    # ========================================
    hp = HyperParams()
    config = {}
    try:
        config = hp.get_config(n_layers)

        if not is_arg_specified('hidden'):
            hidden_sizes = config['hidden']

        if not is_arg_specified('epochs') and args.epochs is None:
            args.epochs = config['epochs']

        if not is_arg_specified('train') and 'train_samples' in config:
            args.train = config['train_samples']
        if not is_arg_specified('test') and 'test_samples' in config:
            args.test = config['test_samples']

        if not is_arg_specified('column_neurons') and 'column_neurons' in config:
            args.column_neurons = config['column_neurons']

    except ValueError as e:
        print(f"Warning: YAML設定の取得に失敗: {e}")
        print("個別パラメータで継続します。\n")

    # エポック数のフォールバック
    if args.epochs is None:
        args.epochs = 10

    # n_layers を再計算（hiddenがYAMLで変わった場合）
    n_layers = len(hidden_sizes)

    # ========================================
    # 5. init_scalesのパース
    # ========================================
    init_scales = None
    if args.init_scales is not None:
        try:
            init_scales = [float(x.strip()) for x in args.init_scales.split(',')]
            expected_length = n_layers + 1
            if len(init_scales) != expected_length:
                print(f"\nエラー: --init_scales には {expected_length} 個の値が必要です")
                print(f"  （{n_layers}層ネットワーク: Layer 0用×1, ..., 出力層用×1）")
                print(f"  指定された値: {len(init_scales)}個 {init_scales}")
                sys.exit(1)
            for i, scale in enumerate(init_scales):
                if scale <= 0:
                    print(f"\nエラー: --init_scales の値は正の数である必要があります: index={i}, value={scale}")
                    sys.exit(1)
        except ValueError as e:
            print(f"\nエラー: --init_scales のパースに失敗: {e}")
            sys.exit(1)
    else:
        # YAMLから取得
        yaml_init_scales = config.get('init_scales', config.get('weight_init_scales'))
        if yaml_init_scales is not None:
            init_scales = list(yaml_init_scales)
        else:
            # フォールバック
            init_scales = [0.3 + (0.7 * i / n_layers) for i in range(n_layers)] + [1.0]

    # ========================================
    # 6. 3系統学習率のYAML自動適用
    # ========================================
    output_lr = config.get('output_lr', 0.15)
    non_column_lr = config.get('non_column_lr', [output_lr] * n_layers)
    column_lr = config.get('column_lr', None)
    column_lr_factors = config.get('column_lr_factors', None)

    # 正規化
    if isinstance(non_column_lr, (int, float)):
        non_column_lr = [float(non_column_lr)] * n_layers
    non_column_lr = list(non_column_lr)
    if len(non_column_lr) < n_layers:
        non_column_lr = non_column_lr + [non_column_lr[-1]] * (n_layers - len(non_column_lr))
    if len(non_column_lr) > n_layers:
        non_column_lr = non_column_lr[:n_layers]

    if column_lr is not None:
        if isinstance(column_lr, (int, float)):
            column_lr = [float(column_lr)] * n_layers
        column_lr = list(column_lr)
        if len(column_lr) < n_layers:
            column_lr = column_lr + [column_lr[-1]] * (n_layers - len(column_lr))
        if len(column_lr) > n_layers:
            column_lr = column_lr[:n_layers]
    elif column_lr_factors is not None:
        if isinstance(column_lr_factors, (int, float)):
            column_lr_factors = [float(column_lr_factors)] * n_layers
        column_lr_factors = list(column_lr_factors)
        if len(column_lr_factors) < n_layers:
            column_lr_factors = column_lr_factors + [column_lr_factors[-1]] * (n_layers - len(column_lr_factors))
        if len(column_lr_factors) > n_layers:
            column_lr_factors = column_lr_factors[:n_layers]
        column_lr = [non_column_lr[i] * column_lr_factors[i] for i in range(n_layers)]

    # その他のYAMLパラメータ
    u1 = config.get('u1', 0.5)
    u2 = config.get('u2', 0.8)
    base_column_radius = config.get('base_column_radius', 1.0)
    participation_rate = config.get('participation_rate', 1.0)
    gradient_clip = config.get('gradient_clip', 0.0)
    hidden_sparsity = config.get('hidden_sparsity', None)
    if hidden_sparsity is not None:
        if isinstance(hidden_sparsity, (int, float)):
            hidden_sparsity = [float(hidden_sparsity)] * n_layers
        hidden_sparsity = list(hidden_sparsity)
        if len(hidden_sparsity) < n_layers:
            hidden_sparsity = hidden_sparsity + [hidden_sparsity[-1]] * (n_layers - len(hidden_sparsity))
        if len(hidden_sparsity) > n_layers:
            hidden_sparsity = hidden_sparsity[:n_layers]

    # Gaborパラメータ
    gabor_orientations = config.get('gabor_orientations', 8)
    gabor_frequencies = config.get('gabor_frequencies', 2)
    gabor_kernel_size = config.get('gabor_kernel_size', 7)

    # ========================================
    # 7. パラメータ表示
    # ========================================
    vprint(f"\n{'='*70}")
    vprint(f"パラメータ設定（{n_layers}層構成）")
    vprint(f"{'='*70}")
    vprint(f"  hidden: {hidden_sizes}")
    vprint(f"  train: {args.train}, test: {args.test}, epochs: {args.epochs}")
    vprint(f"  seed: {args.seed}, dataset: {args.dataset}")
    vprint(f"  output_lr: {output_lr}")
    vprint(f"  non_column_lr: {non_column_lr}")
    vprint(f"  column_lr: {column_lr}")
    vprint(f"  u1: {u1}, u2: {u2}")
    vprint(f"  base_column_radius: {base_column_radius}")
    vprint(f"  column_neurons: {args.column_neurons}")
    vprint(f"  participation_rate: {participation_rate}")
    vprint(f"  gradient_clip: {gradient_clip}")
    vprint(f"  init_scales: {init_scales}")
    vprint(f"  hidden_sparsity: {hidden_sparsity}")
    vprint(f"  gabor: {use_gabor}")
    vprint(f"{'='*70}\n")

    # ========================================
    # 8. データ読み込み
    # ========================================
    dataset = args.dataset
    dataset_path, is_custom = resolve_dataset_path(dataset)

    print(f"データ読み込み中... (訓練:{args.train}, テスト:{args.test}, データセット:{dataset})")

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

    n_input = x_train.shape[1]
    n_classes = len(np.unique(y_train))

    # 表示用画像形状を推定
    if custom_input_shape is not None:
        display_img_shape = custom_input_shape
    elif n_input == 784:
        display_img_shape = (28, 28)
    elif n_input == 3072:
        display_img_shape = (32, 32, 3)
    else:
        side = int(np.sqrt(n_input))
        display_img_shape = (side, side) if side * side == n_input else (1, n_input)

    x_train_raw = None

    print(f"データセット情報: 入力次元={n_input}, クラス数={n_classes}")

    # ========================================
    # 9. Gabor特徴抽出
    # ========================================
    gabor_info = None
    if use_gabor:
        if n_input == 784:
            img_shape = (28, 28)
        elif n_input == 3072:
            print("警告: CIFAR-10はチャンネルが3つあるため、Gabor特徴抽出をスキップします。")
            use_gabor = False
            img_shape = None
        else:
            side = int(np.sqrt(n_input))
            if side * side == n_input:
                img_shape = (side, side)
            else:
                print(f"警告: 入力次元{n_input}から画像形状を推定できません。Gabor特徴抽出をスキップします。")
                use_gabor = False
                img_shape = None

    if use_gabor:
        from modules_simple.gabor_features import GaborFeatureExtractor

        vprint(f"\nGaborフィルタ特徴抽出中... (方位:{gabor_orientations}, 周波数:{gabor_frequencies}, "
               f"カーネル:{gabor_kernel_size})")

        extractor = GaborFeatureExtractor(
            image_shape=img_shape,
            n_orientations=gabor_orientations,
            n_frequencies=gabor_frequencies,
            kernel_size=gabor_kernel_size,
        )

        gabor_info = extractor.get_info()
        vprint(f"  フィルタ数: {gabor_info['n_filters']}")
        vprint(f"  特徴次元: {gabor_info['feature_dim']} (元の入力: {n_input})")

        x_train_raw = x_train.copy()
        x_test_raw = x_test.copy()

        x_train = extractor.transform(x_train)
        x_test = extractor.transform_test(x_test)

        n_input = x_train.shape[1]
        vprint(f"  特徴抽出完了: 入力次元 → {n_input}")

    class_names = get_class_names(dataset, custom_class_names=custom_class_names)
    if class_names:
        vprint(f"クラス名: {class_names}")

    # ========================================
    # 10. ネットワークの構築
    # ========================================
    vprint("\nネットワーク初期化中...")

    network = run_with_log_control(
        RefinedDistributionEDNetwork,
        n_input=n_input,
        n_hidden=hidden_sizes,
        n_output=n_classes,
        output_lr=output_lr,
        non_column_lr=non_column_lr,
        column_lr=column_lr,
        u1=u1,
        u2=u2,
        base_column_radius=base_column_radius,
        column_neurons=args.column_neurons,
        participation_rate=participation_rate,
        use_hexagonal=True,
        gradient_clip=gradient_clip,
        hidden_sparsity=hidden_sparsity,
        init_scales=init_scales,
        seed=args.seed,
        verbose=verbose,
    )

    # ========================================
    # 11. 可視化マネージャーの初期化
    # ========================================
    viz_manager = None
    if args.viz is not None:
        try:
            viz_scale_map = {1: 1.00, 2: 1.30, 3: 1.60, 4: 2.00}
            viz_scale = viz_scale_map.get(args.viz, 1.00)
            viz_manager = VisualizationManager(
                enable_viz=True,
                enable_heatmap=args.heatmap,
                save_path=args.save_viz,
                total_epochs=args.epochs,
                window_scale=viz_scale,
            )
            vprint(f"\n可視化機能: 有効 (サイズレベル{args.viz}, 倍率x{viz_scale:.1f})")
            if use_gabor and gabor_info is not None:
                viz_manager.set_gabor_info(gabor_info)
        except Exception as e:
            print(f"\n警告: 可視化モジュールの初期化に失敗しました: {e}")
            viz_manager = None

    # ========================================
    # 12. 学習ループ
    # ========================================
    print(f"\n{'='*70}")
    print("学習開始")
    print(f"{'='*70}")

    from tqdm import tqdm

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    best_test_acc = 0.0
    best_epoch = 0
    train_acc_history = []
    test_acc_history = []
    class_acc_history = []

    # ヒートマップ中間更新コールバック
    _heatmap_callback = None
    if viz_manager is not None and args.heatmap:
        def _make_heatmap_callback():
            _x_test = x_test
            _y_test_idx = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test
            _x_test_raw = x_test_raw if use_gabor else None
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
                    progress=progress_str
                )

            def set_epoch(ep):
                _epoch_ref[0] = ep

            callback.set_epoch = set_epoch
            return callback
        _heatmap_callback = _make_heatmap_callback()

    train_errors = None
    pbar = tqdm(range(1, args.epochs + 1), desc="Training", ncols=120)
    for epoch in pbar:
        epoch_start = time.time()
        if _heatmap_callback is not None:
            _heatmap_callback.set_epoch(epoch)

        # 最終エポックの不正解収集判定
        is_final_epoch = (epoch == args.epochs)
        collect_errors = args.show_train_errors and is_final_epoch

        # 訓練
        if collect_errors:
            train_acc, train_loss, train_errors = run_with_log_control(
                network.train_epoch,
                x_train, y_train, collect_errors=True, progress_callback=_heatmap_callback
            )
        else:
            train_acc, train_loss = run_with_log_control(
                network.train_epoch,
                x_train, y_train, progress_callback=_heatmap_callback
            )

        # テスト
        test_acc, test_loss, class_accs = run_with_log_control(
            network.evaluate_parallel,
            x_test, y_test, return_per_class=True
        )

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

        print(f"Epoch {epoch:3d}/{args.epochs}: "
              f"Train={train_acc:.4f} (loss={train_loss:.4f}), "
              f"Test={test_acc:.4f} (loss={test_loss:.4f}), "
              f"Time={epoch_time:.2f}s")

        # 可視化更新
        if viz_manager is not None:
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_test_indices = np.argmax(y_test, axis=1)
            else:
                y_test_indices = y_test

            viz_manager.update_learning_curve(
                train_acc_history,
                test_acc_history,
                x_test,
                y_test_indices,
                network
            )

            if args.heatmap:
                sample_idx = np.random.randint(0, len(x_test))
                sample_x = x_test[sample_idx]
                sample_y_true = y_test_indices[sample_idx]

                z_hiddens, z_output, _ = network.forward(sample_x)
                sample_y_pred = np.argmax(z_output)

                true_class_name = class_names[sample_y_true] if class_names else None
                pred_class_name = class_names[sample_y_pred] if class_names else None

                viz_manager.update_heatmap(
                    epoch=epoch,
                    sample_x=sample_x,
                    sample_y_true=sample_y_true,
                    sample_y_true_name=true_class_name,
                    z_hiddens=z_hiddens,
                    z_output=z_output,
                    sample_y_pred=sample_y_pred,
                    sample_y_pred_name=pred_class_name,
                    sample_x_raw=x_train_raw[sample_idx] if use_gabor and x_train_raw is not None else None
                )

    # ========================================
    # 13. 結果サマリー
    # ========================================
    print(f"\n{'='*70}")
    print("学習完了")
    print(f"{'='*70}")
    print(f"最終精度: Train={train_acc:.4f}, Test={test_acc:.4f}")
    print(f"ベスト精度: Test={best_test_acc:.4f} (Epoch {best_epoch})")

    # 混同行列の計算と表示
    if args.heatmap:
        print(f"\n{'='*70}")
        print("混同行列（テストデータ）")
        print(f"{'='*70}")

        y_pred = []
        y_true = []
        for i in range(len(x_test)):
            _, z_output, _ = network.forward(x_test[i])
            pred_class = np.argmax(z_output)
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                true_class = np.argmax(y_test[i])
            else:
                true_class = y_test[i]
            y_pred.append(pred_class)
            y_true.append(true_class)

        cm = confusion_matrix(y_true, y_pred)

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

        print(f"\n{'-'*70}")
        print("クラス別正解率分析")
        print(f"{'-'*70}")
        class_accuracies = []
        for i in range(n_classes):
            class_acc_val = cm[i, i] / cm[i].sum()
            class_accuracies.append(class_acc_val)
            print(f"クラス{i}: {cm[i, i]:>3}/{cm[i].sum():>3} = {class_acc_val*100:>5.1f}%")

        min_class = np.argmin(class_accuracies)
        max_class = np.argmax(class_accuracies)
        print(f"\n最低正解率: クラス{min_class} ({class_accuracies[min_class]*100:.1f}%)")
        print(f"最高正解率: クラス{max_class} ({class_accuracies[max_class]*100:.1f}%)")
        print(f"正解率範囲: {(class_accuracies[max_class] - class_accuracies[min_class])*100:.1f}ポイント")

    # 可視化の最終処理
    if viz_manager is not None:
        viz_manager.save_figures()
        if args.save_viz:
            print(f"\n可視化結果を保存しました: {args.save_viz}")

    # 不正解学習データの一覧表示
    if args.show_train_errors and train_errors is not None:
        from modules_simple.visualization_manager import show_train_errors
        x_display = x_train_raw if x_train_raw is not None else x_train

        if viz_manager is not None:
            viz_manager.close()

        from collections import Counter
        total_rows = 0
        cls_count = Counter(int(t) for _, t, _ in train_errors)
        for n in cls_count.values():
            capped = min(n, args.max_errors_per_class)
            total_rows += 1 + (capped + 9) // 10
        print(f"\n不正解学習データを表示中... ({len(train_errors)}/{len(x_train)} 件, {total_rows} 行)")
        print("  ウィンドウを閉じると終了します。↑↓キーまたはマウスホイールでスクロール")
        show_train_errors(
            error_list=train_errors,
            x_display=x_display,
            y_train=y_train,
            class_names=class_names,
            img_shape=display_img_shape,
            max_per_class=args.max_errors_per_class
        )

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
