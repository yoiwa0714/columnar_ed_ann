#!/usr/bin/env python3
"""
Columnar ED-ANN 簡易版（正式版ベース）

README_simple.md で公開している最小限のCLIだけを受け取り、
内部では columnar_ed_ann.py に処理を委譲する。

方針:
- ユーザー向けインターフェースを簡潔に保つ
- 学習ロジックは正式版の最新実装を利用する
- Gabor特徴は簡易版仕様に合わせてデフォルトON
"""

from __future__ import annotations

import argparse
import runpy
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    """README_simple 準拠の最小引数を解析する。"""
    parser = argparse.ArgumentParser(
        description=(
            "Columnar ED-ANN 簡易版\n"
            "正式版ベースの学習実装を、最小限のCLIで利用するためのフロントエンド。"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--hidden", type=str, default="2048",
                        help="隠れ層ニューロン数（例: 2048 / 2048,1024）")
    parser.add_argument("--train", type=int, default=5000,
                        help="訓練サンプル数（デフォルト: 5000）")
    parser.add_argument("--test", type=int, default=5000,
                        help="テストサンプル数（デフォルト: 5000）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="エポック数（未指定時は正式版側のYAML自動設定）")
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
                        help="コラムニューロン数（未指定時は正式版側のYAML自動設定）")
    parser.add_argument("--init_scales", type=str, default=None,
                        help="層別初期化スケール（例: 0.7,1.8,0.8）")

    parser.add_argument("--list_hyperparams", nargs="?", type=int, const=0, default=None,
                        metavar="N_LAYERS",
                        help="ハイパーパラメータ一覧を表示（引数なしで全体、数値で層別）")
    parser.add_argument("--verbose", action="store_true",
                        help="初期化詳細を表示")

    return parser.parse_args()


def build_full_argv(args: argparse.Namespace) -> List[str]:
    """簡易版引数を正式版の引数へ写像する。"""
    forward = [
        "columnar_ed_ann.py",
        "--hidden", args.hidden,
        "--train", str(args.train),
        "--test", str(args.test),
        "--seed", str(args.seed),
        "--dataset", args.dataset,
    ]

    if args.epochs is not None:
        forward.extend(["--epochs", str(args.epochs)])

    if args.viz is not None:
        forward.extend(["--viz", str(args.viz)])
    if args.heatmap:
        forward.append("--heatmap")
    if args.save_viz is not None:
        forward.extend(["--save_viz", args.save_viz])

    if args.show_train_errors:
        forward.append("--show_train_errors")
    if args.max_errors_per_class != 20:
        forward.extend(["--max_errors_per_class", str(args.max_errors_per_class)])

    # 簡易版仕様: GaborデフォルトON
    if not args.no_gabor:
        forward.append("--gabor_features")

    if args.column_neurons is not None:
        forward.extend(["--column_neurons", str(args.column_neurons)])
    if args.init_scales is not None:
        forward.extend(["--init_scales", args.init_scales])

    if args.list_hyperparams is not None:
        forward.append("--list_hyperparams")
        if args.list_hyperparams != 0:
            forward.append(str(args.list_hyperparams))

    if args.verbose:
        forward.append("--verbose")

    return forward


def main() -> None:
    args = parse_args()
    forward = build_full_argv(args)

    # runpy で正式版を __main__ として実行するため、argv を一時差し替え
    original_argv = sys.argv
    try:
        sys.argv = forward
        runpy.run_path("columnar_ed_ann.py", run_name="__main__")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
