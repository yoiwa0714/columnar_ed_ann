]633;E;echo '#!/usr/bin/env python3';a8caf50e-93ff-4be5-a1ab-2bc37fe50e02]633;C#!/usr/bin/env python3
"""Columnar ED-ANN v1.2.0"""

__version__ = "1.2.0"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import time
import sys

from modules.hyperparameters import HyperParams
from modules.dataset_config import DatasetConfig
from modules.data_loader import load_dataset
from modules.ed_network import SimpleColumnEDNetwork


def parse_args():
    """コマンドライン引数の解析（必要最小限）"""
    parser = argparse.ArgumentParser(
        prog='columnar_ed_ann.py',
        description=f'コラムED法 version {__version__}\n'
                    '微分の連鎖律による誤差逆伝播法を使用せず高精度を実現\n'
                    '\n'
                    '層数に応じた最適パラメータが自動適用されます',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--ver', '-V', action='version',
                       version=f'columnar_ed_ann.py v{__version__}')

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
    parser.add_argument('--gc', '--gradient_clip', dest='gradient_clip', type=float, default=None,
                       help='勾配クリッピング閾値（未指定時: YAML設定値）　短縮形: --gc')
    parser.add_argument('--lgc', '--layer_gc', dest='layer_gc', type=str, default=None,
                       help='層別勾配クリッピング（カンマ区切り、長さ=層数）\n'
                            '例: 0.001,0.005,0.01,0.03,0.05,0.1　短縮形: --lgc\n'
                            '指定時は--gcより優先される')
    parser.add_argument('--lut_base_rate', type=float, default=0.0,
                       help='LUTのベース学習率（ランク外ニューロンの最低学習率、デフォルト: 0.0）\n'
                            '例: 0.01 → ランクcn以降も学習率=0.01で学習に参加')
    parser.add_argument('--is', '--init_scales', dest='init_scales', type=str, default=None,
                       help='層別重み初期化スケール（カンマ区切り、長さ=層数+1）\n'
                            '例: 0.7,1.8,1.8,1.8,1.8,1.8,0.8　短縮形: --is')
    parser.add_argument('--hs', '--hidden_sparsity', dest='hidden_sparsity', type=str, default=None,
                       help='層別非コラム重みスパース率（カンマ区切り）\n'
                            '例: 0.2,0.2,0.4,0.4,0.6,0.6　短縮形: --hs')
    parser.add_argument('--ncl', '--non_column_lr', dest='non_column_lr', type=str, default=None,
                       help='層別非コラム学習率（カンマ区切り、長さ=層数）\n'
                            '例: 0.08,0.08,0.04,0.04,0.02,0.02　短縮形: --ncl')
    parser.add_argument('--olr', '--output_lr', dest='output_lr', type=float, default=None,
                       help='出力層学習率（未指定時: YAML設定値）　短縮形: --olr')
    parser.add_argument('--clf', '--column_lr_factors', dest='column_lr_factors', type=str, default=None,
                       help='コラムニューロンの学習率倍率（カンマ区切り、長さ=層数）\n'
                            '例: 0.005,0.003　短縮形: --clf')
    parser.add_argument('--u1', type=float, default=None,
                       help='アミン拡散係数u1（出力層→最終隠れ層、未指定時: YAML設定値）')
    parser.add_argument('--u2', type=float, default=None,
                       help='アミン拡散係数u2（隠れ層間、多層時に使用、未指定時: YAML設定値）')
    parser.add_argument('--lcn', '--layer_column_neurons', dest='layer_column_neurons', type=str, default=None,
                       help='層別コラムニューロン数（カンマ区切り、0=全コラム化）\n'
                            '例: 0,0,0,10,10,10 → 浅層3層を全コラム化、深層3層はcn=10　短縮形: --lcn')
    parser.add_argument('--gks', '--gabor_kernel_size', dest='gabor_kernel_size', type=int, default=None,
                       help='Gaborフィルタのカーネルサイズ（未指定時: YAML設定値）　短縮形: --gks')
    parser.add_argument('--go', '--gabor_orientations', dest='gabor_orientations', type=int, default=None,
                       help='Gaborフィルタの方位数（未指定時: YAML設定値）　短縮形: --go')
    parser.add_argument('--gf', '--gabor_frequencies', dest='gabor_frequencies', type=int, default=None,
                       help='Gaborフィルタの周波数帯域数（未指定時: YAML設定値）　短縮形: --gf')
    # ハイパーパラメータ一覧
    parser.add_argument('--lh', '--list_hyperparams', dest='list_hyperparams',
                       type=int, nargs='?', const=0, default=None,
                       help='ハイパーパラメータ一覧を表示して終了\n'
                            '層数なし: 全層数の簡易一覧\n'
                            '層数指定: その層数の詳細表示\n'
                            '例: --lh (=全層一覧), --lh 2 (=2層詳細)　短縮形: --lh')
    # 可視化
    viz_group = parser.add_argument_group('可視化')
    viz_group.add_argument('--viz', type=int, nargs='?', const=1, default=None,
                          choices=[1, 2, 3, 4], metavar='SIZE',
                          help='学習曲線のリアルタイム可視化を有効化（サイズ指定: 1-4）\n'
                               '1=基準, 2=1.3倍, 3=1.6倍, 4=2倍（ウィンドウサイズ）\n'
                               '数値省略時は1（--viz == --viz 1）')
    viz_group.add_argument('--heatmap', action='store_true',
                          help='活性化ヒートマップの表示（--vizと併用）')
    viz_group.add_argument('--weight_heatmap', action='store_true',
                          help='重みL2ノルムヒートマップの表示（--vizと併用）\n'
                               '各層ニューロンの入力重みL2ノルムをヒートマップ表示。\n'
                               '--heatmapと同じ配置・更新タイミングで別ウィンドウに表示。')
    viz_group.add_argument('--save_viz', type=str, nargs='?', const='viz_results',
                          default=None, metavar='PATH',
                          help='可視化結果を保存（パス指定可）')

    # 不正解データ表示
    error_group = parser.add_argument_group('不正解データ分析')
    error_group.add_argument('--show_train_errors', action='store_true',
                            help='学習完了後、最終エポックの不正解学習データを一覧表示')
    error_group.add_argument('--max_errors_per_class', type=int, default=20,
                            help='不正解表示のクラスごとの上限数（デフォルト: 20）')

    # 重み保存・読み込み
    weight_group = parser.add_argument_group('重み保存・継続学習・アンサンブル')
    weight_group.add_argument('--save_weights', type=str, default=None, metavar='PATH',
                             help='学習完了後に重みを保存するディレクトリ（例: weights/run1）\n'
                                  'ディレクトリ内に weights_<名前>.npz と weights_<名前>.yaml を作成')
    weight_group.add_argument('--save_best', type=str, default=None, metavar='PATH',
                             help='ベスト精度更新時に重みを保存するディレクトリ（注: 上書き保存）\n'
                                  '--save_weights と同時指定可（独立して動作）')
    weight_group.add_argument('--save_overwrite', action='store_true',
                             help='--save_weights / --save_best で同名ファイルへの上書きを許可\n'
                                  '未指定時、既存ファイルがある場合はエラー')
    weight_group.add_argument('--load_weights', type=str, default=None, metavar='PATH',
                             help='保存済み重みを読み込んで継続学習を行う（PATH: .npzファイルまたはそのディレクトリ）\n'
                                  '学習率が元の値より高い場合は警告を表示')
    weight_group.add_argument('--ensemble', type=str, default=None, metavar='PATHS',
                             help='複数の重みをカンマ区切りで指定してアンサンブル推論\n'
                                  '（学習は行わず推論のみ。例: weights/run1,weights/run2,weights/run3）')

    parser.add_argument('--output_weight_decay', type=float, default=0.0,
                       help='出力層の重み減衰率（0.0で無効、推奨: 0.00001）')
    parser.add_argument('--output_gradient_clip', type=float, default=0.0,
                       help='出力層の勾配クリッピング閾値（0.0で無効）')
    parser.add_argument('--uncertainty_modulation', type=float, default=0.0,
                       help='不確実性変調の強度（0.0で無効）。出力エントロピーに比例してアミン信号を増強')
    parser.add_argument('--hc_strength', type=float, default=0.0,
                       help='D4水平結合の強度（0.0で無効）。同クラスコラムニューロン間のゲイン変調')
    parser.add_argument('--pv_nc_gain', type=float, default=0.0,
                       help='PV型NCゲイン変調の強度（0.0で無効）。NCの集団活性でコラムニューロンをゲイン変調')
    parser.add_argument('--pv_pool_mode', type=str, default='nc', choices=['nc', 'all'],
                       help='PV参照プール。nc=非コラムのみ、all=全ニューロン')
    parser.add_argument('--pv_gain_mode', type=str, default='multiplicative',
                       choices=['multiplicative', 'divisive'],
                       help='PVゲイン方式。multiplicative=現行、divisive=除算正規化型')
    parser.add_argument('--homeostatic_rate', type=float, default=None,
                       help='Phase 2 ホメオスタティック調整のスケール幅（0.0で無効）。'
                            'NCニューロンの平均活性外れ値をエポック媻に正規化'
                            '（デフォルト: YAML common_paramsの値、省略時は0.02）')
    parser.add_argument('--vip_modulation', type=float, default=0.0,
                       help='Phase 3 VIP型学習率変調の強度（0.0で無効）。'
                            '最終隠れ層のコラム-NC整合度でアミン信号を脱抑制変調'
                            '（推奨探索範囲: 0.1〜0.5）。'
                            'uncertainty_modulationと同時有効時は最大(1+um)*(1+vm)倍に増幅')
    parser.add_argument('--sst_rate', type=float, default=0.0,
                       help='Phase 4 SST型動的バイアス補正率（0.0で無効）。'
                            '全ニューロンの平均発火率を目標値(sst_target)に近づけるバイアス更新'
                            '（推奨探索範囲: 0.005〜0.1）')
    parser.add_argument('--sst_target', type=float, default=0.3,
                       help='Phase 4 SST目標平均活性（tanh絶対値、デフォルト0.3）')
    parser.add_argument('--skip', type=str, action='append', default=None,
                       help='D7-4スキップ接続。src,dst,alpha形式。複数指定可\n'
                            '例: --skip 0,3,0.1 --skip 1,4,0.1')
    parser.add_argument('--li_strength', type=float, default=0.0,
                       help='D6-1ハード側抑制の強度（0.0で無効）。勝者コラム以外を減衰')
    parser.add_argument('--li_soft_temp', type=float, default=0.0,
                       help='D6-2ソフト側抑制の温度（0.0で無効）。大=弱い抑制、小=強い抑制')
    parser.add_argument('--hebb_strength', type=float, default=0.0,
                       help='D8-1コラム内ヘブ強化の強度（0.0で無効）')
    parser.add_argument('--nc_hebb_lr', type=float, default=0.0,
                       help='D8-3 NCヘブ自己組織化の学習率（0.0で無効）')
    parser.add_argument('--prediction_error_strength', type=float, default=0.0,
                       help='P1層間予測エラー伝播の強度（0.0で無効）。上位層逆投影でアミン変調')
    parser.add_argument('--input_gate_strength', type=float, default=0.0,
                       help='P2 L6フィードバック入力ゲートの強度（0.0で無効）。深層活性で入力ゲーティング')
    parser.add_argument('--attention_boost_strength', type=float, default=0.0,
                       help='P3 L1注意ブーストの強度（0.0で無効）。出力不確実時に浅層ブースト')
    parser.add_argument('--diagnose_plateau', action='store_true',
                       help='各エポック終了時に学習停滞診断情報を出力する')

    args = parser.parse_args()
    # コマンドラインで明示的に指定された引数名セット（" (変更)" 表示用）
    import sys as _sys
    _specified = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.lstrip('-') in _sys.argv or opt in _sys.argv:
                _specified.add(action.dest)
    args._specified = _specified
    return args


def _show_hyperparams(n_layers_requested, hp):
    """ハイパーパラメータ一覧を表示して終了する。"""
    max_layers = max(hp.layer_configs.keys())

    if n_layers_requested == 0:
        # 層数なし: 全層数の縦並び一覧
        print(f"\n{'='*56}")
        print(f"ハイパーパラメータ設定一覧 ({max_layers}層まで対応)")
        for n in sorted(hp.layer_configs.keys()):
            cfg = hp.get_config(n)
            hidden = cfg.get('hidden', '-')
            cn = cfg.get('column_neurons', '-')
            ep = cfg.get('epochs', '-')
            gc = cfg.get('gradient_clip', '-')
            olr = cfg.get('output_lr', '-')
            ncl = cfg.get('non_column_lr', '-')
            hs = cfg.get('hidden_sparsity', '-')
            isc = cfg.get('weight_init_scales', cfg.get('init_scales', '-'))
            clrf = cfg.get('column_lr_factors', '-')
            u1 = cfg.get('u1', '-')
            u2 = cfg.get('u2', '-')
            has_gabor = 'あり' if 'gabor_orientations' in cfg else 'なし'
            print(f"\n{'─'*56}")
            print(f"{n}層構成:")
            print(f"  hidden:            {hidden}")
            print(f"  column_neurons:    {cn}")
            print(f"  epochs:            {ep}")
            print(f"  gradient_clip:     {gc}")
            print(f"  output_lr:         {olr}")
            print(f"  non_column_lr:     {ncl}")
            print(f"  column_lr_factors: {clrf}")
            print(f"  u1:                {u1}")
            print(f"  u2:                {u2}")
            print(f"  hidden_sparsity:   {hs}")
            print(f"  init_scales:       {isc}")
            print(f"  Gabor特徴:         {has_gabor}")
            if 'gabor_orientations' in cfg:
                print(f"  gabor_orientations:{cfg.get('gabor_orientations', 8)}")
                print(f"  gabor_frequencies: {cfg.get('gabor_frequencies', 2)}")
                print(f"  gabor_kernel_size: {cfg.get('gabor_kernel_size', 11)}")
        print(f"\n{'='*56}")
        print(f"詳細表示: python columnar_ed_ann.py --lh <層数>  例: --lh 2")
        print()
    else:
        # 層数指定: 詳細表示
        n = n_layers_requested
        if n not in hp.layer_configs:
            print(f"エラー: {n}層の設定はまだ定義されていません。利用可能: {sorted(hp.layer_configs.keys())}")
            sys.exit(1)
        cfg = hp.get_config(n)
        hidden = cfg.get('hidden', [2048])
        isc = cfg.get('weight_init_scales', cfg.get('init_scales',
                      [0.7] + [1.8]*(n-1) + [0.8] if n > 1 else [0.4, 1.0]))
        hs = cfg.get('hidden_sparsity', [0.4]*n)
        clrf = cfg.get('column_lr_factors', [0.01]*n)
        ncl = cfg.get('non_column_lr', [cfg.get('output_lr', 0.15)]*n)

        print(f"\n{'='*70}")
        print(f"ハイパーパラメータ設定 ({n}層構成)")
        print(f"{'='*70}")
        print(f"  {cfg.get('description', '-')}")

        print(f"\n[ネットワーク構成]")
        print(f"  hidden:             {hidden}")
        print(f"  column_neurons:     {cfg.get('column_neurons', 10)}  (--lcn で層別指定可)")
        print(f"  init_scales:        {isc}  (--is)")
        print(f"  hidden_sparsity:    {hs}  (--hs)")

        print(f"\n[学習率]")
        print(f"  output_lr:          {cfg.get('output_lr', 0.15)}  (--olr)")
        print(f"  non_column_lr:      {ncl}  (--ncl)")
        print(f"  column_lr_factors:  {clrf}  (--clf)")
        print(f"  u1:                 {cfg.get('u1', 0.5)}  (--u1)")
        print(f"  u2:                 {cfg.get('u2', 0.8)}  (--u2)")

        print(f"\n[訓練設定]")
        print(f"  epochs:             {cfg.get('epochs', 10)}  (--epochs)")
        print(f"  gradient_clip:      {cfg.get('gradient_clip', 0.0001)}  (--gc)")

        if 'gabor_orientations' in cfg:
            print(f"\n[Gabor特徴]")
            print(f"  gabor_orientations: {cfg.get('gabor_orientations', 8)}  (--go)")
            print(f"  gabor_frequencies:  {cfg.get('gabor_frequencies', 2)}  (--gf)")
            print(f"  gabor_kernel_size:  {cfg.get('gabor_kernel_size', 11)}  (--gks)")

        print(f"\n{'='*70}")
        hidden_str = ','.join(str(h) for h in hidden)
        cn = cfg.get('column_neurons', 10)
        ep = cfg.get('epochs', 10)
        gc = cfg.get('gradient_clip', 0.0001)
        is_str = ','.join(str(x) for x in isc)
        hs_str = ','.join(str(x) for x in (hs if isinstance(hs, list) else [hs]*n))
        print(f"\n推奨実行コマンド:")
        print(f"  python columnar_ed_ann.py \\")
        print(f"      --hidden {hidden_str} --train 10000 --test 10000 \\")
        print(f"      --lcn {','.join(['10']*n)} --epochs {ep} \\")
        print(f"      --gc {gc} --is {is_str} --hs {hs_str}")
        print()
    sys.exit(0)


def _resolve_save_dir(path):
    """保存先パスを解決する。

    ディレクトリ区切り文字を含まない単純名（例: "run1"）は
    weights/ ディレクトリ下に自動配置する（例: "weights/run1"）。
    パスを含む指定（例: "weights/run1", "/abs/path/run1"）はそのまま使用する。
    """
    import os
    if os.sep not in path and '/' not in path:
        return os.path.join('weights', path)
    return path


def _save_weights(network, save_dir, args, config, best_test_acc, best_epoch,
                  final_train_acc, final_test_acc, allow_overwrite=False,
                  finetune_source=None):
    """重みをnpz＋yamlの分離形式で保存する。

    save_dir/
      weights_<basename>.npz   ← 重み行列（numpy圧縮バイナリ）
      weights_<basename>.yaml  ← ネットワーク構成・精度情報（人間が読める）

    basename はディレクトリ名の末尾部分（タイムスタンプ等）を使用。
    """
    import os
    import yaml
    from datetime import datetime

    os.makedirs(save_dir, exist_ok=True)
    basename = os.path.basename(save_dir.rstrip('/\\'))
    npz_path  = os.path.join(save_dir, f"weights_{basename}.npz")
    yaml_path = os.path.join(save_dir, f"weights_{basename}.yaml")

    # 上書きチェック: --save_overwrite未指定かつ既存ファイルあり → ユーザーに問い合わせ
    if not allow_overwrite and (os.path.exists(npz_path) or os.path.exists(yaml_path)):
        print()
        print(f"既に存在する重みファイルと同名のファイルが重み保存先として指定されました。")
        print(f"  保存先: {save_dir}/")
        print()
        print("  1) 別のファイル名を指定する")
        print("  2) 上書きする (--save_overwrite 指定と同じ)")
        print("  3) 実行を中止する")
        print()
        while True:
            try:
                choice = input("  1から3の番号を選択してください: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n中止しました。")
                return False
            if choice == '1':
                new_name = input("  新しい保存先を入力してください: ").strip()
                if not new_name:
                    print("  入力が空です。再度入力してください。")
                    continue
                new_dir = _resolve_save_dir(new_name)
                return _save_weights(network, new_dir, args, config,
                                     best_test_acc, best_epoch,
                                     final_train_acc, final_test_acc,
                                     allow_overwrite=False,
                                     finetune_source=finetune_source)
            elif choice == '2':
                allow_overwrite = True
                break
            elif choice == '3':
                print("  重みの保存を中止しました。")
                return False
            else:
                print("  1、2、3のいずれかを入力してください。")

    # --- npz 保存 ---
    save_dict = {'w_output': network.w_output}
    for i, w in enumerate(network.w_hidden):
        save_dict[f'w_hidden_{i}'] = w
    np.savez_compressed(npz_path, **save_dict)

    # --- yaml 保存 ---
    # column_neuronsは層別リストまたは単一値
    cn_val = network.column_neurons
    if isinstance(cn_val, list):
        cn_yaml = cn_val
    else:
        cn_yaml = cn_val

    meta = {
        '# ネットワーク構成': None,
        'hidden': list(network.n_hidden),
        'n_input': int(network.n_input),
        'n_output': int(network.n_output),
        'column_neurons': cn_yaml,
        'init_scales': list(config.get('weight_init_scales', [])),
        'hidden_sparsity': config.get('hidden_sparsity', None),
        'output_lr': float(network.layer_lrs[-1] if hasattr(network, 'layer_lrs') and network.layer_lrs else network.learning_rate),
        'gradient_clip': float(network.gradient_clip),
        'u1': float(network.u1),
        'u2': float(network.u2),
        '# 学習条件': None,
        'dataset': args.dataset,
        'n_train': args.train,
        'n_test': args.test,
        'gabor_features': not args.no_gabor,
        'seed': args.seed,
        '# 達成精度': None,
        'best_test_acc': round(float(best_test_acc), 6),
        'best_epoch': int(best_epoch),
        'final_train_acc': round(float(final_train_acc), 6),
        'final_test_acc': round(float(final_test_acc), 6),
        '# 保存情報': None,
        'saved_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'version': __version__,
    }
    if finetune_source is not None:
        meta['finetune_history'] = [finetune_source]

    # コメントキーを除いてYAMLへ書き出す
    yaml_data = {k: v for k, v in meta.items() if not k.startswith('#') and v is not None}
    with open(yaml_path, 'w', encoding='utf-8') as f:
        # セクションコメントを手動挿入しつつ通常のYAMLで出力
        sections = [
            ('# ネットワーク構成', ['hidden', 'n_input', 'n_output', 'column_neurons',
                                    'init_scales', 'hidden_sparsity', 'output_lr',
                                    'gradient_clip', 'u1', 'u2']),
            ('# 学習条件',         ['dataset', 'n_train', 'n_test', 'gabor_features', 'seed']),
            ('# 達成精度',         ['best_test_acc', 'best_epoch', 'final_train_acc', 'final_test_acc']),
            ('# 保存情報',         ['saved_at', 'version', 'finetune_history']),
        ]
        for comment, keys in sections:
            f.write(f"{comment}\n")
            for k in keys:
                if k in yaml_data:
                    v = yaml_data[k]
                    if isinstance(v, list):
                        f.write(f"{k}: {v}\n")
                    elif isinstance(v, bool):
                        f.write(f"{k}: {str(v).lower()}\n")
                    elif isinstance(v, str):
                        f.write(f"{k}: '{v}'\n")
                    else:
                        f.write(f"{k}: {v}\n")
            f.write("\n")

    print(f"  重み保存完了: {save_dir}/")
    print(f"    weights_{basename}.npz  ({os.path.getsize(npz_path) // 1024} KB)")
    print(f"    weights_{basename}.yaml")
    return True


def _load_weights(network, load_path, current_lr=None):
    """保存済み重みをネットワークに読み込む。

    load_path: .npzファイルへの直接パスまたはそのディレクトリ。
    ディレクトリ指定時は内部の唯一の .npz ファイルを自動検索。
    """
    import os
    import glob

    # .npzファイルの特定
    if os.path.isfile(load_path) and load_path.endswith('.npz'):
        npz_path = load_path
    elif os.path.isdir(load_path):
        npz_files = glob.glob(os.path.join(load_path, '*.npz'))
        if len(npz_files) == 0:
            print(f"エラー: {load_path} に .npz ファイルが見つかりません。")
            return False, None
        if len(npz_files) > 1:
            print(f"エラー: {load_path} に複数の .npz ファイルがあります。直接ファイルパスを指定してください。")
            return False, None
        npz_path = npz_files[0]
    else:
        print(f"エラー: --load_weights の指定が不正です: {load_path}")
        return False, None

    # yamlパスの推定
    yaml_path = npz_path.replace('.npz', '.yaml')
    source_info = None
    if os.path.exists(yaml_path):
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            source_meta = yaml.safe_load(f)
        if source_meta:
            # 継続学習の学習率警告
            if current_lr is not None and source_meta.get('output_lr') is not None:
                orig_lr = source_meta['output_lr']
                if current_lr > orig_lr * 1.5:
                    print(f"  警告: 指定された学習率({current_lr:.5f})が元の学習率({orig_lr:.5f})より"
                          f"かなり高い値です。継続学習では元の0.3〜0.5倍程度を推奨します。")
            source_info = {
                'source_path': npz_path,
                'source_acc': source_meta.get('best_test_acc'),
                'source_epoch': source_meta.get('best_epoch'),
            }
            print(f"  元の精度: Test={source_meta.get('best_test_acc', '?'):.4f} "
                  f"(Epoch {source_meta.get('best_epoch', '?')})")

    # 重みの読み込み
    data = np.load(npz_path)

    # 形状チェック
    if 'w_output' not in data:
        print(f"エラー: npzファイルに w_output が含まれていません。")
        return False, None
    if data['w_output'].shape != network.w_output.shape:
        print(f"エラー: 出力層の形状不一致。"
              f"保存済み: {data['w_output'].shape}, 現在: {network.w_output.shape}")
        return False, None
    for i in range(len(network.w_hidden)):
        key = f'w_hidden_{i}'
        if key not in data:
            print(f"エラー: npzファイルに {key} が含まれていません。")
            return False, None
        if data[key].shape != network.w_hidden[i].shape:
            print(f"エラー: 隠れ層{i}の形状不一致。"
                  f"保存済み: {data[key].shape}, 現在: {network.w_hidden[i].shape}")
            return False, None

    # 重みを上書き
    network.w_output = data['w_output'].copy()
    for i in range(len(network.w_hidden)):
        network.w_hidden[i] = data[f'w_hidden_{i}'].copy()

    print(f"  重み読み込み完了: {npz_path}")
    return True, source_info


def _ensemble_predict(networks, x_test, y_test):
    """複数ネットワークの出力確率を平均してアンサンブル推論を行う。"""
    all_outputs = []
    for net in networks:
        outputs = []
        for x in x_test:
            _, z_output, _ = net.forward(x)
            outputs.append(z_output)
        all_outputs.append(np.array(outputs))  # (n_test, n_classes)

    # 出力スコアの平均
    mean_output = np.mean(all_outputs, axis=0)  # (n_test, n_classes)
    y_pred = np.argmax(mean_output, axis=1)

    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.astype(int)

    acc = np.mean(y_pred == y_true)
    return acc, y_pred


def main():
    """メイン処理"""
    args = parse_args()

    # --lh / --list_hyperparams: ハイパーパラメータ一覧を表示して終了
    if args.list_hyperparams is not None:
        hp = DatasetConfig()
        _show_hyperparams(args.list_hyperparams, hp)

    # ========================================
    # 0. パスの解決（単純名 → weights/ 配下に自動配置）
    # ========================================
    if args.save_weights is not None:
        args.save_weights = _resolve_save_dir(args.save_weights)
    if args.save_best is not None:
        args.save_best = _resolve_save_dir(args.save_best)
    if args.load_weights is not None:
        args.load_weights = _resolve_save_dir(args.load_weights)
    if args.ensemble is not None:
        args.ensemble = ','.join(
            _resolve_save_dir(p.strip()) for p in args.ensemble.split(',')
        )

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
    # 2. YAMLから層数別パラメータ自動選択（データセット別差分適用）
    # ========================================
    hp = DatasetConfig()
    config = hp.get_config(n_layers, dataset=args.dataset)

    # CLI指定がない場合は層数依存の隠れ層サイズを使用
    if args.hidden is None:
        hidden_sizes = config['hidden']

    # エポック数の決定
    epochs = args.epochs if args.epochs is not None else config['epochs']

    # gradient_clip CLIオーバーライド
    if args.gradient_clip is not None:
        config['gradient_clip'] = args.gradient_clip

    # 学習率パラメータ
    output_lr = config['output_lr']
    non_column_lr = config['non_column_lr']
    column_lr_factors = config['column_lr_factors']
    # CLIオーバーライド
    if args.output_lr is not None:
        output_lr = args.output_lr
    if args.non_column_lr is not None:
        non_column_lr = [float(x) for x in args.non_column_lr.split(',')]
    if args.column_lr_factors is not None:
        column_lr_factors = [float(x) for x in args.column_lr_factors.split(',')]
    if args.u1 is not None:
        config['u1'] = args.u1
    if args.u2 is not None:
        config['u2'] = args.u2
    # layer_learning_rates: non_column_lr + output_lr
    layer_lrs = list(non_column_lr) + [output_lr]

    # homeostatic_rate: CLI未指定ならYAML common_params の値を使用（省略時は0.0=無効）
    homeostatic_rate = args.homeostatic_rate
    if homeostatic_rate is None:
        homeostatic_rate = float(config.get('homeostatic_rate', 0.0))

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

    # 未知データセットの場合: データ複雑さを自動推定してパラメータを再取得
    # 既知データセット（mnist, fashion, cifar10）は推定不要
    effective_dataset = args.dataset
    if args.dataset not in DatasetConfig.DATASET_ALIASES:
        estimated = DatasetConfig.estimate_complexity(x_train, n_input)
        print(f"未知データセット '{args.dataset}': 複雑さ推定結果 → '{estimated}' に近いパラメータを使用")
        effective_dataset = estimated
        config = hp.get_config(n_layers, dataset=effective_dataset)
        # CLIオーバーライドを再適用（gradient_clip, hidden サイズは既にCLI値が優先済み）
        if args.gradient_clip is not None:
            config['gradient_clip'] = args.gradient_clip
        # hidden_sizes は直後の hidden 変数に反映済みなので config 側は更新不要

    # データセット別クラス名
    _CLASS_NAMES = {
        'fashion': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
        'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck'],
    }
    class_names = _CLASS_NAMES.get(args.dataset)  # MNISTはNone（数字そのまま）

    # Gabor変換前のデータ保持（ヒートマップ用）
    x_train_raw = None
    x_test_raw = None

    # ========================================
    # 5. Gabor特徴抽出（V1単純型細胞モデル）
    # ========================================
    use_gabor = not args.no_gabor
    gabor_info = None
    n_channels = 1  # RGB=3, モノクロ=1
    if use_gabor:
        from modules.gabor_features import GaborFeatureExtractor

        if n_input == 784:
            img_shape = (28, 28)
            n_channels = 1
        elif n_input == 3072:
            # CIFAR-10: 32×32×3 → RGB 3チャネルGabor
            img_shape = (32, 32)
            n_channels = 3
        else:
            side = int(np.sqrt(n_input))
            if side * side == n_input:
                img_shape = (side, side)
                n_channels = 1
            else:
                # カラー画像(H×W×3)の可能性をチェック
                side = int(np.sqrt(n_input / 3))
                if side * side * 3 == n_input:
                    img_shape = (side, side)
                    n_channels = 3
                else:
                    # npyディレクトリの metadata.json から画像形状を取得
                    meta_shape = None
                    if os.path.isdir(args.dataset):
                        import json
                        meta_path = os.path.join(args.dataset, 'metadata.json')
                        if os.path.exists(meta_path):
                            with open(meta_path, 'r', encoding='utf-8') as _f:
                                _meta = json.load(_f)
                            if 'input_shape' in _meta and len(_meta['input_shape']) >= 2:
                                h_m, w_m = _meta['input_shape'][0], _meta['input_shape'][1]
                                if h_m * w_m == n_input:
                                    meta_shape = (h_m, w_m)
                                    n_channels = 1
                                elif h_m * w_m * 3 == n_input:
                                    meta_shape = (h_m, w_m)
                                    n_channels = 3
                    if meta_shape is not None:
                        img_shape = meta_shape
                        print(f"  metadata.jsonから画像形状を取得: {img_shape[0]}×{img_shape[1]} (チャネル: {n_channels})")
                    else:
                        print(f"警告: 入力次元{n_input}の画像形状を推定できません。Gabor無効化。")
                        use_gabor = False
                        img_shape = None

    if use_gabor and n_channels == 3:
        # ヒートマップ用にカラー元画像を保持
        x_train_raw = x_train.copy()
        x_test_raw = x_test.copy()
        # RGB画像をチャネル別平坦化: (N, H*W*3) → (N, H*W*3) [R全画素|G全画素|B全画素]
        h, w = img_shape
        x_train_color = x_train.reshape(-1, h, w, 3)
        x_test_color = x_test.reshape(-1, h, w, 3)
        # HWC → CHW順に並べ替え（チャネル別独立Gabor用）
        x_train = x_train_color.transpose(0, 3, 1, 2).reshape(-1, 3 * h * w)
        x_test = x_test_color.transpose(0, 3, 1, 2).reshape(-1, 3 * h * w)
        print(f"  RGB 3チャネルGabor: {h}×{w}×3 → チャネル別独立処理")

    if use_gabor:
        gp = hp.gabor_params
        # CLIオーバーライド
        if args.gabor_kernel_size is not None:
            gp['kernel_size'] = args.gabor_kernel_size
        if args.gabor_orientations is not None:
            gp['orientations'] = args.gabor_orientations
        if args.gabor_frequencies is not None:
            gp['frequencies'] = args.gabor_frequencies
        ch_label = f", チャネル: {n_channels}" if n_channels > 1 else ""
        print(f"Gabor特徴抽出中... (方位: {gp['orientations']}, 周波数: {gp['frequencies']}, "
              f"カーネル: {gp['kernel_size']}{ch_label})")

        extractor = GaborFeatureExtractor(
            image_shape=img_shape,
            n_orientations=gp['orientations'],
            n_frequencies=gp['frequencies'],
            kernel_size=gp['kernel_size'],
            pool_size=gp['pool_size'],
            pool_stride=gp['pool_stride'],
            include_edge_filters=True,
            n_channels=n_channels
        )

        gabor_info = extractor.get_info()
        print(f"  フィルタ数: {gabor_info['n_filters']}×{n_channels}ch, "
              f"特徴次元: {gabor_info['feature_dim']} (元: {n_input})")

        # 変換前のデータを保持（ヒートマップ用、カラー画像の場合は既に保存済み）
        if x_train_raw is None:
            x_train_raw = x_train.copy()
            x_test_raw = x_test.copy()

        x_train = extractor.transform(x_train)
        x_test = extractor.transform_test(x_test)
        n_input = x_train.shape[1]

    # ========================================
    # 6. ネットワーク構築
    # ========================================
    print(f"\nネットワーク構築中... ({n_layers}層: {hidden_sizes})")

    # 層別コラムニューロン数
    cn_config = config['column_neurons']
    if args.layer_column_neurons is not None:
        cn_config = [int(x) for x in args.layer_column_neurons.split(',')]

    # CLIからのinit_scales/hidden_sparsityパース
    actual_init_scales = config['weight_init_scales']
    if args.init_scales is not None:
        actual_init_scales = [float(x) for x in args.init_scales.split(',')]
    actual_hidden_sparsity = config.get('hidden_sparsity', 0.4)
    if args.hidden_sparsity is not None:
        actual_hidden_sparsity = [float(x) for x in args.hidden_sparsity.split(',')]

    # 層別勾配クリッピング
    layer_gc_list = None
    if args.layer_gc is not None:
        layer_gc_list = [float(x) for x in args.layer_gc.split(',')]

    # D7-4: スキップ接続の解析
    skip_connections = []
    if args.skip is not None:
        for spec in args.skip:
            parts = spec.split(',')
            if len(parts) == 3:
                skip_connections.append((int(parts[0]), int(parts[1]), float(parts[2])))

    network = SimpleColumnEDNetwork(
        n_input=n_input,
        n_hidden=hidden_sizes,
        n_output=n_classes,
        learning_rate=output_lr,
        u1=config['u1'],
        u2=config['u2'],
        base_column_radius=config.get('base_column_radius', 0.4),
        column_neurons=cn_config,
        participation_rate=config.get('participation_rate', 0.1),
        use_hexagonal=config.get('use_hexagonal', True),
        gradient_clip=config['gradient_clip'],
        layer_gradient_clips=layer_gc_list,
        lut_base_rate=args.lut_base_rate,
        hidden_sparsity=actual_hidden_sparsity,
        column_lr_factors=column_lr_factors,
        init_scales=actual_init_scales,
        layer_learning_rates=layer_lrs,
        output_weight_decay=args.output_weight_decay,
        output_gradient_clip=args.output_gradient_clip,
        uncertainty_modulation=args.uncertainty_modulation,
        hc_strength=args.hc_strength,
        pv_nc_gain=args.pv_nc_gain,
        pv_pool_mode=args.pv_pool_mode,
        pv_gain_mode=args.pv_gain_mode,
        homeostatic_rate=homeostatic_rate,
        vip_modulation=args.vip_modulation,
        sst_rate=args.sst_rate,
        sst_target=args.sst_target,
        skip_connections=skip_connections,
        li_strength=args.li_strength,
        li_soft_temp=args.li_soft_temp,
        hebb_strength=args.hebb_strength,
        nc_hebb_lr=args.nc_hebb_lr,
        prediction_error_strength=args.prediction_error_strength,
        input_gate_strength=args.input_gate_strength,
        attention_boost_strength=args.attention_boost_strength,
        seed=args.seed,
    )

    # ========================================
    # 6-A. --load_weights: 継続学習用の重み読み込み
    # ========================================
    load_source_info = None
    if args.load_weights is not None:
        print(f"\n重みを読み込み中: {args.load_weights}")
        ok, load_source_info = _load_weights(network, args.load_weights,
                                             current_lr=output_lr)
        if not ok:
            sys.exit(1)

    # ========================================
    # 6-B. --ensemble: アンサンブル推論（学習なし）
    # ========================================
    if args.ensemble is not None:
        ensemble_paths = [p.strip() for p in args.ensemble.split(',')]
        print(f"\nアンサンブル推論: {len(ensemble_paths)} モデル")
        networks_ens = []
        for path in ensemble_paths:
            print(f"  読み込み: {path}")
            ok, _ = _load_weights(network, path)
            if not ok:
                sys.exit(1)
            import copy
            networks_ens.append(copy.deepcopy(network))

        print(f"\nアンサンブル推論中 ({len(x_test)} サンプル)...")
        acc, _ = _ensemble_predict(networks_ens, x_test, y_test)
        print(f"\n{'='*60}")
        print(f"アンサンブル結果: Test={acc:.4f} ({len(ensemble_paths)} モデル平均)")
        print(f"{'='*60}")
        return
    # ========================================
    viz_manager = None
    if args.viz is not None:
        try:
            from modules.visualization_manager import VisualizationManager
            viz_scale_map = {1: 1.00, 2: 1.30, 3: 1.60, 4: 2.00}
            viz_scale = viz_scale_map.get(args.viz, 1.00)
            viz_manager = VisualizationManager(
                enable_viz=True,
                enable_heatmap=args.heatmap,
                enable_weight_heatmap=args.weight_heatmap,
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
    def _p(label, value, dest=None):
        """パラメータ1行表示。dest が明示指定された場合は " (変更)" を付ける。"""
        changed = f" (変更)" if dest and dest in args._specified else ""
        print(f"  {label:<20} {value}{changed}")

    print("\n" + "=" * 60)
    print(f"パラメータ設定（{n_layers}層構成）")
    print("=" * 60)
    print(f"  # --- ネットワーク構成 ---")
    _p("hidden:",            hidden_sizes,            "hidden")
    _p("column_neurons:",    cn_config,               "layer_column_neurons")
    _p("init_scales:",       actual_init_scales,      "init_scales")
    _p("hidden_sparsity:",   actual_hidden_sparsity,  "hidden_sparsity")
    print(f"  # --- 学習率 ---")
    _p("output_lr:",         output_lr,               "output_lr")
    _p("non_column_lr:",     non_column_lr,           "non_column_lr")
    _p("column_lr_factors:", column_lr_factors,       "column_lr_factors")
    _p("u1:",                config['u1'],             "u1")
    _p("u2:",                config['u2'],             "u2")
    print(f"  # --- 訓練設定 ---")
    _p("gradient_clip:",     config['gradient_clip'],  "gradient_clip")
    if layer_gc_list:
        _p("layer_gc:",      layer_gc_list,            "layer_gc")
    if args.uncertainty_modulation > 0:
        _p("uncertainty_mod:", args.uncertainty_modulation, "uncertainty_modulation")
    if args.hc_strength > 0:
        _p("hc_strength:", args.hc_strength, "hc_strength")
    if args.pv_nc_gain > 0:
        _p("pv_nc_gain:", args.pv_nc_gain, "pv_nc_gain")
        _p("pv_pool_mode:", args.pv_pool_mode, "pv_pool_mode")
        _p("pv_gain_mode:", args.pv_gain_mode, "pv_gain_mode")
    if homeostatic_rate > 0:
        _p("homeostatic_rate:", homeostatic_rate, "homeostatic_rate")
    if args.vip_modulation > 0:
        _p("vip_modulation:", args.vip_modulation, "vip_modulation")
    if args.sst_rate > 0:
        _p("sst_rate:", args.sst_rate, "sst_rate")
        _p("sst_target:", args.sst_target, "sst_target")
    if skip_connections:
        _p("skip:", skip_connections, "skip")
    if args.li_strength > 0:
        _p("li_strength:", args.li_strength, "li_strength")
    if args.li_soft_temp > 0:
        _p("li_soft_temp:", args.li_soft_temp, "li_soft_temp")
    if args.hebb_strength > 0:
        _p("hebb_strength:", args.hebb_strength, "hebb_strength")
    if args.nc_hebb_lr > 0:
        _p("nc_hebb_lr:", args.nc_hebb_lr, "nc_hebb_lr")
    if args.prediction_error_strength > 0:
        _p("pred_error:", args.prediction_error_strength, "prediction_error_strength")
    if args.input_gate_strength > 0:
        _p("input_gate:", args.input_gate_strength, "input_gate_strength")
    if args.attention_boost_strength > 0:
        _p("attn_boost:", args.attention_boost_strength, "attention_boost_strength")
    _p("epochs:",            epochs,                   "epochs")
    _p("seed:",              args.seed,                "seed")
    print(f"  # --- Gabor特徴 ---")
    if use_gabor:
        gp = hp.gabor_params
        _p("gabor_features:",    True,                    None)
        _p("gabor_orientations:", gp['orientations'],     "gabor_orientations")
        _p("gabor_frequencies:", gp['frequencies'],       "gabor_frequencies")
        _p("gabor_kernel_size:", gp['kernel_size'],       "gabor_kernel_size")
    else:
        _p("gabor_features:",    False,                   None)
    print(f"  # --- 重み保存・継続学習 ---")
    _p("save_weights:",      args.save_weights or "-",  "save_weights")
    _p("save_best:",         args.save_best or "-",     "save_best")
    _p("load_weights:",      args.load_weights or "-",  "load_weights")
    _p("ensemble:",          args.ensemble or "-",      "ensemble")
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
                sample_y_true_name=class_names[y_true] if class_names else str(y_true),
                z_hiddens=z_h,
                z_output=z_o,
                sample_y_pred=y_pred,
                sample_y_pred_name=class_names[y_pred] if class_names else str(y_pred),
                sample_x_raw=x_test_raw[idx] if x_test_raw is not None else None,
                progress=f"{sample_i}/{n_samples}"
            )
            if args.weight_heatmap:
                viz_manager.update_weight_heatmap(
                    epoch=epoch_ref[0],
                    network=net,
                    progress=f"{sample_i}/{n_samples}"
                )

    # 重みヒートマップコールバック（--heatmapなしで--weight_heatmapのみの場合）
    weight_heatmap_callback = None
    if viz_manager is not None and args.weight_heatmap and not args.heatmap:
        weight_heatmap_epoch_ref = [0]

        def weight_heatmap_callback(net, sample_i, n_samples):
            viz_manager.update_weight_heatmap(
                epoch=weight_heatmap_epoch_ref[0],
                network=net,
                progress=f"{sample_i}/{n_samples}"
            )

    pbar = tqdm(range(1, epochs + 1), desc="Training", ncols=100)
    for epoch in pbar:
        epoch_start = time.time()

        if heatmap_callback is not None:
            epoch_ref[0] = epoch
        if weight_heatmap_callback is not None:
            weight_heatmap_epoch_ref[0] = epoch

        # 統計リセット
        network.reset_winner_selection_stats()
        network.reset_class_training_stats()

        # 訓練（オンライン学習: 1サンプルずつ順伝播→重み更新）
        cb = heatmap_callback or weight_heatmap_callback
        train_acc, train_loss = network.train_epoch(
            x_train, y_train, progress_callback=cb
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
            # --save_best: ベスト更新時に都度保存
            if args.save_best is not None:
                _save_weights(network, args.save_best, args, config,
                              best_test_acc, best_epoch,
                              final_train_acc=train_acc, final_test_acc=test_acc,
                              allow_overwrite=True,  # ベストは常に上書き
                              finetune_source=load_source_info)

        pbar.set_postfix({
            'Train': f'{train_acc:.4f}',
            'Test': f'{test_acc:.4f}',
            'Best': f'{best_test_acc:.4f}',
        })

        print(f"Epoch {epoch:3d}/{epochs}: "
              f"Train={train_acc:.4f}, Test={test_acc:.4f}, "
              f"Best={best_test_acc:.4f} (ep{best_epoch}), "
              f"Time={epoch_time:.1f}s")

        # 学習停滞診断
        if args.diagnose_plateau:
            _y_diag = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test
            diag = network.collect_epoch_diagnostics(x_test, _y_diag, n_diag=500)
            print(f"  [診断] 活性化: ", end="")
            for li, la in enumerate(diag['layer_activation']):
                print(f"L{li}(mean={la['abs_mean']:.3f},sat={la['saturation_rate']:.1%},std={la['std']:.3f}) ", end="")
            print()
            print(f"  [診断] アミン: ", end="")
            for li, am in enumerate(diag['layer_amine']):
                print(f"L{li}({am['mean']:.6f}) ", end="")
            print()
            print(f"  [診断] 重み: ", end="")
            for li, ws in enumerate(diag['weight_stats']):
                print(f"L{li}(norm={ws['norm']:.1f},max={ws['max']:.4f}) ", end="")
            ow = diag['output_weight']
            print(f"Out(norm={ow['norm']:.1f},max={ow['max']:.4f})")
            print(f"  [診断] 出力: true_score={diag['output_scores']['true_score_mean']:.4f}, "
                  f"max_score={diag['output_scores']['max_score_mean']:.4f}")
            # クラス別精度（ワースト3）
            ca = diag['class_accuracy']
            worst = sorted(ca.items(), key=lambda x: x[1])[:3]
            class_names_local = class_names if class_names else {i: str(i) for i in range(network.n_output)}
            worst_str = ", ".join([f"C{c}({class_names_local[c] if class_names else c})={a:.1%}" for c, a in worst])
            print(f"  [診断] ワースト3: {worst_str}")

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
                    sample_y_true_name=class_names[y_true] if class_names else str(y_true),
                    z_hiddens=z_hiddens,
                    z_output=z_output,
                    sample_y_pred=y_pred,
                    sample_y_pred_name=class_names[y_pred] if class_names else str(y_pred),
                    sample_x_raw=x_test_raw[sample_idx] if x_test_raw is not None else None,
                )

            if args.weight_heatmap:
                viz_manager.update_weight_heatmap(
                    epoch=epoch,
                    network=network,
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

    # --save_weights: 学習完了後に保存
    if args.save_weights is not None:
        print(f"\n重みを保存中: {args.save_weights}")
        _save_weights(network, args.save_weights, args, config,
                      best_test_acc, best_epoch,
                      final_train_acc=train_acc, final_test_acc=test_acc,
                      allow_overwrite=args.save_overwrite,
                      finetune_source=load_source_info)

    # 可視化の保存
    if viz_manager is not None:
        viz_manager.save_figures()
        if args.save_viz:
            print(f"\n可視化結果を保存: {args.save_viz}")

    # ========================================
    # 11. 不正解学習データの一覧表示
    # ========================================
    if args.show_train_errors:
        _, _, train_errors = network.evaluate_with_errors(x_train, y_train)

        if not train_errors:
            print("\n不正解サンプルはありませんでした。")
        else:
            # データセット別クラス名
            dataset_class_names = {
                'fashion': ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
                'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck'],
            }
            class_names = dataset_class_names.get(args.dataset, None)

            # 画像形状
            dataset_img_shapes = {
                'mnist': (28, 28),
                'fashion': (28, 28),
                'cifar10': (32, 32, 3),
            }
            img_shape = dataset_img_shapes.get(args.dataset, (28, 28))

            # 表示用データ（Gabor変換前のデータがあればそちらを使用）
            x_display = x_train_raw if x_train_raw is not None else x_train

            # viz/heatmapウィンドウを先に閉じる
            if viz_manager is not None:
                viz_manager.close()

            from collections import Counter
            cls_count = Counter(int(t) for _, t, _ in train_errors)
            total_rows = sum(1 + (min(n, args.max_errors_per_class) + 9) // 10
                           for n in cls_count.values())
            print(f"\n不正解学習データを表示中... ({len(train_errors)}/{len(x_train)} 件, {total_rows} 行)")
            print("  ウィンドウを閉じると終了します。↑↓キーまたはマウスホイールでスクロール")

            from modules.visualization_manager import show_train_errors
            show_train_errors(
                error_list=train_errors,
                x_display=x_display,
                y_train=y_train,
                class_names=class_names,
                img_shape=img_shape,
                max_per_class=args.max_errors_per_class
            )

    print()


if __name__ == '__main__':
    main()
