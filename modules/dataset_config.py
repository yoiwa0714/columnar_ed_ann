#!/usr/bin/env python3
"""
データセット別ハイパーパラメータ管理モジュール

コラムED法の本質（ed_network.py 等）とは切り離された補助機能。
config/hyperparameters.yaml の layer_params をベースに、
dataset_overrides セクションの差分を上書き適用して返す。

使い方:
    from modules.dataset_config import DatasetConfig
    dc = DatasetConfig()
    config = dc.get_config(n_layers=2, dataset='fashion')
"""

import numpy as np
from modules.hyperparameters import load_yaml


class DatasetConfig:
    """データセット種別に応じたハイパーパラメータ管理。

    layer_params をベースに dataset_overrides の差分を適用する。
    dataset_overrides セクションがない場合は HyperParams と同じ挙動になる。
    """

    # 既知データセット名の正規化マップ（エイリアス対応）
    DATASET_ALIASES = {
        'mnist':         'mnist',
        'fashion':       'fashion',
        'fashion_mnist': 'fashion',
        'cifar10':       'cifar10',
        'cifar-10':      'cifar10',
    }

    def __init__(self, config_path=None):
        config = load_yaml(config_path)
        self._layer_params = {int(k): v for k, v in config['layer_params'].items()}
        self._common_params = config.get('common_params', {})
        self._gabor_params = config.get('gabor_params', {})
        # dataset_overrides: キーは文字列 or 整数の可能性があるので正規化
        raw_overrides = config.get('dataset_overrides', {})
        self._dataset_overrides = {}
        for ds_key, ds_val in raw_overrides.items():
            # ds_val は {層数: {param: value}} または {} または None
            if ds_val is None:
                ds_val = {}
            normalized_val = {}
            if isinstance(ds_val, dict):
                for layer_key, layer_val in ds_val.items():
                    normalized_val[int(layer_key)] = (layer_val or {})
            self._dataset_overrides[str(ds_key)] = normalized_val

    def get_config(self, n_layers: int, dataset: str = 'default') -> dict:
        """指定層数×データセットのパラメータを返す。

        layer_params[n_layers] をベースに dataset_overrides[dataset][n_layers]
        の差分を上書き適用する。差分がない場合はベースをそのまま返す。

        Args:
            n_layers: 隠れ層数
            dataset: データセット名（'mnist', 'fashion', 'cifar10' または任意の文字列）

        Returns:
            dict: 結合済みパラメータ辞書
        """
        # ベース取得（HyperParams.get_config と同じロジック）
        config = dict(self._common_params)
        if n_layers in self._layer_params:
            config.update(self._layer_params[n_layers])
        else:
            max_layers = max(self._layer_params.keys())
            config.update(self._layer_params[max_layers])
            print(f"注意: {n_layers}層の最適パラメータは未定義です。"
                  f"{max_layers}層の設定をベースに使用します。")

        # データセット名を正規化（エイリアス解決）
        normalized_ds = self.DATASET_ALIASES.get(str(dataset), str(dataset))

        # dataset_overrides から差分を取得
        # 1) 正規化された名前で検索 → 2) 見つからなければ 'default' を使用
        ds_overrides = self._dataset_overrides.get(
            normalized_ds,
            self._dataset_overrides.get('default', {})
        )
        layer_override = ds_overrides.get(n_layers, {})
        if layer_override:
            config.update(layer_override)

        return config

    @property
    def gabor_params(self) -> dict:
        """Gabor特徴抽出パラメータ（HyperParams との互換プロパティ）"""
        return dict(self._gabor_params)

    @property
    def common_params(self) -> dict:
        """共通パラメータ"""
        return dict(self._common_params)

    @property
    def layer_configs(self) -> dict:
        """layer_params の辞書（HyperParams との互換プロパティ）"""
        return dict(self._layer_params)

    def get_layer_counts(self) -> list:
        """定義済み層数の一覧"""
        return sorted(self._layer_params.keys())

    def get_known_datasets(self) -> list:
        """dataset_overrides に定義されたデータセット名の一覧"""
        return sorted(self._dataset_overrides.keys())

    @staticmethod
    def estimate_complexity(x_sample: np.ndarray, n_input: int) -> str:
        """データ複雑さを簡易推定し、近いプリセット名を返す。

        未知データセット向けの補助機能。--dataset で明示指定された場合は使用しない。
        計算コストは最小限（1000サンプルのstd計算、1ms以下）。

        Args:
            x_sample: フラット化済み訓練データ（形状 [N, n_input]、値域 0〜1）
            n_input: 入力次元数

        Returns:
            str: 'cifar10', 'fashion', 'mnist', 'default' のいずれか

        Notes:
            - std_mean の閾値は MNIST(≈0.09) / Fashion-MNIST(≈0.20) の実測値に基づく
            - CIFAR-10: 3チャンネル(n_input=3072) で判定
            - 非標準次元（784, 3072以外）は n_input と std_mean で近似判定
        """
        # カラー画像判定: n_input が 3 の倍数で sqrt(n_input/3) が整数に近い場合
        side3 = int(round((n_input / 3) ** 0.5))
        if side3 * side3 * 3 == n_input and n_input > 1000:
            return 'cifar10'

        # モノクロ画像: 1000サンプルで空間複雑さを計算
        n_sample = min(1000, len(x_sample))
        sample = x_sample[:n_sample]
        # 値域を 0〜1 に正規化（まだされていない場合に備えて）
        if sample.max() > 1.0:
            sample = sample / 255.0
        std_mean = float(np.mean(np.std(sample, axis=0)))

        # 実測値に基づく閾値（Keras経由でダウンロードしたデータで確認済み）:
        #   MNIST:         std_mean ≈ 0.190
        #   Fashion-MNIST: std_mean ≈ 0.275
        #   etlcdb(63×64): std_mean ≈ 0.087
        # MNISTとFashionの中間値 0.23 を境界とする
        if std_mean > 0.23:
            return 'fashion'
        if n_input == 784:
            return 'mnist'

        # 非標準次元（784以外）: std_mean が低く MNIST より単純なデータは 'default'
        return 'default'
