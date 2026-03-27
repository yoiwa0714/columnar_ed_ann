#!/usr/bin/env python3
"""
ハイパーパラメータ管理モジュール（教育用シンプル版）

config/hyperparameters.yaml を読み込み、
層数に応じた最適パラメータを自動選択する。
"""

import yaml
from pathlib import Path


def load_yaml(config_path=None):
    """YAMLファイルからパラメータを読み込む"""
    if config_path is None:
        module_dir = Path(__file__).resolve().parent
        project_root = module_dir.parent
        config_path = project_root / 'config' / 'hyperparameters.yaml'

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if config is None or 'layer_params' not in config:
        raise ValueError(f"設定ファイルに 'layer_params' セクションがありません: {config_path}")

    return config


class HyperParams:
    """層数別パラメータ管理"""

    def __init__(self, config_path=None):
        config = load_yaml(config_path)
        self.layer_configs = {int(k): v for k, v in config['layer_params'].items()}
        self.common_params = config.get('common_params', {})
        self.gabor_params = config.get('gabor_params', {})

    def get_config(self, n_layers):
        """
        指定層数の設定を取得

        共通パラメータをベースに、層別パラメータで上書きして返す。
        """
        config = dict(self.common_params)

        if n_layers in self.layer_configs:
            config.update(self.layer_configs[n_layers])
        else:
            max_layers = max(self.layer_configs.keys())
            config.update(self.layer_configs[max_layers])
            print(f"注意: {n_layers}層の最適パラメータは未定義です。"
                  f"{max_layers}層の設定をベースに使用します。")

        return config
