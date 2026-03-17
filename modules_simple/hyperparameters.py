#!/usr/bin/env python3
"""
層数別ハイパーパラメータ管理モジュール

役割:
  - config/hyperparameters.yaml からパラメータを読み込み
  - 層数に基づく自動パラメータ選択
  - 6層以上のフォールバック処理

使用例:
    from modules_simple.hyperparameters import HyperParams
    
    hp = HyperParams()
    config = hp.get_config(n_layers=2)
"""

import yaml
from pathlib import Path


def load_hyperparameters(config_path=None):
    """
    YAMLファイルからハイパーパラメータを読み込む
    
    Args:
        config_path: YAMLファイルのパス（Noneの場合はデフォルトパスを使用）
    
    Returns:
        dict: YAMLファイルから読み込んだパラメータ辞書
    """
    if config_path is None:
        # columnar_ed_ann/config/ のhyperparameters.yamlを参照
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / 'hyperparameters.yaml'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"設定ファイルが見つかりません: {config_path}\n"
            f"config/hyperparameters_initial.yaml を参照して作成してください。"
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"設定ファイルが空です: {config_path}")
        
        if 'layer_params' not in config:
            raise ValueError(
                f"設定ファイルに 'layer_params' セクションがありません: {config_path}"
            )
        
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"YAMLの解析に失敗しました: {config_path}\n"
            f"エラー: {e}\n"
            f"config/hyperparameters_initial.yaml を参照して修正してください。"
        )


class HyperParams:
    """
    層数依存パラメータをテーブル管理するクラス
    
    config/hyperparameters.yaml から設定を読み込み、
    層数ごとに最適化されたパラメータセットを提供する。
    """
    
    def __init__(self, config_path=None):
        """
        Args:
            config_path: YAMLファイルのパス（Noneの場合はデフォルトパスを使用）
        """
        config = load_hyperparameters(config_path)
        
        self.layer_configs = {}
        for layer_num, params in config['layer_params'].items():
            self.layer_configs[int(layer_num)] = params
        
        self.common_params = config.get('common_params', {})
    
    def get_config(self, n_layers):
        """
        指定層数の設定を取得
        
        Args:
            n_layers: 隠れ層の数（1, 2, 3, 4, 5, または6以上）
        
        Returns:
            dict: 層数に対応した設定辞書
        """
        if n_layers in self.layer_configs:
            config = self.layer_configs[n_layers].copy()
        elif n_layers >= 6:
            config = self.layer_configs[5].copy()
            config['description'] = f'{n_layers}層構成（5層パラメータを使用）'
            print(f"\n*** 注意: {n_layers}層構成は未最適化です。5層のパラメータをフォールバックとして使用します ***\n")
        else:
            supported = list(self.layer_configs.keys())
            raise ValueError(
                f"層数 {n_layers} はサポートされていません。"
                f"サポート層数: {supported} (6層以上は5層のパラメータを使用)"
            )
        
        config.update(self.common_params)
        return config
    
    def list_configs(self):
        """利用可能な設定一覧を表示"""
        print("\n=== 利用可能な層数別設定 ===")
        for n_layers, config in sorted(self.layer_configs.items()):
            print(f"\n[{n_layers}層] {config.get('description', '説明なし')}")
            print(f"  hidden: {config['hidden']}")
            print(f"  output_lr: {config.get('output_lr', 'N/A')}")
            print(f"  non_column_lr: {config.get('non_column_lr', 'N/A')}")
            print(f"  column_lr: {config.get('column_lr', 'N/A')}")
            print(f"  u1: {config.get('u1', 'N/A')}")
            print(f"  u2: {config.get('u2', 'N/A')}")
            print(f"  lateral_lr: {config.get('lateral_lr', 'N/A')}")
            print(f"  base_column_radius: {config.get('base_column_radius', 'N/A')}")
            print(f"  column_radius_per_layer: {config.get('column_radius_per_layer', 'N/A')}")
            print(f"  participation_rate: {config.get('participation_rate', 'N/A')}")
            print(f"  column_neurons: {config.get('column_neurons', 'N/A')}")
            init_scales = config.get('init_scales', config.get('weight_init_scales', 'N/A'))
            print(f"  init_scales: {init_scales}")
            print(f"  hidden_sparsity: {config.get('hidden_sparsity', 'N/A')}")
            print(f"  gradient_clip: {config.get('gradient_clip', 'N/A')}")
            print(f"  epochs: {config['epochs']}")
            if 'gabor_orientations' in config:
                print(f"  gabor_orientations: {config['gabor_orientations']}")
                print(f"  gabor_frequencies: {config['gabor_frequencies']}")
                print(f"  gabor_kernel_size: {config['gabor_kernel_size']}")
        print("\n注: 6層以上の構成は5層のパラメータをフォールバックとして使用します")
