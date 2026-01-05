#!/usr/bin/env python3
"""
層数別ハイパーパラメータ管理モジュール

役割:
  - 1-5層の最適化済みパラメータテーブル管理
  - 層数に基づく自動パラメータ選択
  - 6層以上のフォールバック処理
  - 外部YAMLファイルからの設定読み込み

関数:
  - load_hyperparameters: YAMLファイルからパラメータを読み込む

クラス:
  - HyperParams: パラメータテーブル管理クラス

使用例:
    from modules.hyperparameters import HyperParams
    
    hp = HyperParams()
    config = hp.get_config(n_layers=2)
    # → 2層構成の最適化済みパラメータを取得
"""

import yaml
import os
from pathlib import Path


def load_hyperparameters(config_path=None):
    """
    YAMLファイルからハイパーパラメータを読み込む
    
    Args:
        config_path: YAMLファイルのパス（Noneの場合はデフォルトパスを使用）
    
    Returns:
        dict: YAMLファイルから読み込んだパラメータ辞書
    
    Raises:
        FileNotFoundError: YAMLファイルが見つからない場合
        yaml.YAMLError: YAMLの解析エラー
    
    Notes:
        - デフォルトパス: config/hyperparameters.yaml
        - エラー時はconfig/hyperparameters_initial.yamlを参照してください
    """
    if config_path is None:
        # デフォルトパス: プロジェクトルート/config/hyperparameters.yaml
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
    
    設計方針:
      - 外部YAMLファイルから設定を読み込み
      - 層数ごとに最適化されたパラメータセットを提供
      - column_radius等の層数依存パラメータを一元管理
      - 初学者にも分かりやすいテーブル形式
      - コマンドライン引数でオーバーライド可能
    
    使用例:
      hp = HyperParams()
      config = hp.get_config(n_layers=2)
      network = RefinedDistributionEDNetwork(
          input_dim=784,
          hidden_layers=config['hidden'],
          column_radius=config['column_radius'],
          ...
      )
    """
    
    def __init__(self, config_path=None):
        """
        YAMLファイルから設定を読み込んで初期化
        
        Args:
            config_path: YAMLファイルのパス（Noneの場合はデフォルトパスを使用）
        """
        # YAMLファイルから設定を読み込み
        config = load_hyperparameters(config_path)
        
        # 層数別設定テーブル
        self.layer_configs = {}
        for layer_num, params in config['layer_params'].items():
            self.layer_configs[int(layer_num)] = params
        
        # 共通パラメータ（層数非依存）
        self.common_params = config.get('common_params', {})
    
    def get_config(self, n_layers):
        """
        指定層数の設定を取得
        
        Args:
            n_layers: 隠れ層の数（1, 2, 3, 4, 5, または6以上）
        
        Returns:
            dict: 層数に対応した設定辞書
        
        Notes:
            6層以上の場合は5層のパラメータをフォールバックとして使用
        """
        if n_layers in self.layer_configs:
            # テーブルに存在する層数
            config = self.layer_configs[n_layers].copy()
        elif n_layers >= 6:
            # 6層以上: 5層のパラメータを使用（フォールバック）
            config = self.layer_configs[5].copy()
            config['description'] = f'{n_layers}層構成（5層パラメータを使用）'
            print(f"\n*** 注意: {n_layers}層構成は未最適化です。5層のパラメータをフォールバックとして使用します ***\n")
        else:
            supported = list(self.layer_configs.keys())
            raise ValueError(
                f"層数 {n_layers} はサポートされていません。"
                f"サポート層数: {supported} (6層以上は5層のパラメータを使用)"
            )
        
        # 層数依存パラメータと共通パラメータをマージ
        config.update(self.common_params)
        return config
    
    def list_configs(self):
        """利用可能な設定一覧を表示"""
        print("\n=== 利用可能な層数別設定 ===")
        for n_layers, config in sorted(self.layer_configs.items()):
            print(f"\n[{n_layers}層] {config['description']}")
            print(f"  hidden_layers: {config['hidden']}")
            print(f"  learning_rate: {config['learning_rate']}")
            print(f"  u1: {config.get('u1', 'N/A')}")
            print(f"  u2: {config.get('u2', 'N/A')}")
            print(f"  weight_decay: {config.get('weight_decay', 'N/A')}")
            print(f"  participation_rate: {config.get('participation_rate', 'N/A')}")
            print(f"  weight_init_scales: {config.get('weight_init_scales', 'N/A')}")
            print(f"  epochs: {config['epochs']}")
        print("\n注: 6層以上の構成は5層のパラメータをフォールバックとして使用します")


# モジュール読み込み時に早期エラーチェック
# YAMLファイルの存在と構文の妥当性を確認
try:
    _test_config = load_hyperparameters()
    del _test_config  # テスト用の変数を削除
except Exception as e:
    import sys
    print(f"\n⚠️  警告: ハイパーパラメータ設定の読み込みに失敗しました", file=sys.stderr)
    print(f"エラー: {e}", file=sys.stderr)
    print(f"config/hyperparameters_initial.yaml を参照して修正してください。\n", file=sys.stderr)
    # エラーを出力するが、モジュールの読み込みは継続（他の機能は使える）
