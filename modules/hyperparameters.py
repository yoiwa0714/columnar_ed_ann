#!/usr/bin/env python3
"""
層数別ハイパーパラメータ管理モジュール

役割:
  - 1-5層の最適化済みパラメータテーブル管理
  - 層数に基づく自動パラメータ選択
  - 6層以上のフォールバック処理

クラス:
  - HyperParams: パラメータテーブル管理クラス

使用例:
    from modules.hyperparameters import HyperParams
    
    hp = HyperParams()
    config = hp.get_config(n_layers=2)
    # → 2層構成の最適化済みパラメータを取得
"""


class HyperParams:
    """
    層数依存パラメータをテーブル管理するクラス
    
    設計方針:
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
          base_column_radius=config['base_column_radius'],
          ...
      )
    """
    
    def __init__(self):
        """層数別最適設定テーブルの初期化"""
        
        # 層数別設定テーブル
        # - base=1.0が最適（2025-12-04実験結果）
        # - スケーリング則: radius = base × sqrt(neurons/256)
        # - 各構成は個別に最適化済み
        self.layer_configs = {
            # 1層構成（最適パラメータ確定）
            1: {
                'hidden': [512],
                'learning_rate': 0.20,
                'u1': 0.5,
                'u2': 0.8,
                'lateral_lr': 0.08,
                'base_column_radius': 0.4,
                'column_radius_per_layer': [0.57],  # [sqrt(512/256) * 0.4]
                'participation_rate': 0.1,  # 最適値確定（2025-12-15）
                'epochs': 30,
                'description': '1層最適構成、84.50%テスト精度達成（2025-12-14、pr=0.1確定）'
            },
            
            # 2層構成（最適化中 - 2025-12-19更新）
            2: {
                'hidden': [1024, 512],
                'learning_rate': 0.25,  # 0.7から削減→安定性向上
                'u1': 0.5,
                'u2': 0.8,  # 1層の最適値を適用
                'lateral_lr': 0.08,  # 1層の最適値を適用（0.6は不安定）
                'base_column_radius': 0.4,  # 1層と同じスケール
                'column_radius_per_layer': [0.80, 0.57],  # [sqrt(1024/256) * 0.4, sqrt(512/256) * 0.4]
                'participation_rate': 0.1,  # 1層と共通
                'epochs': 100,  # 十分なエポック数
                'description': '2層最適化中、1024,512構成、1層の安定パラメータを適用（2025-12-19）'
            },
            
            # 3層構成（最適化予定）
            3: {
                'hidden': [256, 128, 64],
                'learning_rate': 0.35,  # 2層の最適値を初期値として使用
                'u1': 0.5,
                'u2': 0.5,
                'lateral_lr': 0.08,
                'base_column_radius': 1.0,
                'column_radius_per_layer': [1.0, 0.71, 0.50],  # [sqrt(256/256), sqrt(128/256), sqrt(64/256)]
                'participation_rate': 1.0,
                'epochs': 50,
                'description': '3層標準構成（最適化予定、現在25.8%）'
            },
            
            # 4層構成（最適化予定）
            4: {
                'hidden': [256, 128, 96, 64],
                'learning_rate': 0.35,  # 2層の最適値を初期値として使用
                'u1': 0.5,
                'u2': 0.5,
                'lateral_lr': 0.08,
                'base_column_radius': 1.0,
                'column_radius_per_layer': [1.0, 0.71, 0.61, 0.50],
                'participation_rate': 1.0,
                'epochs': 50,
                'description': '4層標準構成（最適化予定）'
            },
            
            # 5層構成（最適化予定）
            5: {
                'hidden': [256, 128, 96, 80, 64],
                'learning_rate': 0.35,  # 2層の最適値を初期値として使用
                'u1': 0.5,
                'u2': 0.5,
                'lateral_lr': 0.08,
                'base_column_radius': 1.0,
                'column_radius_per_layer': [1.0, 0.71, 0.61, 0.56, 0.50],
                'participation_rate': 1.0,
                'epochs': 50,
                'description': '5層標準構成（最適化予定）'
            },
        }
        
        # 共通パラメータ（層数非依存）
        self.common_params = {
            'w1': 1.0,  # 重み初期化範囲
            'column_overlap': 0.1,
            'column_ratio': 0.2,
            'time_loops': 2,
            'dataset': 'mnist',
            'train_samples': 10000,
            'test_samples': 2000,
        }
    
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
            print(f"  lateral_lr: {config.get('lateral_lr', 'N/A')}")
            print(f"  base_column_radius: {config['base_column_radius']}")
            print(f"  column_radius_per_layer: {config['column_radius_per_layer']}")
            print(f"  participation_rate: {config.get('participation_rate', 'N/A')}")
            print(f"  epochs: {config['epochs']}")
        print("\n注: 6層以上の構成は5層のパラメータをフォールバックとして使用します")
