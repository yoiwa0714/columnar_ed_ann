"""
ED-SNN プロジェクトメインモジュール

モジュール構成:
- snn: スパイキングニューラルネットワーク
- ed_learning: エラー拡散学習法 (後で追加)
- utils: 共通ユーティリティ (後で追加)
"""

# SNNモジュール
from .snn import *

# ED学習モジュール
from .ed_learning import *

# ユーティリティモジュール (後で追加)
# from .utils import *

# 現在利用可能なモジュールのみインポート
from .snn import *
# from .ed_learning import *  # 後で追加
# from .utils import *        # 後で追加

__version__ = "0.0.1"
__author__ = "ED-SNN Team"