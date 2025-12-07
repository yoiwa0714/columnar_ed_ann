"""
ユーティリティ モジュール

共通のユーティリティ関数:
- 可視化
- パフォーマンス測定・プロファイリング
- デバッグツール
- 日本語フォント設定
"""

from .font_config import setup_japanese_font, ensure_japanese_font, print_font_status
from .profiler import EDSNNProfiler, profile_function, TimingContext, profiler

__all__ = [
    'setup_japanese_font', 'ensure_japanese_font', 'print_font_status',
    'EDSNNProfiler', 'profile_function', 'TimingContext', 'profiler'
]

# ユーティリティコンポーネントは後で実装
# from .data_loader import DataLoader
# from .visualization import Visualizer
# from .performance import PerformanceMonitor

# __all__ = ['DataLoader', 'Visualizer', 'PerformanceMonitor']
__all__ = []