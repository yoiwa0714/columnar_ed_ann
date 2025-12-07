"""
ED Learning モジュール

Error Diffusion 学習アルゴリズムの実装:
- 純粋なED法（金子勇氏の理論準拠）
- アミン拡散メカニズム
- 独立出力ニューロンアーキテクチャ
- 興奮性・抑制性制約
"""

# 機能するマルチクラス分類ED法実装
from .ed_core import EDCore
# from .amine_diffusion import AmineDiffusion  # 後で実装

__all__ = ['EDCore']