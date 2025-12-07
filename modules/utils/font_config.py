"""
日本語フォント設定モジュール
ED-SNNプロジェクト共通フォント設定

作成者: ED-SNN開発チーム
作成日: 2025年9月28日
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
from typing import Optional

def setup_japanese_font() -> str:
    """
    Linuxシステム用日本語フォント自動設定
    
    Returns:
    --------
    str: 設定されたフォント名
    """
    
    # 利用可能なフォント一覧を取得
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Linuxシステム用日本語フォントリスト (優先順)
    japanese_fonts = [
        'Noto Sans CJK JP',        # 標準的なLinux日本語フォント
        'Noto Sans JP',            # Google Noto フォント  
        'Takao Gothic',            # Takaoフォント
        'IPAexGothic',            # IPA フォント
        'VL Gothic',              # Vine Linuxフォント
        'Liberation Sans',         # RedHat系フォント
        'DejaVu Sans',             # 最終フォールバック
        'sans-serif'              # システムフォールバック
    ]
    
    # 強制的に Noto Sans CJK JP を最優先で試行
    selected_font = 'Noto Sans CJK JP'
    
    # フォント存在確認（より詳細なマッチング）
    for font in japanese_fonts:
        # 完全一致をまず試行
        if font in available_fonts:
            selected_font = font
            break
        # 部分一致も試行（CJKフォントの名前バリエーション対応）
        for available_font in available_fonts:
            if 'Noto Sans CJK' in available_font and 'JP' in available_font:
                selected_font = available_font
                break
        if selected_font != 'Noto Sans CJK JP' and 'Noto Sans CJK' in selected_font:
            break
    
    # matplotlib設定
    plt.rcParams['font.family'] = selected_font
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False  # マイナス記号文字化け対策
    
    return selected_font

def get_font_info() -> dict:
    """
    現在のフォント設定情報を取得
    
    Returns:
    --------
    dict: フォント設定情報
    """
    return {
        'current_font': plt.rcParams['font.family'],
        'font_size': plt.rcParams['font.size'],
        'platform': platform.system(),
        'available_japanese_fonts': [
            f.name for f in fm.fontManager.ttflist 
            if any(jp in f.name.lower() for jp in ['noto', 'cjk', 'jp', 'japanese', 'gothic', 'ipa'])
        ][:10]  # 上位10個のみ表示
    }

def print_font_status():
    """
    フォント設定状況を表示
    """
    info = get_font_info()
    
    print("📝 日本語フォント設定状況")
    print("=" * 40)
    print(f"現在のフォント: {info['current_font']}")
    print(f"フォントサイズ: {info['font_size']}")
    print(f"プラットフォーム: {info['platform']}")
    print("\n利用可能な日本語関連フォント:")
    for font in info['available_japanese_fonts']:
        print(f"  - {font}")

# 初回インポート時に自動設定
_font_initialized = False

def ensure_japanese_font():
    """
    日本語フォント設定確認・初期化
    """
    global _font_initialized
    if not _font_initialized:
        selected_font = setup_japanese_font()
        print(f"🎨 日本語フォント設定完了: {selected_font}")
        _font_initialized = True

# インポート時に自動実行
ensure_japanese_font()