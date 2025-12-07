"""
データ管理モジュール

MNIST・Fashion-MNISTデータセットの読み込み・前処理機能
"""

from .dataset_manager import MNISTDataManager, create_data_manager, EDSNNDataset

__all__ = ['MNISTDataManager', 'create_data_manager', 'EDSNNDataset']