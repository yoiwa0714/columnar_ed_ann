"""
PyTorch標準機能を使用したMNIST/Fashion-MNISTデータセット管理

汎用性とed_multi_snn.prompt.md準拠の最適化を重視
プロファイリング機能統合によるボトルネック特定

作成者: ED-SNN開発チーム  
作成日: 2025年9月28日
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional, List, Dict, Any
import os
import time

from .profiler import profile_function, TimingContext


class MNISTDatasetManager:
    """
    PyTorch標準機能を使用したMNIST/Fashion-MNISTデータセット管理
    
    特徴:
    - PyTorchの標準transformsを使用
    - 効率的なバッチ処理
    - ED-SNN向けデータ前処理
    - プロファイリング統合
    """
    
    def __init__(
        self,
        dataset_type: str = 'MNIST',
        data_dir: str = './data',
        batch_size: int = 32,
        normalize: bool = True,
        download: bool = True
    ):
        """
        データセットマネージャー初期化
        
        Parameters:
        -----------
        dataset_type : str
            'MNIST' or 'FashionMNIST'
        data_dir : str
            データ保存ディレクトリ
        batch_size : int
            バッチサイズ
        normalize : bool
            正規化実行フラグ
        download : bool
            自動ダウンロードフラグ
        """
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.normalize = normalize
        
        # データ変換定義
        self.transform = self._create_transform()
        
        # データセット選択
        if dataset_type.upper() == 'MNIST':
            self.dataset_class = torchvision.datasets.MNIST
        elif dataset_type.upper() == 'FASHIONMNIST':
            self.dataset_class = torchvision.datasets.FashionMNIST
        else:
            raise ValueError(f"サポートされていないデータセット: {dataset_type}")
        
        # データセット初期化
        self._initialize_datasets(download)
        
        print(f"✅ {dataset_type}データセット初期化完了")
        print(f"   訓練データ: {len(self.train_dataset)}サンプル")
        print(f"   テストデータ: {len(self.test_dataset)}サンプル")
        print(f"   バッチサイズ: {batch_size}")
    
    def _create_transform(self) -> transforms.Compose:
        """データ変換パイプライン作成"""
        transform_list = [transforms.ToTensor()]
        
        if self.normalize:
            # MNIST/Fashion-MNIST標準正規化
            transform_list.append(
                transforms.Normalize((0.1307,), (0.3081,))
            )
        
        return transforms.Compose(transform_list)
    
    @profile_function("dataset_initialization")
    def _initialize_datasets(self, download: bool):
        """データセット初期化"""
        # 訓練データセット
        self.train_dataset = self.dataset_class(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=download
        )
        
        # テストデータセット
        self.test_dataset = self.dataset_class(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=download
        )
        
        # データローダー
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    @profile_function("get_batch_data")
    def get_batch(self, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        バッチデータ取得
        
        Parameters:
        -----------
        train : bool
            訓練データフラグ
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (images, labels) - ED-SNN形式
        """
        loader = self.train_loader if train else self.test_loader
        
        for batch_images, batch_labels in loader:
            # PyTorchテンソル → NumPy配列
            images = batch_images.numpy()
            labels = batch_labels.numpy()
            
            # 形状変換: (batch, 1, 28, 28) → (batch, 784)
            images = images.reshape(images.shape[0], -1)
            
            # one-hot エンコーディング
            labels_onehot = np.eye(10)[labels]
            
            return images, labels_onehot
    
    @profile_function("get_single_sample")
    def get_single_sample(self, index: int, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        単一サンプル取得（デバッグ用）
        
        Parameters:
        -----------
        index : int
            サンプルインデックス
        train : bool
            訓練データフラグ
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (image, label) - ED-SNN形式
        """
        dataset = self.train_dataset if train else self.test_dataset
        
        image, label = dataset[index]
        
        # Tensor → NumPy, 形状変換
        image_np = image.numpy().reshape(-1)  # (784,)
        label_onehot = np.eye(10)[label]
        
        return image_np, label_onehot
    
    @profile_function("create_small_dataset")
    def create_small_dataset(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        デバッグ用小規模データセット作成
        
        Parameters:
        -----------
        n_samples : int
            サンプル数
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (train_X, train_y, test_X, test_y)
        """
        print(f"🔍 小規模データセット作成 (n={n_samples})")
        
        train_X, train_y = [], []
        test_X, test_y = [], []
        
        # 訓練データ
        for i in range(min(n_samples, len(self.train_dataset))):
            image, label = self.get_single_sample(i, train=True)
            train_X.append(image)
            train_y.append(label)
        
        # テストデータ
        test_size = min(n_samples // 5, len(self.test_dataset))  # 20%をテスト用
        for i in range(test_size):
            image, label = self.get_single_sample(i, train=False)
            test_X.append(image)
            test_y.append(label)
        
        return (np.array(train_X), np.array(train_y), 
                np.array(test_X), np.array(test_y))
    
    def get_class_names(self) -> List[str]:
        """クラス名取得"""
        if self.dataset_type.upper() == 'MNIST':
            return [str(i) for i in range(10)]
        elif self.dataset_type.upper() == 'FASHIONMNIST':
            return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        return []
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """データセット情報取得"""
        return {
            'dataset_type': self.dataset_type,
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset),
            'batch_size': self.batch_size,
            'input_shape': (784,),  # 28x28 flattened
            'num_classes': 10,
            'class_names': self.get_class_names(),
            'normalize': self.normalize
        }


class EDSNNDataProcessor:
    """
    ED-SNN用データ前処理クラス
    
    PyTorchデータとED-SNN形式の効率的な変換
    """
    
    @staticmethod
    @profile_function("normalize_images")
    def normalize_for_ed_snn(images: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        ED-SNN用画像正規化
        
        Parameters:
        -----------
        images : np.ndarray
            入力画像 (batch_size, 784)
        method : str
            正規化方法 ('minmax', 'standard', 'sigmoid')
            
        Returns:
        --------
        np.ndarray
            正規化済み画像 [0, 1]
        """
        if method == 'minmax':
            # Min-Max正規化 [0, 1]
            img_min = np.min(images, axis=1, keepdims=True)
            img_max = np.max(images, axis=1, keepdims=True)
            return (images - img_min) / (img_max - img_min + 1e-8)
        
        elif method == 'standard':
            # Z-score正規化 → Sigmoid変換
            img_mean = np.mean(images, axis=1, keepdims=True)
            img_std = np.std(images, axis=1, keepdims=True)
            z_scores = (images - img_mean) / (img_std + 1e-8)
            return 1.0 / (1.0 + np.exp(-z_scores))
        
        elif method == 'sigmoid':
            # Sigmoid正規化
            return 1.0 / (1.0 + np.exp(-images * 6.0))
        
        else:
            raise ValueError(f"未対応の正規化方法: {method}")
    
    @staticmethod
    @profile_function("augment_data")  
    def augment_for_training(images: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        学習用データ拡張
        
        Parameters:
        -----------
        images : np.ndarray
            入力画像
        noise_level : float
            ノイズレベル
            
        Returns:
        --------
        np.ndarray
            拡張済み画像
        """
        # ガウシアンノイズ追加
        noise = np.random.normal(0, noise_level, images.shape)
        augmented = np.clip(images + noise, 0, 1)
        
        return augmented


def benchmark_dataset_loading():
    """データセット読み込み性能ベンチマーク"""
    print("🔍 データセット読み込み性能測定")
    print("=" * 50)
    
    # MNIST
    with TimingContext("mnist_loading"):
        mnist_manager = MNISTDatasetManager('MNIST', batch_size=64)
        train_X, train_y = mnist_manager.create_small_dataset(1000)
    
    # Fashion-MNIST
    with TimingContext("fashion_mnist_loading"):
        fashion_manager = MNISTDatasetManager('FashionMNIST', batch_size=64)
        fashion_X, fashion_y = fashion_manager.create_small_dataset(1000)
    
    print(f"MNIST形状: {train_X.shape}, {train_y.shape}")
    print(f"Fashion-MNIST形状: {fashion_X.shape}, {fashion_y.shape}")
    
    from .profiler import profiler
    print(profiler.get_performance_report())


if __name__ == "__main__":
    benchmark_dataset_loading()