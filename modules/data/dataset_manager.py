"""
PyTorchベースデータセット管理モジュール

MNIST・Fashion-MNISTデータの標準的な読み込み・前処理機能
ED-SNNプロジェクト用データローダー

作成者: ED-SNN開発チーム
作成日: 2025年9月28日
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Dict, Any, Optional
import os
from pathlib import Path


class EDSNNDataset(Dataset):
    """
    ED-SNN用データセットラッパー
    
    PyTorchのMNIST/Fashion-MNISTデータをED-SNN形式に変換
    """
    
    def __init__(self, pytorch_dataset, transform=None):
        """
        Parameters:
        -----------
        pytorch_dataset : torchvision.datasets
            PyTorchの標準データセット
        transform : callable, optional
            追加の変換処理
        """
        self.pytorch_dataset = pytorch_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.pytorch_dataset)
    
    def __getitem__(self, idx):
        image, label = self.pytorch_dataset[idx]
        
        # PIL Image → Tensor → NumPy (flatten)
        if isinstance(image, torch.Tensor):
            # 既にTensorの場合
            image_array = image.view(-1).numpy()  # flatten to 784
        else:
            # PIL Imageの場合
            image_tensor = transforms.ToTensor()(image)
            image_array = image_tensor.view(-1).numpy()  # flatten to 784
            
        # 正規化 (0-1範囲)
        image_array = image_array.astype(np.float32)
        
        # ラベルをone-hot エンコーディング
        label_onehot = np.zeros(10, dtype=np.float32)
        label_onehot[label] = 1.0
        
        if self.transform:
            image_array = self.transform(image_array)
            
        return image_array, label_onehot, label


class MNISTDataManager:
    """
    MNIST・Fashion-MNISTデータ管理クラス
    
    PyTorch標準機能を使用した汎用的なデータローダー
    """
    
    def __init__(
        self, 
        data_root: str = './data',
        dataset_type: str = 'mnist',
        batch_size: int = 32,
        num_workers: int = 2,
        download: bool = True
    ):
        """
        Parameters:
        -----------
        data_root : str
            データ保存ディレクトリ
        dataset_type : str
            'mnist' or 'fashion_mnist'
        batch_size : int
            バッチサイズ
        num_workers : int
            データローダーのワーカー数
        download : bool
            データセット自動ダウンロード
        """
        self.data_root = Path(data_root)
        self.dataset_type = dataset_type.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        
        # データ保存ディレクトリ作成
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # データセット初期化
        self._initialize_datasets()
        
        # データローダー作成
        self._create_dataloaders()
        
        print(f"📊 {self.dataset_type.upper()}データセット準備完了")
        print(f"   訓練データ: {len(self.train_dataset):,}サンプル")
        print(f"   テストデータ: {len(self.test_dataset):,}サンプル")
        print(f"   バッチサイズ: {batch_size}")
        
    def _initialize_datasets(self):
        """データセット初期化"""
        
        # 基本変換
        transform = transforms.Compose([
            transforms.ToTensor(),
            # 正規化は不要（EDSNNDatasetで実施）
        ])
        
        if self.dataset_type == 'mnist':
            # MNIST
            train_dataset = torchvision.datasets.MNIST(
                root=str(self.data_root), 
                train=True,
                transform=transform,
                download=self.download
            )
            test_dataset = torchvision.datasets.MNIST(
                root=str(self.data_root),
                train=False, 
                transform=transform,
                download=self.download
            )
            
        elif self.dataset_type == 'fashion_mnist':
            # Fashion-MNIST
            train_dataset = torchvision.datasets.FashionMNIST(
                root=str(self.data_root),
                train=True,
                transform=transform, 
                download=self.download
            )
            test_dataset = torchvision.datasets.FashionMNIST(
                root=str(self.data_root),
                train=False,
                transform=transform,
                download=self.download
            )
        else:
            raise ValueError(f"未対応データセット: {self.dataset_type}")
            
        # ED-SNN形式ラッパー
        self.train_dataset = EDSNNDataset(train_dataset)
        self.test_dataset = EDSNNDataset(test_dataset)
        
    def _create_dataloaders(self):
        """データローダー作成"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def get_train_loader(self) -> DataLoader:
        """訓練データローダー取得"""
        return self.train_loader
        
    def get_test_loader(self) -> DataLoader:
        """テストデータローダー取得"""
        return self.test_loader
        
    def get_sample_batch(self, from_train: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        サンプルバッチ取得
        
        Parameters:
        -----------
        from_train : bool
            訓練データから取得するか
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (画像データ, one-hotラベル, 元ラベル)
        """
        loader = self.train_loader if from_train else self.test_loader
        images, labels_onehot, labels_orig = next(iter(loader))
        
        return images.numpy(), labels_onehot.numpy(), labels_orig.numpy()
        
    def get_class_names(self) -> list:
        """クラス名取得"""
        if self.dataset_type == 'mnist':
            return [str(i) for i in range(10)]  # '0', '1', ..., '9'
        elif self.dataset_type == 'fashion_mnist':
            return [
                'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
            ]
        else:
            return [f'Class_{i}' for i in range(10)]
            
    def get_dataset_info(self) -> Dict[str, Any]:
        """データセット情報取得"""
        return {
            'dataset_type': self.dataset_type,
            'train_samples': len(self.train_dataset),
            'test_samples': len(self.test_dataset), 
            'num_classes': 10,
            'input_shape': (28, 28),
            'input_size': 784,
            'class_names': self.get_class_names(),
            'batch_size': self.batch_size
        }
        
    def visualize_samples(self, num_samples: int = 8, from_train: bool = True):
        """
        サンプル可視化
        
        Parameters:
        -----------
        num_samples : int
            表示サンプル数
        from_train : bool
            訓練データから表示するか
        """
        import matplotlib.pyplot as plt
        from modules.utils.font_config import ensure_japanese_font
        
        # 日本語フォント設定
        ensure_japanese_font()
        
        # サンプル取得
        images, labels_onehot, labels_orig = self.get_sample_batch(from_train)
        
        # 表示用準備
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        class_names = self.get_class_names()
        
        for i in range(min(num_samples, len(images))):
            # 28x28に変形
            image = images[i].reshape(28, 28)
            label = labels_orig[i]
            
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'{class_names[label]} ({label})')
            axes[i].axis('off')
            
        plt.suptitle(f'{self.dataset_type.upper()} サンプル画像')
        plt.tight_layout()
        
        # 保存
        output_path = f'images/{self.dataset_type}_samples.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"サンプル画像を {output_path} に保存しました")


def create_data_manager(dataset_type: str = 'mnist', **kwargs) -> MNISTDataManager:
    """
    データマネージャー作成のファクトリー関数
    
    Parameters:
    -----------
    dataset_type : str
        'mnist' or 'fashion_mnist'
    **kwargs
        MNISTDataManagerの追加引数
        
    Returns:
    --------
    MNISTDataManager
        設定済みデータマネージャー
    """
    return MNISTDataManager(dataset_type=dataset_type, **kwargs)


# 使用例とテスト関数
def test_data_manager():
    """データマネージャーのテスト"""
    print("🧪 PyTorchデータマネージャーテスト開始")
    
    # MNIST
    print("\n📊 MNISTデータセット:")
    mnist_manager = create_data_manager('mnist', batch_size=16)
    mnist_info = mnist_manager.get_dataset_info()
    
    for key, value in mnist_info.items():
        print(f"   {key}: {value}")
        
    # サンプルバッチ取得
    images, labels, orig_labels = mnist_manager.get_sample_batch()
    print(f"\nサンプルバッチ形状:")
    print(f"   画像: {images.shape}")
    print(f"   ラベル: {labels.shape}")
    print(f"   元ラベル: {orig_labels.shape}")
    
    # Fashion-MNIST
    print("\n📊 Fashion-MNISTデータセット:")
    fashion_manager = create_data_manager('fashion_mnist', batch_size=16)
    fashion_info = fashion_manager.get_dataset_info()
    
    for key, value in fashion_info.items():
        if key != 'class_names':  # 長いので省略
            print(f"   {key}: {value}")
    
    print("\n✅ データマネージャーテスト完了")


if __name__ == "__main__":
    test_data_manager()