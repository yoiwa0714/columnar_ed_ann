"""
ED法用ミニバッチデータローダー

ed_multi_snn.prompt.md拡張機能2準拠
- ミニバッチ学習システム実装
- バッチサイズ可変対応
- データシャッフル機能
- イテレータプロトコル実装

注：金子勇氏のED理論にはバッチ処理概念なし
    大規模データ対応のための現代的機能拡張
"""

import numpy as np
from typing import Tuple


class MiniBatchDataLoader:
    """
    ED法用ミニバッチデータローダー
    
    特徴:
    - バッチサイズ可変対応（1〜任意のサイズ）
    - データシャッフル機能（エポックごと）
    - Pythonイテレータプロトコル実装
    - 最終バッチの不完全サイズ対応
    
    使用例:
        loader = MiniBatchDataLoader(train_data, train_labels, batch_size=32)
        for batch_inputs, batch_labels in loader:
            # バッチ処理...
    """
    
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, 
                 batch_size: int, shuffle: bool = True):
        """
        ミニバッチデータローダーの初期化
        
        Args:
            inputs: 入力データ配列 (N, D) - N: サンプル数, D: 特徴次元
            labels: ラベル配列 (N,)
            batch_size: バッチサイズ（1以上）
            shuffle: エポックごとのデータシャッフル有効化
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        
        if len(inputs) != len(labels):
            raise ValueError(f"inputs and labels length mismatch: {len(inputs)} vs {len(labels)}")
        
        self.inputs = inputs.copy()  # 元データを保護
        self.labels = labels.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(inputs)
        
        # バッチ数計算（最終バッチが不完全でも含む）
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        
        # イテレータ状態
        self.current_batch = 0
    
    def _reset(self):
        """エポック開始時のリセット処理（シャッフル含む）"""
        if self.shuffle:
            # ランダムにインデックスをシャッフル
            indices = np.random.permutation(self.num_samples)
            self.inputs = self.inputs[indices]
            self.labels = self.labels[indices]
        
        # バッチカウンタリセット
        self.current_batch = 0
    
    def __iter__(self):
        """
        イテレータ初期化
        
        Returns:
            self: イテレータオブジェクト
        """
        self._reset()
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        次のバッチを取得
        
        Returns:
            (batch_inputs, batch_labels): 次のバッチのデータとラベル
            
        Raises:
            StopIteration: 全バッチ処理完了時
        """
        # 全バッチ処理完了チェック
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        # 現在のバッチの開始・終了インデックス
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        # バッチデータ取得
        batch_inputs = self.inputs[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        # 次のバッチへ
        self.current_batch += 1
        
        return batch_inputs, batch_labels
    
    def __len__(self) -> int:
        """
        バッチ数を返す
        
        Returns:
            int: 総バッチ数
        """
        return self.num_batches
    
    def get_batch_info(self) -> dict:
        """
        バッチ処理情報を取得
        
        Returns:
            dict: バッチ情報
        """
        last_batch_size = self.num_samples - (self.num_batches - 1) * self.batch_size
        
        return {
            'num_samples': self.num_samples,
            'batch_size': self.batch_size,
            'num_batches': self.num_batches,
            'last_batch_size': last_batch_size,
            'shuffle': self.shuffle
        }


# 使用例とテスト
if __name__ == "__main__":
    # テストデータ生成
    test_inputs = np.random.randn(100, 784)
    test_labels = np.random.randint(0, 10, 100)
    
    print("🧪 MiniBatchDataLoaderテスト")
    print("=" * 60)
    
    # テスト1: 通常のバッチ処理
    print("\n📦 テスト1: batch_size=32, shuffle=True")
    loader = MiniBatchDataLoader(test_inputs, test_labels, batch_size=32, shuffle=True)
    print(f"  総サンプル数: {loader.num_samples}")
    print(f"  バッチサイズ: {loader.batch_size}")
    print(f"  総バッチ数: {loader.num_batches}")
    
    batch_count = 0
    for batch_inputs, batch_labels in loader:
        batch_count += 1
        print(f"  バッチ{batch_count}: inputs shape={batch_inputs.shape}, labels shape={batch_labels.shape}")
    
    # テスト2: 逐次処理（batch_size=1）
    print("\n📦 テスト2: batch_size=1 (逐次処理)")
    loader = MiniBatchDataLoader(test_inputs[:10], test_labels[:10], batch_size=1, shuffle=False)
    print(f"  総バッチ数: {len(loader)}")
    
    # テスト3: 大バッチ
    print("\n📦 テスト3: batch_size=100 (全データ1バッチ)")
    loader = MiniBatchDataLoader(test_inputs, test_labels, batch_size=100, shuffle=False)
    print(f"  総バッチ数: {len(loader)}")
    
    for batch_inputs, batch_labels in loader:
        print(f"  バッチサイズ: {len(batch_inputs)}")
    
    print("\n✅ MiniBatchDataLoaderテスト完了")
