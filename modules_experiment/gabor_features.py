#!/usr/bin/env python3
"""
固定畳み込みフィルタによる特徴抽出モジュール

生物学的背景:
  大脳皮質V1の単純型細胞はGaborフィルタと等価な方位選択性応答を持つ。
  これらは生得的（学習不要）な特徴検出器であり、ED法の入力前処理として
  誤差逆伝播なしで使用できる。

実装:
  - Gaborフィルタバンク（複数方位×複数空間周波数）
  - 固定フィルタ（学習不要）で畳み込み → プーリング → 平坦化
  - 入力: 28×28画像（784次元） → 出力: フィルタ数×プーリング後次元

使用例:
  from modules.gabor_features import GaborFeatureExtractor
  extractor = GaborFeatureExtractor(n_orientations=8, n_frequencies=2)
  X_features = extractor.transform(X_flat)  # (N, 784) → (N, feature_dim)
"""

import numpy as np


class GaborFeatureExtractor:
    """Gaborフィルタバンクによる固定特徴抽出器"""
    
    def __init__(self, image_shape=(28, 28), n_orientations=8, n_frequencies=2,
                 kernel_size=7, pool_size=4, pool_stride=4,
                 include_edge_filters=True):
        """
        Args:
            image_shape: 入力画像の形状 (H, W)
            n_orientations: Gaborフィルタの方位数（V1の方位選択性コラムに対応）
            n_frequencies: 空間周波数の数（V1の空間周波数帯域に対応）
            kernel_size: フィルタカーネルサイズ（奇数）
            pool_size: 平均プーリングのウィンドウサイズ
            pool_stride: プーリングのストライド
            include_edge_filters: Sobelエッジフィルタを追加するか
        """
        self.image_shape = image_shape
        self.n_orientations = n_orientations
        self.n_frequencies = n_frequencies
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.include_edge_filters = include_edge_filters
        
        # フィルタバンク構築
        self.filters = self._build_filter_bank()
        
        # プーリング後の空間サイズを計算
        h, w = image_shape
        # 畳み込み後サイズ（same padding）
        conv_h, conv_w = h, w
        # プーリング後サイズ
        self.pool_h = (conv_h - pool_size) // pool_stride + 1
        self.pool_w = (conv_w - pool_size) // pool_stride + 1
        
        self.n_filters = len(self.filters)
        self.feature_dim = self.n_filters * self.pool_h * self.pool_w
        
        # im2col + 行列積用のカーネル行列を事前構築 (n_filters, ksize*ksize)
        self._kernel_matrix = np.array(self.filters).reshape(self.n_filters, -1)
        
    def _build_filter_bank(self):
        """Gaborフィルタバンク + 基本エッジフィルタの構築"""
        filters = []
        
        # Gaborフィルタ: 方位 × 空間周波数
        orientations = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        # MNISTに適した空間周波数（低〜中周波数帯域）
        frequencies = np.linspace(0.1, 0.3, self.n_frequencies)
        
        for freq in frequencies:
            for theta in orientations:
                gabor = self._create_gabor_kernel(
                    self.kernel_size, theta, freq, sigma=2.0
                )
                filters.append(gabor)
        
        # Sobelエッジフィルタ（水平・垂直）
        if self.include_edge_filters:
            # パッドして kernel_size に合わせる
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
            filters.append(self._pad_kernel(sobel_x))
            filters.append(self._pad_kernel(sobel_y))
        
        return filters
    
    def _create_gabor_kernel(self, size, theta, frequency, sigma=2.0):
        """
        Gaborフィルタカーネルの生成
        
        V1単純型細胞のモデル:
          g(x,y) = exp(-(x'^2 + γ²y'^2)/(2σ²)) × cos(2π·f·x')
          x' = x·cos(θ) + y·sin(θ)
          y' = -x·sin(θ) + y·cos(θ)
        """
        half = (size - 1) / 2.0
        y, x = np.mgrid[0:size, 0:size].astype(np.float64)
        x = x - half
        y = y - half
        
        # 回転
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        # Gaborフィルタ（実部のみ = 対称型、V1の偶対称細胞）
        gamma = 0.5  # アスペクト比
        gaussian = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
        sinusoidal = np.cos(2 * np.pi * frequency * x_theta)
        
        kernel = gaussian * sinusoidal
        
        # ゼロ平均化（DCオフセット除去）
        kernel -= kernel.mean()
        # 正規化
        norm = np.sqrt(np.sum(kernel**2))
        if norm > 0:
            kernel /= norm
        
        return kernel
    
    def _pad_kernel(self, small_kernel):
        """小さなカーネルをkernel_sizeにゼロパディング"""
        padded = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float64)
        sh, sw = small_kernel.shape
        oh = (self.kernel_size - sh) // 2
        ow = (self.kernel_size - sw) // 2
        padded[oh:oh+sh, ow:ow+sw] = small_kernel
        # 正規化
        norm = np.sqrt(np.sum(padded**2))
        if norm > 0:
            padded /= norm
        return padded
    
    def transform_single(self, flat_image):
        """単一画像の特徴抽出（im2col + 行列積による高速実装）
        
        全フィルタの畳み込みを一括行列積で計算:
          patches: (ksize*ksize, H*W)  — im2col展開
          kernel_matrix: (n_filters, ksize*ksize)
          結果: kernel_matrix @ patches → (n_filters, H*W)
        
        Args:
            flat_image: 1D配列 (784,)
            
        Returns:
            features: 1D配列 (feature_dim,)
        """
        image = flat_image.reshape(self.image_shape)
        h, w = self.image_shape
        ksize = self.kernel_size
        pad = ksize // 2
        
        # ゼロパディング
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')
        
        # im2col: パッチ行列を構築 (ksize*ksize, H*W)
        patches = np.zeros((ksize * ksize, h * w), dtype=np.float64)
        idx = 0
        for ki in range(ksize):
            for kj in range(ksize):
                patches[idx] = padded[ki:ki+h, kj:kj+w].ravel()
                idx += 1
        
        # 全フィルタ一括行列積: (n_filters, ksize*ksize) @ (ksize*ksize, H*W)
        conv_results = self._kernel_matrix @ patches  # (n_filters, H*W)
        conv_results = conv_results.reshape(self.n_filters, h, w)
        
        # 半波整流（V1単純型細胞の応答）
        np.maximum(conv_results, 0, out=conv_results)
        
        # ベクトル化平均プーリング（stride_tricks使用）
        ps, st = self.pool_size, self.pool_stride
        shape = (self.n_filters, self.pool_h, self.pool_w, ps, ps)
        strides = (conv_results.strides[0],
                   conv_results.strides[1] * st,
                   conv_results.strides[2] * st,
                   conv_results.strides[1],
                   conv_results.strides[2])
        windows = np.lib.stride_tricks.as_strided(conv_results, shape=shape, strides=strides)
        pooled = windows.mean(axis=(3, 4))  # (n_filters, pool_h, pool_w)
        
        return pooled.reshape(-1)
    
    def transform(self, X_flat):
        """バッチ特徴抽出
        
        Args:
            X_flat: 2D配列 (N, 784)
            
        Returns:
            X_features: 2D配列 (N, feature_dim)
        """
        N = X_flat.shape[0]
        X_features = np.zeros((N, self.feature_dim), dtype=np.float64)
        
        for i in range(N):
            X_features[i] = self.transform_single(X_flat[i])
        
        # 特徴量の正規化（各特徴を[0,1]にスケーリング）
        feat_max = X_features.max(axis=0, keepdims=True)
        feat_max[feat_max == 0] = 1.0  # ゼロ除算防止
        X_features /= feat_max
        
        # 正規化パラメータを保存（テストデータには訓練データの統計を使用）
        self._feat_max = feat_max
        
        return X_features
    
    def transform_test(self, X_flat):
        """テストデータの特徴抽出（訓練データの統計で正規化）
        
        Args:
            X_flat: 2D配列 (N, 784)
            
        Returns:
            X_features: 2D配列 (N, feature_dim)
        """
        N = X_flat.shape[0]
        X_features = np.zeros((N, self.feature_dim), dtype=np.float64)
        
        for i in range(N):
            X_features[i] = self.transform_single(X_flat[i])
        
        # 訓練データの統計で正規化
        X_features /= self._feat_max
        # クリッピング（テストデータが訓練データの範囲を超える場合）
        np.clip(X_features, 0, 1, out=X_features)
        
        return X_features
    
    def get_info(self):
        """特徴抽出器の情報を返す"""
        return {
            'n_filters': self.n_filters,
            'n_gabor_filters': self.n_orientations * self.n_frequencies,
            'n_edge_filters': 2 if self.include_edge_filters else 0,
            'n_orientations': self.n_orientations,
            'n_frequencies': self.n_frequencies,
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
            'pool_stride': self.pool_stride,
            'pool_output_shape': (self.pool_h, self.pool_w),
            'feature_dim': self.feature_dim,
            'image_shape': self.image_shape,
        }
