#!/usr/bin/env python3
"""
Gabor特徴抽出モジュール（教育用シンプル版）

生物学的背景:
  大脳皮質V1の単純型細胞はGaborフィルタと等価な方位選択性応答を持つ。
  これらは生得的（学習不要）な特徴検出器であり、ED法の入力前処理として
  誤差逆伝播なしで使用できる。

実装:
  - Gaborフィルタバンク（方位×空間周波数）+ Sobelエッジフィルタ
  - im2col＋行列積による高速畳み込み → 平均プーリング → 平坦化
  - 入力: 28×28画像（784次元）→ 出力: フィルタ数×プーリング後次元
  - RGB画像対応: チャネル別独立方式（各チャネルに同一フィルタバンクを適用→結合）
"""

import numpy as np


class GaborFeatureExtractor:
    """Gaborフィルタバンクによる固定特徴抽出器"""

    def __init__(self, image_shape=(28, 28), n_orientations=8, n_frequencies=2,
                 kernel_size=11, pool_size=4, pool_stride=4,
                 include_edge_filters=True, n_channels=1):
        self.image_shape = image_shape
        self.n_orientations = n_orientations
        self.n_frequencies = n_frequencies
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.include_edge_filters = include_edge_filters
        self.n_channels = n_channels

        # フィルタバンク構築
        self.filters = self._build_filter_bank()

        # プーリング後の空間サイズ
        h, w = image_shape
        self.pool_h = (h - pool_size) // pool_stride + 1
        self.pool_w = (w - pool_size) // pool_stride + 1
        self.n_filters = len(self.filters)
        self._single_channel_dim = self.n_filters * self.pool_h * self.pool_w
        self.feature_dim = self._single_channel_dim * n_channels

        # im2col用カーネル行列を事前構築
        self._kernel_matrix = np.array(self.filters).reshape(self.n_filters, -1)

    def _build_filter_bank(self):
        """Gaborフィルタバンク＋Sobelエッジフィルタの構築"""
        filters = []
        orientations = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        frequencies = np.linspace(0.1, 0.3, self.n_frequencies)

        for freq in frequencies:
            for theta in orientations:
                gabor = self._create_gabor_kernel(self.kernel_size, theta, freq, sigma=2.0)
                filters.append(gabor)

        if self.include_edge_filters:
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
            filters.append(self._pad_kernel(sobel_x))
            filters.append(self._pad_kernel(sobel_y))

        return filters

    def _create_gabor_kernel(self, size, theta, frequency, sigma=2.0):
        """
        Gaborフィルタカーネル生成（V1単純型細胞モデル）

        g(x,y) = exp(-(x'^2 + γ²y'^2)/(2σ²)) × cos(2π·f·x')
        """
        half = (size - 1) / 2.0
        y, x = np.mgrid[0:size, 0:size].astype(np.float64)
        x, y = x - half, y - half

        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gamma = 0.5
        gaussian = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
        sinusoidal = np.cos(2 * np.pi * frequency * x_theta)
        kernel = gaussian * sinusoidal

        kernel -= kernel.mean()
        norm = np.sqrt(np.sum(kernel**2))
        if norm > 0:
            kernel /= norm
        return kernel

    def _pad_kernel(self, small_kernel):
        """小カーネルをkernel_sizeにゼロパディング"""
        padded = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float64)
        sh, sw = small_kernel.shape
        oh = (self.kernel_size - sh) // 2
        ow = (self.kernel_size - sw) // 2
        padded[oh:oh+sh, ow:ow+sw] = small_kernel
        norm = np.sqrt(np.sum(padded**2))
        if norm > 0:
            padded /= norm
        return padded

    def transform_single(self, flat_image):
        """単一画像の特徴抽出（im2col＋行列積による高速実装）"""
        if self.n_channels > 1:
            pixels_per_ch = flat_image.size // self.n_channels
            features = np.empty(self.feature_dim, dtype=np.float64)
            for c in range(self.n_channels):
                ch_data = flat_image[c * pixels_per_ch : (c + 1) * pixels_per_ch]
                features[c * self._single_channel_dim : (c + 1) * self._single_channel_dim] = \
                    self._transform_single_channel(ch_data)
            return features
        return self._transform_single_channel(flat_image)

    def _transform_single_channel(self, flat_image):
        """1チャネル分の特徴抽出（im2col＋行列積）"""
        image = flat_image.reshape(self.image_shape)
        h, w = self.image_shape
        ksize = self.kernel_size
        pad = ksize // 2

        padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')

        # im2col: パッチ行列を構築
        patches = np.zeros((ksize * ksize, h * w), dtype=np.float64)
        idx = 0
        for ki in range(ksize):
            for kj in range(ksize):
                patches[idx] = padded[ki:ki+h, kj:kj+w].ravel()
                idx += 1

        # 全フィルタ一括行列積
        conv_results = self._kernel_matrix @ patches
        conv_results = conv_results.reshape(self.n_filters, h, w)

        # 半波整流（V1単純型細胞の応答）
        np.maximum(conv_results, 0, out=conv_results)

        # 平均プーリング
        ps, st = self.pool_size, self.pool_stride
        shape = (self.n_filters, self.pool_h, self.pool_w, ps, ps)
        strides = (conv_results.strides[0],
                   conv_results.strides[1] * st,
                   conv_results.strides[2] * st,
                   conv_results.strides[1],
                   conv_results.strides[2])
        windows = np.lib.stride_tricks.as_strided(conv_results, shape=shape, strides=strides)
        pooled = windows.mean(axis=(3, 4))

        return pooled.reshape(-1)

    def transform(self, X_flat):
        """バッチ特徴抽出（訓練データ用 — 正規化パラメータを保存）"""
        N = X_flat.shape[0]
        X_features = np.zeros((N, self.feature_dim), dtype=np.float64)
        for i in range(N):
            X_features[i] = self.transform_single(X_flat[i])

        feat_max = X_features.max(axis=0, keepdims=True)
        feat_max[feat_max == 0] = 1.0
        X_features /= feat_max
        self._feat_max = feat_max
        return X_features

    def transform_test(self, X_flat):
        """バッチ特徴抽出（テストデータ用 — 訓練データの統計で正規化）"""
        N = X_flat.shape[0]
        X_features = np.zeros((N, self.feature_dim), dtype=np.float64)
        for i in range(N):
            X_features[i] = self.transform_single(X_flat[i])

        X_features /= self._feat_max
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
            'n_channels': self.n_channels,
        }
