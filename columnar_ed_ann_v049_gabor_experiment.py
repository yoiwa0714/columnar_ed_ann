#!/usr/bin/env python3
"""
columnar_ed_ann_v049_gabor_experiment.py

目的:
- 既存の columnar_ed_ann_v049.py を変更せず、実験用に MATLAB互換寄りGabor特徴抽出を切替可能にする。
- --matlab_compat 指定時のみ、modules.gabor_features.GaborFeatureExtractor を
  本ファイル内の実験実装へ差し替える。

使い方:
- 従来動作: python columnar_ed_ann_v049_gabor_experiment.py [v049の引数...]
- MATLAB互換寄り: python columnar_ed_ann_v049_gabor_experiment.py --matlab_compat [v049の引数...]

追加オプション（--matlab_compat と併用）:
- --mc_wavelengths 4,8,16
- --mc_bandwidth 1.0
- --mc_aspect_ratio 0.5
- --mc_smooth_k 3.0
"""

from __future__ import annotations

import math
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np


@dataclass
class MatlabCompatConfig:
    wavelengths: List[float] | None = None
    bandwidth: float = 1.0
    aspect_ratio: float = 0.5
    smooth_k: float = 3.0


_MC_CONFIG = MatlabCompatConfig()


def _build_gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(truncate * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k_sum = k.sum()
    if k_sum > 0:
        k /= k_sum
    return k


def _gaussian_blur_2d(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image

    k = _build_gaussian_kernel_1d(sigma)
    pad = len(k) // 2

    # horizontal
    padded_h = np.pad(image, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.zeros_like(image, dtype=np.float64)
    for c in range(image.shape[1]):
        tmp[:, c] = (padded_h[:, c : c + len(k)] * k).sum(axis=1)

    # vertical
    padded_v = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float64)
    for r in range(image.shape[0]):
        out[r, :] = (padded_v[r : r + len(k), :] * k[:, None]).sum(axis=0)

    return out


class MatlabCompatGaborFeatureExtractor:
    """MATLAB gabor/imgaborfilt の設計思想に寄せた実験用抽出器。"""

    def __init__(
        self,
        image_shape=(28, 28),
        n_orientations=8,
        n_frequencies=2,
        kernel_size=7,
        pool_size=4,
        pool_stride=4,
        include_edge_filters=True,
    ):
        self.image_shape = image_shape
        self.n_orientations = n_orientations
        self.n_frequencies = n_frequencies
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.include_edge_filters = include_edge_filters

        self.wavelengths = self._select_wavelengths(n_frequencies)
        self.orientations_deg = np.linspace(0.0, 180.0, n_orientations, endpoint=False)

        self.filters = self._build_filter_bank()
        self.n_filters = len(self.filters)

        h, w = image_shape
        self.pool_h = (h - pool_size) // pool_stride + 1
        self.pool_w = (w - pool_size) // pool_stride + 1
        self.feature_dim = self.n_filters * self.pool_h * self.pool_w

        self._kernel_matrix = np.array(self.filters, dtype=np.float64).reshape(self.n_filters, -1)
        self._filter_wavelengths = self._build_filter_wavelength_table()

        self._feat_max = None
        self.last_conv_mean = None

    def _select_wavelengths(self, n_frequencies: int) -> List[float]:
        if _MC_CONFIG.wavelengths:
            return [float(v) for v in _MC_CONFIG.wavelengths]

        num_rows, num_cols = self.image_shape
        w_min = 4.0 / math.sqrt(2.0)
        w_max = math.hypot(num_rows, num_cols)
        n = max(2, int(math.floor(math.log2(w_max / w_min))))
        # MATLAB例: 2.^(0:(n-2))*w_min
        base = [w_min * (2.0**i) for i in range(max(1, n - 1))]
        if len(base) <= n_frequencies:
            return base
        idx = np.linspace(0, len(base) - 1, n_frequencies).round().astype(int)
        return [base[i] for i in idx]

    def _sigma_from_wavelength_bandwidth(self, wavelength: float, bandwidth: float) -> float:
        # よく使われる変換式 (Daugman/Jain系)
        # sigma/lambda = (1/pi)*sqrt(log(2)/2) * ((2^b + 1)/(2^b - 1))
        b = max(1e-6, float(bandwidth))
        ratio = (1.0 / math.pi) * math.sqrt(math.log(2.0) / 2.0) * ((2.0**b + 1.0) / (2.0**b - 1.0))
        return max(0.5, wavelength * ratio)

    def _create_gabor_kernel(self, size: int, theta_deg: float, wavelength: float) -> np.ndarray:
        theta = np.deg2rad(theta_deg)
        sigma = self._sigma_from_wavelength_bandwidth(wavelength, _MC_CONFIG.bandwidth)
        gamma = max(1e-6, float(_MC_CONFIG.aspect_ratio))

        half = (size - 1) / 2.0
        y, x = np.mgrid[0:size, 0:size].astype(np.float64)
        x = x - half
        y = y - half

        x_t = x * np.cos(theta) + y * np.sin(theta)
        y_t = -x * np.sin(theta) + y * np.cos(theta)

        gaussian = np.exp(-(x_t**2 + (gamma**2) * y_t**2) / (2.0 * sigma * sigma))
        sinusoid = np.cos((2.0 * np.pi / wavelength) * x_t)

        kernel = gaussian * sinusoid
        kernel -= kernel.mean()
        norm = np.sqrt(np.sum(kernel**2))
        if norm > 0:
            kernel /= norm
        return kernel

    def _pad_kernel(self, k3: np.ndarray) -> np.ndarray:
        out = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float64)
        sh, sw = k3.shape
        oh = (self.kernel_size - sh) // 2
        ow = (self.kernel_size - sw) // 2
        out[oh : oh + sh, ow : ow + sw] = k3
        norm = np.sqrt(np.sum(out**2))
        if norm > 0:
            out /= norm
        return out

    def _build_filter_bank(self) -> List[np.ndarray]:
        filters: List[np.ndarray] = []
        for wl in self.wavelengths:
            for od in self.orientations_deg:
                filters.append(self._create_gabor_kernel(self.kernel_size, float(od), float(wl)))

        if self.include_edge_filters:
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
            filters.append(self._pad_kernel(sobel_x))
            filters.append(self._pad_kernel(sobel_y))

        return filters

    def _build_filter_wavelength_table(self) -> List[float]:
        wl = []
        for w in self.wavelengths:
            for _ in self.orientations_deg:
                wl.append(float(w))
        # edge filters は最小波長相当として扱う
        if self.include_edge_filters:
            edge_w = float(min(self.wavelengths)) if self.wavelengths else 4.0
            wl.extend([edge_w, edge_w])
        return wl

    def transform_single(self, flat_image: np.ndarray) -> np.ndarray:
        image = flat_image.reshape(self.image_shape)
        h, w = self.image_shape
        ksize = self.kernel_size
        pad = ksize // 2

        padded = np.pad(image, ((pad, pad), (pad, pad)), mode="constant")

        patches = np.zeros((ksize * ksize, h * w), dtype=np.float64)
        idx = 0
        for ki in range(ksize):
            for kj in range(ksize):
                patches[idx] = padded[ki : ki + h, kj : kj + w].ravel()
                idx += 1

        conv = self._kernel_matrix @ patches
        conv = conv.reshape(self.n_filters, h, w)

        # MATLAB imgaborfilt の magnitude 寄りに絶対値を使用
        np.abs(conv, out=conv)

        # 波長依存平滑化（MATLAB例: sigma=0.5*wavelength, K*sigma）
        smooth_k = max(0.0, float(_MC_CONFIG.smooth_k))
        if smooth_k > 0:
            for fi in range(self.n_filters):
                sigma = 0.5 * self._filter_wavelengths[fi]
                conv[fi] = _gaussian_blur_2d(conv[fi], smooth_k * sigma)

        self.last_conv_mean = conv.mean(axis=0)

        ps, st = self.pool_size, self.pool_stride
        shape = (self.n_filters, self.pool_h, self.pool_w, ps, ps)
        strides = (
            conv.strides[0],
            conv.strides[1] * st,
            conv.strides[2] * st,
            conv.strides[1],
            conv.strides[2],
        )
        windows = np.lib.stride_tricks.as_strided(conv, shape=shape, strides=strides)
        pooled = windows.mean(axis=(3, 4))
        return pooled.reshape(-1)

    def transform(self, X_flat: np.ndarray) -> np.ndarray:
        n = X_flat.shape[0]
        out = np.zeros((n, self.feature_dim), dtype=np.float64)
        for i in range(n):
            out[i] = self.transform_single(X_flat[i])

        feat_max = out.max(axis=0, keepdims=True)
        feat_max[feat_max == 0] = 1.0
        out /= feat_max
        self._feat_max = feat_max
        return out

    def transform_test(self, X_flat: np.ndarray) -> np.ndarray:
        n = X_flat.shape[0]
        out = np.zeros((n, self.feature_dim), dtype=np.float64)
        for i in range(n):
            out[i] = self.transform_single(X_flat[i])

        if self._feat_max is None:
            feat_max = out.max(axis=0, keepdims=True)
            feat_max[feat_max == 0] = 1.0
            self._feat_max = feat_max

        out /= self._feat_max
        np.clip(out, 0, 1, out=out)
        return out

    def get_info(self):
        return {
            "n_filters": self.n_filters,
            "n_gabor_filters": len(self.wavelengths) * len(self.orientations_deg),
            "n_edge_filters": 2 if self.include_edge_filters else 0,
            "n_orientations": len(self.orientations_deg),
            "n_frequencies": len(self.wavelengths),
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "pool_stride": self.pool_stride,
            "pool_output_shape": (self.pool_h, self.pool_w),
            "feature_dim": self.feature_dim,
            "image_shape": self.image_shape,
            "matlab_compat": True,
            "wavelengths": list(self.wavelengths),
            "spatial_frequency_bandwidth": float(_MC_CONFIG.bandwidth),
            "spatial_aspect_ratio": float(_MC_CONFIG.aspect_ratio),
            "smooth_k": float(_MC_CONFIG.smooth_k),
        }


def _parse_experiment_args(argv: Sequence[str]):
    matlab_compat = False
    wavelengths = None
    bandwidth = 1.0
    aspect_ratio = 0.5
    smooth_k = 3.0

    cleaned = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--matlab_compat":
            matlab_compat = True
            i += 1
            continue
        if a == "--mc_wavelengths" and i + 1 < len(argv):
            wavelengths = [float(x.strip()) for x in argv[i + 1].split(",") if x.strip()]
            i += 2
            continue
        if a.startswith("--mc_wavelengths="):
            v = a.split("=", 1)[1]
            wavelengths = [float(x.strip()) for x in v.split(",") if x.strip()]
            i += 1
            continue
        if a == "--mc_bandwidth" and i + 1 < len(argv):
            bandwidth = float(argv[i + 1])
            i += 2
            continue
        if a.startswith("--mc_bandwidth="):
            bandwidth = float(a.split("=", 1)[1])
            i += 1
            continue
        if a == "--mc_aspect_ratio" and i + 1 < len(argv):
            aspect_ratio = float(argv[i + 1])
            i += 2
            continue
        if a.startswith("--mc_aspect_ratio="):
            aspect_ratio = float(a.split("=", 1)[1])
            i += 1
            continue
        if a == "--mc_smooth_k" and i + 1 < len(argv):
            smooth_k = float(argv[i + 1])
            i += 2
            continue
        if a.startswith("--mc_smooth_k="):
            smooth_k = float(a.split("=", 1)[1])
            i += 1
            continue

        cleaned.append(a)
        i += 1

    cfg = MatlabCompatConfig(
        wavelengths=wavelengths,
        bandwidth=bandwidth,
        aspect_ratio=aspect_ratio,
        smooth_k=smooth_k,
    )
    return matlab_compat, cfg, cleaned


def _activate_matlab_compat(cfg: MatlabCompatConfig):
    global _MC_CONFIG
    _MC_CONFIG = cfg

    import modules.gabor_features as mg

    mg.GaborFeatureExtractor = MatlabCompatGaborFeatureExtractor

    print("[gabor-experiment] MATLAB互換寄りGaborモード: ON")
    print(
        f"[gabor-experiment] wavelengths={cfg.wavelengths or 'auto'}, "
        f"bandwidth={cfg.bandwidth}, aspect_ratio={cfg.aspect_ratio}, smooth_k={cfg.smooth_k}"
    )


def main():
    # argv[0] は実験スクリプト名、残りを v049 へ渡す
    matlab_compat, cfg, cleaned = _parse_experiment_args(sys.argv[1:])

    if matlab_compat:
        _activate_matlab_compat(cfg)
    else:
        print("[gabor-experiment] MATLAB互換寄りGaborモード: OFF（従来実装）")

    # v049 側argparseに通るように sys.argv を再構築
    sys.argv = ["columnar_ed_ann_v049.py"] + cleaned

    script_path = Path(__file__).with_name("columnar_ed_ann_v049.py")
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
