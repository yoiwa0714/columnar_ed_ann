"""
データ拡張モジュール (Data Augmentation)

生物学的背景:
  - サッケード（視線の微細運動）→ 画像シフト
  - 網膜上の像のゆらぎ → 微小回転、微小スケール変化
  - 神経ノイズ → ガウスノイズ

ED法の学習ボトルネック（隠れ層がほぼ固定のランダム射影）に対して、
入力側の多様性を増やすことで出力層の学習を改善する。
"""

import numpy as np


def augment_image(x, image_shape=(28, 28), shift_range=2, rotation_range=10.0,
                  noise_std=0.03, scale_range=0.0, seed=None):
    """
    1サンプルの画像データを拡張する
    
    Args:
        x: フラット化された画像 (shape: [n_pixels])
        image_shape: 元の画像サイズ (height, width)
        shift_range: シフト範囲（ピクセル、整数）。±shift_range の範囲でランダムシフト
        rotation_range: 回転範囲（度数）。±rotation_range の範囲でランダム回転
        noise_std: ガウスノイズの標準偏差
        scale_range: スケール変動範囲。1±scale_range の範囲
        seed: 乱数シード（Noneで非固定）
    
    Returns:
        拡張後のフラット化画像 (shape: [n_pixels])
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    h, w = image_shape
    
    # 2D画像に復元
    img = x.reshape(h, w)
    
    # 1. ランダムシフト（最も効果的な拡張）
    if shift_range > 0:
        dy = rng.randint(-shift_range, shift_range + 1)
        dx = rng.randint(-shift_range, shift_range + 1)
        img = _shift_image(img, dy, dx)
    
    # 2. ランダム回転（軽微な回転）
    if rotation_range > 0:
        angle = rng.uniform(-rotation_range, rotation_range)
        img = _rotate_image(img, angle)
    
    # 3. ガウスノイズ
    if noise_std > 0:
        noise = rng.normal(0, noise_std, img.shape)
        img = img + noise
    
    # [0, 1] にクリッピング
    img = np.clip(img, 0.0, 1.0)
    
    return img.flatten()


def augment_batch(x_batch, image_shape=(28, 28), shift_range=2, rotation_range=10.0,
                  noise_std=0.03, scale_range=0.0, seed=None):
    """
    バッチ単位のデータ拡張（ベクトル化版）
    
    Args:
        x_batch: 画像バッチ (shape: [n_samples, n_pixels])
        その他引数はaugment_imageと同じ
    
    Returns:
        拡張後の画像バッチ (shape: [n_samples, n_pixels])
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    n_samples = len(x_batch)
    h, w = image_shape
    
    # 2D画像バッチに復元
    imgs = x_batch.reshape(n_samples, h, w)
    result = np.empty_like(imgs)
    
    for i in range(n_samples):
        img = imgs[i].copy()
        
        # 1. ランダムシフト
        if shift_range > 0:
            dy = rng.randint(-shift_range, shift_range + 1)
            dx = rng.randint(-shift_range, shift_range + 1)
            img = _shift_image(img, dy, dx)
        
        # 2. ランダム回転
        if rotation_range > 0:
            angle = rng.uniform(-rotation_range, rotation_range)
            img = _rotate_image(img, angle)
        
        # 3. ガウスノイズ
        if noise_std > 0:
            noise = rng.normal(0, noise_std, img.shape)
            img = img + noise
        
        result[i] = img
    
    # クリッピング & フラット化
    result = np.clip(result, 0.0, 1.0)
    return result.reshape(n_samples, -1)


def create_augmented_dataset(x_train, y_train, n_augmented=1, image_shape=(28, 28),
                             shift_range=2, rotation_range=10.0, noise_std=0.03,
                             scale_range=0.0, seed=None):
    """
    訓練データセットを拡張して新しいデータセットを作成する（事前拡張方式）
    
    Args:
        x_train: 元の訓練データ (shape: [n_samples, n_pixels])
        y_train: 元の訓練ラベル
        n_augmented: 拡張回数（1=元データと同数の拡張データを追加 → 2倍）
        image_shape: 画像サイズ
        shift_range: シフト範囲
        rotation_range: 回転範囲（度数）
        noise_std: ノイズ標準偏差
        scale_range: スケール変動範囲
        seed: 乱数シード
    
    Returns:
        (x_augmented, y_augmented): 元データ + 拡張データの結合
    """
    rng = np.random.RandomState(seed)
    
    all_x = [x_train]
    all_y = [y_train]
    
    for aug_idx in range(n_augmented):
        aug_seed = rng.randint(0, 2**31)
        x_aug = augment_batch(
            x_train, image_shape=image_shape,
            shift_range=shift_range, rotation_range=rotation_range,
            noise_std=noise_std, scale_range=scale_range,
            seed=aug_seed
        )
        all_x.append(x_aug)
        all_y.append(y_train.copy())
    
    x_combined = np.concatenate(all_x, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    
    # シャッフル
    shuffle_idx = rng.permutation(len(x_combined))
    x_combined = x_combined[shuffle_idx]
    y_combined = y_combined[shuffle_idx]
    
    return x_combined, y_combined


def _shift_image(img, dy, dx):
    """画像をシフトする（ゼロパディング）"""
    h, w = img.shape
    result = np.zeros_like(img)
    
    # ソース範囲の計算
    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)
    
    # ターゲット範囲の計算
    dst_y_start = max(0, dy)
    dst_y_end = min(h, h + dy)
    dst_x_start = max(0, dx)
    dst_x_end = min(w, w + dx)
    
    if src_y_end > src_y_start and src_x_end > src_x_start:
        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            img[src_y_start:src_y_end, src_x_start:src_x_end]
    
    return result


def _rotate_image(img, angle_deg):
    """
    画像を回転する（純NumPy実装、バイリニア補間）
    
    外部ライブラリ（scipy, PIL, cv2）に依存しない軽量実装。
    """
    if abs(angle_deg) < 0.1:
        return img
    
    h, w = img.shape
    cy, cx = h / 2.0, w / 2.0
    
    # 逆回転行列（デスティネーション→ソース座標変換）
    angle_rad = -np.radians(angle_deg)  # 逆変換
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # デスティネーション座標グリッド
    yy, xx = np.mgrid[0:h, 0:w]
    
    # 中心基準に回転（ソース座標算出）
    src_x = cos_a * (xx - cx) - sin_a * (yy - cy) + cx
    src_y = sin_a * (xx - cx) + cos_a * (yy - cy) + cy
    
    # バイリニア補間
    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # 境界チェック
    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    
    # 安全なインデックス（範囲外はクリップ）
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    
    # 補間重み
    wx = src_x - x0
    wy = src_y - y0
    
    # バイリニア補間
    result = (img[y0c, x0c] * (1 - wx) * (1 - wy) +
              img[y0c, x1c] * wx * (1 - wy) +
              img[y1c, x0c] * (1 - wx) * wy +
              img[y1c, x1c] * wx * wy)
    
    # 範囲外は0
    result = np.where(valid, result, 0.0)
    
    return result
