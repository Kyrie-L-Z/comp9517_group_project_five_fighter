import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.io import imread


def extract_lbp_descriptors(image_paths, radius=1, n_points=8, method='uniform', augmentation_fn=None):
    """
    对图像提取 LBP 特征（每张图像 → 统一长度的 LBP 直方图）

    参数：
        image_paths: 图像路径列表
        radius: LBP 半径
        n_points: 采样点数
        method: LBP 模式（默认 uniform）
        augmentation_fn: 可选的数据增强函数

    返回：
        descriptors: List[np.ndarray]，每张图像对应一个LBP直方图
    """
    descriptors = []

    for path in image_paths:
        img = imread(path)
        if augmentation_fn is not None:
            img = augmentation_fn(img)

        gray = rgb2gray(img)
        lbp = local_binary_pattern(gray, P=n_points, R=radius, method=method)

        # 计算 LBP 直方图（统一维度）
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)  # 归一化

        descriptors.append(hist)

    return descriptors
