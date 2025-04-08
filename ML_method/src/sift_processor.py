"""
SIFT 特征提取：可选择对图像进行数据增强
"""

import cv2
import numpy as np

def extract_sift_descriptors(image_paths, augmentation_fn=None):
    """
    提取每张图像的SIFT描述符，支持可选的数据增强函数
    参数：
        image_paths: 图像路径列表
        augmentation_fn: 数据增强函数 (image -> image)
    返回：
        descriptors_list: 每张图像对应的局部描述符数组
    """
    sift = cv2.SIFT_create()
    descriptors_list = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            descriptors_list.append(np.array([]))
            continue

        # 可选数据增强
        if augmentation_fn is not None:
            img = augmentation_fn(img)

        # 转RGB分3个通道提取SIFT
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        b, g, r = cv2.split(img_rgb)

        channel_descriptors = []
        for channel in [r, g, b]:
            kpts, desc = sift.detectAndCompute(channel, None)
            if desc is not None:
                channel_descriptors.append(desc)

        if channel_descriptors:
            all_desc = np.vstack(channel_descriptors)
            descriptors_list.append(all_desc)
        else:
            descriptors_list.append(np.array([]))

    return descriptors_list
