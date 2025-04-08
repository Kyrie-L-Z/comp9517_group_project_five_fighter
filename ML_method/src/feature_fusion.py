"""
数据增强 + 颜色直方图提取
"""
import numpy as np
import cv2
import random

def augment_image(img):
    """
    对图像进行随机数据增强：翻转/旋转/缩放/模糊等
    """
    # 随机翻转
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # 随机旋转 [0,90,180,270]
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        center = (img.shape[1] // 2, img.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

    # 随机缩放（0.9 ~ 1.1倍）
    scale = random.uniform(0.9, 1.1)
    matrix = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 0, scale)
    img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

    # 随机高斯模糊 [1,3,5]
    kernel_size = random.choice([1, 3, 5])
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    return img

def extract_rgb_histogram(img_path, bins=(8, 8, 8)):
    """
    提取图像的RGB三通道直方图, 返回向量
    """
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((bins[0]*bins[1]*bins[2],), dtype=np.float32)

    img = cv2.resize(img, (128, 128))
    hist = cv2.calcHist([img], [0,1,2], None, bins, [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
