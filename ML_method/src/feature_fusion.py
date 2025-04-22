"""
Data Augmentation + RGB Color Histogram Extraction
"""

import numpy as np
import cv2
import random

def augment_image(img):
    """
    Apply random data augmentations to an image: flip, rotate, scale, blur.

    Args:
        img: Input image in BGR format (as read by OpenCV).

    Returns:
        Augmented image.
    """
    # Random horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # Random rotation (0, 90, 180, or 270 degrees)
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        center = (img.shape[1] // 2, img.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

    # Random scaling (scale factor between 0.9 and 1.1)
    scale = random.uniform(0.9, 1.1)
    matrix = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), 0, scale)
    img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

    # Random Gaussian blur with kernel size of 1, 3, or 5
    kernel_size = random.choice([1, 3, 5])
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    return img

def extract_rgb_histogram(img_path, bins=(8, 8, 8)):
    """
    Extract a normalized RGB 3D color histogram from an image.

    Args:
        img_path: Path to the input image.
        bins: Number of bins per channel (R, G, B).

    Returns:
        Flattened and normalized histogram vector.
    """
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((bins[0] * bins[1] * bins[2],), dtype=np.float32)

    img = cv2.resize(img, (128, 128))
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
