import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.io import imread


def extract_lbp_descriptors(image_paths, radius=1, n_points=8, method='uniform', augmentation_fn=None):
    """
    Extract LBP (Local Binary Pattern) features from a list of images.
    Each image is converted into a fixed-length LBP histogram.

    Args:
        image_paths: List of image file paths.
        radius: Radius of the LBP pattern.
        n_points: Number of sampling points around each pixel.
        method: LBP method (default is 'uniform').
        augmentation_fn: Optional augmentation function to apply to each image before LBP.

    Returns:
        descriptors: List[np.ndarray], where each element is an LBP histogram for an image.
    """
    descriptors = []

    for path in image_paths:
        img = imread(path)
        if augmentation_fn is not None:
            img = augmentation_fn(img)

        gray = rgb2gray(img)
        lbp = local_binary_pattern(gray, P=n_points, R=radius, method=method)

        # Compute LBP histogram (ensure fixed dimension)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)  # Normalize histogram

        descriptors.append(hist)

    return descriptors