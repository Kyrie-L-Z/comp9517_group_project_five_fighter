"""
SIFT Feature Extraction: Supports optional image augmentation.
"""

import cv2
import numpy as np

def extract_sift_descriptors(image_paths, augmentation_fn=None):
    """
    Extract SIFT descriptors from each image. Supports optional image augmentation.

    Args:
        image_paths: List of image file paths.
        augmentation_fn: Optional augmentation function to apply before extraction (image -> image).

    Returns:
        descriptors_list: A list of arrays where each element contains the local descriptors of an image.
    """
    sift = cv2.SIFT_create()
    descriptors_list = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            descriptors_list.append(np.array([]))
            continue

        # Apply augmentation if provided
        if augmentation_fn is not None:
            img = augmentation_fn(img)

        # Convert to RGB and extract SIFT from each channel separately
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
