"""
Build visual dictionary (BoW) and convert descriptors into BoW vectors.
"""

import numpy as np
from sklearn.cluster import KMeans

def build_visual_dictionary(descriptor_list, cluster_count=300, max_sample=15000, seed=42):
    """
    Build a visual dictionary using KMeans clustering.

    Args:
        descriptor_list: List of descriptors from each image.
        cluster_count: Number of visual words (dictionary size).
        max_sample: Maximum number of descriptors to use in KMeans.
        seed: Random seed.

    Returns:
        Trained KMeans model.
    """
    # Collect all non-empty descriptors
    total_desc = [desc for desc in descriptor_list if len(desc) > 0]
    if len(total_desc) == 0:
        return None

    stacked = np.vstack(total_desc)  # Stack all descriptors

    if len(stacked) > max_sample:
        indices = np.random.choice(len(stacked), size=max_sample, replace=False)
        stacked = stacked[indices]

    kmeans = KMeans(n_clusters=cluster_count, random_state=seed)
    kmeans.fit(stacked)
    return kmeans

def transform_to_bow(descriptor_list, kmeans_model):
    """
    Convert a list of descriptors into a matrix of BoW vectors.
    Each image's descriptors are mapped to a histogram over visual words.

    Args:
        descriptor_list: List of local descriptors for each image.
        kmeans_model: Trained KMeans model representing the visual vocabulary.

    Returns:
        A NumPy array of shape (n_images, cluster_count) with BoW feature vectors.
    """
    cluster_count = kmeans_model.n_clusters
    bow_vectors = []

    for desc in descriptor_list:
        hist = np.zeros(cluster_count, dtype=np.float32)

        if desc is not None and len(desc) > 0:
            # Ensure desc is 2D
            if desc.ndim == 1:
                desc = desc.reshape(1, -1)

            predictions = kmeans_model.predict(desc)
            for idx in predictions:
                hist[idx] += 1

            # Optionally normalize the histogram
            if hist.sum() > 0:
                hist /= hist.sum()

        bow_vectors.append(hist)

    return np.array(bow_vectors)
