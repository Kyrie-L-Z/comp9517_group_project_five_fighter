"""
构建视觉词典（BoW）并将描述符转换为BoW向量
"""

import numpy as np
from sklearn.cluster import KMeans

def build_visual_dictionary(descriptor_list, cluster_count=300, max_sample=15000, seed=42):
    """
    使用KMeans聚类生成视觉词典
    参数：
        descriptor_list: 每张图像的描述符列表
        cluster_count: 词典大小
        max_sample: 用于KMeans的最大描述符数
        seed: 随机种子
    返回：
        KMeans 模型
    """
    # 汇总所有描述符
    total_desc = [desc for desc in descriptor_list if len(desc) > 0]
    if len(total_desc) == 0:
        return None

    stacked = np.vstack(total_desc)  # 统一叠加

    if len(stacked) > max_sample:
        indices = np.random.choice(len(stacked), size=max_sample, replace=False)
        stacked = stacked[indices]

    kmeans = KMeans(n_clusters=cluster_count, random_state=seed)
    kmeans.fit(stacked)
    return kmeans

def transform_to_bow(descriptor_list, kmeans_model):
    """
    将一组描述符列表转换为 BoW 向量矩阵。
    每个图像的描述符 -> 直方图
    """
    cluster_count = kmeans_model.n_clusters
    bow_vectors = []

    for desc in descriptor_list:
        hist = np.zeros(cluster_count, dtype=np.float32)

        if desc is not None and len(desc) > 0:
            # 保证 desc 是二维的
            if desc.ndim == 1:
                desc = desc.reshape(1, -1)

            predictions = kmeans_model.predict(desc)
            for idx in predictions:
                hist[idx] += 1

            # 可以加上归一化
            if hist.sum() > 0:
                hist /= hist.sum()

        bow_vectors.append(hist)

    return np.array(bow_vectors)
