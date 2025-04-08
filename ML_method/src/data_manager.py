"""
数据加载与划分模块
"""

import os
import numpy as np
from glob import glob
from sklearn.utils import shuffle

def load_and_split_dataset(root_folder, test_ratio=0.2, sample_ratio=1.0, random_seed=42):
    """
    加载图像并划分为训练和测试
    参数：
        root_folder: 数据根目录（子文件夹 = 类别）
        test_ratio: 测试集占比
        sample_ratio: 每类最大使用比例
        random_seed: 随机种子
    返回：
        (train_paths, train_labels, test_paths, test_labels, class_names)
    """
    np.random.seed(random_seed)
    class_names = sorted(os.listdir(root_folder))

    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    for label_id, cname in enumerate(class_names):
        class_dir = os.path.join(root_folder, cname)
        if not os.path.isdir(class_dir):
            continue

        images = glob(os.path.join(class_dir, "*.jpg")) + \
                 glob(os.path.join(class_dir, "*.png")) + \
                 glob(os.path.join(class_dir, "*.jpeg"))

        images = shuffle(images, random_state=random_seed)
        # 按比例取部分样本
        sample_count = int(len(images) * sample_ratio)
        samples = images[:sample_count]

        # 划分训练/测试
        split_index = int(sample_count * (1 - test_ratio))
        train_samples = samples[:split_index]
        test_samples  = samples[split_index:]

        train_paths.extend(train_samples)
        train_labels.extend([label_id]*len(train_samples))

        test_paths.extend(test_samples)
        test_labels.extend([label_id]*len(test_samples))

    return train_paths, train_labels, test_paths, test_labels, class_names
