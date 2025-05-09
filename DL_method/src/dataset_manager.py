import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler


class CustomImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.transform:
            img_rgb = self.transform(img_rgb)
        return img_rgb, label


def load_imbalanced_data(root_folder, test_ratio=0.2, sample_ratio=1.0, reduce_factor=0.1, minority_classes=[0],
                         random_seed=42):
    """
    Read image paths and labels from root_folder (default ImageFolder format),
    Simulate long-tail distribution: keep only `reduce_factor` proportion for specified minority_classes.
    """
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(root_folder)
    file_paths = [s[0] for s in dataset.samples]
    labels = [s[1] for s in dataset.samples]
    class_names = dataset.classes

    # Imbalance processing
    random.seed(random_seed)
    all_indices = []
    # Select all samples initially
    for c in range(len(class_names)):
        idx_c = [i for i, lab in enumerate(labels) if lab == c]
        random.shuffle(idx_c)
        if c in minority_classes:
            keep_num = int(len(idx_c) * reduce_factor)
            idx_c = idx_c[:keep_num]
        # Apply sample_ratio
        keep_num2 = int(len(idx_c) * sample_ratio)
        idx_c = idx_c[:keep_num2]
        all_indices.extend(idx_c)

    random.shuffle(all_indices)
    split = int(len(all_indices) * (1 - test_ratio))
    train_idx = all_indices[:split]
    test_idx = all_indices[split:]

    train_paths = [file_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_paths = [file_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_paths, train_labels, test_paths, test_labels, class_names


def make_data_loaders(train_paths, train_labels, test_paths, test_labels, class_names,
                      batch_size=32, random_seed=42, im_size=224):
    """
    1) Define transforms
    2) Build Dataset
    3) Construct WeightedRandomSampler
    4) Return train_loader and test_loader
    """
    import torch
    from torchvision import transforms

    # Count number of samples per class in training set
    from collections import Counter
    label_counter = Counter(train_labels)
    # WeightedRandomSampler
    counts = np.array([label_counter[i] for i in range(len(class_names))])
    class_weights = 1.0 / counts
    sample_weights = [class_weights[lab] for lab in train_labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(im_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = CustomImageDataset(test_paths, test_labels, transform=test_transform)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader


def load_dataset(root_folder="../Aerial_Landscapes", test_ratio=0.2, augment=False, imbalance=False,
                 reduce_factor=0.1, sample_ratio=1.0, batch_size=32, im_size=224, random_seed=42):
    """
    General entry: supports loading for normal / augment / imbalance settings

    Returns:
        train_loader, test_loader, class_names
    """
    # Decide whether to simulate imbalance
    if imbalance:
        # By default, reduce the last two classes
        minority_classes = [-2, -1]  # Last two classes
        train_paths, train_labels, test_paths, test_labels, class_names = load_imbalanced_data(
            root_folder=root_folder,
            test_ratio=test_ratio,
            sample_ratio=sample_ratio,
            reduce_factor=reduce_factor,
            minority_classes=list(range(len(os.listdir(root_folder)))[-2:]),
            random_seed=random_seed
        )
    else:
        # Balanced setting
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(root_folder)
        file_paths = [s[0] for s in dataset.samples]
        labels = [s[1] for s in dataset.samples]
        class_names = dataset.classes

        # Split dataset
        indices = list(range(len(file_paths)))
        random.seed(random_seed)
        random.shuffle(indices)
        split = int(len(indices) * (1 - test_ratio))
        train_idx, test_idx = indices[:split], indices[split:]
        train_paths = [file_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_paths = [file_paths[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

    # Return data loaders
    return make_data_loaders(
        train_paths=train_paths,
        train_labels=train_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        class_names=class_names,
        batch_size=batch_size,
        random_seed=random_seed,
        im_size=im_size
    ) + (class_names,)


def load_data_path_only(root_folder="../Aerial_Landscapes", imbalance=False):
    if imbalance:
        # Use imbalanced setup
        train_paths, train_labels, test_paths, test_labels, class_names = load_imbalanced_data(root_folder)
    else:
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(root_folder)
        file_paths = [s[0] for s in dataset.samples]
        labels = [s[1] for s in dataset.samples]
        class_names = dataset.classes

        from sklearn.model_selection import train_test_split
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            file_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )

    return train_paths, train_labels, test_paths, test_labels, class_names
