import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# === 训练一个 epoch ===
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


# === 验证模型性能 ===
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


# === 完整训练 + 验证流程 ===
def train_and_evaluate_model(model, train_loader, val_loader, device, class_names, label="model", epochs=5):
    import torch.nn as nn
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"\n⏳ Training {label} for {epochs} epochs...")
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    train_time = time.time() - start_time
    print(f"✅ Finished Training {label} in {train_time:.2f} seconds.\n")

    # === 验证模型 ===
    val_loss, val_acc, all_preds, all_labels = evaluate(model, val_loader, criterion, device)

    print("📊 Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    print(report)

    print("📉 Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # === 返回评分指标 ===
    return {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds, average='weighted'), 4),
        "recall": round(recall_score(all_labels, all_preds, average='weighted'), 4),
        "f1": round(f1_score(all_labels, all_preds, average='weighted'), 4),
        "train_time": round(train_time, 2)
    }
