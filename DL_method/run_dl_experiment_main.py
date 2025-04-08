import sys
sys.path.append('.')     # 保证能import到src/
sys.path.append('./src')

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score, precision_score, recall_score)

# ============ 导入自定义模块 ============
from src.dataset_manager import load_imbalanced_data, make_data_loaders
from src.model_builder import create_model
from src.train_utils import train_one_epoch, evaluate
from src.occlusion_utils import occlusion_test_evaluation

# ============ 全局变量配置 =============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"        # 数据集根目录
TEST_RATIO = 0.2
SAMPLE_RATIO = 1.0       # 默认用全部，后面做不平衡
BATCH_SIZE = 16
IM_SIZE = 224
RANDOM_SEED = 42
EPOCHS = 3
LR = 1e-4

# 结果表
all_results = []

def main():
    print(f"Using device: {DEVICE}")
    # ========= Step 1: 普通数据训练 =========
    print("\n=== Step 1: Normal Data Training ===")
    train_paths, train_labels, test_paths, test_labels, class_names = load_imbalanced_data(
        root_folder=DATA_DIR,
        test_ratio=TEST_RATIO,
        sample_ratio=SAMPLE_RATIO,
        reduce_factor=1.0,     # 不减少任何类
        minority_classes=[],
        random_seed=RANDOM_SEED
    )
    train_loader, test_loader = make_data_loaders(
        train_paths, train_labels,
        test_paths, test_labels,
        class_names,
        batch_size=BATCH_SIZE,
        im_size=IM_SIZE
    )
    num_classes = len(class_names)

    def train_and_return_metrics(model_name, scenario="NormalData", epochs=3, lr=1e-4):
        start_time = time.time()
        model = create_model(model_name, num_classes=num_classes, pretrained=True)
        model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_acc = 0
        for ep in range(1, epochs+1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_acc, preds, labs = evaluate(model, test_loader, criterion, DEVICE)
            if val_acc > best_acc:
                best_acc = val_acc
            print(f"[{scenario}-{model_name}] Epoch {ep}/{epochs}, train_acc={tr_acc:.4f}, test_acc={val_acc:.4f}")

        train_time = time.time() - start_time

        # 计算更多指标
        acc = accuracy_score(labs, preds)
        f1  = f1_score(labs, preds, average='weighted')
        prec= precision_score(labs, preds, average='weighted')
        rec = recall_score(labs, preds, average='weighted')

        # 存 all_results
        all_results.append({
            "Scenario": scenario,
            "Model": model_name,
            "Accuracy": f"{acc*100:.2f}%",
            "F1": f"{f1:.3f}",
            "Precision": f"{prec:.3f}",
            "Recall": f"{rec:.3f}",
            "TrainTime(s)": f"{train_time:.1f}"
        })

        return model

    # 训练2种模型
    resnet_normal = train_and_return_metrics("resnet18", scenario="NormalData", epochs=EPOCHS, lr=LR)
    mobilenet_normal = train_and_return_metrics("mobilenet_v2", scenario="NormalData", epochs=EPOCHS, lr=LR)

    # ========= Step 2: 不平衡数据训练 =========
    print("\n=== Step 2: Imbalanced Data Training ===")
    # 例如减少类别0,1到10%
    minority_classes = [0,1]
    reduce_factor = 0.1

    train_paths_imb, train_labels_imb, test_paths_imb, test_labels_imb, class_names_imb = load_imbalanced_data(
        root_folder=DATA_DIR,
        test_ratio=TEST_RATIO,
        sample_ratio=1.0,
        reduce_factor=reduce_factor,
        minority_classes=minority_classes,
        random_seed=RANDOM_SEED
    )
    train_loader_imb, test_loader_imb = make_data_loaders(
        train_paths_imb, train_labels_imb,
        test_paths_imb, test_labels_imb,
        class_names_imb,
        batch_size=BATCH_SIZE,
        im_size=IM_SIZE
    )
    num_classes_imb = len(class_names_imb)

    def train_imb_and_return_metrics(model_name, scenario="Imbalanced", epochs=3, lr=1e-4):
        import numpy as np
        from collections import Counter

        start_time = time.time()
        model = create_model(model_name, num_classes=num_classes_imb, pretrained=True)
        model.to(DEVICE)

        # class_weight
        label_counts = Counter(train_labels_imb)
        c_array = np.array([label_counts.get(i,0) for i in range(num_classes_imb)], dtype=np.float32)
        inv_c = 1.0/(c_array+1e-7)
        inv_c /= inv_c.sum()/num_classes_imb
        weight_tensor = torch.tensor(inv_c).to(DEVICE)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_acc = 0
        for ep in range(1, epochs+1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader_imb, optimizer, criterion, DEVICE)
            val_loss, val_acc, preds, labs = evaluate(model, test_loader_imb, criterion, DEVICE)
            if val_acc>best_acc:
                best_acc = val_acc
            print(f"[{scenario}-{model_name}] Epoch {ep}/{epochs}, train_acc={tr_acc:.4f}, test_acc={val_acc:.4f}")

        train_time = time.time() - start_time

        acc = accuracy_score(labs, preds)
        f1  = f1_score(labs, preds, average='weighted')
        prec= precision_score(labs, preds, average='weighted')
        rec = recall_score(labs, preds, average='weighted')

        all_results.append({
            "Scenario": scenario,
            "Model": model_name,
            "Accuracy": f"{acc*100:.2f}%",
            "F1": f"{f1:.3f}",
            "Precision": f"{prec:.3f}",
            "Recall": f"{rec:.3f}",
            "TrainTime(s)": f"{train_time:.1f}"
        })
        return model

    resnet_imb = train_imb_and_return_metrics("resnet18", scenario="Imbalanced", epochs=EPOCHS, lr=LR)
    mobilenet_imb = train_imb_and_return_metrics("mobilenet_v2", scenario="Imbalanced", epochs=EPOCHS, lr=LR)

    # ========== Step 3: 遮挡测试(批量)==========
    print("\n=== Step 3: Occlusion Test ===")
    import torchvision.transforms as T
    test_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # 1) NormalData ResNet
    occlusion_test_evaluation(
        model=resnet_normal,
        test_paths=test_paths,
        test_labels=test_labels,
        transform=test_transform,
        device=DEVICE,
        occlude_size=60,
        scenario="OcclusionTest_Normal_ResNet",
        model_name="resnet_normal",
        class_names=class_names,
        results_list=all_results
    )

    # 2) NormalData MobileNet
    occlusion_test_evaluation(
        model=mobilenet_normal,
        test_paths=test_paths,
        test_labels=test_labels,
        transform=test_transform,
        device=DEVICE,
        occlude_size=60,
        scenario="OcclusionTest_Normal_Mobile",
        model_name="mobilenet_normal",
        class_names=class_names,
        results_list=all_results
    )

    # 3) Imbalanced ResNet
    occlusion_test_evaluation(
        model=resnet_imb,
        test_paths=test_paths_imb,
        test_labels=test_labels_imb,
        transform=test_transform,
        device=DEVICE,
        occlude_size=60,
        scenario="OcclusionTest_Imb_ResNet",
        model_name="resnet_imb",
        class_names=class_names_imb,
        results_list=all_results
    )

    # 4) Imbalanced MobileNet
    occlusion_test_evaluation(
        model=mobilenet_imb,
        test_paths=test_paths_imb,
        test_labels=test_labels_imb,
        transform=test_transform,
        device=DEVICE,
        occlude_size=60,
        scenario="OcclusionTest_Imb_Mobile",
        model_name="mobilenet_imb",
        class_names=class_names_imb,
        results_list=all_results
    )

    # ========== Step 4: 打印最终结果 ==========
    df_all = pd.DataFrame(all_results)
    print(df_all)
    print("\n=== Summary Table ===")
    print(df_all.to_markdown(index=False))

    df_all.to_csv("final_comparison.csv", index=False)
    print("All experiments completed. Results saved in final_comparison.csv.")

# ============ 入口 ============
if __name__ == "__main__":
    main()
