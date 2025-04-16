# 🛰️ COMP9517 小组项目 - 遥感图像分类系统

本项目实现了两种遥感图像分类方法：

1. 🔍 传统机器学习方法（ML）：SIFT / LBP + 视觉词袋（BoW）+ 颜色直方图 + 4 种分类器
2. 🤖 深度学习方法（DL）：ResNet18 / MobileNetV2 + 数据增强 + Grad-CAM + 遮挡鲁棒性测试

---

## 📁 项目结构说明

```
comp9517_group_project/
│
├── Aerial_Landscapes/               # 数据集文件夹（已按类别分类）
│
├── DL_method/                       # 深度学习方法实现
│   ├── src/
│   │   ├── dataset_manager.py       # 加载划分数据集（支持不平衡模拟）
│   │   ├── model_builder.py         # 创建 ResNet18 / MobileNetV2 模型
│   │   ├── train_utils.py           # 模型训练与评估工具函数
│   │   ├── gradcam_utils.py         # Grad-CAM 可视化
│   │   ├── occlusion_utils.py       # 遮挡测试函数（模型鲁棒性分析）
│   └── run_dl_experiment_main.py    # 脚本入口（运行所有深度模型实验）
└── └── demo_dl_method.ipynb         # Notebook 演示完整 DL 流程
│
│
├── ML_method/                       # 传统机器学习方法实现
│   ├── src/
│   │   ├── data_manager.py          # 图像路径采样与划分
│   │   ├── sift_processor.py        # 提取 SIFT 描述符
│   │   ├── lbp_processor.py         # 提取 LBP 描述符
│   │   ├── bow_encoder.py           # 建立视觉词袋并提取 BoW 特征
│   │   ├── feature_fusion.py        # 颜色直方图提取与增强
│   │   ├── evaluator.py             # 各项评价指标输出
│   └── demo_ml_method.py      # 脚本入口（运行全部分类器组合）
└── └── run_ml_experiment_main.py      # Notebook 演示完整 ML 流程
│
├── README.md                        # 项目说明文档（中英双语版本）
└── *.md                             # 方法/模型说明文件
```

---

## ✅ 项目功能亮点

- ✅ 支持 SIFT 和 LBP 特征提取与对比
- ✅ 构建视觉词袋模型（BoW）+ 颜色直方图特征融合
- ✅ 集成 4 种分类器（SVM / RF / XGBoost / KNN）
- ✅ 支持 ResNet18 与 MobileNetV2 网络结构训练与测试
- ✅ 支持不平衡数据集训练模拟（类间图像数不均）
- ✅ Grad-CAM 可视化神经网络注意力区域
- ✅ 遮挡/噪声测试分析模型鲁棒性
- ✅ 输出完整对比实验表格（准确率/F1/训练时间等）

---

## 📊 SIFT vs. LBP + 4种分类器和数据增强对比（完整评估指标）

📋 Markdown Table - BoW Only:

| Feature | Model | Accuracy | F1 | Precision | Recall | Test Time (s) | Setting |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SIFT_plain | SVM_RBF | 69.42% | 0.691 | 0.694 | 0.694 | 1.171 | BoW Only |
| SIFT_plain | RandomForest | 63.08% | 0.618 | 0.621 | 0.631 | 0.035 | BoW Only |
| SIFT_plain | XGBoost | 66.92% | 0.67 | 0.673 | 0.669 | 0.014 | BoW Only |
| SIFT_plain | KNN | 55.42% | 0.539 | 0.589 | 0.554 | 0.06 | BoW Only |
| LBP_plain | SVM_RBF | 41.42% | 0.411 | 0.42 | 0.414 | 0.763 | BoW Only |
| LBP_plain | RandomForest | 40.83% | 0.402 | 0.407 | 0.408 | 0.065 | BoW Only |
| LBP_plain | XGBoost | 40.25% | 0.401 | 0.419 | 0.403 | 0.009 | BoW Only |
| LBP_plain | KNN | 24.83% | 0.206 | 0.274 | 0.248 | 0.049 | BoW Only |

📋 Markdown Table - With ColorHist + Aug:

| Feature | Model | Accuracy | F1 | Precision | Recall | Test Time (s) | Setting |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SIFT_full | SVM_RBF | 73.83% | 0.737 | 0.742 | 0.738 | 2.629 | With ColorHist + Aug |
| SIFT_full | RandomForest | 71.08% | 0.704 | 0.716 | 0.711 | 0.04 | With ColorHist + Aug |
| SIFT_full | XGBoost | 76.50% | 0.766 | 0.774 | 0.765 | 0.019 | With ColorHist + Aug |
| SIFT_full | KNN | 56.67% | 0.555 | 0.576 | 0.567 | 0.157 | With ColorHist + Aug |
| LBP_full | SVM_RBF | 58.25% | 0.575 | 0.579 | 0.583 | 2.938 | With ColorHist + Aug |
| LBP_full | RandomForest | 67.50% | 0.669 | 0.673 | 0.675 | 0.044 | With ColorHist + Aug |
| LBP_full | XGBoost | 70.83% | 0.708 | 0.711 | 0.708 | 0.013 | With ColorHist + Aug |
| LBP_full | KNN | 44.58% | 0.433 | 0.454 | 0.446 | 0.127 | With ColorHist + Aug |

---

## 📊 深度学习方法评估结果对比（ResNet18 vs. MobileNetV2）

| Scenario                     | Model            | Accuracy |   F1   | Precision | Recall | Train Time (s) |
|:-----------------------------|:------------------|:---------|:------:|:----------|:-------|----------------:|
| NormalData                   | resnet18           | 94.58%   | 0.946  | 0.948     | 0.946  | 73.7            |
| NormalData                   | mobilenet_v2       | 95.17%   | 0.952  | 0.954     | 0.952  | 73.6            |
| Imbalanced                   | resnet18           | 94.65%   | 0.947  | 0.951     | 0.946  | 116.7           |
| Imbalanced                   | mobilenet_v2       | 96.16%   | 0.962  | 0.963     | 0.962  | 125.9           |
| OcclusionTest_Normal_ResNet | resnet_normal      | 82.33%   | 0.827  | 0.876     | 0.823  | N/A             |
| OcclusionTest_Normal_Mobile | mobilenet_normal   | 82.83%   | 0.844  | 0.910     | 0.828  | N/A             |
| OcclusionTest_Imb_ResNet    | resnet_imb         | 74.24%   | 0.759  | 0.904     | 0.742  | N/A             |
| OcclusionTest_Imb_Mobile    | mobilenet_imb      | 73.30%   | 0.769  | 0.920     | 0.733  | N/A             |

---

## 🧪 如何运行实验

### ▶ 运行传统 ML 方法
```bash
cd ML_method
python run_ml_experiment_main.py
```

### ▶ 运行深度学习 DL 方法
```bash
cd DL_method
python run_dl_experiment_main.py
```

或者直接运行对应的 Jupyter Notebook 交互式执行并展示图像结果。

---

## 👥 作者信息

- Zhi Li & Group five fighter | COMP9517 T1 2025



## 🧠 所用模型与版本信息

### 📌 深度学习模型
- `ResNet18`：`torchvision.models.resnet18` (预训练)
- `MobileNetV2`：`torchvision.models.mobilenet_v2` (预训练)
- 框架版本：
  - `PyTorch` >= 1.13
  - `torchvision` >= 0.14
  - `pytorch-grad-cam` >= 1.3.8

### 🧪 传统机器学习模型
- `SVM_RBF`：来自 `sklearn.svm.SVC`
- `RandomForest`：来自 `sklearn.ensemble.RandomForestClassifier`
- `XGBoost`：`xgboost.XGBClassifier`
- `KNN`：`sklearn.neighbors.KNeighborsClassifier`

### 🔧 主要依赖库版本
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `opencv-python`, `scikit-image`
- `scikit-learn >= 1.1`
- `xgboost >= 1.7`
- `pytorch-grad-cam`（用于可解释性可视化）

---

## 💡 建议改进项 / 可拓展方向

- ✅ 支持更多深度模型（如 EfficientNet / ConvNeXt）
- ✅ 增加模型训练曲线（可视化 loss/acc）
- ✅ 模型保存与加载功能（持久化 checkpoint）
- ✅ 更多类型的攻击与防御方法集成（Adversarial Attack）
- ✅ 图像分割、目标检测等其他遥感任务拓展