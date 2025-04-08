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
│   │   └── demo_dl_method.ipynb     # Notebook 演示完整 DL 流程
│   └── ml_main.py                   # 脚本入口（运行所有深度模型实验）
│
├── ML_method/                       # 传统机器学习方法实现
│   ├── src/
│   │   ├── data_manager.py          # 图像路径采样与划分
│   │   ├── sift_processor.py        # 提取 SIFT 描述符
│   │   ├── lbp_processor.py         # 提取 LBP 描述符
│   │   ├── bow_encoder.py           # 建立视觉词袋并提取 BoW 特征
│   │   ├── feature_fusion.py        # 颜色直方图提取与增强
│   │   ├── evaluator.py             # 各项评价指标输出
│   │   └── demo_sift_lbp.ipynb      # Notebook 演示完整 ML 流程
│   └── run_experiment_main.py      # 脚本入口（运行全部分类器组合）
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

## 📊 传统方法实验结果（SIFT vs. LBP + 4分类器）

| 特征   | 分类器        | 准确率   | 训练时间(s) | 测试时间(s) |
|:--------|:---------------|:----------|-------------:|-------------:|
| SIFT    | SVM_RBF        | 72.25%   | 3.472       | 2.349       |
| SIFT    | RandomForest   | 72.42%   | 6.338       | 0.052       |
| SIFT    | XGBoost        | 76.75%   | 24.854      | 0.015       |
| SIFT    | KNN            | 59.75%   | 0.005       | 0.060       |
| LBP     | SVM_RBF        | 53.33%   | 4.170       | 2.431       |
| LBP     | RandomForest   | 67.75%   | 3.159       | 0.056       |
| LBP     | XGBoost        | 71.33%   | 16.325      | 0.016       |
| LBP     | KNN            | 33.92%   | 0.005       | 0.056       |

---

## 🤖 深度学习方法结果（ResNet18 vs MobileNetV2）

| 实验场景               | 模型             | 准确率  | F1值   | 精确率 | 召回率 | 训练时长(s) |
|:------------------------|:------------------|:--------|:-------|:--------|:--------|--------------|
| 正常数据集              | resnet18          | 95.17% | 0.952 | 0.954  | 0.952  | 903.1        |
| 正常数据集              | mobilenet_v2      | 95.00% | 0.950 | 0.952  | 0.950  | 945.0        |
| 不平衡数据集            | resnet18          | 91.57% | 0.918 | 0.932  | 0.916  | 1494.1       |
| 不平衡数据集            | mobilenet_v2      | 95.45% | 0.955 | 0.957  | 0.955  | 1425.2       |
| 遮挡测试（正常数据）    | resnet_normal     | 73.00% | 0.760 | 0.888  | 0.730  | N/A          |
| 遮挡测试（正常数据）    | mobilenet_normal  | 83.08% | 0.846 | 0.909  | 0.831  | N/A          |
| 遮挡测试（不平衡数据）  | resnet_imb        | 69.08% | 0.720 | 0.899  | 0.691  | N/A          |
| 遮挡测试（不平衡数据）  | mobilenet_imb     | 82.34% | 0.837 | 0.918  | 0.823  | N/A          |

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