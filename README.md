# 🛰️ COMP9517 Group Project – Remote Sensing Image Classification System

This project implements **two** approaches for remote‑sensing image classification:

1. 🔍 **Traditional Machine‑Learning (ML) pipeline**  
   *SIFT / LBP*  +  *Bag of Visual Words (BoW)*  +  *Color Histograms*  +  **4 classifiers**

2. 🤖 **Deep‑Learning (DL) pipeline**  
   *ResNet18 / MobileNetV2*  +  *Data Augmentation*  +  *Imbalanced‑data training*  +  *Occlusion‑robustness testing*

---

## 📁 Project Structure

```text
comp9517_group_project/
│
├── Aerial_Landscapes/               # Dataset (already organised by class)
│
├── DL_method/                       # Deep‑learning implementation
│   ├── src/
│   │   ├── dataset_manager.py       # Dataset loading & splitting (supports imbalance simulation)
│   │   ├── model_builder.py         # Build ResNet18 / MobileNetV2 models
│   │   ├── train_utils.py           # Training & evaluation helpers
│   │   ├── gradcam_utils.py         # Grad‑CAM visualisation
│   │   ├── occlusion_utils.py       # Occlusion testing (robustness analysis)
│   ├── run_dl_experiment_main.py    # Entry script – run all DL experiments
│   └── demo_dl_method.ipynb         # Notebook showing the full DL workflow
│
├── ML_method/                       # Traditional ML implementation
│   ├── src/
│   │   ├── data_manager.py          # Sample image paths and train/val/test split
│   │   ├── sift_processor.py        # Extract SIFT descriptors
│   │   ├── lbp_processor.py         # Extract LBP descriptors
│   │   ├── bow_encoder.py           # Build BoW vocabulary & extract BoW features
│   │   ├── feature_fusion.py        # Extract + fuse colour‑histogram features
│   │   ├── evaluator.py             # Output evaluation metrics
│   ├── demo_ml_method.py            # Entry script – run all classifier combos
│   └── run_ml_experiment_main.py    # Notebook showing the full ML workflow
│
├── README.md                        # Project documentation (bilingual CN/EN)
└── *.md                             # Supplementary method/model notes
```

---

## ✅ Key Features

- **SIFT** and **LBP** feature extraction with side‑by‑side comparison  
- **BoW model** + **colour‑histogram fusion**  
- Integration of **four classifiers** – SVM, Random Forest, XGBoost, KNN  
- Training & testing with **ResNet18** and **MobileNetV2** architectures  
- **Imbalanced‑dataset simulation** (uneven image counts per class)  
- **Occlusion / noise** robustness testing  
- Automatic export of **comprehensive comparison tables** (Accuracy, F1, training time, …)

---

## 📊 SIFT vs LBP + 4 Classifiers & Data‑Augmentation Comparison (complete metrics)

### 📋 BoW‑only

| Feature    | Model         | Accuracy |   F1   | Precision | Recall | Test Time (s) | Setting |
|:----------:|:-------------:|:-------:|:------:|:---------:|:------:|:-------------:|:-------:|
| SIFT_plain | SVM_RBF       | 69.42 % | 0.691 | 0.694 | 0.694 | 1.171 | BoW |
| SIFT_plain | RandomForest  | 63.08 % | 0.618 | 0.621 | 0.631 | 0.035 | BoW |
| SIFT_plain | XGBoost       | 66.92 % | 0.670 | 0.673 | 0.669 | 0.014 | BoW |
| SIFT_plain | KNN           | 55.42 % | 0.539 | 0.589 | 0.554 | 0.060 | BoW |
| LBP_plain  | SVM_RBF       | 41.42 % | 0.411 | 0.420 | 0.414 | 0.763 | BoW |
| LBP_plain  | RandomForest  | 40.83 % | 0.402 | 0.407 | 0.408 | 0.065 | BoW |
| LBP_plain  | XGBoost       | 40.25 % | 0.401 | 0.419 | 0.403 | 0.009 | BoW |
| LBP_plain  | KNN           | 24.83 % | 0.206 | 0.274 | 0.248 | 0.049 | BoW |

### 📋 With Colour‑Histogram Fusion + Augmentation

| Feature   | Model        | Accuracy |   F1   | Precision | Recall | Test Time (s) | Setting |
|:---------:|:------------:|:--------:|:------:|:---------:|:------:|:-------------:|:-------:|
| SIFT_full | SVM_RBF      | 73.83 % | 0.737 | 0.742 | 0.738 | 2.629 | +ColourHist +Aug |
| SIFT_full | RandomForest | 71.08 % | 0.704 | 0.716 | 0.711 | 0.040 | +ColourHist +Aug |
| SIFT_full | XGBoost      | 76.50 % | 0.766 | 0.774 | 0.765 | 0.019 | +ColourHist +Aug |
| SIFT_full | KNN          | 56.67 % | 0.555 | 0.576 | 0.567 | 0.157 | +ColourHist +Aug |
| LBP_full  | SVM_RBF      | 58.25 % | 0.575 | 0.579 | 0.583 | 2.938 | +ColourHist +Aug |
| LBP_full  | RandomForest | 67.50 % | 0.669 | 0.673 | 0.675 | 0.044 | +ColourHist +Aug |
| LBP_full  | XGBoost      | 70.83 % | 0.708 | 0.711 | 0.708 | 0.013 | +ColourHist +Aug |
| LBP_full  | KNN          | 44.58 % | 0.433 | 0.454 | 0.446 | 0.127 | +ColourHist +Aug |

---

## 📊 Deep‑Learning Results (ResNet18 vs MobileNetV2)

### Balanced Dataset – plain vs augmented

| Model       | Setting          | Accuracy |   F1   | Precision | Recall | Train Time (s) |
|-------------|------------------|:--------:|:------:|:---------:|:------:|:--------------:|
| ResNet18    | Balanced         | 0.9817 | 0.9816 | 0.9818 | 0.9817 | 2128.19 |
| ResNet18    | Balanced + Aug   | 0.9779 | 0.9779 | 0.9781 | 0.9779 | 1788.94 |
| MobileNetV2 | Balanced         | 0.9754 | 0.9754 | 0.9756 | 0.9754 | 2042.97 |
| MobileNetV2 | Balanced + Aug   | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 1210.77 |

### Imbalanced Dataset – plain vs augmented

| Model       | Setting            | Accuracy |   F1   | Precision | Recall | Train Time (s) |
|-------------|--------------------|:--------:|:------:|:---------:|:------:|:--------------:|
| ResNet18    | Imbalanced         | 0.9593 | 0.9592 | 0.9598 | 0.9593 | 512.33 |
| ResNet18    | Imbalanced + Aug   | 0.9536 | 0.9536 | 0.9553 | 0.9536 | 508.44 |
| MobileNetV2 | Imbalanced         | 0.9593 | 0.9593 | 0.9601 | 0.9593 | 505.32 |
| MobileNetV2 | Imbalanced + Aug   | 0.9688 | 0.9687 | 0.9697 | 0.9688 | 506.82 |

#### Performance on augmented classes only

| Model       | Class        | Setting            |   F1   | Precision | Recall |
|-------------|--------------|--------------------|:------:|:---------:|:------:|
| ResNet18    | Residential  | Imbalanced         | 0.8750 | 0.8750 | 0.8750 |
| ResNet18    | River        | Imbalanced         | 0.7000 | 0.7778 | 0.6364 |
| ResNet18    | Residential  | Imbalanced + Aug   | 0.7778 | 0.7000 | 0.8750 |
| ResNet18    | River        | Imbalanced + Aug   | 0.8571 | 0.9000 | 0.8182 |
| MobileNetV2 | Residential  | Imbalanced         | 0.8235 | 0.7778 | 0.8750 |
| MobileNetV2 | River        | Imbalanced         | 0.8000 | 0.8889 | 0.7273 |
| MobileNetV2 | Residential  | Imbalanced + Aug   | 0.8000 | 0.8571 | 0.7500 |
| MobileNetV2 | River        | Imbalanced + Aug   | 0.9524 | 1.0000 | 0.9091 |

### 🧪 Robustness under Occlusion

| Scenario                     | Model          | Accuracy |   F1   | Precision | Recall |
|------------------------------|----------------|:--------:|:------:|:---------:|:------:|
| OcclusionTest_ResNet_Plain   | resnet_plain   | 92.67 % | 0.930 | 0.953 | 0.927 |
| OcclusionTest_Mobile_Plain   | mobilenet_plain| 77.38 % | 0.808 | 0.942 | 0.774 |
| OcclusionTest_ResNet_Aug     | resnet_aug     | 89.60 % | 0.870 | 0.939 | 0.850 |
| OcclusionTest_Mobile_Aug     | mobilenet_aug  | 86.25 % | 0.878 | 0.944 | 0.863 |

---

## 🧪 How to Run

### ▶ Traditional ML pipeline
```bash
cd ML_method
python run_ml_experiment_main.py
```

### ▶ Deep‑learning pipeline
```bash
cd DL_method
python run_dl_experiment_main.py
```

Or open the corresponding Jupyter notebooks for interactive execution and visualisation.

---

## 👥 Authors

*Zhi Li & Group Five Fighters* · COMP9517 T1 2025

---

## 🧠 Models & Versions

### 📌 Deep‑learning models
- **ResNet18** – `torchvision.models.resnet18` (pre‑trained)  
- **MobileNetV2** – `torchvision.models.mobilenet_v2` (pre‑trained)  

> Frameworks  
> • **PyTorch** ≥ 1.13  
> • **torchvision** ≥ 0.14  
> • **pytorch‑grad-cam** ≥ 1.3.8  

### 🧪 Traditional ML models
- **SVM_RBF** – `sklearn.svm.SVC`  
- **RandomForest** – `sklearn.ensemble.RandomForestClassifier`  
- **XGBoost** – `xgboost.XGBClassifier`  
- **KNN** – `sklearn.neighbors.KNeighborsClassifier`

### 🔧 Main dependencies
`numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python`, `scikit-image`,  
`scikit‑learn` ≥ 1.1, `xgboost` ≥ 1.7, `pytorch‑grad-cam`

---

## 💡 Future Improvements

- Add more CNN backbones (e.g., EfficientNet, ConvNeXt)  
- Visualise training curves (loss / accuracy)  
- Checkpoint saving & loading  
- Integrate adversarial attack/defence modules  
- Extend to further remote‑sensing tasks (segmentation, detection)
