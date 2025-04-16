import numpy as np
import pandas as pd
import time
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# === Import project modules ===
from src.data_manager import load_and_split_dataset
from src.sift_processor import extract_sift_descriptors
from src.lbp_processor import extract_lbp_descriptors
from src.bow_encoder import build_visual_dictionary, transform_to_bow
from src.feature_fusion import extract_rgb_histogram, augment_image

SEED = 42

def run():
    # STEP 1-2: Load dataset
    train_paths, train_labels, test_paths, test_labels, class_names = load_and_split_dataset(
        root_folder="data", test_ratio=0.2, sample_ratio=0.5, random_seed=SEED
    )
    y_train, y_test = np.array(train_labels), np.array(test_labels)

    # STEP 3: Define classifiers
    models = {
        "SVM_RBF": SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=SEED),
        "XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                     eval_metric='mlogloss', random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    # STEP 4: SIFT Features
    sift_desc_all = extract_sift_descriptors(train_paths, augmentation_fn=None)
    kmeans_model_sift = build_visual_dictionary(sift_desc_all, 300, 15000, SEED)
    sift_train_plain = extract_sift_descriptors(train_paths, augmentation_fn=None)
    sift_test_plain = extract_sift_descriptors(test_paths, augmentation_fn=None)
    X_train_sift_plain = transform_to_bow(sift_train_plain, kmeans_model_sift)
    X_test_sift_plain = transform_to_bow(sift_test_plain, kmeans_model_sift)
    sift_train_aug = extract_sift_descriptors(train_paths, augmentation_fn=augment_image)
    sift_test_aug = extract_sift_descriptors(test_paths, augmentation_fn=None)
    bow_train_sift_full = transform_to_bow(sift_train_aug, kmeans_model_sift)
    bow_test_sift_full = transform_to_bow(sift_test_aug, kmeans_model_sift)
    hist_train_sift = np.array([extract_rgb_histogram(p) for p in train_paths])
    hist_test_sift = np.array([extract_rgb_histogram(p) for p in test_paths])
    X_train_sift_full = np.hstack((bow_train_sift_full, hist_train_sift))
    X_test_sift_full = np.hstack((bow_test_sift_full, hist_test_sift))

    # STEP 5: LBP Features
    lbp_desc_all = extract_lbp_descriptors(train_paths, augmentation_fn=None)
    kmeans_model_lbp = build_visual_dictionary(lbp_desc_all, 300, 15000, SEED)
    lbp_train_plain = extract_lbp_descriptors(train_paths, augmentation_fn=None)
    lbp_test_plain = extract_lbp_descriptors(test_paths, augmentation_fn=None)
    X_train_lbp_plain = transform_to_bow(lbp_train_plain, kmeans_model_lbp)
    X_test_lbp_plain = transform_to_bow(lbp_test_plain, kmeans_model_lbp)
    lbp_train_aug = extract_lbp_descriptors(train_paths, augmentation_fn=augment_image)
    lbp_test_aug = extract_lbp_descriptors(test_paths, augmentation_fn=None)
    bow_train_lbp_full = transform_to_bow(lbp_train_aug, kmeans_model_lbp)
    bow_test_lbp_full = transform_to_bow(lbp_test_aug, kmeans_model_lbp)
    hist_train_lbp = np.array([extract_rgb_histogram(p) for p in train_paths])
    hist_test_lbp = np.array([extract_rgb_histogram(p) for p in test_paths])
    X_train_lbp_full = np.hstack((bow_train_lbp_full, hist_train_lbp))
    X_test_lbp_full = np.hstack((bow_test_lbp_full, hist_test_lbp))

    # STEP 6-7: Evaluation
    results = []
    for setting_name, feature_set in [
        ("BoW Only", [("SIFT_plain", X_train_sift_plain, X_test_sift_plain),
                      ("LBP_plain", X_train_lbp_plain, X_test_lbp_plain)]),
        ("With ColorHist + Aug", [("SIFT_full", X_train_sift_full, X_test_sift_full),
                                  ("LBP_full", X_train_lbp_full, X_test_lbp_full)])
    ]:
        for feature_tag, X_train_feat, X_test_feat in feature_set:
            for model_name, model in models.items():
                print(f"🔍 Training: {feature_tag} + {model_name}")
                model.fit(X_train_feat, y_train)

                start = time.time()
                y_pred = model.predict(X_test_feat)
                test_time = time.time() - start

                acc = accuracy_score(y_test, y_pred)
                print(classification_report(y_test, y_pred, target_names=class_names))

                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
                plt.title(f"Confusion Matrix: {feature_tag} + {model_name}")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                plt.show()

                report_dict = classification_report(y_test, y_pred, output_dict=True)
                results.append({
                    "Feature": feature_tag,
                    "Model": model_name,
                    "Accuracy": f"{acc*100:.2f}%",
                    "F1": round(report_dict["weighted avg"]["f1-score"], 3),
                    "Precision": round(report_dict["weighted avg"]["precision"], 3),
                    "Recall": round(report_dict["weighted avg"]["recall"], 3),
                    "Test Time (s)": round(test_time, 3),
                    "Setting": setting_name
                })

    # STEP 8: Summary
    df_all = pd.DataFrame(results)
    print("📊 Summary Table:")
    print(df_all)

    # STEP 9: Markdown Table Output
    def format_markdown_table(df):
        headers = "| " + " | ".join(df.columns) + " |"
        separators = "|" + "|".join([":--:" for _ in df.columns]) + "|"
        rows = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in df.values
        ]
        return "\n".join([headers, separators] + rows)

    print("\n📋 Markdown Table for Report:")
    print(format_markdown_table(df_all))

if __name__ == "__main__":
    run()
