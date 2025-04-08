import numpy as np
import pandas as pd
import time
import sys

# 确保能够import到src/中的模块
sys.path.append('.')
sys.path.append('./src')
sys.path.append('..')

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 自定义模块
from src.data_manager import load_and_split_dataset
from src.sift_processor import extract_sift_descriptors
from src.lbp_processor import extract_lbp_descriptors
from src.bow_encoder import build_visual_dictionary, transform_to_bow
from src.feature_fusion import extract_rgb_histogram, augment_image
from src.evaluator import print_classification_metrics

def run_full_experiment(
    data_dir="../Aerial_Landscapes",
    test_ratio=0.2,
    sample_ratio=0.5,
    seed=42,
    n_clusters=100,
    max_samples=5000,
    lbp_radius=1,
    lbp_points=None
):
    """
    一键运行SIFT和LBP两种特征 + 4分类器(SVM, RF, XGBoost, KNN)的所有流程，
    并输出性能对比表。

    参数：
      data_dir: 数据目录
      test_ratio: 测试集比例
      sample_ratio: 每类抽取图片比例
      seed: 随机种子
      n_clusters: KMeans聚类中心数
      max_samples: KMeans聚类时最大描述符数
      lbp_radius: LBP半径
      lbp_points: LBP采样点个数(默认为8*radius)
    返回：
      df_all: 包含 8 行 (2特征 x 4分类器) 的 DataFrame，记录准确率、训练/测试时间等
    """
    if lbp_points is None:
        lbp_points = 8 * lbp_radius

    # ---------- 1. 加载&划分数据 ----------
    train_paths, train_labels, test_paths, test_labels, class_names = load_and_split_dataset(
        root_folder=data_dir,
        test_ratio=test_ratio,
        sample_ratio=sample_ratio,
        random_seed=seed
    )
    print(f"[DATA] 训练图像数: {len(train_paths)}, 测试图像数: {len(test_paths)}")
    print("类别名称:", class_names)

    # ---------- 2. 模型训练+评估函数 ----------
    def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_name):
        model_list = [
            ("SVM_RBF", SVC(kernel='rbf', C=10, gamma='scale', random_state=seed)),
            ("RandomForest", RandomForestClassifier(n_estimators=150, random_state=seed)),
            ("XGBoost", xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                          eval_metric='mlogloss', random_state=seed)),
            ("KNN", KNeighborsClassifier(n_neighbors=5))
        ]
        results = []
        for model_name, clf in model_list:
            print(f"\n=== [{feature_name} - {model_name}] ===")
            start_train = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - start_train

            start_test = time.time()
            y_pred = clf.predict(X_test)
            test_time = time.time() - start_test

            # 打印分类报告 & 混淆矩阵
            print_classification_metrics(y_test, y_pred, class_names)
            acc = accuracy_score(y_test, y_pred)

            results.append({
                "Feature": feature_name,
                "Model": model_name,
                "Accuracy": f"{acc*100:.2f}%",
                "Train Time (s)": f"{train_time:.3f}",
                "Test Time (s)": f"{test_time:.3f}"
            })
        return results

    all_results = []

    # ---------- 3. SIFT 流程 ----------
    print("\n========== [SIFT 流程] ==========")
    sift_train_desc = extract_sift_descriptors(train_paths, augmentation_fn=augment_image)
    sift_test_desc  = extract_sift_descriptors(test_paths, augmentation_fn=None)

    kmeans_sift = build_visual_dictionary(sift_train_desc, n_clusters, max_samples, seed)
    bow_sift_train = transform_to_bow(sift_train_desc, kmeans_sift)
    bow_sift_test  = transform_to_bow(sift_test_desc, kmeans_sift)

    hist_sift_train = np.array([extract_rgb_histogram(p) for p in train_paths])
    hist_sift_test  = np.array([extract_rgb_histogram(p) for p in test_paths])

    X_sift_train = np.hstack((bow_sift_train, hist_sift_train))
    X_sift_test  = np.hstack((bow_sift_test,  hist_sift_test))

    scaler_sift = StandardScaler()
    X_sift_train_scaled = scaler_sift.fit_transform(X_sift_train)
    X_sift_test_scaled  = scaler_sift.transform(X_sift_test)

    results_sift = train_and_evaluate_models(
        X_train=X_sift_train_scaled,
        X_test=X_sift_test_scaled,
        y_train=train_labels,
        y_test=test_labels,
        feature_name="SIFT"
    )
    all_results.extend(results_sift)

    # ---------- 4. LBP 流程 ----------
    print("\n========== [LBP 流程] ==========")
    from src.lbp_processor import extract_lbp_descriptors
    lbp_train_desc = extract_lbp_descriptors(
        train_paths, radius=lbp_radius, n_points=lbp_points, method='uniform',
        augmentation_fn=augment_image
    )
    lbp_test_desc = extract_lbp_descriptors(
        test_paths, radius=lbp_radius, n_points=lbp_points, method='uniform',
        augmentation_fn=None
    )

    kmeans_lbp = build_visual_dictionary(lbp_train_desc, n_clusters, max_samples, seed)
    bow_lbp_train = transform_to_bow(lbp_train_desc, kmeans_lbp)
    bow_lbp_test  = transform_to_bow(lbp_test_desc, kmeans_lbp)

    hist_lbp_train = np.array([extract_rgb_histogram(p) for p in train_paths])
    hist_lbp_test  = np.array([extract_rgb_histogram(p) for p in test_paths])

    X_lbp_train = np.hstack((bow_lbp_train, hist_lbp_train))
    X_lbp_test  = np.hstack((bow_lbp_test,  hist_lbp_test))

    scaler_lbp = StandardScaler()
    X_lbp_train_scaled = scaler_lbp.fit_transform(X_lbp_train)
    X_lbp_test_scaled  = scaler_lbp.transform(X_lbp_test)

    results_lbp = train_and_evaluate_models(
        X_train=X_lbp_train_scaled,
        X_test=X_lbp_test_scaled,
        y_train=train_labels,
        y_test=test_labels,
        feature_name="LBP"
    )
    all_results.extend(results_lbp)

    # ---------- 5. 打印&保存结果 ----------
    df_all = pd.DataFrame(all_results)
    print("\n===== 全部结果汇总 =====")
    print(df_all.to_markdown(index=False))

    # 可以保存输出CSV
    df_all.to_csv("sift_lbp_comparison.csv", index=False)

    return df_all

def main():
    """
    main函数：执行 run_full_experiment 并打印结果
    """
    df = run_full_experiment(
        data_dir="../Aerial_Landscapes",      # 数据路径
        test_ratio=0.2,
        sample_ratio=0.5,
        seed=42,
        n_clusters=100,
        max_samples=5000,
        lbp_radius=1,
        lbp_points=None
    )
    print("\n[INFO] 实验已完成, 最终结果如下:")
    print(df)

if __name__ == "__main__":
    main()
