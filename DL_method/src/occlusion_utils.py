import cv2
import numpy as np
import torch
import torch.nn.functional as F
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def occlude_image(img_bgr, occlude_size=60):
    """
    在图像中随机位置画一个方块进行遮挡，返回新图像。
    """
    h, w, _ = img_bgr.shape
    y0 = random.randint(0, h - occlude_size)
    x0 = random.randint(0, w - occlude_size)
    occluded = img_bgr.copy()
    cv2.rectangle(occluded, (x0,y0), (x0+occlude_size,y0+occlude_size), (0,0,0), -1)
    return occluded

def predict_single(model, img_bgr, transform, device):
    """
    将BGR格式图像转换后给model做单张预测，返回预测类别ID (int)。
    """
    import torchvision.transforms as T
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        pred_idx = out.argmax(dim=1).item()
    return pred_idx

def occlusion_test_evaluation(model, test_paths, test_labels, transform, device,
                              occlude_size=60, scenario="OcclusionTest",
                              model_name="model", class_names=None, results_list=None):
    """
    对test_paths里每张图做 "遮挡后" 预测(不另做遮挡前预测)，
    并与真实标签对比，统计宏观Accuracy/F1/Precision/Recall。

    - model: 已训练模型
    - test_paths, test_labels: 测试图像路径 & 对应真实标签ID
    - transform: 与训练相同的图像变换
    - device: CPU or CUDA
    - occlude_size: 方块大小
    - scenario, model_name: 记录到结果里的名称
    - class_names: 类别名列表 (若你要打印分类报告可用)
    - results_list: 如果传入一个list，就把结果dict加入其中

    返回: (preds_after, labs)
    """
    preds_after = []
    labs = []

    for path, label in zip(test_paths, test_labels):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # 如果图像读取失败, 跳过或print
            continue

        # 遮挡
        occ_img = occlude_image(img_bgr, occlude_size=occlude_size)
        # 推理
        pred_aft = predict_single(model, occ_img, transform, device)
        preds_after.append(pred_aft)
        labs.append(label)

    # 统计宏观指标
    acc = accuracy_score(labs, preds_after)
    f1  = f1_score(labs, preds_after, average='weighted')
    prec= precision_score(labs, preds_after, average='weighted')
    rec = recall_score(labs, preds_after, average='weighted')

    print(f"\n===[{scenario}]===  {model_name} occlude_size={occlude_size}")
    print(f"遮挡后: Acc={acc:.3f}, F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")

    # 做混淆矩阵或分类报告
    if class_names is not None and len(class_names) > 0:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(labs, preds_after)
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{scenario}-{model_name}: Occlusion Confusion")
        plt.show()

        print("Classification Report (遮挡后):")
        print(classification_report(labs, preds_after, target_names=class_names))

    # 如果results_list不为空, 将指标append进去
    if results_list is not None:
        results_list.append({
            "Scenario": scenario,
            "Model": model_name,
            "Accuracy": f"{acc*100:.2f}%",
            "F1": f"{f1:.3f}",
            "Precision": f"{prec:.3f}",
            "Recall": f"{rec:.3f}",
            "TrainTime(s)": "N/A"
        })

    return preds_after, labs
