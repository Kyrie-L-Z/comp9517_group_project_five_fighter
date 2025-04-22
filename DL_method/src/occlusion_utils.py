import cv2
import numpy as np
import torch
import torch.nn.functional as F
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def occlude_image(img_bgr, occlude_size=60):
    """
    Randomly draw a square block on the image for occlusion and return the new image.
    """
    h, w, _ = img_bgr.shape
    y0 = random.randint(0, h - occlude_size)
    x0 = random.randint(0, w - occlude_size)
    occluded = img_bgr.copy()
    cv2.rectangle(occluded, (x0, y0), (x0 + occlude_size, y0 + occlude_size), (0, 0, 0), -1)
    return occluded

def predict_single(model, img_bgr, transform, device):
    """
    Convert a BGR image and make a single prediction using the model.
    Return the predicted class ID (int).
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
    Perform prediction on each image in test_paths after applying occlusion only,
    then compare with ground truth and compute macro Accuracy/F1/Precision/Recall.

    - model: trained model
    - test_paths, test_labels: test image paths & corresponding ground truth label IDs
    - transform: same image transform as used in training
    - device: CPU or CUDA
    - occlude_size: square block size
    - scenario, model_name: identifiers for logging results
    - class_names: list of class names (for printing classification report)
    - results_list: if a list is provided, append result dictionary to it

    Returns: (preds_after, labs)
    """
    preds_after = []
    labs = []

    for path, label in zip(test_paths, test_labels):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # Skip or print if image failed to load
            continue

        # Apply occlusion
        occ_img = occlude_image(img_bgr, occlude_size=occlude_size)
        # Inference
        pred_aft = predict_single(model, occ_img, transform, device)
        preds_after.append(pred_aft)
        labs.append(label)

    # Compute macro metrics
    acc = accuracy_score(labs, preds_after)
    f1  = f1_score(labs, preds_after, average='weighted')
    prec = precision_score(labs, preds_after, average='weighted')
    rec = recall_score(labs, preds_after, average='weighted')

    print(f"\n===[{scenario}]===  {model_name} occlude_size={occlude_size}")
    print(f"After occlusion: Acc={acc:.3f}, F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")

    # Plot confusion matrix or classification report if class names are available
    if class_names is not None and len(class_names) > 0:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(labs, preds_after)
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{scenario}-{model_name}: Occlusion Confusion")
        plt.show()

        print("Classification Report (after occlusion):")
        print(classification_report(labs, preds_after, target_names=class_names))

    # Append metrics to results list if provided
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
