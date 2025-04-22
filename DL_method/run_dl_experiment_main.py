from src.dataset_manager import load_dataset, load_data_path_only
from src.model_builder import create_model
from src.train_utils import train_and_evaluate_model
from src.occlusion_utils import occlusion_test_evaluation

import torch
import pandas as pd
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_FOLDER = "../Aerial_Landscapes"
BATCH_SIZE = 128
IM_SIZE = 224

all_results = []

# === Step 1: Balanced Dataset (Plain)
print("🔹 Step 1: Balanced Dataset (No Augmentation)")
train_loader_plain, val_loader_plain, class_names = load_dataset(
    root_folder=ROOT_FOLDER, augment=False, imbalance=False, batch_size=BATCH_SIZE, im_size=IM_SIZE
)
resnet_plain = create_model("resnet18", num_classes=len(class_names)).to(DEVICE)
mobilenet_plain = create_model("mobilenet_v2", num_classes=len(class_names)).to(DEVICE)
resnet_plain_metrics = train_and_evaluate_model(resnet_plain, train_loader_plain, val_loader_plain, DEVICE, class_names, label="resnet_plain")
mobilenet_plain_metrics = train_and_evaluate_model(mobilenet_plain, train_loader_plain, val_loader_plain, DEVICE, class_names, label="mobilenet_plain")

# === Step 2: Balanced Dataset + Augmentation
print("\n🔹 Step 2: Balanced Dataset with Augmentation")
train_loader_aug, val_loader_aug, _ = load_dataset(
    root_folder=ROOT_FOLDER, augment=True, imbalance=False, batch_size=BATCH_SIZE, im_size=IM_SIZE
)
resnet_aug = create_model("resnet18", num_classes=len(class_names)).to(DEVICE)
mobilenet_aug = create_model("mobilenet_v2", num_classes=len(class_names)).to(DEVICE)
resnet_aug_metrics = train_and_evaluate_model(resnet_aug, train_loader_aug, val_loader_aug, DEVICE, class_names, label="resnet_aug")
mobilenet_aug_metrics = train_and_evaluate_model(mobilenet_aug, train_loader_aug, val_loader_aug, DEVICE, class_names, label="mobilenet_aug")

# === Step 3: Imbalanced Dataset (Plain)
print("\n🔹 Step 3: Imbalanced Dataset (No Augmentation)")
train_loader_imb, val_loader_imb, _ = load_dataset(
    root_folder=ROOT_FOLDER, augment=False, imbalance=True, batch_size=BATCH_SIZE, im_size=IM_SIZE
)
resnet_imb = create_model("resnet18", num_classes=len(class_names)).to(DEVICE)
mobilenet_imb = create_model("mobilenet_v2", num_classes=len(class_names)).to(DEVICE)
resnet_imb_metrics = train_and_evaluate_model(resnet_imb, train_loader_imb, val_loader_imb, DEVICE, class_names, label="resnet_imb")
mobilenet_imb_metrics = train_and_evaluate_model(mobilenet_imb, train_loader_imb, val_loader_imb, DEVICE, class_names, label="mobilenet_imb")

# === Step 4: Imbalanced Dataset + Targeted Augmentation
print("\n🔹 Step 4: Imbalanced Dataset with Augmentation")
train_loader_imb_aug, val_loader_imb_aug, _ = load_dataset(
    root_folder=ROOT_FOLDER, augment=True, imbalance=True, batch_size=BATCH_SIZE, im_size=IM_SIZE
)
resnet_imb_aug = create_model("resnet18", num_classes=len(class_names)).to(DEVICE)
mobilenet_imb_aug = create_model("mobilenet_v2", num_classes=len(class_names)).to(DEVICE)
resnet_imb_aug_metrics = train_and_evaluate_model(resnet_imb_aug, train_loader_imb_aug, val_loader_imb_aug, DEVICE, class_names, label="resnet_imb_aug")
mobilenet_imb_aug_metrics = train_and_evaluate_model(mobilenet_imb_aug, train_loader_imb_aug, val_loader_imb_aug, DEVICE, class_names, label="mobilenet_imb_aug")

# === Step 5: Occlusion Evaluation
print("\n🔹 Step 5: Occlusion Evaluation")
test_paths, test_labels, _, _, _ = load_data_path_only(ROOT_FOLDER, imbalance=False)
test_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
occlusion_test_evaluation(resnet_plain, test_paths, test_labels, test_transform, DEVICE, 60, "OcclusionTest_ResNet_Plain", "resnet_plain", class_names, all_results)
occlusion_test_evaluation(mobilenet_plain, test_paths, test_labels, test_transform, DEVICE, 60, "OcclusionTest_MobileNet_Plain", "mobilenet_plain", class_names, all_results)
occlusion_test_evaluation(resnet_aug, test_paths, test_labels, test_transform, DEVICE, 60, "OcclusionTest_ResNet_Aug", "resnet_aug", class_names, all_results)
occlusion_test_evaluation(mobilenet_aug, test_paths, test_labels, test_transform, DEVICE, 60, "OcclusionTest_MobileNet_Aug", "mobilenet_aug", class_names, all_results)

# === 保存结果
df_final = pd.DataFrame(all_results)
df_final.to_csv("final_dl_comparison.csv", index=False)
print("✅ All deep learning experiments completed and saved.")
