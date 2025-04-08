import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def show_gradcam(model, img_path, device, transform=None, target_class=None):
    """
    使用 Grad-CAM 可视化模型注意力区域
    参数：
      model: 已训练的 PyTorch 模型 (ResNet18, MobileNetV2等)
      img_path: 图像路径
      device: CPU或CUDA
      transform: 可选图像预处理，与训练时一致
      target_class: 若不为空，则可视化指定类别；否则可视化模型预测的最高分类别
    """
    # 读取图像
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 如果没有传入transform, 用默认Normalize
    if transform is not None:
        input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    else:
        default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        input_tensor = default_transform(img_rgb).unsqueeze(0).to(device)

    # 根据模型选择目标卷积层(以ResNet18/MobileNetV2为例)
    if isinstance(model, torchvision.models.ResNet):
        target_layer = model.layer4[-1].conv2
    elif isinstance(model, torchvision.models.MobileNetV2):
        target_layer = model.features[-1]
    else:
        raise ValueError("Need to specify GradCAM target_layer for this model type")

    # 这里必须使用 target_layers=[...] 而非 target_layer=...
    cam = GradCAM(
        model=model,
        target_layers=[target_layer]  # 注意这里是列表
    )

    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    # 生成 Grad-CAM 热力图
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # 返回值 shape: (batch_size, height, width) -> 我们只处理 batch=1 的[0]
    grayscale_cam = grayscale_cam[0, :]

    # 叠加到原图上
    img_float = img_rgb / 255.0
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # 显示
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.subplot(1,2,2)
    plt.imshow(visualization)
    plt.title("Grad-CAM Visualization")
    plt.show()
