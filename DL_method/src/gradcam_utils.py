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
    Visualize model attention regions using Grad-CAM.

    Args:
        model: Trained PyTorch model (e.g., ResNet18, MobileNetV2)
        img_path: Path to the input image
        device: CPU or CUDA
        transform: Optional preprocessing (should match training preprocessing)
        target_class: If specified, visualize this class. Otherwise, use model's predicted class.
    """
    # Load image
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Apply preprocessing
    if transform is not None:
        input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    else:
        default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = default_transform(img_rgb).unsqueeze(0).to(device)

    # Select target convolutional layer (e.g., ResNet18 or MobileNetV2)
    if isinstance(model, torchvision.models.ResNet):
        target_layer = model.layer4[-1].conv2
    elif isinstance(model, torchvision.models.MobileNetV2):
        target_layer = model.features[-1]
    else:
        raise ValueError("Please specify target_layer for this model type in GradCAM.")

    # Create GradCAM object (target_layers must be a list)
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Optional: specify target class
    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    # Generate Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Only process first image in batch

    # Overlay heatmap on original image
    img_float = img_rgb / 255.0
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM Visualization")
    plt.axis("off")

    plt.show()
