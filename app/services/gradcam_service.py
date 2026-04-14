import os
import uuid
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import app.services.classifier_service as classifier
from app.config import settings

_gradcam_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_cam_instance = None


def _get_cam():
    global _cam_instance
    if _cam_instance is None:
        classifier._load_model()
        target_layer = [classifier._model.layer4[-1]]
        _cam_instance = GradCAM(model=classifier._model, target_layers=target_layer)
    return _cam_instance


def generate_gradcam(cropped_image: np.ndarray, predicted_class_idx: int) -> str:
    classifier._load_model()
    cam = _get_cam()

    rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    input_tensor = _gradcam_transform(pil_image).unsqueeze(0)

    targets = [ClassifierOutputTarget(predicted_class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    original_resized = np.array(pil_image.resize((224, 224))) / 255.0
    heatmap_overlay = show_cam_on_image(original_resized, grayscale_cam, use_rgb=True)

    heatmap_filename = f"{uuid.uuid4().hex}_gradcam.png"
    heatmap_path = os.path.join(settings.UPLOAD_DIR, "gradcam", heatmap_filename)
    heatmap_bgr = cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(heatmap_path, heatmap_bgr)

    return heatmap_path