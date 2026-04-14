import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from app.config import settings

# Global model variable - loaded once when the server starts
_model = None
_class_names = None
_transform = None


def _load_model():
    global _model, _class_names, _transform

    if _model is not None:
        return

    checkpoint = torch.load(settings.RESNET_MODEL_PATH, map_location=torch.device("cpu"))

    _class_names = checkpoint.get("class_names", ["fracture", "normal"])

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _model = model

    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"ResNet50 model loaded. Classes: {_class_names}")


def classify_roi(cropped_image: np.ndarray) -> dict:
    _load_model()

    rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    input_tensor = _transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        output = _model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class = _class_names[predicted_idx.item()]
    confidence_value = confidence.item()

    return {
        "classification": predicted_class,
        "confidence": confidence_value
    }