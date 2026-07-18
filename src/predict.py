"""
predict.py
Run inference on a single image using a trained checkpoint.

Usage:
    python src/predict.py --image path/to/leaf.jpg --checkpoint models/best_model.pth
"""

import argparse

import torch
from PIL import Image

from dataset import get_transforms
from model import build_model


def predict(image_path: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_model(num_classes=len(class_names), freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = get_transforms(train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = probs.max(0)

    return class_names[top_idx.item()], top_prob.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    args = parser.parse_args()

    label, confidence = predict(args.image, args.checkpoint)
    print(f"Prediction: {label} ({confidence*100:.1f}% confidence)")
