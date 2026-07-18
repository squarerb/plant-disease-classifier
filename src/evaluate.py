"""
evaluate.py
Loads a trained checkpoint and produces a confusion matrix plus
a classification report (precision/recall/F1 per class).

Usage:
    python src/evaluate.py --checkpoint models/best_model.pth
"""

import argparse

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from dataset import get_dataloaders
from model import build_model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]

    _, _, test_loader, _ = get_dataloaders(args.data_dir)

    model = build_model(num_classes=len(class_names), freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    print("Saved confusion matrix to outputs/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    args = parser.parse_args()
    main(args)
