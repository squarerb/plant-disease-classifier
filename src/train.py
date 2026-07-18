"""
train.py
Trains the plant disease classifier and saves the best checkpoint.

Usage:
    python src/train.py --epochs 10 --lr 0.001
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import build_model, unfreeze_backbone


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(args.data_dir)
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(num_classes=len(class_names), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        start = time.time()

        # Unfreeze backbone halfway through for fine-tuning
        if epoch == args.epochs // 2:
            print("Unfreezing backbone for fine-tuning...")
            model = unfreeze_backbone(model)
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }, "models/best_model.pth")
            print(f"  -> Saved new best model (val_acc={val_acc:.4f})")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal test accuracy: {test_acc:.4f}")

    with open("outputs/history.json", "w") as f:
        json.dump({**history, "test_acc": test_acc, "test_loss": test_loss}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    main(args)
