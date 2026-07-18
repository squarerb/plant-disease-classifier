"""
app.py
A simple Gradio web demo for the plant disease classifier.
Run locally with: python app.py
"""

import gradio as gr
import torch

from src.dataset import get_transforms
from src.model import build_model

CHECKPOINT_PATH = "models/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
class_names = checkpoint["class_names"]

model = build_model(num_classes=len(class_names), freeze_backbone=False).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = get_transforms(train=False)


def classify(image):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a leaf photo"),
    outputs=gr.Label(num_top_classes=3, label="Predicted disease"),
    title="Plant Disease Classifier",
    description="Upload a photo of a plant leaf to detect common diseases. "
                 "Trained via transfer learning on a ResNet18 backbone.",
)

if __name__ == "__main__":
    demo.launch()
