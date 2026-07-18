# 🌿 Plant Disease Classifier

A transfer-learning image classifier that identifies plant diseases from leaf
photographs. Given a photo of a leaf, it predicts whether the plant is healthy
or which disease it's showing signs of — useful for early detection in home
gardening or small-scale farming.

## Why this project

Manual disease diagnosis in crops requires expert knowledge that's not always
accessible. This project shows an end-to-end pipeline for turning a folder of
labeled images into a usable diagnostic tool: data loading, transfer learning,
evaluation, and a deployable demo.

## Overview

- **Model:** ResNet18 pretrained on ImageNet, fine-tuned on plant leaf images
- **Approach:** Freeze the backbone and train a new classifier head first,
  then unfreeze and fine-tune the whole network at a lower learning rate
- **Framework:** PyTorch
- **Demo:** Gradio web app for interactive predictions

## Project structure

```
plant-disease-classifier/
├── app.py                  # Gradio web demo
├── data/raw/                # Place dataset here (see Setup)
├── models/                  # Saved checkpoints (not committed)
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
├── outputs/                 # Training history, confusion matrix
├── src/
│   ├── dataset.py           # Data loading & augmentation
│   ├── model.py              # Model architecture
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation + confusion matrix
│   └── predict.py             # Single-image inference
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/YOUR_USERNAME/plant-disease-classifier.git
   cd plant-disease-classifier
   pip install -r requirements.txt
   ```

2. Download a plant disease dataset — the
   [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
   on Kaggle works well — and unzip it so each disease class is its own
   subfolder under `data/raw/`:
   ```
   data/raw/Tomato___Healthy/
   data/raw/Tomato___Early_Blight/
   data/raw/Potato___Late_Blight/
   ...
   ```

## Usage

**Train the model:**
```bash
python src/train.py --epochs 10 --lr 0.001
```

**Evaluate on the test set:**
```bash
python src/evaluate.py --checkpoint models/best_model.pth
```
This prints a precision/recall/F1 report and saves a confusion matrix to
`outputs/confusion_matrix.png`.

**Predict on a single image:**
```bash
python src/predict.py --image path/to/leaf.jpg
```

**Launch the interactive demo:**
```bash
python app.py
```

## Results

*(Fill this in after training on your dataset — e.g. "Achieved 94% test
accuracy across 15 classes" plus a screenshot of the confusion matrix.)*

## Future improvements

- Add Grad-CAM visualizations to show which part of the leaf drove each
  prediction
- Try a larger backbone (EfficientNet, ConvNeXt) for higher accuracy
- Expand to more crop species
- Deploy the Gradio app publicly (Hugging Face Spaces)

## License

MIT — see [LICENSE](LICENSE)
