"""
dataset.py
Handles loading, transforming, and splitting the plant leaf image dataset.

Expected folder structure (standard ImageFolder format):

data/
  raw/
    Tomato___Healthy/
      img1.jpg
      img2.jpg
    Tomato___Early_Blight/
      img1.jpg
      ...
    Potato___Late_Blight/
      ...

Download a dataset like PlantVillage (https://www.kaggle.com/datasets/emmarex/plantdisease)
and unzip it into data/raw/ following the structure above.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# ImageNet normalization stats (required since we use a pretrained backbone)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def get_dataloaders(data_dir: str = "data/raw"):
    """
    Loads the full dataset, splits into train/val/test, and returns
    DataLoaders plus the list of class names.
    """
    full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(train=True))
    class_names = full_dataset.classes

    n = len(full_dataset)
    n_val = int(n * VAL_SPLIT)
    n_test = int(n * TEST_SPLIT)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Validation/test sets shouldn't use training augmentations
    val_set.dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, class_names
