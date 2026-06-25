"""
Pneumonia X-Ray Classification Using PyTorch And ResNet18

Dataset:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

This Project Classifies Chest X-Ray Images Into:

- NORMAL
- PNEUMONIA

Model:
- ResNet18

Configuration:
- Epochs: 5
- Batch Size: 32
- Image Size: 128x128
- Optimizer: Adam
- Learning Rate: 0.0001

Training Time:
Approximately 50 Minutes On Google Colab.

Final Accuracy:
77.88%

Increasing The Number Of Epochs May Improve Accuracy.

Developed For AltayBioAI.

For Educational And Research Purposes Only.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

dataset_path = "chest_xray"

train_path = os.path.join(
    dataset_path,
    "train"
)

test_path = os.path.join(
    dataset_path,
    "test"
)

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5],
        [0.5]
    )
])

train_dataset = ImageFolder(
    train_path,
    transform=transform
)

test_dataset = ImageFolder(
    test_path,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(
            outputs,
            labels
        )
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch+1}/{epochs}"
        f" - Loss: {epoch_loss:.4f}"
    )

correct = 0
total = 0

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(
            outputs,
            1
        )
        total += labels.size(0)
        correct += (
            predicted == labels
        ).sum().item()

accuracy = 100 * correct / total

print(
    f"Accuracy: {accuracy:.2f}%"
)

torch.save(
    model.state_dict(),
    "pneumonia_resnet18.pth"
)

print(
    "Training Completed Successfully."
)

print(
    f"Final Accuracy: {accuracy:.2f}%"
)
