"""
Malaria Cell Classification With PyTorch

AltayBioAI Project

A Lightweight CNN Model For Malaria Cell Classification.

Dataset:
Cell Images For Detecting Malaria

Repository:
AltayBioAI
"""

import os
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# Device

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Model

class MalariaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        return x

def main():
    print("Downloading Dataset...")
    path = kagglehub.dataset_download(
        "iarunava/cell-images-for-detecting-malaria"
    )

    dataset_path = os.path.join(
        path,
        "cell_images"
    )

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(
        dataset_path,
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32
    )

    model = MalariaCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 2

    print("Training Started...")

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
            optimizer.zero_grad()
            
        epoch_loss = (
            running_loss /
            len(train_loader)
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {epoch_loss:.4f}"
        )

    print("Evaluating Model...")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
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
        f"Validation Accuracy: "
        f"{accuracy:.2f}%"
    )

    torch.save(
        model.state_dict(),
        "malaria_cnn.pth"
    )

    print(
        "Model Saved As "
        "malaria_cnn.pth"
    )

if __name__ == "__main__":
    main()
