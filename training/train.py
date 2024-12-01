import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model import MNISTModel

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def train_and_evaluate():
    # Transformations and data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Model, loss, and optimizer
    model = MNISTModel()
    model_size = get_model_size(model)
    print(f"Model Size: {model_size:.2f} MB")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    total_correct = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()

    # Training accuracy
    train_accuracy = total_correct / len(train_loader.dataset)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Total Parameters: {total_params}")

    # Validation loop
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()

    # Validation accuracy
    val_accuracy = val_correct / len(val_loader.dataset)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Save the model for GitHub Actions testing
    torch.save(model.state_dict(), "mnist_model.pth")

    return train_accuracy, total_params

if __name__ == "__main__":
    train_and_evaluate()
