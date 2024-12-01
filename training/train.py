import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model import MNISTModel




def get_scheduler(optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, threshold=0.001, threshold_mode='abs', eps=0.001, verbose=True)


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def train_and_evaluate(max_epochs=20):
    # Transformations and data loading
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-10., 10.), fill=0),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transforms, download=True)
    val_dataset = datasets.MNIST(root='./data', train=False, transform=test_transforms, download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Model, loss, and optimizer
    model = MNISTModel()
    model_size = get_model_size(model)
    print(f"Model Size: {model_size:.2f} MB")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = get_scheduler(optimizer)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    train_accuracy = 0
    maxval_accuracy = 0
    val_acc = []
    for epoch in range(max_epochs):
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
        train_accuracy = 100 * total_correct / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{max_epochs}], Training Accuracy: {train_accuracy:.2f}%")

        # Validation loop
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        # Validation accuracy
        val_accuracy = 100 * val_correct / len(val_loader.dataset)
        #val_acc.append(val_accuracy)

        scheduler.step(val_accuracy)

        print(f"Epoch [{epoch + 1}/{max_epochs}], Validation Accuracy: {val_accuracy:.2f}%")

        if maxval_accuracy < val_accuracy:
            maxval_accuracy = val_accuracy

    # Save the model for GitHub Actions testing
    torch.save(model.state_dict(), "mnist_model.pth")

    return total_params, train_accuracy, maxval_accuracy

if __name__ == "__main__":
    train_and_evaluate()
