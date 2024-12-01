import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.05)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(0.05)
        )

        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(12, 8, 1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 12, 3),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(0.05)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05)
        )

        self.transition2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 20, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout2d(0.05)
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(3)
        )

        self.fc = nn.Linear(20, 10)


    def forward(self, x):
        x = self.conv1(x)               #in: 28x28 out: 28x28
        x = self.conv2(x)               #in: 28x28 out: 28x28
        x = self.transition1(x)          #in: 28x28 out: 14x14
        x = self.conv3(x)               #in: 14x14 out: 12x12
        x = self.conv4(x)               #in: 12x12 out: 10x10
        x = self.transition2(x)          #in: 10x10 out: 5x5
        x = self.conv5(x)               #in: 5x5 out: 3x3
        x = self.gap(x)
        x = x.view(x.size(0), 20)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
