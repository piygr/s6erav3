import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.05)
        )

        self.transition = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 8, 1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.05)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05)
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(3)
        )

        self.fc = nn.Linear(16, 10)


    def forward(self, x):
        x = self.conv1(x)               #in: 28x28 out: 28x28
        x = self.conv2(x)               #in: 28x28 out: 28x28
        x = self.transition(x)          #in: 28x28 out: 14x14
        x = self.conv3(x)               #in: 14x14 out: 12x12
        x = self.conv4(x)               #in: 12x12 out: 10x10
        x = self.transition(x)          #in: 10x10 out: 5x5
        x = self.conv5(x)               #in: 5x5 out: 3x3
        x = self.gap(x)
        x = x.view(x.size(0), 16)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
