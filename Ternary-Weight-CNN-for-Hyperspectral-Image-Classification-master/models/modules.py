import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fp_weights: torch.Tensor, delta: float) -> torch.Tensor:
        ternary_weights = torch.where(fp_weights > delta, 1.0, torch.where(fp_weights < -delta, -1.0, 0.0))
        return ternary_weights

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad_input = grad_output
        return grad_input, None


class TernaryWeightLayer(nn.Module):
    def __init__(self, num_channels: int, delta: float = 0.5) -> None:
        super().__init__()
        self.delta = delta
        self.num_channels = num_channels
        self.full_precision_weights = nn.Parameter(
            torch.empty(num_channels, 1, 1, 1).uniform_(0, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ternary_weights = TernaryActivation.apply(self.full_precision_weights, self.delta)
        x = F.conv2d(x, ternary_weights, groups=self.num_channels)
        tw_sum = ternary_weights.abs().sum()
        return x, tw_sum


class DenseClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Inception(nn.Module):    
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        c1 = c3 = c5 = 16
        self.conv1x1 = nn.ModuleList(
            [nn.Conv2d(in_channels, c1, kernel_size=1, padding='same') for _ in range(4)]
        )
        self.conv3x3 = nn.Conv2d(c1, c3, kernel_size=3, padding='same')
        self.conv5x5 = nn.Conv2d(c5, c5, kernel_size=5, padding='same')
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(c) for c in [c1, c1, c3, c5, c1, c1]]
        )
        self.pool = nn.MaxPool2d(3, 1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.bn[0](self.conv1x1[0](x)))
        x2 = F.relu(self.bn[1](self.conv3x3(
                F.relu(self.bn[4](self.conv1x1[1](x)))
            )))
        x3 = F.relu(self.bn[2](self.conv5x5(
                F.relu(self.bn[5](self.conv1x1[2](x)))
            )))
        x4 = F.relu(self.bn[3](self.conv1x1[3](
                self.pool(x)
            )))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x
