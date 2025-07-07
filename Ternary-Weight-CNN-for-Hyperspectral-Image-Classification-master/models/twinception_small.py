import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import DenseClassifier, Inception, TernaryWeightLayer


class FeatureExtractionCNN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.inception = Inception(in_channels)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inception(x)
        
        z = self.bn2(F.relu(self.conv2(z)))
        z = self.bn3(F.relu(self.conv3(z)))
        z = self.pool(z)
        
        z = self.bn4(F.relu(self.conv4(z)))
        z = self.pool(z)
        
        features = z.reshape(z.size(0), -1)
        return features


class TWInceptionSmall(nn.Module):
    def __init__(self, num_channels: int = 200, num_classes: int = 16, delta: float = 0.5) -> None:
        super().__init__()
        self.ternary_layer = TernaryWeightLayer(num_channels, delta)
        self.feature_extractor = FeatureExtractionCNN(num_channels)
        self.classifier = DenseClassifier(256 * 1 * 1, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, tw_sum = self.ternary_layer(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x, tw_sum
