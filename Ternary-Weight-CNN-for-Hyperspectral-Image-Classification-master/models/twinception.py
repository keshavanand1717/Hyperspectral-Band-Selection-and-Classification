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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.inception(x)
        aux_features = z.reshape(z.size(0), -1)
        z = F.relu(self.bn2(self.conv2(z)))
        z = F.relu(self.bn3(self.conv3(z)))
        z = self.pool(z)
        z = F.relu(self.bn4(self.conv4(z)))
        z = self.pool(z)
        features = z.reshape(z.size(0), -1)
        return features, aux_features


class TWInception(nn.Module):
    def __init__(self, num_channels: int = 200, num_classes: int = 16, delta: float = 0.5) -> None:
        super().__init__()
        self.ternary_layer = TernaryWeightLayer(num_channels, delta)
        self.feature_extractor = FeatureExtractionCNN(num_channels)
        self.auxiliary_classifier = DenseClassifier(64 * 15 * 15, num_classes)
        self.classifier = DenseClassifier(256 * 1 * 1, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, tw_sum = self.ternary_layer(x)
        x, x_aux = self.feature_extractor(x)
        x_aux = self.auxiliary_classifier(x_aux)
        x = self.classifier(x)
        return x, x_aux, tw_sum
