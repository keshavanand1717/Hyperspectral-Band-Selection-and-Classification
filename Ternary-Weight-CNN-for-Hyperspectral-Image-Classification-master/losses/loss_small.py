import torch
import torch.nn as nn
import torch.nn.functional as F

class TWLossSmall(nn.Module):
    def __init__(self, num_bands: int = 10, lambda_: float = 0.01) -> None:
        super().__init__()
        self.lambda_ = lambda_
        self.num_bands = num_bands

    def forward(self, outputs: tuple[torch.Tensor, torch.Tensor], true_labels: torch.Tensor) -> torch.Tensor:
        predicted_labels, ternarized_weights_sum = outputs

        classifier_cost = F.cross_entropy(predicted_labels, true_labels)
        ternarized_weight_regularization = (ternarized_weights_sum - self.num_bands) ** 2
        
        total_cost = classifier_cost + self.lambda_ * ternarized_weight_regularization
        return total_cost
