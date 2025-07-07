import torch
import torch.nn as nn
import torch.nn.functional as F


class TWLoss(nn.Module):
    def __init__(self, num_bands: int = 10, lambda1: float = 0.05, lambda2: float = 0.01) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.num_bands = num_bands

    def forward(self, outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], true_labels: torch.Tensor) -> torch.Tensor:
        predicted_labels, auxiliary_classifier_output, ternarized_weights_sum = outputs

        classifier_cost = F.cross_entropy(predicted_labels, true_labels)
        auxiliary_classifier_cost = F.cross_entropy(auxiliary_classifier_output, true_labels)
        ternarized_weight_regularization = (ternarized_weights_sum - self.num_bands) ** 2
        
        total_cost = classifier_cost + self.lambda1 * auxiliary_classifier_cost + self.lambda2 * ternarized_weight_regularization
        return total_cost
