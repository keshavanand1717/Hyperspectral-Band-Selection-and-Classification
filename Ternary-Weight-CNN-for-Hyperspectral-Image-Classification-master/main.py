import os

import torch

from data import create_dataloaders
from losses.loss import TWLoss
from losses.loss_small import TWLossSmall
from models.twcnn import TWCNN
from models.twinception import TWInception
from models.twinception_small import TWInceptionSmall
from train_test import test_model, train_model

# Paths to the hyperspectral image and ground truth data
hsi_path = os.path.join('data', 'Indian_pines_corrected.mat')
gt_path = os.path.join('data', 'Indian_pines_gt.mat')
num_channels = 200
num_classes = 16

# hsi_path = os.path.join('data', 'PaviaU.mat')
# gt_path = os.path.join('data', 'PaviaU_gt.mat')
# num_channels = 103
# num_classes = 9

# No of required bands
num_bands = 10

train_dataloader, test_dataloader = create_dataloaders(hsi_path, gt_path, batch_size=16)

model = TWCNN(num_channels, num_classes)
# model = TWInception(num_channels, num_classes)
# model = TWInceptionSmall(num_channels, num_classes)
loss_fn = TWLoss(num_bands)
# loss_fn = TWLossSmall(num_bands)

train_model(model, train_dataloader, loss_fn, epochs=100)
print()
test_model(model, test_dataloader, loss_fn)
print()

# Band Selection
w = model.ternary_layer.full_precision_weights.cpu().squeeze()
delta = 0.5

selected_bands = sorted(torch.nonzero(w.abs() > delta).squeeze().tolist())
print(f"Number of required bands = {num_bands}")
print(f"Number of selected bands = {len(selected_bands)}")
print(f"Selected bands: {selected_bands}")
