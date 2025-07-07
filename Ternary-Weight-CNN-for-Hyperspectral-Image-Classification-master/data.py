import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io

class HSI_Dataset(Dataset):
    def __init__(self, hsi_path: str, gt_path: str, spatial_context: int = 15) -> None:
        hsi = scipy.io.loadmat(hsi_path)
        self.hyperspectral_image = torch.tensor(
            hsi[next(key for key in hsi.keys() if not key.startswith('__'))],
            dtype=torch.float32
        )
        gt = scipy.io.loadmat(gt_path)
        self.ground_truth = torch.tensor(
            gt[next(key for key in gt.keys() if not key.startswith('__'))],
            dtype=torch.long
        )
        self.spatial_context = spatial_context

        height, width, num_channels = self.hyperspectral_image.size()
        self.hyperspectral_image = self.hyperspectral_image.permute(2, 0, 1)
        pad = self.spatial_context // 2
        self.hyperspectral_image = F.pad(self.hyperspectral_image, (pad, pad, pad, pad), mode='constant', value=0)
        
        self.indices = [(i,j) for i in range(height) for j in range(width) if self.ground_truth[i, j] != 0]

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i, j = self.indices[idx]
        sample = self.hyperspectral_image[:, i:i + self.spatial_context, j:j + self.spatial_context]
        label = self.ground_truth[i, j] - 1
        
        return sample, label

def create_dataloaders(hsi_path: str, gt_path: str, test_size: float = 0.95, batch_size: int = 1) -> tuple[DataLoader, DataLoader]:
    dataset = HSI_Dataset(hsi_path, gt_path)
    train_dataset, test_dataset = random_split(dataset, [1 - test_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader
