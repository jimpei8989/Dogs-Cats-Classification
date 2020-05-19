import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

trainTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(22.5),
    transforms.RandomResizedCrop((64, 64), scale = (0.75, 1), ratio = (0.8, 1.25)),
    transforms.ColorJitter(brightness = 0.15, contrast = 0.15, saturation = 0.15, hue = 0.1),
    transforms.ToTensor()
])

testTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

class ImageDataset(Dataset):
    def __init__(self, X, Y = None, transform = None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.transform(self.X[idx]), self.Y[idx]
        else:
            return self.transform(self.X[idx])

