import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        mnist,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader
