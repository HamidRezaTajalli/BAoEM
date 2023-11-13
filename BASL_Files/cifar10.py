from typing import Tuple
import torchvision.datasets
from torchvision import transforms


def get_dataset(root='data/CIFAR10/') -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.247, 0.243, 0.261]),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform)

        return trainset, testset