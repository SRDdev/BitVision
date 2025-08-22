from torchvision.transforms import Compose,RandomCrop,Resize,RandomHorizontalFlip,ToTensor,Normalize
import torchvision
import torch
from config import *

def dataset_builder(root='data', download=False, batch_size=batch_size):
    """
    Download and prepare the CIFAR-10 dataset.

    Args:
        root (str): path to save the dataset.
        download (bool): if True, download the dataset from the internet.
        batch_size (int): size of the batches.

    Returns:
        torch.utils.data.DataLoader: the dataset with specified batch size.
    """
    # Resize the input data to 224x224, since that is the training resolution used in the paper.
    # The mean and std values used to normalize CIFAR10 data are from: https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
    transform_training_data = Compose([
        RandomCrop(32, padding=4),
        Resize((224)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_data = torchvision.datasets.CIFAR10(
        root=root, train=True, download=download, transform=transform_training_data)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    return trainloader