import torch
import torchvision
from torchvision.transforms import Compose, RandomCrop, Resize, RandomHorizontalFlip, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Tuple, Optional

class DatasetLoader:
    """
    A class to handle dataset loading and preprocessing for vision tasks.
    Includes functionality for subset sampling and data augmentation.
    """
    
    def __init__(self, root: str = 'data', dataset_size: Optional[float] = 0.1, batch_size: int = 32, num_workers: int = 2,image_size: int = 224):
        self.root = root
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        self.transform = self._create_transform()
        
    def _create_transform(self) -> Compose:
        """Creates the transformation pipeline."""
        return Compose([
            RandomCrop(32, padding=4),
            Resize((self.image_size, self.image_size)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=(0.4914, 0.4822, 0.4465), 
                     std=(0.2023, 0.1994, 0.2010))
        ])
    
    def _create_subset_indices(self, dataset, subset_ratio: float) -> np.ndarray:
        """
        Creates balanced subset indices for the dataset.
        
        Args:
            dataset: The full dataset
            subset_ratio: Fraction of data to use (0 to 1)
            
        Returns:
            np.ndarray: Indices for the balanced subset
        """
        labels = np.array(dataset.targets)
        classes = np.unique(labels)
        
        # Calculate samples per class
        samples_per_class = int(len(dataset) * subset_ratio / len(classes))
        
        subset_indices = []
        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            subset_indices.extend(
                np.random.choice(cls_indices, samples_per_class, replace=False)
            )
            
        return np.array(subset_indices)
    
    def load_cifar10(self) -> Tuple[DataLoader, DataLoader]:
        """
        Loads CIFAR-10 dataset with optional subset sampling.
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=True, download=True,
            transform=self.transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=False, download=True,
            transform=self.transform
        )
        
        # Create subsets if specified
        if self.dataset_size and self.dataset_size < 1.0:
            train_indices = self._create_subset_indices(train_dataset, self.dataset_size)
            test_indices = self._create_subset_indices(test_dataset, self.dataset_size)
            
            train_dataset = Subset(train_dataset, train_indices)
            test_dataset = Subset(test_dataset, test_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")
        
        return train_loader, test_loader

if __name__ == "__main__":
    dataset_loader = DatasetLoader(dataset_size=0.1, batch_size=32,num_workers=2)
    train_loader, test_loader = dataset_loader.load_cifar10()