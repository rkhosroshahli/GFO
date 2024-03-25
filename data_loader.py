import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import (
    DataLoader,
    Subset,
    RandomSampler,
    Sampler,
    WeightedRandomSampler,
)


class GFO_data:
    def __init__(self, dataset: str, num_samples: int, seed: int = 42):
        self.dataset = dataset
        self.num_samples = num_samples
        self.seed = seed

        self.num_classes = None
        self.transform = None
        self.train_data = None
        self.test_data = None

        if dataset == 'cifar10':
            self.num_classes = 10
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
                ]
            )
            self.train_data = torchvision.datasets.CIFAR10(
                "./data", train=True, download=True, transform=self.transform
            )
            self.test_data = torchvision.datasets.CIFAR10(
                "./data", train=False, download=True, transform=self.transform
            )
        elif dataset == 'cifar100':
            self.num_classes = 100
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
                ]
            )
            self.train_data = torchvision.datasets.CIFAR100(
                "./data", train=True, download=True, transform=self.transform
            )
            self.test_data = torchvision.datasets.CIFAR100(
                "./data", train=False, download=True, transform=self.transform
            )
        elif dataset == 'svhn':
            self.num_classes = 10
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
                ]
            )
            self.train_data = torchvision.datasets.SVHN(
                "./data", split='train', download=True, transform=self.transform
            )
            self.test_data = torchvision.datasets.SVHN(
                "./data", split='test', download=True, transform=self.transform
            )
        elif dataset == 'mnist':
            self.num_classes = 10
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            self.train_data = torchvision.datasets.MNIST(
                "./data", train=True, download=True, transform=self.transform
            )
            self.test_data = torchvision.datasets.MNIST(
                "./data", train=False, download=True, transform=self.transform
            )
        else:
            raise ValueError('Dataset not recognized')

        self.train_loader = DataLoader(self.train_data, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=128, shuffle=False)

    def generate_balanced_dataloader(self, num_samples: int = None, seed: int = None):
        rng = np.random.default_rng(seed)
        if num_samples is None:
            num_samples = self.num_samples
        samples_per_class = num_samples // self.num_classes
        if num_samples % self.num_classes > 0:
            samples_per_class += 1
        # Create an empty list to store the balanced dataset
        balanced_indices = []
        # Randomly select samples from each class for the training dataset
        for i in range(self.num_classes):
            class_indices = np.where(np.array(self.train_data.targets) == i)[0]
            selected_indices = rng.choice(class_indices, samples_per_class, replace=False)
            balanced_indices.extend(selected_indices)
        # Create a subset of the original dataset using the balanced indices
        balanced_dataset = Subset(self.train_data, balanced_indices)
        # Create DataLoaders for the balanced datasets
        data_loader = DataLoader(
            balanced_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        return data_loader


