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


class CircularBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, num_samples, max_num_call):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.data_size = len(data_source)
        self.super_batch_step = 0
        self.num_calls = max_num_call
        self.max_num_call = max_num_call

    def __iter__(self):
        # np.random.randint(0, )
        indices = torch.arange(
            self.super_batch_step * self.num_samples,
            (self.super_batch_step + 1) * self.num_samples,
        )
        # print(indices)
        super_batch = torch.cat([indices])

        for i in range(0, 1):
            yield super_batch[i : i + self.num_samples]

        self.num_calls -= 1
        if self.num_calls == 0:
            self.super_batch_step += 1
            self.super_batch_step %= self.data_size // self.num_samples
            self.num_calls = self.max_num_call

    def __len__(self):
        return self.data_size // self.num_samples


class FirstNBatchSampler(Sampler):
    def __init__(self, data_source, n_samples):
        self.data_source = data_source
        self.n_samples = n_samples

    def __iter__(self):
        return iter(range(self.n_samples))

    def __len__(self):
        return self.n_samples


def load_mnist_train_fixed_selection(num_samples=100, seed=42, batch_size=64):
    # Set a random seed for reproducibility
    # seed = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the original MNIST dataset
    mnist_train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )

    # Create an empty list to store the balanced dataset
    balanced_train_dataset = []
    samples_per_class = num_samples // 10
    # Randomly select samples from each class for the training dataset
    for i in range(10):
        class_indices = np.where(np.array(mnist_train.targets) == i)[0]
        selected_indices = rng.choice(class_indices, samples_per_class, replace=False)
        balanced_train_dataset.extend(Subset(mnist_train, selected_indices))

    # Check the sizes of the resulting datasets
    print("Size of balanced training dataset:", len(balanced_train_dataset))

    # Create DataLoaders for the balanced datasets
    train_loader = DataLoader(
        balanced_train_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader


def load_mnist_train_each_step(num_samples=100, seed=42, batch_size=64, max_num_call=1):
    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the original MNIST dataset
    train_set = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )

    custom_sampler = CircularBatchSampler(
        train_set, batch_size=batch_size, max_num_call=max_num_call
    )

    # Create DataLoaders for the balanced datasets
    train_loader = DataLoader(train_set, batch_sampler=custom_sampler)
    return train_loader


def load_mnist_train_selection(num_samples=100, seed=42, batch_size=64):
    # Set a random seed for reproducibility
    # seed = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the original MNIST dataset
    mnist_train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )

    # Create DataLoaders for the balanced datasets
    train_loader = DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=False,
        sampler=RandomSampler(mnist_train, replacement=False, num_samples=num_samples),
        num_workers=0,
        pin_memory=False,
    )
    return train_loader


def load_mnist_train_full(batch_size=64):
    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )

    print("Size of balanced training dataset:", len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def load_mnist_test(batch_size=64):
    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    print("Size of balanced test dataset:", len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def load_cifar10_train_fixed_selection(num_samples=100, seed=42, batch_size=64):
    # Set a random seed for reproducibility
    # seed = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load the original MNIST dataset
    train_dataset = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform
    )

    # Create an empty list to store the balanced dataset
    # balanced_train_dataset = []
    # samples_per_class = num_samples // 10
    # # Randomly select samples from each class for the training dataset
    # for i in range(10):
    #     class_indices = np.where(np.array(mnist_train.targets) == i)[0]
    #     selected_indices = rng.choice(class_indices, samples_per_class, replace=False)
    #     balanced_train_dataset.extend(selected_indices)

    # Check the sizes of the resulting datasets
    # print("Size of balanced training dataset:", len(balanced_train_dataset))
    # balanced_subset = Subset(mnist_train, balanced_train_dataset)
    # class_sample_count = np.array(
    #     [
    #         len(np.where(mnist_train.targets == t)[0])
    #         for t in np.unique(mnist_train.targets)
    #     ]
    # )
    # weight = 1.0 / class_sample_count
    # samples_weight = np.array([weight[t] for t in mnist_train.targets])

    # samples_weight = torch.from_numpy(samples_weight)
    # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    # Split the dataset into classes
    # classwise_indices = [[] for _ in range(10)]
    # for idx, (_, label) in enumerate(train_dataset):
    #     classwise_indices[label].append(idx)

    # # Randomly sample from each class to create a balanced dataset
    # balanced_indices = []
    # num_samples_per_class = 1000  # We want 1000 samples per class
    # for indices in classwise_indices:
    #     balanced_indices.extend(torch.randperm(len(indices))[:num_samples_per_class])

    # # Create a subset of the original dataset using the balanced indices
    # balanced_dataset = Subset(train_dataset, balanced_indices)
    custom_sampler = FirstNBatchSampler(train_dataset, n_samples=num_samples)
    # Create DataLoaders for the balanced datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=custom_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader


def load_cifar10_train_each_step(
    num_samples=100, seed=42, batch_size=64, max_num_call=1
):
    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load the original MNIST dataset
    train_set = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform
    )

    custom_sampler = CircularBatchSampler(
        train_set,
        batch_size=batch_size,
        num_samples=num_samples,
        max_num_call=max_num_call,
    )

    # Create DataLoaders for the balanced datasets
    train_loader = DataLoader(train_set, batch_sampler=custom_sampler)
    return train_loader


def load_cifar10_train_selection(num_samples=100, seed=42, batch_size=64, **args):
    # Set a random seed for reproducibility
    # seed = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load the original MNIST dataset
    train_set = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform
    )

    custom_sampler = FirstNBatchSampler(train_set, n_samples=num_samples)
    # Create DataLoaders for the balanced datasets
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=custom_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader

    # subset = Subset(train_set, torch.arange(0, num_samples))
    # # Create DataLoaders for the balanced datasets
    # train_loader = DataLoader(
    #     subset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     # sampler=RandomSampler(mnist_train, replacement=True, num_samples=num_samples),
    #     num_workers=0,
    #     pin_memory=False,
    # )
    # return train_loader


def load_cifar10_train_full(batch_size=64):
    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform
    )

    # print("Size of balanced training dataset:", len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def load_cifar10_test(batch_size=64):
    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=transform, download=True
    )

    # print("Size of balanced test dataset:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return test_loader


def load_cifar100_train_fixed_selection(num_samples=100, seed=42, batch_size=64):
    # Set a random seed for reproducibility
    # seed = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load the original MNIST dataset
    mnist_train = torchvision.datasets.CIFAR100(
        "./data", train=True, download=True, transform=transform
    )

    # Create an empty list to store the balanced dataset
    balanced_train_dataset = []
    samples_per_class = num_samples // 100
    # Randomly select samples from each class for the training dataset
    for i in range(100):
        class_indices = np.where(np.array(mnist_train.targets) == i)[0]
        selected_indices = rng.choice(class_indices, samples_per_class, replace=False)
        balanced_train_dataset.extend(Subset(mnist_train, selected_indices))

    # Check the sizes of the resulting datasets
    # print("Size of balanced training dataset:", len(balanced_train_dataset))

    # Create DataLoaders for the balanced datasets
    train_loader = DataLoader(
        balanced_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader


def load_cifar100_train_each_step(
    num_samples=100, seed=42, batch_size=64, max_num_call=1
):
    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load the original MNIST dataset
    train_set = torchvision.datasets.CIFAR100(
        "./data", train=True, download=True, transform=transform
    )

    custom_sampler = CircularBatchSampler(
        train_set,
        batch_size=batch_size,
        num_samples=num_samples,
        max_num_call=max_num_call,
    )

    # Create DataLoaders for the balanced datasets
    train_loader = DataLoader(train_set, batch_sampler=custom_sampler)
    return train_loader


def load_cifar100_train_selection(num_samples=100, seed=42, batch_size=64, **args):
    # Set a random seed for reproducibility
    # seed = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load the original MNIST dataset
    train_dataset = torchvision.datasets.CIFAR100(
        "./data", train=True, download=True, transform=transform
    )

    custom_sampler = FirstNBatchSampler(train_dataset, n_samples=num_samples)
    # Create DataLoaders for the balanced datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=custom_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader

    # Create DataLoaders for the balanced datasets
    # train_loader = DataLoader(
    #     mnist_train,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     sampler=RandomSampler(mnist_train, replacement=True, num_samples=num_samples),
    #     num_workers=0,
    #     pin_memory=False,
    # )
    # return train_loader


def load_cifar100_train_full(batch_size=64):
    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR100(
        "./data", train=True, download=True, transform=transform
    )

    # print("Size of balanced training dataset:", len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def load_cifar100_test(batch_size=64):
    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, transform=transform, download=True
    )

    # print("Size of balanced test dataset:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )
    return test_loader


def data_loader(dataset, batch_size, sample_size, max_num_call=100, seed=None):
    num_classes = None
    sample_train_loader = None
    full_train_loader = None
    test_loader = None
    if dataset == "mnist":
        num_classes = 10
        data_shape = (28, 28, 1)
        full_train_loader = load_mnist_train_full(batch_size=batch_size)
        sample_train_loader = load_mnist_train_each_step(
            num_samples=sample_size,
            seed=seed,
            batch_size=batch_size,
            max_num_call=max_num_call,
        )
        test_loader = load_mnist_test(batch_size=batch_size)
    elif dataset == "cifar10":
        num_classes = 10
        data_shape = (32, 32, 3)
        full_train_loader = load_cifar10_train_full(batch_size=batch_size)
        sample_train_loader = load_cifar10_train_selection(
            num_samples=sample_size,
            seed=seed,
            batch_size=batch_size,
            max_num_call=max_num_call,
        )
        test_loader = load_cifar10_test(batch_size=batch_size)
    elif dataset == "cifar100":
        num_classes = 100
        data_shape = (32, 32, 3)
        full_train_loader = load_cifar100_train_full(batch_size=batch_size)
        sample_train_loader = load_cifar100_train_selection(
            num_samples=sample_size,
            seed=seed,
            batch_size=batch_size,
            max_num_call=max_num_call,
        )
        test_loader = load_cifar100_test(batch_size=batch_size)

    return sample_train_loader, full_train_loader, test_loader, num_classes


def data_random_each_step_sampler(
    dataset, batch_size, sample_size=0, max_num_call=100, seed=None
):
    sample_train_loader = None
    if dataset == "mnist":

        sample_train_loader = load_mnist_train_each_step(
            num_samples=sample_size,
            seed=seed,
            batch_size=batch_size,
            max_num_call=max_num_call,
        )

    elif dataset == "cifar10":

        sample_train_loader = load_cifar10_train_each_step(
            num_samples=sample_size,
            seed=seed,
            batch_size=batch_size,
            max_num_call=max_num_call,
        )

    elif dataset == "cifar100":

        sample_train_loader = load_cifar100_train_each_step(
            num_samples=sample_size,
            seed=seed,
            batch_size=batch_size,
            max_num_call=max_num_call,
        )

    return sample_train_loader


def data_fixed_sampler(dataset, batch_size, sample_size, seed):
    sample_train_loader = None
    if dataset == "mnist":

        sample_train_loader = load_mnist_train_fixed_selection(
            num_samples=sample_size,
            seed=seed,
            batch_size=batch_size,
        )

    elif dataset == "cifar10":

        sample_train_loader = load_cifar10_train_fixed_selection(
            num_samples=sample_size,
            seed=seed,
            batch_size=batch_size,
        )

    elif dataset == "cifar100":

        sample_train_loader = load_cifar100_train_fixed_selection(
            num_samples=sample_size,
            seed=seed,
            batch_size=batch_size,
        )

    return sample_train_loader
