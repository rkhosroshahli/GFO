import torch
import torch.nn as nn


# Define a simple neural network class
class ANN1H1D(nn.Module):
    def __init__(
        self,
        weights=None,
        num_classes=10,
    ):
        super(ANN2H1D, self).__init__()
        self.input_shape = 28 * 28 * 1
        self.fc1 = nn.Linear(self.input_shape, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class ANN1H3D(nn.Module):
    def __init__(self, weights=None, num_classes=10):
        super(ANN2H1D, self).__init__()
        self.input_shape = 32 * 32 * 3
        self.fc1 = nn.Linear(self.input_shape, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class ANN2H1D(nn.Module):
    def __init__(self, weights=None, num_classes=10):
        super(ANN2H1D, self).__init__()
        self.input_shape = 28 * 28 * 1

        self.fc1 = nn.Linear(self.input_shape, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc(x)
        return x


class LeNet5V1(nn.Module):
    def __init__(self, weights=None, num_classes=10):
        super().__init__()
        self.feature = nn.Sequential(
            # 1
            nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2
            ),  # 28*28->32*32-->28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            # 2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feature(x))


# Define the LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self, weights=None, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.Tanh()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x)
        return x


def model_loader(arch, dataset):
    model = None
    weights = None
    if arch.lower() == "ann":
        if dataset == "mnist":
            model = ANN1H1D
        elif dataset == "cifar10" or dataset == "cifar100":
            model = ANN1H3D
        else:
            model = ANN2H1D
    elif arch.lower() == "resnet18":
        from torchvision.models import resnet18

        model = resnet18
        weights = "DEFAULT"
    elif arch.lower() == "vgg16":
        from torchvision.models import vgg16

        model = vgg16
        weights = "DEFAULT"
    elif arch.lower() == "lenet5":
        if dataset == "mnist":
            model = LeNet5V1
        else:
            model = LeNet5
    else:
        ValueError("Please enter a valid model, choose between:", ["ANN", "resnet18"])

    return model, weights
