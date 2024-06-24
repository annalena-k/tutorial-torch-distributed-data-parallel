import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import AlexNet_Weights


def load_datasets():
    """
    Load CIFAR-10 dataset.
    """
    # Load data transform
    transform_train = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load CIFAR-10 training and test datasets
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    return train_dataset, test_dataset


def load_model():
    # Load pre-trained AlexNet model and modify the final layer for CIFAR-10
    model = models.alexnet(AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, 10)
    return model