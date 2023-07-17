import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


NUM_WORKERS = os.cpu_count()


def from_path(dataset_dir: str, batch_size: int = 32, transform: transforms.Compose = transforms.ToTensor(), shuffle=True, dataset_type=datasets.ImageFolder, pin_memory=True):
    dataset = dataset_type(dataset_dir, transform=transform)
    labels = dataset.classes
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    return dataloader, labels


def traintest_from_path(trainset_dir: str, testset_dir: str, batch_size: int = 32, transform: transforms.Compose = transforms.ToTensor(), dataset_type=datasets.ImageFolder, pin_memory=True):
    trainset = dataset_type(trainset_dir, transform=transform)
    testset = dataset_type(testset_dir, transform=transform)
    labels = trainset.classes

    if trainset.classes == testset.classes:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

        return trainloader, testloader, labels
    else:
        print("Error: trainset and testset classes do not match")
