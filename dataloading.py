import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


NUM_WORKERS = os.cpu_count()


def from_path(dataset_dir: str, batch_size: int, transform: transforms.Compose, shuffle=True, dataset_type=datasets.ImageFolder, num_workers: int = NUM_WORKERS) -> DataLoader:
    dataset = dataset_type(dataset_dir, transforms=transform)
    labels = dataset.classes
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return dataloader, labels


def traintest_from_path(trainset_dir: str, testset_dir: str, batch_size: int, transform: transforms.Compose, dataset_type=datasets.ImageFolder, num_workers: int = NUM_WORKERS) -> tuple(DataLoader, DataLoader):
    if trainset.classes == testset.classes:
        trainset = dataset_type(trainset_dir, transforms=transform)
        testset = dataset_type(testset_dir, transforms=transform)
        labels = trainset.classes

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return trainloader, testloader, labels
    else:
        print("Error: trainset and testset classes do not match")
