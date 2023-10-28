import os
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


def from_path(dataset_dir: str, batch_size: int = 32, transform: transforms.Compose = transforms.ToTensor(), shuffle=True, dataset_type=datasets.ImageFolder, pin_memory=True):
    """
    Loads a dataset from a given directory path and returns a DataLoader object and the labels of the dataset.

    Args:
    - dataset_dir (str): The directory path of the dataset.
    - batch_size (int): The batch size for the DataLoader object. Default is 32.
    - transform (torchvision.transforms.Compose): The transformation to apply to the dataset. Default is transforms.ToTensor().
    - shuffle (bool): Whether to shuffle the dataset. Default is True.
    - dataset_type (torchvision.datasets): The type of dataset to load. Default is datasets.ImageFolder.
    - pin_memory (bool): Whether to use pinned memory for the DataLoader object. Default is True.

    Returns:
    - dataloader (torch.utils.data.DataLoader): The DataLoader object for the loaded dataset.
    - labels (list): The labels of the loaded dataset.
    """
    dataset = dataset_type(dataset_dir, transform=transform)
    labels = dataset.classes
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    return dataloader, labels


def traintest_from_path(trainset_dir: str, testset_dir: str, batch_size: int = 32, transform: transforms.Compose = transforms.ToTensor(), dataset_type=datasets.ImageFolder, pin_memory=True):
    """
    Loads train and test datasets from given directory paths and returns DataLoader objects and the labels of the datasets.

    Args:
    - train_dir (str): The directory path of the train dataset.
    - test_dir (str): The directory path of the test dataset.
    - batch_size (int): The batch size for the DataLoader objects. Default is 32.
    - transform (torchvision.transforms.Compose): The transformation to apply to the datasets. Default is transforms.ToTensor().
    - shuffle (bool): Whether to shuffle the datasets. Default is True.
    - dataset_type (torchvision.datasets): The type of dataset to load. Default is datasets.ImageFolder.
    - pin_memory (bool): Whether to use pinned memory for the DataLoader objects. Default is True.

    Returns:
    - train_dataloader (torch.utils.data.DataLoader): The DataLoader object for the loaded train dataset.
    - test_dataloader (torch.utils.data.DataLoader): The DataLoader object for the loaded test dataset.
    - labels (list): The labels of the loaded datasets.
    """
    trainset = dataset_type(trainset_dir, transform=transform)
    testset = dataset_type(testset_dir, transform=transform)
    labels = trainset.classes

    if trainset.classes == testset.classes:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

        return trainloader, testloader, labels
    else:
        print("Error: trainset and testset classes do not match")


class DatasetFromDF(Dataset):
    """
    A custom Dataset class that takes a pandas DataFrame as input.
    
    Args:
    - dataframe (pandas.DataFrame): The input DataFrame containing the data.
    - label_column (str): The name of the column containing the labels. If empty, the first column is used as the label column.
    
    Returns:
    - tuple: A tuple containing the features and labels.
    """
    def __init__(self, dataframe, label_column="", transform=None):
        self.dataframe = pd.DataFrame()
        self.transform = transform

        if label_column != dataframe.columns[0]:
            if label_column == "":
                self.dataframe["index"] = dataframe.index.values

                for col in dataframe.columns:
                    self.dataframe[col] = dataframe[col]
            else:
                self.dataframe[label_column] = dataframe[label_column]

                for col in dataframe.columns:
                    if col != label_column:
                        self.dataframe[col] = dataframe[col]
        else:
            self.dataframe = dataframe


    def __getitem__(self, index):
        row = self.dataframe.iloc[index].values

        if self.transform:
            features = self.transform(torch.tensor(row[1:], dtype=torch.float32))
        else:
            features = torch.tensor(row[1:], dtype=torch.float32)
        label = torch.tensor(row[0], dtype=torch.long)
        return features, label


    def __len__(self):
        return len(self.dataframe)


def from_df(dataframe, label_column="", batch_size: int = 32, transform: transforms.Compose = None, shuffle=True, pin_memory=True):
    """
    Loads a dataset from a given pandas DataFrame and returns a DataLoader object and the labels of the dataset.
    The dataframe must contain one column for the labels and the rest of the columns for the features.

    Args:
    - dataframe (pandas.DataFrame): The pandas DataFrame containing the dataset.
    - label_column (str): The name of the column containing the labels of the dataset. Default is an empty string.
    - batch_size (int): The batch size for the DataLoader object. Default is 32.
    - transform (torchvision.transforms.Compose): The transformation to apply to the dataset. Default is transforms.ToTensor().
    - shuffle (bool): Whether to shuffle the dataset. Default is True.
    - pin_memory (bool): Whether to use pinned memory for the DataLoader object. Default is True.

    Returns:
    - dataloader (torch.utils.data.DataLoader): The DataLoader object for the loaded dataset.
    - labels (list): The labels of the loaded dataset.
    """
    dataset = DatasetFromDF(dataframe, label_column=label_column, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    return dataloader


def traintest_from_df(train_dataframe, test_dataframe, label_column="", batch_size: int = 32, transform: transforms.Compose = None, pin_memory=True):
    """
    Loads train and test datasets from given pandas DataFrames and returns DataLoader objects and the labels of the datasets.

    Args:
    - train_dataframe (pandas.DataFrame): The pandas DataFrame containing the train dataset.
    - test_dataframe (pandas.DataFrame): The pandas DataFrame containing the test dataset.
    - label_column (str): The name of the column containing the labels of the dataset. Default is an empty string.
    - batch_size (int): The batch size for the DataLoader objects. Default is 32.
    - transform (torchvision.transforms.Compose): The transformation to apply to the datasets. Default is transforms.ToTensor().
    - pin_memory (bool): Whether to use pinned memory for the DataLoader objects. Default is True.

    Returns:
    - train_dataloader (torch.utils.data.DataLoader): The DataLoader object for the loaded train dataset.
    - test_dataloader (torch.utils.data.DataLoader): The DataLoader object for the loaded test dataset.
    - labels (list): The labels of the loaded datasets.
    """
    trainset = DatasetFromDF(train_dataframe, label_column=label_column, transform=transform)
    testset = DatasetFromDF(test_dataframe, label_column=label_column, transform=transform)


    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    return trainloader, testloader
