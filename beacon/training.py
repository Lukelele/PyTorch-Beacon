import torch
from . import metrics
import numpy as np
from tqdm.auto import tqdm


def fit(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimiser=torch.optim.Adam, loss_type=torch.nn.CrossEntropyLoss, epochs=100, lr=0.01, device="cpu"):
    """
    Trains a PyTorch model on a given dataset using the specified optimizer and loss function.

    Args:
    - model (torch.nn.Module): The PyTorch model to be trained.
    - dataloader (torch.utils.data.DataLoader): The data loader for the training dataset.
    - optimiser (torch.optim.Optimizer): The optimizer to be used for training. Default is Adam.
    - loss_type (torch.nn.modules.loss._Loss): The loss function to be used for training. Default is CrossEntropyLoss.
    - epochs (int): The number of epochs to train the model for. Default is 100.
    - lr (float): The learning rate for the optimizer. Default is 0.01.
    - device (str): The device to be used for training. Default is "cpu".

    Returns:
    - losses (numpy.ndarray): An array of the losses for each epoch.
    - accuracies (numpy.ndarray): An array of the accuracies for each epoch.
    """
    model.to(device)

    loss_func = loss_type()
    optimiser = optimiser(model.parameters(), lr=lr)

    losses = np.zeros(epochs)
    accuracies = np.zeros(epochs)

    model.train()

    for epoch in tqdm(range(epochs)):
        batch_loss = 0
        
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            loss = loss_func(y_pred, y)
            batch_loss += loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
 
        epoch_loss = batch_loss / len(dataloader)
        losses[epoch] = epoch_loss.item()

    return losses, accuracies


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_type=torch.nn.CrossEntropyLoss, device="cpu"):
    """
    Evaluates a PyTorch model on a given dataset using the specified loss function.

    Args:
    - model (torch.nn.Module): The PyTorch model to be evaluated.
    - dataloader (torch.utils.data.DataLoader): The data loader for the evaluation dataset.
    - loss_type (torch.nn.modules.loss._Loss): The loss function to be used for evaluation. Default is CrossEntropyLoss.
    - device (str): The device to be used for evaluation. Default is "cpu".

    Returns:
    - loss (float): The loss for the evaluation dataset.
    """
    model.to(device)
    model.eval()

    loss_func = loss_type()

    loss = 0

    with torch.inference_mode():
        loss = 0
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss += loss_func(y_pred, y)

        loss /= len(dataloader)

    return loss.item()


def predict(model: torch.nn.Module, inputs: torch.Tensor, device="cpu"):
    """
    Predicts the output of a PyTorch model on a given input tensor.

    Args:
    - model (torch.nn.Module): The PyTorch model to be used for prediction.
    - inputs (torch.Tensor): The input tensor for the prediction.
    - device (str): The device to be used for prediction. Default is "cpu".

    Returns:
    - y_pred (torch.Tensor): The predicted output tensor.
    """
    model.to(device)
    model.eval()

    with torch.inference_mode():
        y_pred = model(inputs.to(device))

    return y_pred
