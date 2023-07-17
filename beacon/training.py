import torch
from . import metrics
import numpy as np
from tqdm.auto import tqdm


def fit(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_type=torch.nn.MSELoss, optimiser=torch.optim.SGD, accuracy_func=metrics.categorical_accuracy, epochs=100, lr=0.01, device="cpu"):
    model.to(device)

    loss_func = loss_type()
    optimiser = optimiser(model.parameters(), lr=lr)

    losses = np.zeros(epochs)
    accuracies = np.zeros(epochs)

    for epoch in tqdm(range(epochs)):
        model.train()

        batch_loss = 0
        batch_accuracy = 0
        
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            loss = loss_func(y_pred, y)
            batch_loss += loss
            accuracy = accuracy_func(y_pred, y)
            batch_accuracy += accuracy

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
 
        epoch_loss = batch_loss / len(dataloader)
        epoch_accuracy = batch_accuracy / len(dataloader)
        losses[epoch] = epoch_loss.item()
        accuracies[epoch] = epoch_accuracy.item()

    return losses, accuracies


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_type=torch.nn.MSELoss, accuracy_func=metrics.categorical_accuracy, device="cpu"):
    model.to(device)
    model.eval()

    loss_func = loss_type()

    loss = 0
    accuracy = 0

    with torch.inference_mode():
        loss = 0
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss += loss_func(y_pred, y)
            accuracy = accuracy_func(y_pred, y)

        loss /= len(dataloader)
        accuracy /= len(dataloader)

    return loss.item(), accuracy.item()
