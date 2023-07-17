import torch
import metrics
import numpy as np


def fit(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_func=torch.nn.MSELoss(), optimiser=torch.optim.Adam, accuracy_func=metrics.categorical_accuracy, epochs=100, lr=0.01, device="cpu"):
    model.to(device)
    optimiser = optimiser(model.parameters(), lr=lr)

    losses = np.zeros(epochs)
    accuracies = np.zeros(epochs)

    for epoch in range(epochs):
        model.train()

        batch_loss = 0
        batch_accuracy = 0
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            optimiser.zero_grad()

            loss = loss_func(y_pred, y)
            batch_loss += loss
            accuracy = accuracy_func(y_pred, y)
            batch_accuracy += accuracy

            loss.backward()
            optimiser.step()
 
        epoch_loss = batch_loss / len(dataloader)
        epoch_accuracy = batch_accuracy / len(dataloader)
        losses[epoch] = epoch_loss.item()
        accuracies[epoch] = epoch_accuracy.item()

    return losses, accuracies


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_func=torch.nn.MSELoss(), accuracy_func=metrics.categorical_accuracy, device="cpu"):
    model.to(device)
    model.eval()
    loss = 0

    with torch.inference_mode():
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss += loss_func(y_pred, y)
            accuracy = accuracy_func(y_pred, y)

        loss /= len(dataloader)
        accuracy /= len(dataloader)

    return loss.item(), accuracy.item()
