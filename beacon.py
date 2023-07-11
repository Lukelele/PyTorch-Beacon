import torch
import numpy as np


def fit(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimiser=torch.optim.Adam, loss_func=torch.nn.MSELoss(), epochs=100, lr=0.01, device="cpu"):
    model.to(device)
    optimiser = optimiser(model.parameters(), lr=lr)

    losses = np.zeros(epochs)

    for epoch in range(epochs):
        model.train()

        batch_loss = 0
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            optimiser.zero_grad()
            loss = loss_func(y_pred, y)
            batch_loss += loss
            loss.backward()
            optimiser.step()

        epoch_loss = batch_loss / len(dataloader)
        losses[epoch] = epoch_loss.item()

    return losses


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_func=torch.nn.MSELoss(), device="cpu"):
    model.to(device)
    model.eval()
    loss = 0

    with torch.inference_mode():
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss += loss_func(y_pred, y)

        loss /= len(dataloader)

    return loss.item()
