import numpy as np
import torch
from tqdm.auto import tqdm
from collections import OrderedDict
from typing import Iterator
from . import metrics


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def compile(self, optimiser: torch.optim.Optimizer=torch.optim.Adam, learning_rate=0.1, loss_function=torch.nn.CrossEntropyLoss, accuracy_function=metrics.categorical_accuracy, device: str = "cpu", optimisations=False):
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.accuracy_function = accuracy_function
        self.device = device
        
        if optimisations == True:
            self = torch.compile(self)
            
            
    def fit(self, dataloader: torch.utils.data.DataLoader, epochs=10):
        self.to(self.device)
        
        loss_function = self.loss_function()
        optimiser = self.optimiser(self.parameters(), self.learning_rate)
        accuracy_function = self.accuracy_function

        losses = np.zeros(epochs)
        accuracies = np.zeros(epochs)

        self.train()

        for epoch in tqdm(range(epochs)):
            batch_loss = 0
            batch_accuracy = 0

            for (x, y) in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                y_pred = self(x)

                loss = loss_function(y_pred, y)
                batch_loss += loss
                accuracy = accuracy_function(y_pred, y)
                batch_accuracy += accuracy

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
    
            epoch_loss = batch_loss / len(dataloader)
            epoch_accuracy = batch_accuracy / len(dataloader)
            losses[epoch] = epoch_loss.item()
            accuracies[epoch] = epoch_accuracy.item()

        return (losses, accuracies)


    def evaluate(self, dataloader: torch.utils.data.DataLoader):
        self.to(self.device)
        self.eval()
        
        loss_function = self.loss_function()
        accuracy_function = self.accuracy_function

        loss = 0
        accuracy = 0

        with torch.inference_mode():
            loss = 0
            for (x, y) in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self(x)
                loss += loss_function(y_pred, y)
                accuracy += accuracy_function(y_pred, y)

            loss /= len(dataloader)
            accuracy /= len(dataloader)

        return loss.item(), accuracy.item()



class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        
        for i, module in enumerate(args):
            self.add_module(str(i), module)
        
                
    def forward(self, inputs):
        for module in self:
            inputs = module(inputs)
        return inputs
    
    
    def __iter__(self):
        return iter(self._modules.values())
    