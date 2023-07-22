import torch
from . import metrics


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def compile(self, optimiser: torch.optim.Optimizer = torch.optim.Adam, learning_rate=0.1, loss_function=torch.nn.CrossEntropyLoss, accuracy_function=metrics.categorical_accuracy):
        self.loss_function = loss_function()
        self.optimiser = optimiser(self.parameters(), learning_rate)
        self.learning_rate = learning_rate
        
        self.layers = torch.compile(self.layers)


class Sequential(Module):
    def __init__(self, *layers: torch.nn.Module, optimiser: torch.optim.Optimizer = torch.optim.Adam, learning_rate=0.1, loss_function=torch.nn.CrossEntropyLoss, accuracy_function=metrics.categorical_accuracy):
        if len(layers) > 0:
            self.layers = torch.nn.Sequential(*layers)
        else:
            self.layers = None
            
        self.compile(optimiser, learning_rate, loss_function, accuracy_function)
    