import numpy as np
import torch
from tqdm.auto import tqdm


class Module(torch.nn.Module):
    """
    A base class for creating neural network modules in PyTorch.
    
    Attributes:
    -----------
    loss_function: torch.nn.modules.loss._Loss
        The loss function to be used during training.
    optimiser: torch.optim.Optimizer
        The optimizer to be used during training.
    learning_rate: float
        The learning rate to be used during training.
    device: str
        The device to be used for training.
    
    Methods:
    --------
    compile(optimiser: torch.optim.Optimizer=torch.optim.Adam, learning_rate=0.1, loss_function=torch.nn.CrossEntropyLoss, device: str = "cpu", optimisations=False):
        Configures the module for training.
        
    fit(dataloader: torch.utils.data.DataLoader, epochs=10):
        Trains the module on the given data.
    """
    def __init__(self):
        super().__init__()
        
        
    def compile(self, optimiser: torch.optim.Optimizer=torch.optim.Adam, learning_rate=0.1, loss_function=torch.nn.CrossEntropyLoss, device: str = "cpu", optimisations=False):
        """
        Compiles the model with the specified optimizer, learning rate, loss function, device and optimisations.

        Args:
        - optimiser: The optimizer to use for training the model. Default is torch.optim.Adam.
        - learning_rate: The learning rate to use for training the model. Default is 0.1.
        - loss_function: The loss function to use for training the model. Default is torch.nn.CrossEntropyLoss.
        - device: The device to use for training the model. Default is "cpu".
        - optimisations: Whether to apply optimizations to the model. Default is False.

        Returns:
        - None
        """
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.device = device
        
        if optimisations == True:
            self = torch.compile(self)
            
            
    def fit(self, dataloader: torch.utils.data.DataLoader, epochs=10):
        """
        Trains the model on the specified dataloader for the specified number of epochs.

        Args:
        - dataloader: The dataloader to use for training the model.
        - epochs: The number of epochs to train the model for. Default is 10.

        Returns:
        - None
        """
        self.to(self.device)
        
        loss_function = self.loss_function()
        optimiser = self.optimiser(self.parameters(), self.learning_rate)

        losses = np.zeros(epochs)

        self.train()

        for epoch in tqdm(range(epochs)):
            batch_loss = 0

            for (x, y) in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                y_pred = self(x)

                loss = loss_function(y_pred, y)
                batch_loss += loss

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
    
            epoch_loss = batch_loss / len(dataloader)
            losses[epoch] = epoch_loss.item()

        return losses


    def evaluate(self, dataloader: torch.utils.data.DataLoader, *metrics):
        """
        Evaluates the model on the specified dataloader.

        Args:
        - dataloader: The dataloader to use for evaluating the model.

        Returns:
        """
        self.to(self.device)
        self.eval()
        
        loss_function = self.loss_function()

        loss = 0

        with torch.inference_mode():
            loss = 0
            for (x, y) in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self(x)
                loss += loss_function(y_pred, y)

            loss /= len(dataloader)

        return loss.item()

    
    def fit_tensor(self, X: torch.Tensor, y: torch.Tensor, epochs=10):
        """
        Trains the model on the specified input and target tensors for the specified number of epochs.

        Args:
        - X: The input tensor to use for training the model.
        - y: The target tensor to use for training the model.
        - epochs: The number of epochs to train the model for. Default is 10.

        Returns:
        - None
        """
        self.to(self.device)
        X, y = X.to(self.device), y.to(self.device)
        
        loss_function = self.loss_function()
        optimiser = self.optimiser(self.parameters(), self.learning_rate)

        losses = np.zeros(epochs)

        self.train()

        for epoch in tqdm(range(epochs)):
            y_pred = self(X).squeeze()
            loss = loss_function(y_pred, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            losses[epoch] = loss.item()

        return losses
    
    
    def evaluate_tensor(self, X: torch.Tensor, y: torch.Tensor):
        """
        Evaluates the model on the specified input and target tensors.

        Args:
        - X: The input tensor to use for evaluating the model.
        - y: The target tensor to use for evaluating the model.

        Returns:
        - Tuple of loss.
        """
        self.to(self.device)
        X, y = X.to(self.device), y.to(self.device)
        self.eval()
        
        loss_function = self.loss_function()

        with torch.inference_mode():
            y_pred = self(X).squeeze()
            loss = loss_function(y_pred, y)

        return loss.item()
    
    
    def predict(self, inputs: torch.Tensor):
        """
        Predicts the output of the model for the given input tensor.

        Args:
        - inputs: The input tensor for which to predict the output.

        Returns:
        - The predicted output tensor.
        """
        self.to(self.device)
        self.eval()
        
        with torch.inference_mode():
            y_pred = self(inputs.to(self.device))
            
        return y_pred
    
    
    def save(self, filepath: str):
        """
        Saves the state dictionary of the model to the specified file path.

        Args:
        - filepath: The file path to save the model state dictionary to. If the file path does not end with ".pt" or ".pth", ".pt" will be appended to the file path.

        Returns:
        - None
        """
        if filepath.endswith(".pt") or filepath.endswith(".pth"):
            torch.save(self.state_dict(), filepath)
        else:
            torch.save(self.state_dict(), filepath + ".pt")
            
    
    def load(self, filepath: str):
        """
        Loads the state dictionary of the model from the specified file path.

        Args:
        - filepath: The file path to load the model state dictionary from.

        Returns:
        - None
        """
        self.load_state_dict(torch.load(filepath))



class Sequential(Module):
    """
    A sequential container for holding a sequence of modules.
    Modules will be added to the container in the order they are passed as arguments.
    The forward method will call each module in the sequence in the order they were added.
    
    Args:
        *args: Variable length argument list of modules to be added to the container.
    """
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
    