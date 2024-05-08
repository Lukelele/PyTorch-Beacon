
![beacon](https://github.com/Lukelele/PyTorch-Beacon/assets/44749665/3e1a344e-e506-40b3-8dcc-2b7a51fb00c1)

PyTorch-Beacon
--------------

A lightweight wrapper library for PyTorch with the purpose of simplifying training and testing models. Designed to mimic the standard PyTorch syntax, but replaces time consuming and repetitive tasks such as writing training and evaluating loops with one line code.

Installation
------------
```pip install pytorch-beacon```

Getting Started
---------------

We will build a simple neural network which learns to solve the XOR problem.

```
# Required imports
import torch
import beacon


# Create a randomised XOR dataset
X = torch.randint(0, 2, (1000, 2)).float()
y = torch.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).float()
```

```
# Create a simple neural network using beacon.Module
class XORNet(beacon.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 4)
        self.layer2 = torch.nn.Linear(4, 1)
        self.ReLU = torch.nn.ReLU()
        self.Sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.Sigmoid(self.layer2(self.ReLU(self.layer1(x))))


# Create an instance of the model
model = XORNet()

# Compile the model, use CPU as the device, specify device="cuda" to use GPU
model.compile(loss_function=torch.nn.BCELoss, optimiser=torch.optim.Adam, learning_rate=0.03, device="cpu")

# Train the model
model.fit_tensor(X, y, epochs=2000)
```

```
# Evaluate the model for the loss
model.evaluate_tensor(X, y)
```


```
# Predict the XOR outcomes using the trained model
model.predict(torch.tensor([1, 0]).float())
```
