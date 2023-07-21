import torch


class Sequential:
    def __init__(self, *layers: torch.nn.Module):
        self.layers = torch.nn.Sequential(*layers)
