"""
    Defines all NN architecture.

    Author: Adrian Rahul Kamal Rajkamal
"""
import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ModuleWrapper(nn.Module):
    """
    This class has been sourced from
    https://github.com/RixonC/invexifying-regularization
    It appears in its original form
    """
    def __init__(self, module, lamda=0.0):
        super().__init__()
        self.module = module
        self.lamda = lamda
        self.batch_idx = 0

    def init_ps(self, train_dataloader):
        if self.lamda != 0.0:
            self.module.eval()
            ps = []
            for inputs, targets in iter(train_dataloader):
                outputs = self.module(inputs)
                p = torch.zeros_like(outputs)
                ps.append(torch.nn.Parameter(p, requires_grad=True))
            self.ps = torch.nn.ParameterList(ps)
            self.module.train()

    def set_batch_idx(self, batch_idx):
        self.batch_idx = batch_idx

    def forward(self, x):
        x = self.module(x)
        if self.lamda != 0.0 and self.training:
            x = x + self.lamda * self.ps[self.batch_idx]
        return x
