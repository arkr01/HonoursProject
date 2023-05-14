"""
    Defines all model architectures.

    Author: Adrian Rahul Kamal Rajkamal
"""
import torch
from torch import nn


class ModuleWrapper(nn.Module):
    """
    This class has been sourced from
    https://github.com/RixonC/invexifying-regularization
    It appears in its original form

    This acts as a wrapper for any logistic_model to perform invex regularisation (https://arxiv.org/abs/2111.11027v1)
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


class NNClassifier(nn.Module):
    """
    Arbitrary NN Classifier just to get things started.
    """
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


class MultinomialLogisticRegression(nn.Module):
    """
    Implements multinomial logistic regression

    :param input_dim - input dimension of data (e.g. 28 x 28 image has input_dim = 28)
    :param num_classes - number of classes to consider
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(input_dim * input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.hidden(x)
        return logits  # no need for softmax - performed by cross-entropy loss
