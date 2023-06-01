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

    This acts as a wrapper for any model to perform invex regularisation (https://arxiv.org/abs/2111.11027v1)
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


# Used for zero-initialisation
def _init_weights_zero(module):
    if isinstance(module, nn.Linear):
        module.weight.data.zero_()
        module.bias.data.zero_()


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
        self.apply(_init_weights_zero)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.hidden(x)
        return logits  # no need for softmax - performed by cross-entropy loss


class VAE(nn.Module):

    def __init__(self, latent_dim, ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU()
        )

        self.mu = nn.Linear(256, 1)
        self.logvar = nn.Linear(256, 1)
        self.final = nn.Linear(1, 256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(265, 128, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 1, 3, 2, 1),
            nn.Sigmoid()  # Only to scale results to [0, 1]
        )

    def encode(self, x):
        # encode
        x = self.encoder(x)
        return torch.flatten(x, start_dim=1)

    def reparameterize(self, x):
        mu, logvar = self.mu(x), self.logvar(x)
        std_dev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std_dev)
        return mu + epsilon * std_dev

    def decode(self, z):
        z = nn.Unflatten(1, (256, 1, 1))
        return self.decoder(z)

    def forward(self, x):
        x = self.encode(x)
        z = self.reparameterize(x)
        z = self.final(z)
        return self.decode(z)

    def sample(self, num_samples, device='cuda'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)

    def reconstruct(self, x):
        return self.decode(x)[0]