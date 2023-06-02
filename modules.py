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

    def __init__(self, input_dim, num_input_channels, latent_dim=200, num_conv=4):
        super().__init__()
        self.input_dim = input_dim
        self.num_input_channels = num_input_channels
        self.latent_dim = latent_dim
        self.num_conv = num_conv

        # Encoder doubles the number of convolutional filters each time, whilst also halving the image dimensions.
        # Starts (arbitrarily) with the initial number of output filters equal to the input dimension
        self.conv_channel_nums = [self.num_input_channels] + [self.input_dim * 2 ** i for i in range(self.num_conv)]
        encoder_layer_pairs = [(nn.Conv2d(self.conv_channel_nums[i], self.conv_channel_nums[i + 1], 3, 2, 1),
                                nn.LeakyReLU()) for i in range(self.num_conv)]
        self.encoder = nn.Sequential(*[layer for pair in encoder_layer_pairs for layer in pair])

        # After last layer, as each layer halves the image dimensions, the total number of neurons in the last layer is
        # the number of convolutional filters in the last layer, multiplied by the dimensions of the image of the
        # output layer (think 'volume of prism')
        encoder_output_shape = (self.conv_channel_nums[-1]) * ((self.input_dim // (2 ** self.num_conv)) ** 2)

        self.mu = nn.Linear(encoder_output_shape, self.latent_dim)
        self.logvar = nn.Linear(encoder_output_shape, self.latent_dim)

        # Project back from latent space to (flattened) encoder output space
        self.projection = nn.Linear(self.latent_dim, encoder_output_shape)

        decoder_layer_pairs = [(
            nn.ConvTranspose2d(self.conv_channel_nums[i], self.conv_channel_nums[i - 1], 3, 2, 1, 1), nn.LeakyReLU())
            for i in range(self.num_conv, 0, -1)]
        self.decoder = nn.Sequential(*[layer for pair in decoder_layer_pairs for layer in pair])
        self.decoder.append(nn.Sigmoid())  # To scale decoded pixels to [0, 1]

    def encode(self, x):
        # encode
        x = self.encoder(x)
        return torch.flatten(x, start_dim=1)

    def reparameterize(self, x):
        mu, logvar = self.mu(x), self.logvar(x)
        std_dev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std_dev)
        return mu + epsilon * std_dev, mu, logvar

    def decode(self, z):
        # Unflatten input
        z_unflattened = z.view(-1, self.conv_channel_nums[-1], (self.input_dim // (2 ** self.num_conv)),
                               (self.input_dim // (2 ** self.num_conv)))
        return self.decoder(z_unflattened)

    def forward(self, x):
        x = self.encode(x)
        z, mu, logvar = self.reparameterize(x)
        z = self.projection(z)
        return self.decode(z), mu, logvar

    def sample(self, num_samples, device='cuda'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(self.projection(z))
