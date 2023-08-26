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
    It appears primarily in its original form, barring a small modification in the constructor to initialise self.ps to
    None, as well as generalisations to handle NN architectures with multiple outputs, and those requiring
    logarithmic output. Generalisations have also been made to consider a scalar variable p multiplied by a vector of
    ones.

    This acts as a wrapper for any model to perform invex regularisation (https://arxiv.org/abs/2111.11027v1)
    """
    def __init__(self, module, lamda=0.0, p_ones=False, multi_output=False, log_out=False):
        super().__init__()
        self.module = module
        self.lamda = lamda
        self.batch_idx = 0
        self.ps = None
        self.p_ones = p_ones
        self.multi_output = multi_output
        self.log_out = log_out

    def init_ps(self, train_dataloader):
        if self.lamda != 0.0:
            self.module.eval()
            ps = []
            for inputs, targets in iter(train_dataloader):
                if self.p_ones:
                    p = torch.tensor([0.0])
                else:
                    outputs = self.module(inputs)
                    p = torch.zeros_like(outputs[0] if self.multi_output else outputs)
                ps.append(torch.nn.Parameter(p, requires_grad=True))
            self.ps = torch.nn.ParameterList(ps)
            self.module.train()

    def set_batch_idx(self, batch_idx):
        self.batch_idx = batch_idx

    def forward(self, x):
        x = self.module(x)
        if self.lamda != 0.0 and self.training:
            if self.multi_output:
                updated_x = list(x)
                updated_x[0] = updated_x[0] + self.lamda * torch.mul(self.ps[self.batch_idx], torch.ones_like(x[0]))
                x = tuple(updated_x)
            else:
                x = x + self.lamda * torch.mul(self.ps[self.batch_idx], torch.ones_like(x))
        if self.log_out:
            x = torch.log(x)
        return x


# Used for zero-initialisation
def _init_weights_zero(module):
    if isinstance(module, nn.Linear):
        module.weight.detach().zero_()
        module.bias.detach().zero_()


class LinearLeastSquares(nn.Module):
    """
    Implements linear least squares regression
    """
    def __init__(self, input_dim):
        super().__init__()
        self.x = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

    def forward(self, A):
        return torch.mv(A, self.x)


class BinaryClassifier(nn.Module):
    """
    Implements a binary classifier
    """
    def __init__(self, input_dim):
        super().__init__()
        self.x = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

    def forward(self, A):
        return torch.sigmoid(torch.mv(A, self.x))


class MultinomialLogisticRegression(nn.Module):
    """
    Implements multinomial logistic regression
    """
    def __init__(self, input_dim, num_classes):
        """
        Constructor

        :param input_dim: input dimension of data (e.g. 28 x 28 image has input_dim = 28)
        :param num_classes: number of classes to consider
        """
        super().__init__()
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(input_dim * input_dim, self.num_classes)
        self.apply(_init_weights_zero)
        self.log_softmax = nn.LogSoftmax(dim=1)  # if not invex else nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.hidden(x)
        return self.log_softmax(logits) if self.num_classes > 1 else self.sigmoid(logits)


class VAE(nn.Module):
    """
    Implements a Variational Autoencoder
    """

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


class ConvBlock(nn.Module):
    """
    Convolutional Block as part of ResNet50.
    """
    def __init__(self, input_channels, output_channels, kernel, stride, padding, compare_batch_norm):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel,
                              stride=stride, padding=padding)
        self.compare_batch_norm = compare_batch_norm
        if self.compare_batch_norm:
            self.bn = nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x):
        output = self.conv(x)
        if self.compare_batch_norm:
            output = self.bn(output)
        return output


class ResidualBlock(nn.Module):
    """
    Residual Block as part of ResNet50.
    """
    def __init__(self, input_channels, output_channels, compare_batch_norm, first_layer=False):
        super().__init__()
        residual_channels = input_channels // 4
        stride = 1
        self.is_projection = input_channels != output_channels

        if self.is_projection or first_layer:
            stride += self.is_projection and not first_layer
            self.proj = ConvBlock(input_channels, output_channels, 1, stride, 0, compare_batch_norm)
            residual_channels = input_channels // stride

        self.conv1 = ConvBlock(input_channels, residual_channels, 1, 1, 0, compare_batch_norm)
        self.conv2 = ConvBlock(residual_channels, residual_channels, 3, stride, 1, compare_batch_norm)
        self.conv3 = ConvBlock(residual_channels, output_channels, 1, 1, 0, compare_batch_norm)
        self.relu = nn.ReLU()

    def forward(self, x):
        activation = self.relu(self.conv1(x))
        activation = self.relu(self.conv2(activation))
        activation = self.conv3(activation)

        if self.is_projection:
            x = self.proj(x)
        return self.relu(torch.add(activation, x))


class ResNet50(nn.Module):
    """
    Implements ResNet50.
    """
    def __init__(self, compare_batch_norm, compare_dropout, dropout_param=0.5, input_channels=3, num_classes=1000):
        super().__init__()
        num_blocks = [3, 4, 6, 3]
        output_features = [256, 512, 1024, 2048]
        self.res_blocks = nn.ModuleList([ResidualBlock(64, 256, compare_batch_norm, True)])

        for i in range(len(output_features)):
            if i > 0:
                self.res_blocks.append(ResidualBlock(output_features[i - 1], output_features[i], compare_batch_norm))
            for _ in range(num_blocks[i] - 1):
                self.res_blocks.append(ResidualBlock(output_features[i], output_features[i], compare_batch_norm))

        self.conv1 = ConvBlock(input_channels, 64, 7, 2, 3, compare_batch_norm)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.compare_dropout = compare_dropout
        if self.compare_dropout:
            self.dropout = nn.Dropout(p=dropout_param)
        self.initialise_weights()

    def initialise_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        activation = self.relu(self.conv1(x))
        activation = self.max_pool(activation)
        for res_block in self.res_blocks:
            activation = res_block(activation)
            if self.compare_dropout:
                activation = self.dropout(activation)
        pooled = self.avg_pool(activation)
        flattened = torch.flatten(pooled, 1)
        return self.linear(flattened)


class ResNet50LastLayer(nn.Module):
    """
    Implementation of regularisation techniques to modify last (hidden) layer, when training only the last (overall)
    layer of (pretrained) ResNet50.
    """
    def __init__(self, previous_layer_output_shape, compare_batch_norm, compare_dropout, dropout_param=0.5):
        super().__init__()
        self.compare_batch_norm = compare_batch_norm
        self.compare_dropout = compare_dropout

        if self.compare_batch_norm:
            self.bn = nn.BatchNorm2d(num_features=previous_layer_output_shape)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.compare_dropout:
            self.dropout = nn.Dropout(p=dropout_param)

    def forward(self, x):
        if self.compare_batch_norm:
            out = self.avgpool(self.bn(x))
        else:
            out = self.avgpool(x)
        if self.compare_dropout:
            out = self.dropout(out)
        return out
