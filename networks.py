"""
    Defines all model architectures.

    Author: Adrian Rahul Kamal Rajkamal
"""
import torch
from torch import nn


class InvexRegulariser(nn.Module):
    """
    This class has been sourced from
    https://github.com/RixonC/invexifying-regularization
    It appears primarily in its original form, barring a new name, a small modification in the constructor to
    initialise self.ps to None, as well as generalisations to handle NN architectures with multiple outputs,
    and those requiring logarithmic output. Generalisations have also been made to consider a scalar variable p
    multiplied by a vector of ones.

    This acts as a wrapper for any model to perform invex regularisation (https://arxiv.org/abs/2111.11027v1)
    """

    def __init__(self, module, lamda=0.0, p_ones=False, multi_output=False, log_out=False, diffusion=False):
        super().__init__()
        self.module = module
        self.lamda = lamda
        self.batch_idx = 0
        self.ps = None
        self.p_ones = p_ones
        self.multi_output = multi_output
        self.log_out = log_out
        self.diffusion = diffusion

    def init_ps(self, train_dataloader):
        if self.lamda != 0.0:
            self.module.eval()
            ps = []
            for inputs, targets in iter(train_dataloader):
                if self.p_ones:
                    p = torch.tensor([0.0])
                else:
                    outputs = inputs if self.diffusion else self.module(inputs)  # diffusion model preserves shape
                    p = torch.zeros_like(outputs[0] if self.multi_output else outputs)
                ps.append(torch.nn.Parameter(p, requires_grad=True))
            self.ps = torch.nn.ParameterList(ps)
            self.module.train()

    def set_batch_idx(self, batch_idx):
        self.batch_idx = batch_idx

    def forward(self, x, *args):
        x = self.module(x, *args)
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
    Convolutional Block as part of ResNet.
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
    Residual Block as part of ResNet.
    """

    def __init__(self, input_channels, output_channels, smaller_resnet, compare_batch_norm, first_block=False):
        """
        Constructor.

        :param input_channels: number of input channels (for entire ResidualBlock)
        :param output_channels: number of output channels (for entire ResidualBlock)
        :param smaller_resnet: True if implementing ResNet18/34, False otherwise
        :param compare_batch_norm: True if activating batch normalisation within ConvBlocks, False otherwise
        :param first_block: True if this is the first ResidualBlock, False otherwise
        """
        super().__init__()
        self.smaller_resnet = smaller_resnet
        residual_channels = input_channels if smaller_resnet else input_channels // 4
        stride = 1
        self.is_projection = input_channels != output_channels

        if self.is_projection or first_block:
            stride += self.is_projection and not first_block
            self.proj = ConvBlock(input_channels, output_channels, 1, stride, 0, compare_batch_norm)
            residual_channels = input_channels * stride if smaller_resnet else input_channels // stride

        self.conv1 = ConvBlock(input_channels, residual_channels, 3 if self.smaller_resnet else 1, 1,
                               1 if self.smaller_resnet else 0, compare_batch_norm)
        self.conv2 = ConvBlock(residual_channels, residual_channels, 3, stride, 1, compare_batch_norm)
        if not smaller_resnet:
            self.conv3 = ConvBlock(residual_channels, output_channels, 1, 1, 0, compare_batch_norm)
        self.relu = nn.ReLU()

    def forward(self, x):
        activation = self.relu(self.conv1(x))

        # ReLU must be activated only *after* adding skip connection to final ConvBlock output
        activation = self.conv2(activation)
        if not self.smaller_resnet:
            activation = self.relu(activation)
            activation = self.conv3(activation)
        if self.is_projection:
            x = self.proj(x)  # Dimension matching
        return self.relu(torch.add(activation, x))


class ResNet(nn.Module):
    """
    Implements a general ResNet. Dimension constants (e.g. num_blocks, output_features, etc.) specified by original
    paper: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, variant, compare_batch_norm, compare_dropout, dropout_param=0.2, input_channels=3,
                 num_classes=1000):
        super().__init__()
        # ResNet18/34 have a shared ResidualBlock architecture, which differs from ResNet50/101/152 - must specify
        smaller_resnet = variant < 50

        # Number of ResidualBlocks before each downsample (specified by ResNet variant)
        if variant == 18:
            num_blocks = [2, 2, 2, 2]
        elif variant == 34 or variant == 50:
            num_blocks = [3, 4, 6, 3]
        elif variant == 101:
            num_blocks = [3, 4, 23, 3]
        elif variant == 152:
            num_blocks = [3, 8, 36, 3]
        else:
            raise Exception("Invalid ResNet variant")
        output_features = [64, 128, 256, 512] if smaller_resnet else [256, 512, 1024, 2048]

        # Define ResidualBlocks (and ConvBlocks within ResidualBlocks) as per Table 1 from ResNet paper (see docstring)
        self.res_blocks = nn.ModuleList([ResidualBlock(64, output_features[0], smaller_resnet, compare_batch_norm,
                                                       True)])
        for i in range(len(output_features)):
            if i > 0:
                self.res_blocks.append(ResidualBlock(output_features[i - 1], output_features[i], smaller_resnet,
                                                     compare_batch_norm))
            for _ in range(num_blocks[i] - 1):
                self.res_blocks.append(ResidualBlock(output_features[i], output_features[i], smaller_resnet,
                                                     compare_batch_norm))

        self.conv1 = ConvBlock(input_channels, 64, 7, 2, 3, compare_batch_norm)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(output_features[-1], num_classes)
        self.relu = nn.ReLU()
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

    def __init__(self, previous_layer_output_shape, compare_batch_norm, compare_dropout, dropout_param=0.2):
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


class DiffusionSetup:
    def __init__(self, input_dim, device, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.input_dim = input_dim
        self.device = device
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Standard Linear scheduler as per original DDPM paper
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_steps).to(dtype=torch.float64)
        self.beta = self.beta.to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timestep(self, n):
        return torch.randint(low=1, high=self.num_steps, size=(n,))

    def forward_process(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.input_dim, self.input_dim)).to(dtype=torch.float64)
            x = x.to(self.device)
            for i in reversed(range(1, self.num_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                pred_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                noise = torch.zeros_like(x)  # don't add noise to final iteration
                if i > 1:
                    noise = torch.randn_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + \
                    torch.sqrt(beta) * noise
        model.train()

        # Clip to correct pixel range
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


class SelfAttention(nn.Module):
    def __init__(self, num_channels, size):
        super(SelfAttention, self).__init__()
        self.num_channels = num_channels
        self.size = size
        self.mha = nn.MultiheadAttention(self.num_channels, 4, batch_first=True)
        self.layer_norm = nn.LayerNorm([self.num_channels])
        self.feed_forward = nn.Sequential(
            nn.LayerNorm([self.num_channels]),
            nn.Linear(self.num_channels, self.num_channels),
            nn.GELU(),
            nn.Linear(self.num_channels, self.num_channels),
        )

    def forward(self, x):
        x = x.view(-1, self.num_channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.layer_norm(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.feed_forward(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.num_channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, compare_dropout=False,
                 dropout_param=0.2, compare_batch_norm=False):
        super().__init__()
        self.residual = residual
        self.gelu = nn.GELU()
        if not mid_channels:
            mid_channels = out_channels
        self.compare_dropout = compare_dropout
        self.compare_batch_norm = compare_batch_norm
        if self.compare_dropout:
            self.dropout = nn.Dropout(p=dropout_param)
        if self.compare_batch_norm:
            self.batch_norm = nn.BatchNorm2d(num_features=mid_channels)
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        x = self.first_conv(x)
        if self.compare_dropout:
            x = self.dropout(x)
        if self.compare_batch_norm:
            x = self.batch_norm(x)
        if self.residual:
            return self.gelu(x + self.second_conv(x))
        else:
            return self.second_conv(x)


class DownSample(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, embedding_dim=256, compare_dropout=False,
                 dropout_param=0.2, compare_batch_norm=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(num_input_channels, num_input_channels, residual=True, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm),
            DoubleConv(num_input_channels, num_output_channels, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, num_output_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpSample(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, embedding_dim=256, compare_dropout=False,
                 dropout_param=0.2, compare_batch_norm=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(num_input_channels, num_input_channels, residual=True, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm),
            DoubleConv(num_input_channels, num_output_channels, num_input_channels // 2,
                       compare_dropout=compare_dropout, dropout_param=dropout_param,
                       compare_batch_norm=compare_batch_norm),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, num_output_channels),
        )

    def forward(self, x, skip_x, t):
        x = x.to('cpu')  # workaround for lack of deterministic implementation of Upsample
        x = self.up(x)
        x = x.to('cuda')
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, input_dim, device, num_input_channels=3, num_output_channels=3, time_dim=256,
                 compare_dropout=False, dropout_param=0.2, compare_batch_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.time_dim = time_dim
        
        self.inc = DoubleConv(self.num_input_channels, 64, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm)

        # Encoder
        self.down1 = DownSample(64, 128, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm)
        self.sa1 = SelfAttention(128, input_dim // 2)
        self.down2 = DownSample(128, 256, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm)
        self.sa2 = SelfAttention(256, input_dim // 4)
        self.down3 = DownSample(256, 256, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm)
        self.sa3 = SelfAttention(256, input_dim // 8)

        # Bottleneck
        self.bottleneck1 = DoubleConv(256, 512, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm)
        self.bottleneck2 = DoubleConv(512, 512, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm)
        self.bottleneck3 = DoubleConv(512, 256, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm)

        # Decoder
        self.up1 = UpSample(512, 128, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm)
        self.sa4 = SelfAttention(128, input_dim // 4)
        self.up2 = UpSample(256, 64, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm)
        self.sa5 = SelfAttention(64, input_dim // 2)
        self.up3 = UpSample(128, 64, compare_dropout=compare_dropout,
                       dropout_param=dropout_param, compare_batch_norm=compare_batch_norm)
        self.sa6 = SelfAttention(64, input_dim)

        # Output projection
        self.out_conv = nn.Conv2d(64, self.num_output_channels, kernel_size=1)

    def sinusoidal_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).double() / channels))
        positional_encoding_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        positional_encoding_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        positional_encoding = torch.cat([positional_encoding_a, positional_encoding_b], dim=-1)
        return positional_encoding

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float64)
        t = self.sinusoidal_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.sa1(self.down1(x1, t))
        x3 = self.sa2(self.down2(x2, t))
        x4 = self.sa3(self.down3(x3, t))

        x4 = self.bottleneck3(self.bottleneck2(self.bottleneck1(x4)))

        x = self.sa4(self.up1(x4, x3, t))
        x = self.sa5(self.up2(x, x2, t))
        x = self.sa6(self.up3(x, x1, t))

        return self.out_conv(x)
