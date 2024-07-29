"""
a convolutional variational autoencoder on mnist, with some
light annotations

architecture inspired (read: copied) from Apple's MLX CVAE
https://github.com/ml-explore/mlx-examples/blob/main/cvae/vae.py
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class UpsamplingConv2d(nn.Module):
    """
    a polyfill for torch.nn.ConvTranspose2d, which isn't supported
    on metal backends. approximate transposed convolution using
    nearest neightbor upsampling and convolution, similar to
    stable diffusion's unet 
    https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/unet.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, /, stride, padding):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding)

    @staticmethod
    def upsample(x, scale = 2):
        B, C, H, W = x.shape
        x = torch.broadcast_to(x[:, :, :, None, :, None], (B, C, H, scale, W, scale))
        x = x.reshape(B, C, H * scale, W * scale)
        return x


    def forward(self, x):
        return self.conv(UpsamplingConv2d.upsample(x))

class Encoder(nn.Module):
    """
    convolutional encoder

    input -> normal distribution, then sample a latent vector

    some notation in comments:
        B = batch dimension
        C = input channels
        H = image height
        W = image width
    """

    def __init__(self, input_shape, latent_dim, max_filters):
        super().__init__()

        in_channels = input_shape[0] + 1
        filters = [max_filters // 4, max_filters // 2, max_filters]

        layers = []

        for filter in filters:
            """
            downsampling convolution

            our goal is to reduce the dimensionality of our input data
            for processing. we have three filter layers where we half
            the number of filters each iteration, so this looks something
            like:
                conv1: B x H x W x C -> B x H/2 x W/2 x filters[0]
                conv2: <prev output> -> B x H/4 x W/4 x filters[1]
                conv3: <prev output> -> B x H/8 x W/8 x filters[2]

            we apply batch normalization and a (leaky) relu between
            convolutions to normalize (and stabilize) our distribution
            and introduce nonlinearities to our dataset.

            convolution: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
            batchnorm: https://arxiv.org/abs/1502.03167
            """
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, filter, 3, stride=2, padding=1),
                nn.BatchNorm2d(filter),
                nn.LeakyReLU()
            ))
            in_channels = filter

        self.encoder = nn.Sequential(*layers)

        output_shape = [filters[-1]] + [
                dim // 8 for dim in input_shape[1:]
        ]

        flattened_dim = np.prod(output_shape, dtype=int)

        # linear map mean and standard deviation
        self.proj_mu = nn.Linear(flattened_dim, latent_dim)
        self.proj_logvar = nn.Linear(flattened_dim, latent_dim)

    @staticmethod
    def reparametrize(mu, logvar):
        # variance -> stdev
        sigma = torch.exp(logvar * 0.5)

        # sample normal distribution
        eps = torch.rand_like(sigma)

        return eps * sigma + mu

    def forward(self, x):
        x = self.encoder(x)

        # flatten all dims except batch
        x = torch.flatten(x, start_dim=1)

        # get mean and variance of gaussian
        mu = self.proj_mu(x)
        logvar = self.proj_logvar(x)

        # reparametrization trick
        z = Encoder.reparametrize(mu, logvar)

        return z, mu, logvar

class Decoder(nn.Module):
    """
    convolutional decoder
    """

    def __init__(self, input_shape, latent_dim, num_classes, num_filters):
        super().__init__()

        self.latent_dim = latent_dim
        in_channels = input_shape[0]
        self.num_filters = num_filters

        # note the opposite of the encoder
        filters = [num_filters, num_filters // 2, num_filters // 4]

        self.input_shape = [filters[0]] + [dim // 8 for dim in input_shape[1:]]
        flattened_dim = np.prod(self.input_shape, dtype=int)

        # latent dim -> flattened dim
        self.linear = nn.Linear(latent_dim + 384, flattened_dim)

        layers = []
        for i in range(len(filters)):
            """
            upsampling convolution

            we're essentially undoing the downsampling performed in the 
            encoder. after we've sampled a vector from our latent space,
            we throw it through this series to unreduce (?) the dimensionality
            of out output. this gives us an image with the same dimensions as
            our input. here, it looks like:
                conv1: B x H/8 x W/8 x filters[2] -> B x H/4 x H/4 x filters[1]
                conv2: <prev output> -> B x H/2 x W/2 x filters[0]
                conv3: <prev output> -> B x H x W x C

            we apply batchnorms and (leaky) relus between convolutions for the
            same reading as before. however, we now sigmoid the output of our
            last layer to normalize the pixel values.
            """
            if i < len(filters) - 1:
                layers.append(nn.Sequential(
                    UpsamplingConv2d(filters[i], filters[i+1], 3, stride=1, padding=1),
                    nn.BatchNorm2d(filters[i+1]),
                    nn.LeakyReLU()
                ))
            else:
                layers.append(nn.Sequential(
                    UpsamplingConv2d(filters[i], in_channels, 3, stride=1, padding=1),
                    nn.Sigmoid()
                ))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        # reshape to BCHW format
        x = x.view(-1, self.num_filters, self.input_shape[1], self.input_shape[2])
        return self.decoder(x)

class CVAE(nn.Module):
    """
    and finally, our convolutional variational autoencoder
    """

    def __init__(self, input_shape, latent_dim, num_classes, num_filters):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.class_embedder = nn.Linear(384, np.prod(input_shape[1:], dtype=int))
        self.input_embedder = nn.Conv2d(input_shape[0], input_shape[0], kernel_size=1)

        self.encoder = Encoder(input_shape, latent_dim, num_filters)
        self.decoder = Decoder(input_shape, latent_dim, num_classes, num_filters)

    def forward(self, x, embeddings):
        y = embeddings
        embed_class = self.class_embedder(y)\
            .view(-1, self.input_shape[1], self.input_shape[2])\
            .unsqueeze(1)
        embed_input = self.input_embedder(x)
        x = torch.cat([embed_input, embed_class], dim=1)
        z, mu, logvar = self.encoder(x)
        z = torch.cat([z, y], dim=1)
        x = self.decoder(z)
        return x, mu, logvar

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z):
        return self.decoder(z)

    def generate(self, embedding, /, device):
        y = embedding
        z = (torch.randn(1, self.latent_dim)
            .to(device))
        z = torch.cat([z, y], dim=1)
        return self.decode(z)

class CVAEEmbedder:
    def __init__(self):
        super().__init__()

        self.out_dim = 384
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, inputs, /, device):
        embeddings = self.model.encode(inputs)
        return torch.tensor(embeddings).to(device)
