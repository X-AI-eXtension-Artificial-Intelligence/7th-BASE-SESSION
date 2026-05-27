import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Goodfellow 2014 Generator: z ~ N(0,1) -> MLP -> image
    Input:  latent vector z (batch, latent_dim)
    Output: flattened image (batch, 784), range [-1, 1] via Tanh
    """
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 784),
            nn.Tanh(),          # output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    Goodfellow 2014 Discriminator: image -> MLP -> probability
    Input:  flattened image (batch, 784)
    Output: scalar probability in (0, 1) via Sigmoid
    Dropout(0.3) after each activation to prevent D from overpowering G.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
