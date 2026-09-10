"""VAE convolucional para MNIST (28x28, 1 canal).

Respecto a la versión original (fully-connected, una sola capa oculta de
400 unidades): un encoder/decoder convolucional captura mucha mejor la
estructura espacial de los dígitos con un número de parámetros similar,
lo que se traduce directamente en reconstrucciones más nítidas — tanto
para el propio VAE como para la rama geodésica, que parte de estos
mismos latentes.

El latente sigue siendo un único vector por imagen (no espacial): es la
representación sobre la que opera después la cuantización geodésica
"a posteriori", así que mantenemos ese diseño y mejoramos únicamente la
capacidad del encoder/decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 28x28x1 -> 14x14x32 -> 7x7x64
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
        )
        self.enc_flat_dim = 64 * 7 * 7
        self.fc_mu = nn.Linear(self.enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_flat_dim, latent_dim)

        # Decoder: latent -> 7x7x64 -> 14x14x32 -> 28x28x1
        self.fc_dec = nn.Linear(latent_dim, self.enc_flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 14 -> 28
        )

    def encode(self, x: torch.Tensor):
        h = self.enc_conv(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        h = self.fc_dec(z)
        h = h.view(-1, 64, 7, 7)
        return torch.sigmoid(self.dec_conv(h))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    """Devuelve (total, bce, kl) para poder loguear cada término por separado."""
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl, bce, kl
