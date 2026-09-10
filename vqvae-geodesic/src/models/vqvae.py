"""VQ-VAE convolucional para MNIST, con cuantización espacial.

Cambio clave respecto a la versión original: el encoder ya no colapsa toda
la imagen en un único vector cuantizado a uno de K=64 códigos (lo que
limitaba la capacidad expresiva del modelo a 64 reconstrucciones posibles
en total). Aquí el encoder produce una rejilla espacial de vectores
latentes (7x7 posiciones), y CADA posición se cuantiza de forma independiente
contra el mismo codebook — igual que en el VQ-VAE original de van den Oord
et al. (2017). Con K=64 códigos y una rejilla 7x7, el número de
combinaciones posibles es 64^49, no 64.

También se sustituye la actualización del codebook por MSE directo (menos
estable) por una media móvil exponencial (EMA), que es el estándar actual
y evita que el codebook colapse a unos pocos vectores.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """Codebook compartido, actualizado con EMA (van den Oord et al., 2017, Appx A.1)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        embedding = torch.randn(num_embeddings, embedding_dim) * 0.1
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embedding_avg", embedding.clone())

    def forward(self, z: torch.Tensor):
        """
        z: (B, H, W, D) — vectores latentes en cada posición espacial.
        Devuelve z_q (mismo shape), commitment loss, índices (B, H, W) y perplejidad.
        """
        flat_z = z.reshape(-1, self.embedding_dim)  # (B*H*W, D)

        with torch.no_grad():
            dists = (
                flat_z.pow(2).sum(1, keepdim=True)
                + self.embedding.pow(2).sum(1)
                - 2 * flat_z @ self.embedding.t()
            )
            indices = dists.argmin(1)  # (B*H*W,)
            one_hot = F.one_hot(indices, self.num_embeddings).float()

        z_q_flat = self.embedding[indices]
        z_q = z_q_flat.view_as(z)

        if self.training:
            with torch.no_grad():
                self.cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
                embed_sum = one_hot.t() @ flat_z
                self.embedding_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
                self.embedding.copy_(self.embedding_avg / cluster_size.unsqueeze(1))

        commitment = self.beta * F.mse_loss(z_q.detach(), z)
        z_q_st = z + (z_q - z).detach()  # straight-through

        # Perplejidad: cuántos códigos del codebook se usan de verdad (K = uso uniforme ideal)
        avg_probs = one_hot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, commitment, indices.view(z.shape[:-1]), perplexity


class VQVAE(nn.Module):
    def __init__(self, embedding_dim: int = 16, num_embeddings: int = 64, beta: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Encoder: 28x28x1 -> 14x14x32 -> 7x7x(embedding_dim)  (rejilla espacial, no vector único)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),        # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, embedding_dim, kernel_size=4, stride=2, padding=1),  # 14 -> 7
        )

        self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, beta)

        # Decoder: 7x7x(embedding_dim) -> 14x14x32 -> 28x28x1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 32, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),              # 14 -> 28
        )

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)                       # (B, D, 7, 7)
        z_e = z_e.permute(0, 2, 3, 1)                # (B, 7, 7, D)  -> canal al final para el codebook
        z_q, commitment, indices, perplexity = self.quantizer(z_e)
        z_q = z_q.permute(0, 3, 1, 2)                # (B, D, 7, 7)  -> de vuelta para el decoder
        recon = torch.sigmoid(self.decoder(z_q))
        return recon, commitment, indices, perplexity


def vqvae_loss(recon_x: torch.Tensor, x: torch.Tensor, commitment: torch.Tensor):
    """Con EMA el codebook ya no necesita su propio término de pérdida (se actualiza
    directamente, no por gradiente) — la pérdida total es BCE + commitment únicamente."""
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    return bce + commitment, bce, commitment
