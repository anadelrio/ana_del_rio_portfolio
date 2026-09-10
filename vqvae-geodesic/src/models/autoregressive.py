"""Modelo autoregresivo sobre los códigos discretos del VQ-VAE.

Cambio respecto a la versión original: el script original entrenaba el
modelo a predecir "el código de la imagen nº33 dados los códigos de las
imágenes 1-32", tratando imágenes *distintas y sin relación* del dataset
como si fueran una secuencia temporal — no tiene ningún sentido semántico
(el orden del dataset es arbitrario).

Con la nueva arquitectura del VQ-VAE, cada imagen produce una REJILLA de
códigos (7x7=49 posiciones), no un único código. Esto permite hacer lo que
de verdad se hace en el VQ-VAE original (van den Oord et al., 2017): un
prior autoregresivo que predice cada código de la rejilla a partir de los
códigos anteriores de esa MISMA imagen (orden raster), lo cual sí captura
una dependencia real — un 7 y un 1 tienen patrones espaciales de códigos
distintos y aprendibles.
"""
import torch
import torch.nn as nn


class CodeGridAR(nn.Module):
    """GRU autoregresiva sobre la secuencia raster de códigos de una imagen."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        # +1 para el token de inicio de secuencia (BOS)
        self.embed = nn.Embedding(vocab_size + 1, embed_dim)
        self.bos_token = vocab_size
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, codes: torch.Tensor):
        """
        codes: (B, L) códigos de una rejilla ya aplanada en orden raster.
        Entrena con teacher forcing: predice el token t a partir de BOS + tokens[:t].
        """
        B, L = codes.shape
        bos = torch.full((B, 1), self.bos_token, dtype=torch.long, device=codes.device)
        inputs = torch.cat([bos, codes[:, :-1]], dim=1)  # desplazado una posición
        h = self.embed(inputs)
        out, _ = self.rnn(h)
        return self.fc(out)  # (B, L, vocab_size) — logits para cada posición

    @torch.no_grad()
    def generate(self, batch_size: int, seq_len: int, device, temperature: float = 1.0):
        """Genera `batch_size` rejillas de códigos nuevas, token a token."""
        self.eval()
        generated = torch.full((batch_size, 1), self.bos_token, dtype=torch.long, device=device)
        hidden = None
        for _ in range(seq_len):
            h = self.embed(generated[:, -1:])
            out, hidden = self.rnn(h, hidden)
            logits = self.fc(out[:, -1, :]) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated[:, 1:]  # sin el BOS
