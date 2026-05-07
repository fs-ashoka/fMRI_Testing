"""TopK sparse autoencoder."""
from __future__ import annotations

import torch
from torch import nn


class TopKSparseAutoencoder(nn.Module):
    """A linear encoder/decoder SAE with exactly TopK retained latents per sample."""

    def __init__(self, input_dim: int, latent_dim: int, top_k: int) -> None:
        super().__init__()
        if top_k <= 0 or top_k > latent_dim:
            raise ValueError("top_k must be in [1, latent_dim]")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.top_k = top_k
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def topk(self, latents: torch.Tensor) -> torch.Tensor:
        """Keep only largest absolute-valued latents per sample."""
        values, indices = torch.topk(latents.abs(), k=self.top_k, dim=1)
        del values
        mask = torch.zeros_like(latents)
        mask.scatter_(1, indices, 1.0)
        return latents * mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, sparse latents, and dense pre-TopK latents."""
        dense = self.encoder(x)
        sparse = self.topk(dense)
        recon = self.decoder(sparse)
        return recon, sparse, dense
