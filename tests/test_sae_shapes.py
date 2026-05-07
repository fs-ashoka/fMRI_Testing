import torch

from fmri_testing.sae.model import TopKSparseAutoencoder


def test_sae_forward_shape_and_topk_sparsity():
    model = TopKSparseAutoencoder(input_dim=10, latent_dim=7, top_k=3)
    x = torch.randn(5, 10)
    recon, sparse, dense = model(x)
    assert recon.shape == x.shape
    assert sparse.shape == (5, 7)
    assert dense.shape == (5, 7)
    assert torch.all((sparse != 0).sum(dim=1) <= 3)
