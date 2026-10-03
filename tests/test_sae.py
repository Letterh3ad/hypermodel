import numpy as np
import torch
from safetensors.torch import save_file

from hypermodel.sae import TopKSAE, active_latents, fvu


def _write_sparsify(path, n_lat=64, d=16, seed=0, k=4):
    g = torch.Generator().manual_seed(seed)
    W_dec = torch.nn.functional.normalize(torch.randn(n_lat, d, generator=g), dim=1)
    save_file({"W_dec": W_dec, "b_dec": torch.randn(d, generator=g) * 0.1,
               "encoder.weight": W_dec.clone(), "encoder.bias": torch.zeros(n_lat)}, str(path))
    (path.parent / "cfg.json").write_text(f'{{"activation": "topk", "k": {k}, "d_in": {d}}}')
    return W_dec


def test_encode_keeps_exactly_k_nonnegative_latents(tmp_path):
    _write_sparsify(tmp_path / "sae.safetensors")
    sae = TopKSAE.from_dir(tmp_path, device="cpu")
    acts = sae.encode(np.random.default_rng(0).standard_normal((10, 16)).astype(np.float32))
    assert acts.shape == (10, 64)
    assert ((acts > 0).sum(1) <= 4).all() and (acts >= 0).all()


def test_one_sparse_atom_per_row_is_reconstructed(tmp_path):
    W_dec = _write_sparsify(tmp_path / "sae.safetensors", k=1)
    sae = TopKSAE.from_dir(tmp_path, device="cpu")
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 64, 200)
    X = (rng.uniform(1, 3, (200, 1)) * W_dec[rows].numpy() + sae.b_dec.numpy()).astype(np.float32)
    assert fvu(X, sae) < 1e-6
    assert fvu(rng.standard_normal((200, 16)).astype(np.float32), sae) > 0.3


def test_active_latents_keeps_columns_that_fire_often_enough():
    acts = np.zeros((10, 5), np.float32)
    acts[:6, 1] = 1
    acts[:2, 3] = 1
    assert active_latents(acts, np.ones(10, bool), min_count=3).tolist() == [1]
    assert active_latents(acts, np.arange(10) < 2, min_count=2).tolist() == [1, 3]
