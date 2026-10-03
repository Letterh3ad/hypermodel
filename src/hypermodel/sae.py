"""TopK sparse autoencoders in EleutherAI `sparsify` layout (e.g. gzxiong/sae-qwen3-0.6b)."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TopKSAE:
    def __init__(self, W_enc, b_enc, W_dec, b_dec, k: int):
        self.W_enc, self.b_enc, self.W_dec, self.b_dec, self.k = W_enc, b_enc, W_dec, b_dec, k

    @classmethod
    def from_dir(cls, path: Path, device: str = _DEVICE) -> "TopKSAE":
        path = Path(path)
        t = {name: v.to(device, torch.float32) for name, v in load_file(str(path / "sae.safetensors")).items()}
        k = json.loads((path / "cfg.json").read_text())["k"]
        return cls(t["encoder.weight"], t["encoder.bias"], t["W_dec"], t["b_dec"], k)

    @classmethod
    def from_hub(cls, repo: str, layer: int, device: str = _DEVICE) -> "TopKSAE":
        from huggingface_hub import hf_hub_download
        for f in ("sae.safetensors", "cfg.json"):
            local = hf_hub_download(repo, f"layers.{layer}/{f}")
        return cls.from_dir(Path(local).parent, device)

    @torch.no_grad()
    def _codes(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values)

    def encode(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        dev = self.W_enc.device
        return np.concatenate([self._codes(torch.as_tensor(np.asarray(X[i:i + batch_size], np.float32),
                                                           device=dev)).cpu().numpy()
                               for i in range(0, len(X), batch_size)])

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(X, np.float32), device=self.W_enc.device)
        return (self._codes(x) @ self.W_dec + self.b_dec).cpu().numpy()


def fvu(X: np.ndarray, sae: TopKSAE) -> float:
    """Fraction of variance unexplained by the reconstruction (0 is perfect, 1 is no better than the mean)."""
    X = np.asarray(X, np.float64)
    err = X - sae.reconstruct(X)
    return float((err ** 2).sum() / ((X - X.mean(0)) ** 2).sum())


def active_latents(acts: np.ndarray, rows: np.ndarray, min_count: int = 5) -> np.ndarray:
    """Columns that fire on at least min_count of the given rows; the rest are constant for the probe."""
    return np.flatnonzero((acts[rows] > 0).sum(0) >= min_count)
