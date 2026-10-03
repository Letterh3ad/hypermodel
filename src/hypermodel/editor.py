"""Transient edits: a bank of LoRA experts on the residual writers, mixed per example."""

import math
from contextlib import contextmanager

import torch
from torch import nn

from hypermodel.adapter import ModelAdapter


class EditLog:
    """Squared edit outputs seen while a bank is applied, for the edit-norm penalty."""

    def __init__(self):
        self._sq: list[torch.Tensor] = []

    def add(self, delta: torch.Tensor) -> None:
        self._sq.append(delta.float().pow(2).mean())

    def norm(self) -> torch.Tensor:
        """Mean squared edit per output element, averaged over hooked writer calls."""
        return torch.stack(self._sq).mean() if self._sq else torch.zeros(())


class LoRABank(nn.Module):
    """Writer l gains sum_e g[b, l, e] * (alpha / rank) * (h @ V[l, e]) @ U[l, e]^T, h being its input."""

    def __init__(self, adapter: ModelAdapter, layers: list[int], n_experts: int = 8, rank: int = 8,
                 alpha: float = 16.0):
        super().__init__()
        self.layers = list(layers)
        self.scale = alpha / rank
        self._writers = [adapter.residual_writers()[l] for l in self.layers]
        d_in, d_out = self._writers[0].in_features, self._writers[0].out_features
        L, E = len(self.layers), n_experts
        self.V = nn.Parameter(torch.empty(L, E, d_in, rank))
        nn.init.kaiming_uniform_(self.V.view(L * E, d_in, rank), a=math.sqrt(5))
        self.U = nn.Parameter(torch.zeros(L, E, d_out, rank))  # zero, so an untrained bank is the identity
        self.to(adapter.device)

    @property
    def shape(self) -> tuple[int, int]:
        """(layers, experts): the trailing shape of a mixing tensor g."""
        return self.V.shape[0], self.V.shape[1]

    def delta(self, l: int, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Edit added to writer l's output; g is [E] or [B, E], h is [B, T, d_in]."""
        low = torch.einsum("btd,edr->bter", h, self.V[l].to(h.dtype))
        g = g.to(h.dtype).expand(h.shape[0], -1)
        return self.scale * torch.einsum("bter,be,eor->bto", low, g, self.U[l].to(h.dtype))

    @contextmanager
    def apply(self, g: torch.Tensor):
        """Hooks the writers for the duration; g is [L, E] (shared) or [B, L, E] (per example)."""
        if g.shape[-2:] != self.shape:
            raise ValueError(f"mixing shape {tuple(g.shape)} does not end in {self.shape}")

        log = EditLog()

        def hook(l):
            def fn(_m, inp, out):
                d = self.delta(l, inp[0], g[..., l, :])
                log.add(d)
                return out + d
            return fn

        handles = [w.register_forward_hook(hook(l)) for l, w in enumerate(self._writers)]
        try:
            yield log
        finally:
            for h in handles:
                h.remove()
