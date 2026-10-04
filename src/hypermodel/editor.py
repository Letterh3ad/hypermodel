"""Transient edits: a bank of LoRA experts on the residual writers, mixed per example or per token."""

import math
from collections.abc import Callable
from contextlib import contextmanager

import torch
from torch import nn

from hypermodel.adapter import ModelAdapter

Mixing = torch.Tensor | Callable[[int, torch.Tensor], torch.Tensor]


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
    """Writer l gains sum_e g[b, t, l, e] * (alpha / rank) * (h @ V[l, e]) @ U[l, e]^T, h being its input."""

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

    @property
    def d_in(self) -> int:
        return self.V.shape[2]

    def delta(self, l: int, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Edit added to writer l's output; g is [E], [B, E] or [B, T, E], h is [B, T, d_in]."""
        low = torch.einsum("btd,edr->bter", h, self.V[l].to(h.dtype))
        g = g.to(h.dtype)
        g = (g if g.dim() == 3 else g.unsqueeze(-2)).expand(*h.shape[:2], -1)
        return self.scale * torch.einsum("bter,bte,eor->bto", low, g, self.U[l].to(h.dtype))

    @contextmanager
    def apply(self, mixing: Mixing):
        """Hooks the writers for the duration.

        mixing is g [L, E] (shared), g [B, L, E] (per example), or a router called in each hook as
        router(l, h) -> [B, T, E] (per token, from the writer's own input h)."""
        if isinstance(mixing, torch.Tensor):
            if mixing.shape[-2:] != self.shape:
                raise ValueError(f"mixing shape {tuple(mixing.shape)} does not end in {self.shape}")
            g = mixing
            mixing = lambda l, h: g[..., l, :]  # noqa: E731

        log = EditLog()

        def hook(l):
            def fn(_m, inp, out):
                d = self.delta(l, inp[0], mixing(l, inp[0]))
                log.add(d)
                return out + d
            return fn

        handles = [w.register_forward_hook(hook(l)) for l, w in enumerate(self._writers)]
        try:
            yield log
        finally:
            for h in handles:
                h.remove()
