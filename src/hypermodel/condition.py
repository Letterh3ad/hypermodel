"""What the editor conditions on: the unedited model's trace of an input, read by the step 2 observer."""

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from hypermodel.adapter import ModelAdapter
from hypermodel.arithmetic import Question
from hypermodel.retain import RetainItem
from hypermodel.trace import capture, operand_chars

_QUESTION = re.compile(r"(\d+)[+*%](\d+)=")


@dataclass(frozen=True)
class ObsInput:
    """Text the unedited pass reads, and the last digit of each operand (None for plain text)."""
    text: str
    a_char: int | None = None
    b_char: int | None = None


def obs_input(item: Question | RetainItem, prefix: str) -> ObsInput:
    """The question without its answer; a retain paragraph whole, read at its last token."""
    if isinstance(item, Question):
        return ObsInput(prefix + item.prompt, *operand_chars(prefix, item))
    if item.source == "text":
        return ObsInput(item.text)
    m = _QUESTION.match(item.text, item.start)
    return ObsInput(item.text[:m.end()], m.end(1) - 1, m.end(2) - 1)


_OPS = {"+": "add", "*": "mul", "%": "mod"}
_RETAIN_QUESTION = re.compile(r"(\d+)([+*%])(\d+)=(\d+)")


def question_features(items: list, prefix: str) -> tuple[list[str], np.ndarray]:
    """Phase 1 question-only features on one schema for all items, plus an is-text column (text rows are 0 else)."""
    from hypermodel.probe import baseline_features

    records, rows = [], []
    for i, it in enumerate(items):
        if isinstance(it, Question):
            records.append({"a": it.a, "b": it.b, "op": it.op, "answer": it.answer})
        elif it.source != "text":
            a, op, b, ans = _RETAIN_QUESTION.match(it.text, it.start).groups()
            records.append({"a": int(a), "b": int(b), "op": _OPS[op], "answer": int(ans)})
        else:
            continue
        rows.append(i)
    base = baseline_features(records, with_answer=False) if records else np.zeros((0, 0), np.float32)
    X = np.zeros((len(items), base.shape[1] + 1), np.float32)
    X[rows, :-1] = base
    X[:, -1] = [not (isinstance(it, Question) or it.source != "text") for it in items]
    return [obs_input(it, prefix).text for it in items], X


class FeatureStore:
    """Precomputed conditioning inputs by key; the unedited pass is deterministic, so it runs once per input."""

    def __init__(self, keys: list[str], features: torch.Tensor):
        self.index = {k: i for i, k in enumerate(keys)}
        self.features = features  # kept on CPU; one batch at a time moves to the GPU

    def __call__(self, keys: list[str], device=None) -> torch.Tensor:
        try:
            rows = [self.index[k] for k in keys]
        except KeyError as e:
            raise KeyError(f"no conditioning features for {e.args[0]!r}") from None
        return self.features[rows].to(device, torch.float32)

    def extend(self, keys: list[str], features: torch.Tensor) -> None:
        new = [i for i, k in enumerate(keys) if k not in self.index]
        self.index.update({keys[i]: len(self.index) + j for j, i in enumerate(new)})
        self.features = torch.cat([self.features, features[new].to(self.features.dtype)])


class ConditionedMixer(nn.Module):
    """g = g0 + head(encoder(features)); the head's last layer starts at zero, so training begins at plain LoRA."""

    def __init__(self, shape: tuple[int, int], encoder: nn.Module, enc_dim: int, store: FeatureStore,
                 key: Callable, hidden: int = 64, freeze_encoder: bool = False):
        super().__init__()
        self.shape, self.store, self.key, self.freeze_encoder = shape, store, key, freeze_encoder
        self.g0 = nn.Parameter(torch.ones(shape))
        self.encoder = encoder.requires_grad_(not freeze_encoder)
        self.head = nn.Sequential(nn.Linear(enc_dim, hidden), nn.GELU(), nn.Linear(hidden, shape[0] * shape[1]))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, items: list) -> torch.Tensor:
        x = self.store([self.key(it) for it in items], self.g0.device)
        return self.g0 + self.head(self.encoder(x)).view(-1, *self.shape)

    def param_groups(self, lr: float, encoder_lr: float) -> list[dict]:
        own = [p for n, p in self.named_parameters() if not n.startswith("encoder.")]
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        return [{"params": own, "lr": lr}] + ([{"params": enc, "lr": encoder_lr}] if enc else [])


class RouterMixer(nn.Module):
    """MoLE-style routing: g[b, t, l] = g0[l] + h[b, t] @ W[l] + c[l], h being writer l's input at token t.

    Reads the current (edited) pass, so it needs no observer or precomputed inputs. W and c start at zero,
    so training begins at plain LoRA."""

    def __init__(self, shape: tuple[int, int], d_in: int):
        super().__init__()
        self.g0 = nn.Parameter(torch.ones(shape))
        self.W = nn.Parameter(torch.zeros(shape[0], d_in, shape[1]))
        self.c = nn.Parameter(torch.zeros(shape))

    def forward(self, items: list) -> Callable[[int, torch.Tensor], torch.Tensor]:
        return self.route

    def route(self, l: int, h: torch.Tensor) -> torch.Tensor:
        """[B, T, E] mixing for writer l from its input h [B, T, d_in]."""
        return self.g0[l] + h.to(self.W.dtype) @ self.W[l] + self.c[l]


class Standardize(nn.Module):
    """Fixed z-scoring, the encoder for question features."""

    def __init__(self, mu: torch.Tensor, sd: torch.Tensor):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sd


def trace_features(adapter: ModelAdapter, inputs: list[ObsInput], layers: list[int],
                   batch_size: int = 64) -> np.ndarray:
    """raw/all observer features [N, layers * positions * d], laid out as in phase 1."""
    operands = [None if o.a_char is None else (o.a_char, o.b_char) for o in inputs]
    resid, _ = capture(adapter, [o.text for o in inputs], operands, layers, batch_size)
    return resid.reshape(len(inputs), -1)
