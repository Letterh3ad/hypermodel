"""Linear correctness probes, scored by test AUROC, plus features that see only the question."""

import json
import math
from pathlib import Path

import numpy as np
import torch


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes")
    _, inv, counts = np.unique(np.asarray(scores, np.float64), return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inv]  # ties share their average 1-based rank
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_probe(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
              l2: float = 1e-2, seed: int = 0) -> float:
    """L2 logistic regression on a bootstrap of the train rows; the fit is convex, so the seed only picks the resample."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.flatnonzero(train), size=int(train.sum()), replace=True)
    Xtr = torch.from_numpy(np.asarray(X[idx], np.float32))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    ytr = torch.from_numpy(np.asarray(y[idx], np.float32))
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + l2 / 2 * w.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        Xte = (torch.from_numpy(np.asarray(X[test], np.float32)) - mu) / sd
        return auroc((Xte @ w + b).numpy(), y[test])


def probe_auroc(X, y, train, test, seeds=(0, 1, 2), **kw) -> tuple[float, float]:
    scores = [fit_probe(X, y, train, test, seed=s, **kw) for s in seeds]
    return float(np.mean(scores)), float(np.std(scores))


def _digits(n: int, width: int) -> np.ndarray:
    """One-hot per digit, right-aligned so the units digit always lands in the same slot."""
    out = np.zeros((width, 10), np.float32)
    for i, ch in enumerate(reversed(str(abs(n)))):
        out[width - 1 - i, int(ch)] = 1
    return out.ravel()


def baseline_features(records: list[dict]) -> np.ndarray:
    """Question-only features: if these predict correctness, a probe may just be reading difficulty."""
    wa = max(len(str(abs(r["a"]))) for r in records)
    wb = max(len(str(abs(r["b"]))) for r in records)
    ops = sorted({r["op"] for r in records})
    rows = []
    for r in records:
        a, b = abs(r["a"]), abs(r["b"])
        scalars = [len(str(a)), len(str(b)), math.log10(a + 1), math.log10(b + 1),
                   math.log10(abs(r["answer"]) + 1)]
        lead = np.zeros(20, np.float32)
        lead[int(str(a)[0])] = lead[10 + int(str(b)[0])] = 1
        op = np.array([r["op"] == o for o in ops], np.float32)
        rows.append(np.concatenate([np.array(scalars, np.float32), lead, op, _digits(a, wa), _digits(b, wb)]))
    return np.stack(rows)


def layer_sweep(acts: np.ndarray, y, train, test, seeds=(0, 1, 2), out_dir: Path | None = None,
                **kw) -> list[dict]:
    """acts is [N, L, d]; one probe per layer."""
    rows = []
    for layer in range(acts.shape[1]):
        mean, std = probe_auroc(np.asarray(acts[:, layer], np.float32), y, train, test, seeds, **kw)
        rows.append({"layer": layer, "auroc_mean": mean, "auroc_std": std})
    if out_dir is not None:
        _write_sweep(rows, Path(out_dir))
    return rows


def _write_sweep(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layer_sweep.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.errorbar([r["layer"] for r in rows], [r["auroc_mean"] for r in rows],
                yerr=[r["auroc_std"] for r in rows], marker="o", capsize=3)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set(xlabel="layer", ylabel="test AUROC", title="Correctness probe by layer (last token)")
    fig.tight_layout()
    fig.savefig(out_dir / "layer_sweep.png", dpi=120)
    plt.close(fig)
