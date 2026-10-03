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


L2_GRID = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 10.0)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fit_probe(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, l2: float = 1.0) -> float:
    """Standardized L2 logistic regression fit on train rows, scored by AUROC on test rows."""
    return auroc(probe_scores(X, y, train, test, l2), y[test])


def probe_scores(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, l2: float = 1.0) -> np.ndarray:
    Xtr = torch.as_tensor(np.asarray(X[train], np.float32), device=_DEVICE)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    ytr = torch.as_tensor(np.asarray(y[train], np.float32), device=_DEVICE)
    w = torch.zeros(Xtr.shape[1], device=_DEVICE, requires_grad=True)
    b = torch.zeros(1, device=_DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + l2 / 2 * w.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        Xte = (torch.as_tensor(np.asarray(X[test], np.float32), device=_DEVICE) - mu) / sd
        return (Xte @ w + b).cpu().numpy()


def probe_auroc(X, y, pool, test, seeds=(0, 1, 2), grid=L2_GRID) -> tuple[float, float]:
    """Per seed: re-split pool 75/25 into train/val, pick l2 on val, report test AUROC. Test rows stay fixed."""
    scores = []
    for seed in seeds:
        rows = np.flatnonzero(pool)
        val_rows = np.random.default_rng(seed).choice(rows, size=len(rows) // 4, replace=False)
        val = np.zeros_like(pool, dtype=bool)
        val[val_rows] = True
        train = pool & ~val
        best = max(grid, key=lambda l2: fit_probe(X, y, train, val, l2))
        scores.append(fit_probe(X, y, train, test, best))
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


def layer_sweep(acts: np.ndarray, y, pool, test, seeds=(0, 1, 2), out_dir: Path | None = None,
                **kw) -> list[dict]:
    """acts is [N, L, d]; one probe per layer."""
    rows = []
    for layer in range(acts.shape[1]):
        mean, std = probe_auroc(np.asarray(acts[:, layer], np.float32), y, pool, test, seeds, **kw)
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


def feature_sets(ts, fit, sae_repo: str | None = None) -> dict[str, np.ndarray]:
    from hypermodel.contrastive import contrastive_basis, difficulty_strata
    from hypermodel.trace import POSITIONS

    n = len(ts.records)
    resid = np.asarray(ts.resid, np.float32)
    strata = difficulty_strata(ts.records)
    named = {"baseline": baseline_features(ts.records), "raw/all": resid.reshape(n, -1)}
    contrast = []
    for i, layer in enumerate(ts.meta["layers"]):
        named[f"raw/L{layer}/all-pos"] = resid[:, i].reshape(n, -1)
        for j, pos in enumerate(POSITIONS):
            acts = resid[:, i, j]
            named[f"raw/L{layer}/{pos}"] = acts
            # Bases are fit on train rows only; val selects l2 and early stopping, test stays unseen.
            for tag, kw in {"k1": {}, "strat-k4": {"strata": strata, "k": 4}}.items():
                proj = acts @ contrastive_basis(acts, ts.labels, fit, **kw)
                named[f"contrastive/{tag}/L{layer}/{pos}"] = proj
                if tag == "strat-k4":
                    contrast.append(proj)
    named["contrastive/strat-k4/all"] = np.concatenate(contrast, 1)
    named["raw/all+contrastive"] = np.concatenate([named["raw/all"], named["contrastive/strat-k4/all"]], 1)
    # What the internals add beyond the question itself.
    last = resid[:, :, POSITIONS.index("last")].reshape(n, -1)
    named["baseline+raw/last"] = np.concatenate([named["baseline"], last], 1)
    named["baseline+contrastive"] = np.concatenate([named["baseline"], named["contrastive/strat-k4/all"]], 1)
    if sae_repo:
        named.update(_sae_features(ts, fit, resid, sae_repo))
        named["baseline+sae"] = np.concatenate([named["baseline"], named["sae/all"]], 1)
    return named


def _sae_features(ts, fit, resid, repo) -> dict[str, np.ndarray]:
    from hypermodel.sae import TopKSAE, active_latents
    from hypermodel.trace import POSITIONS

    named, parts = {}, []
    for i, layer in enumerate(ts.meta["layers"]):
        sae = TopKSAE.from_hub(repo, layer)
        for j, pos in enumerate(POSITIONS):
            acts = sae.encode(resid[:, i, j])
            feats = acts[:, active_latents(acts, fit)]
            named[f"sae/L{layer}/{pos}"] = feats
            parts.append(feats)
    named["sae/all"] = np.concatenate(parts, 1)
    return named


def main():
    import argparse

    from hypermodel.trace import TraceSet

    p = argparse.ArgumentParser(description="Correctness probes on a trace run: `python -m hypermodel.probe --help`.")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--only", help="substring filter on feature names")
    p.add_argument("--sae", help="HF repo of sparsify SAEs for the traced layers, e.g. gzxiong/sae-qwen3-0.6b")
    args = p.parse_args()
    ts = TraceSet.load(args.traces)
    y, test = ts.labels, ts.mask("test")
    pool = ts.mask("train") | ts.mask("val")
    rows = []
    if not args.skip_sweep:
        rows += [{"feature": f"sweep/L{r['layer']}", **r}
                 for r in layer_sweep(ts.last_all, y, pool, test, out_dir=args.out)]
    for name, X in feature_sets(ts, ts.mask("train"), args.sae).items():
        if args.only and args.only not in name:
            continue
        mean, std = probe_auroc(X, y, pool, test)
        rows.append({"feature": name, "auroc_mean": mean, "auroc_std": std})
        print(f"{name:>32}  {mean:.3f} +- {std:.3f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    name = "probes.jsonl" if not args.only else f"probes-{args.only.replace('/', '_')}.jsonl"
    (args.out / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
