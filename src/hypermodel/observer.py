"""Observer E_phi: trace features -> descriptor z. `python -m hypermodel.observer --help`."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hypermodel.probe import L2_GRID, auroc, probe_scores

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Observer(nn.Module):
    def __init__(self, in_dim: int, k: int = 64, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.config = {"in_dim": in_dim, "k": k, "hidden": hidden, "dropout": dropout}
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.encoder = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, k))
        # Step 2 trains z through a correctness head; step 3 retrains it end to end with the editor.
        self.head = nn.Linear(k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder((x - self.mu) / self.sd)

    @torch.no_grad()
    def encode(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.eval()
        dev = self.mu.device
        return np.concatenate([self(torch.as_tensor(np.asarray(X[i:i + batch_size], np.float32), device=dev))
                               .cpu().numpy() for i in range(0, len(X), batch_size)])

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Correctness logits from the training head."""
        z = torch.as_tensor(self.encode(X), device=self.mu.device)
        return self.head(z).squeeze(1).cpu().numpy()

    def save(self, path: Path) -> None:
        torch.save({"config": self.config, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, device: str = _DEVICE) -> "Observer":
        blob = torch.load(path, map_location=device)
        obs = cls(**blob["config"]).to(device)
        obs.load_state_dict(blob["state"])
        return obs.eval()


def train_observer(X: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, k: int = 64,
                   seed: int = 0, max_epochs: int = 300, patience: int = 20, lr: float = 1e-3,
                   weight_decay: float = 1e-2, batch_size: int = 256, **arch) -> Observer:
    """Fit on train rows, keep the epoch with the best val AUROC."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=_DEVICE)
    yt = torch.as_tensor(np.asarray(y, np.float32), device=_DEVICE)
    tr = torch.as_tensor(np.flatnonzero(train), device=_DEVICE)
    obs = Observer(X.shape[1], k, **arch).to(_DEVICE)
    obs.mu.copy_(Xt[tr].mean(0))
    obs.sd.copy_(Xt[tr].std(0) + 1e-6)
    opt = torch.optim.AdamW(obs.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, stale = -1.0, None, 0
    for _ in range(max_epochs):
        obs.train()
        for idx in tr[torch.randperm(len(tr), generator=gen).to(_DEVICE)].split(batch_size):
            loss = nn.functional.binary_cross_entropy_with_logits(obs.head(obs(Xt[idx])).squeeze(1), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        obs.eval()
        with torch.no_grad():
            score = auroc(obs.head(obs(Xt[val])).squeeze(1).cpu().numpy(), y[val])
        if score > best:
            best, stale = score, 0
            best_state = {k_: v.clone() for k_, v in obs.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    obs.load_state_dict(best_state)
    return obs.eval()


def paired_auroc_ci(s_a: np.ndarray, s_b: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0,
                    alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of AUROC(s_a) - AUROC(s_b), resampling the same rows for both."""
    rng = np.random.default_rng(seed)
    diffs = []
    while len(diffs) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if y[idx].all() or not y[idx].any():
            continue
        diffs.append(auroc(s_a[idx], y[idx]) - auroc(s_b[idx], y[idx]))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


MARGIN = 0.01


def main():
    import argparse

    from hypermodel.probe import feature_sets
    from hypermodel.trace import TraceSet

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--inputs", nargs="+", default=["raw/all", "contrastive/strat-k4/all"],
                   help="feature sets to concatenate, e.g. raw/all contrastive/strat-k4/all sae/all")
    p.add_argument("--sae", help="HF repo of sparsify SAEs, needed when inputs include sae/all")
    p.add_argument("--conflict", type=Path, help="conflict.npy (hypermodel.conflict): adds the conflict feature set")
    p.add_argument("--reference", type=Path,
                   help="observer dir to beat: paired CI of this observer minus its observer-s<seed>.pt, per seed")
    p.add_argument("--reference-inputs", nargs="+", default=["raw/all"])
    args = p.parse_args()
    if any(name.startswith("sae") for name in args.inputs) and not args.sae:
        p.error("--inputs with sae features needs --sae <repo>")
    ts = TraceSet.load(args.traces)
    y = ts.labels
    train, val, test = ts.mask("train"), ts.mask("val"), ts.mask("test")
    named = feature_sets(ts, train, args.sae)
    B = named["baseline"]
    if args.conflict:
        named["conflict"] = np.load(args.conflict)
        named["baseline+conflict"] = np.concatenate([B, named["conflict"]], 1)
    X = np.concatenate([named[name] for name in args.inputs], 1)

    def linear(F):
        l2 = max(L2_GRID, key=lambda v: auroc(probe_scores(F, y, train, val, v), y[val]))
        return probe_scores(F, y, train, test, l2)

    # Linear baseline with l2 chosen on val, plus an observer-sized MLP on the same question-only features.
    base_scores = linear(B)
    base = auroc(base_scores, y[test])
    extra = {}
    if args.conflict:
        # Does the model's own processing add anything to the question, before any observer is involved?
        extra["question_plus_conflict_ci"] = paired_auroc_ci(linear(named["baseline+conflict"]), base_scores, y[test])
    mlp_base = auroc(train_observer(B, y, train, val, k=args.k, seed=0).predict(B[test]), y[test])
    print(f"baseline linear {base:.3f}, baseline MLP {mlp_base:.3f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        obs = train_observer(X, y, train, val, k=args.k, seed=seed)
        # Scored by the head on test rows only: no second probe fit on rows the observer trained on.
        scores = obs.predict(X[test])
        a = auroc(scores, y[test])
        lo, hi = paired_auroc_ci(scores, base_scores, y[test], seed=seed)
        obs.save(args.out / f"observer-s{seed}.pt")
        np.save(args.out / f"z-s{seed}.npy", obs.encode(X))
        rows.append({"seed": seed, "k": args.k, "inputs": args.inputs, "z_auroc": a,
                     "diff_vs_baseline_ci": [lo, hi]})
        if args.reference:
            ref = Observer.load(args.reference / f"observer-s{seed}.pt")
            R = np.concatenate([named[name] for name in args.reference_inputs], 1)
            rows[-1]["diff_vs_reference_ci"] = paired_auroc_ci(scores, ref.predict(R[test]), y[test], seed=seed)
        print(json.dumps(rows[-1]), flush=True)
    aucs = [r["z_auroc"] for r in rows]
    verdict = {"z_auroc_mean": float(np.mean(aucs)), "z_auroc_std": float(np.std(aucs)),
               "baseline_auroc": base, "baseline_mlp_auroc": mlp_base, "n_test": int(test.sum()),
               "margin": MARGIN, **extra,
               # Non-inferiority: every seed's CI must clear -MARGIN.
               "pass": min(aucs) >= 0.75 and all(r["diff_vs_baseline_ci"][0] >= -MARGIN for r in rows)}
    if args.reference:
        verdict["beats_reference"] = all(r["diff_vs_reference_ci"][0] > 0 for r in rows)
    rows.append(verdict)
    (args.out / "observer.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
