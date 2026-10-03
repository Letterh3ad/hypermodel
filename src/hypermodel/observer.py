"""Observer E_phi: trace features -> descriptor z. `python -m hypermodel.observer --help`."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hypermodel.probe import auroc, probe_auroc

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


def main():
    import argparse

    from hypermodel.probe import feature_sets
    from hypermodel.trace import TraceSet

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()
    ts = TraceSet.load(args.traces)
    y = ts.labels
    train, val, test = ts.mask("train"), ts.mask("val"), ts.mask("test")
    pool = train | val
    named = feature_sets(ts, pool)
    X = np.concatenate([named["raw/all"], named["contrastive/strat-k4/all"]], 1)
    base, _ = probe_auroc(named["baseline"], y, pool, test)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        obs = train_observer(X, y, train, val, k=args.k, seed=seed)
        z = obs.encode(X)
        mean, _ = probe_auroc(z, y, pool, test, seeds=(seed,))
        obs.save(args.out / f"observer-s{seed}.pt")
        np.save(args.out / f"z-s{seed}.npy", z)
        rows.append({"seed": seed, "k": args.k, "z_auroc": mean})
        print(f"seed {seed}: z AUROC {mean:.3f}", flush=True)
    z_mean = float(np.mean([r["z_auroc"] for r in rows]))
    verdict = {"z_auroc_mean": z_mean, "z_auroc_std": float(np.std([r["z_auroc"] for r in rows])),
               "baseline_auroc": base, "pass": z_mean >= 0.75 and z_mean >= base - 0.01}
    rows.append(verdict)
    (args.out / "observer.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
