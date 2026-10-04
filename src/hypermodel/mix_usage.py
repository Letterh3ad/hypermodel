"""How much a conditioned editor's mix actually varies across test questions: `python -m hypermodel.mix_usage --help`.

A tie with plain LoRA could mean the mix barely moves, so the conditioned bank acts as one averaged LoRA."""

import json
from pathlib import Path

import numpy as np
import torch

COLLAPSE = 0.10  # pre-registered (ticket 07): spread below this means the conditioning is unused


def mixing_stats(g: torch.Tensor) -> dict[str, float]:
    """g [N, L, E] for N questions.

    spread: mean over questions of |g_q - mean g| / |mean g|, per layer then averaged (0 = one shared mix).
    effective_experts: exp(entropy) of |g| normalised over experts, averaged (E = all equal, 1 = one expert)."""
    g = g.float()
    mean = g.mean(0, keepdim=True)
    spread = ((g - mean).norm(dim=-1) / mean.norm(dim=-1).clamp_min(1e-12)).mean()
    p = g.abs() / g.abs().sum(-1, keepdim=True).clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
    return {"spread": spread.item(), "effective_experts": entropy.exp().mean().item(), "experts": g.shape[-1]}


def mixing_on_test(run: Path, ts) -> torch.Tensor | None:
    """g for the run's test questions, rebuilt from its checkpoint without loading the LM; None for plain
    LoRA and the per-token router."""
    from hypermodel.condition import ConditionedMixer, FeatureStore, Standardize, question_features
    from hypermodel.edit_train import split_questions
    from hypermodel.observer import Observer

    result = json.loads((run / "result.json").read_text())
    kind = result["conditioning"]
    if kind in ("none", "router"):
        return None
    state = torch.load(run / f"editor-s{result['seed']}.pt", map_location="cpu")["mixer"]
    prefix = ts.meta["prefix"]
    test = split_questions(ts.records, ts.split)["test"]
    keys = [prefix + q.prompt for q in test]
    if kind == "features":
        _, X = question_features(test, prefix)
        encoder, store = Standardize(state["encoder.mu"], state["encoder.sd"]), FeatureStore(keys, torch.as_tensor(X))
    else:
        rows = np.flatnonzero(ts.mask("test"))
        X = np.asarray(ts.resid[rows], np.float32).reshape(len(rows), -1)
        if result.get("conflict"):
            X = np.concatenate([X, np.load(Path(result["traces"]) / "conflict.npy")[rows]], 1)
        w1, w2 = state["encoder.encoder.1.weight"], state["encoder.encoder.3.weight"]
        encoder = Observer(w1.shape[1], k=w2.shape[0], hidden=w1.shape[0])
        store = FeatureStore(keys, torch.as_tensor(X))
    mode = {"z-gain": "gain", "z-gain-shuffled": "gain", "z-sparse": "topk"}.get(kind, "mix")
    mixer = ConditionedMixer(tuple(state["g0"].shape), encoder, state["head.0.weight"].shape[1], store,
                             key=lambda q: prefix + q.prompt, mode=mode, top_k=result.get("top_k", 2))
    mixer.load_state_dict(state)
    with torch.no_grad():
        return torch.cat([mixer.eval()(test[i:i + 512]) for i in range(0, len(test), 512)])


def main():
    import argparse

    from hypermodel.trace import TraceSet

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path, help="edit run directories with result.json")
    p.add_argument("--traces", type=Path, default=Path("runs/traces-mul-2x3-20k"))
    args = p.parse_args()
    ts = TraceSet.load(args.traces)
    for run in args.runs:
        g = mixing_on_test(run, ts)
        if g is None:
            continue
        stats = {**mixing_stats(g), "collapse_threshold": COLLAPSE}
        stats["collapsed"] = stats["spread"] < COLLAPSE
        (run / "mix_usage.json").write_text(json.dumps(stats, indent=1))
        print(f"{run.name}: {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
