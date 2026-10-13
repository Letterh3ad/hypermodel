"""Paired-bootstrap accuracy difference between edit runs: `python -m hypermodel.compare --help`.

Runs are paired by test prompt, so the CI reflects disagreement between the two runs, not question difficulty."""

import json
from pathlib import Path

import numpy as np


def paired_records(a: Path, b: Path, keep=lambda r: True) -> tuple[np.ndarray, np.ndarray]:
    """Correctness from two records files (jsonl with prompt and correct), aligned by prompt; keep filters rows."""
    def rows(path):
        return {r["prompt"]: r["correct"] for r in map(json.loads, path.open()) if keep(r)}
    A, B = rows(a), rows(b)
    if set(A) != set(B):
        raise ValueError(f"{a} and {b} were scored on different questions")
    keys = sorted(A)
    return np.array([A[k] for k in keys], bool), np.array([B[k] for k in keys], bool)


def load_correct(a: Path, b: Path) -> tuple[np.ndarray, np.ndarray]:
    return paired_records(a / "test_records.jsonl", b / "test_records.jsonl")


def paired_bootstrap(pairs: list[tuple[np.ndarray, np.ndarray]], n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean of a - b over questions, pooled over pairs (seeds); questions are resampled jointly across pairs."""
    return stratified_bootstrap([pairs], n_boot, seed)


def stratified_bootstrap(strata: list[list[tuple[np.ndarray, np.ndarray]]], n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean over strata (tasks) of each stratum's paired_bootstrap mean, so every task weighs the same whatever its
    size; questions are resampled within their stratum, jointly across its pairs (seeds)."""
    Ds = [np.stack([a.astype(float) - b for a, b in pairs]) for pairs in strata]
    rng = np.random.default_rng(seed)
    boots = [np.mean([D[:, rng.integers(0, D.shape[1], D.shape[1])].mean() for D in Ds]) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": np.mean([D.mean() for D in Ds]), "lo": lo, "hi": hi,
            "disagree": np.mean([(D != 0).mean() for D in Ds])}


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pairs", nargs="+", help="A:B run directory pairs, one per seed; pooled when several")
    args = p.parse_args()
    pairs = [load_correct(*map(Path, s.split(":"))) for s in args.pairs]
    d = paired_bootstrap(pairs)
    print(f"{' + '.join(args.pairs)}: {100 * d['diff']:+.2f} pts [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]"
          f"  disagree {100 * d['disagree']:.0f}%")


if __name__ == "__main__":
    main()
