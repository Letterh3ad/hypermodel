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


def paired_bootstrap(pairs: list[tuple[np.ndarray, np.ndarray]], n_boot: int = 5000, seed: int = 0,
                     resample_seeds: bool = False) -> dict:
    """Mean of a - b over questions, pooled over pairs (seeds); questions are resampled jointly across pairs."""
    return stratified_bootstrap([pairs], n_boot, seed, resample_seeds)


def stratified_bootstrap(strata: list[list[tuple[np.ndarray, np.ndarray]]], n_boot: int = 5000, seed: int = 0,
                         resample_seeds: bool = False) -> dict:
    """Mean over strata (tasks) of each stratum's paired_bootstrap mean, so every task weighs the same whatever its
    size; questions are resampled within their stratum, jointly across its pairs (seeds).

    resample_seeds makes it a crossed two-way bootstrap: pair s is A seed s vs B seed s, and each resample also draws
    seeds with replacement, so seed-to-seed variance enters the CI. With 3 seeds that draw has only 10 distinct
    multisets, so the CI is coarse."""
    Ds = [np.stack([a.astype(float) - b for a, b in pairs]) for pairs in strata]
    n_seeds = {len(D) for D in Ds}
    if resample_seeds and len(n_seeds) > 1:
        raise ValueError(f"every stratum needs the same seeds, got counts {sorted(n_seeds)}")
    k = len(Ds[0])
    rng = np.random.default_rng(seed)
    # Seeds get their own stream so the question draws match the question-only bootstrap; 1 seed reproduces it.
    seed_rng = np.random.default_rng([seed, 1])

    def resample():
        # One seed draw for all strata (a seed is one run spanning every task); one question draw per stratum shared
        # by all drawn seeds, which keeps A and B paired by question.
        s = seed_rng.integers(0, k, k) if resample_seeds else np.arange(k)
        return np.mean([D[s][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for D in Ds])
    lo, hi = np.percentile([resample() for _ in range(n_boot)], [2.5, 97.5])
    return {"diff": np.mean([D.mean() for D in Ds]), "lo": lo, "hi": hi,
            "disagree": np.mean([(D != 0).mean() for D in Ds])}


def format_ci(d: dict | None, digits: int = 1, na: str = "n/a") -> str:
    if d is None:
        return na
    diff, lo, hi = (f"{100 * d[k]:+.{digits}f}" for k in ("diff", "lo", "hi"))
    return f"{diff} [{lo}, {hi}]"


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pairs", nargs="+", help="A:B run directory pairs, one per seed; pooled when several")
    args = p.parse_args()
    pairs = [load_correct(*map(Path, s.split(":"))) for s in args.pairs]
    d = paired_bootstrap(pairs)
    d2 = paired_bootstrap(pairs, resample_seeds=True) if len(pairs) > 1 else None
    print(f"{' + '.join(args.pairs)}: {format_ci(d, 2)} pts by questions, {format_ci(d2, 2, 'n/a (1 seed)')} by "
          f"seeds x questions, {len(pairs)} seeds  disagree {100 * d['disagree']:.0f}%")


if __name__ == "__main__":
    main()
