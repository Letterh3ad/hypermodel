"""Correct-vs-wrong directions in the residual stream (abliteration's mean-difference, generalised to k)."""

import numpy as np

MIN_PER_CLASS = 10


def difficulty_strata(records: list[dict], n_bins: int = 6) -> np.ndarray:
    """Quantile bins of answer magnitude; contrasts inside a bin compare questions of similar difficulty."""
    size = np.log10(np.abs([r["answer"] for r in records]) + 1)
    edges = np.quantile(size, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, size, side="right")


def contrastive_basis(acts: np.ndarray, y: np.ndarray, fit: np.ndarray, strata: np.ndarray | None = None,
                      k: int = 1) -> np.ndarray:
    """Orthonormal [d, k] basis from mean(correct) - mean(wrong) over the fit rows.

    Without strata this is the single abliteration-style direction. With strata, the per-stratum
    differences are stacked and their top-k right singular vectors returned.
    """
    acts = np.asarray(acts, np.float64)
    groups = np.zeros(len(y), int) if strata is None else np.asarray(strata)
    diffs = []
    for g in np.unique(groups[fit]):
        rows = fit & (groups == g)
        pos, neg = rows & y, rows & ~y
        if pos.sum() >= MIN_PER_CLASS and neg.sum() >= MIN_PER_CLASS:
            diffs.append(acts[pos].mean(0) - acts[neg].mean(0))
    if len(diffs) < k:
        raise ValueError(f"only {len(diffs)} strata have both classes, need k={k}")
    _, _, vt = np.linalg.svd(np.stack(diffs), full_matrices=False)
    basis = vt[:k].T
    # Orient the leading direction to point toward correct answers.
    if basis[:, 0] @ np.mean(diffs, 0) < 0:
        basis[:, 0] *= -1
    return basis.astype(np.float32)
