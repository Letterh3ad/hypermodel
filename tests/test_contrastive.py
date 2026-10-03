import numpy as np

from hypermodel.contrastive import contrastive_basis, difficulty_strata


def _planted(rng, n=3000, d=64, conf=None):
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.random(n) < 0.4
    u = np.zeros(d, np.float32)
    u[5] = 1
    X[y] += 1.5 * u
    return X, y, u


def test_mean_difference_recovers_the_planted_direction():
    rng = np.random.default_rng(0)
    X, y, u = _planted(rng)
    D = contrastive_basis(X, y, np.ones(len(y), bool))
    assert D.shape == (64, 1)
    assert abs(float(D[:, 0] @ u)) > 0.9


def test_stratified_top_k_is_orthonormal_and_contains_the_direction():
    rng = np.random.default_rng(1)
    X, y, u = _planted(rng)
    strata = rng.integers(0, 6, len(y))
    D = contrastive_basis(X, y, np.ones(len(y), bool), strata=strata, k=3)
    assert D.shape == (64, 3)
    np.testing.assert_allclose(D.T @ D, np.eye(3), atol=1e-5)
    assert np.linalg.norm(D.T @ u) > 0.9


def test_stratifying_removes_a_difficulty_confound():
    rng = np.random.default_rng(2)
    n, d = 4000, 32
    hard = rng.random(n) < 0.5
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[hard, 0] += 3  # difficulty shows up on dim 0 ...
    y = rng.random(n) < np.where(hard, 0.2, 0.8)  # ... and drives correctness
    X[y, 1] += 1  # the per-question correctness signal lives on dim 1
    plain = contrastive_basis(X, y, np.ones(n, bool))[:, 0]
    strat = contrastive_basis(X, y, np.ones(n, bool), strata=hard.astype(int), k=1)[:, 0]
    assert abs(plain[0]) > abs(plain[1])
    assert abs(strat[1]) > 0.9


def test_only_rows_in_the_fit_mask_matter():
    rng = np.random.default_rng(3)
    X, y, _ = _planted(rng)
    train = rng.random(len(y)) < 0.6
    y2 = y.copy()
    y2[~train] = ~y2[~train]
    np.testing.assert_array_equal(contrastive_basis(X, y, train), contrastive_basis(X, y2, train))


def test_difficulty_strata_bins_by_answer_size():
    recs = [{"answer": a} for a in [1, 9, 50, 500, 900, 5000, 99999, 100000]]
    s = difficulty_strata(recs, n_bins=4)
    assert len(s) == 8 and s.min() == 0 and s.max() == 3
    assert (np.diff(s) >= 0).all()
