import json

import numpy as np
import pytest

from hypermodel.probe import auroc, baseline_features, fit_probe, layer_sweep, probe_auroc


def _split(n, rng):
    train = rng.random(n) < 0.6
    return train, ~train


@pytest.mark.parametrize("scores,y,expected", [
    ([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1], 0.75),
    ([1, 2, 3, 4], [0, 0, 1, 1], 1.0),
    ([4, 3, 2, 1], [0, 0, 1, 1], 0.0),
    ([5, 5, 5, 5], [0, 1, 0, 1], 0.5),
    ([1, 2, 2, 3], [0, 0, 1, 1], 0.875),  # one tied pos/neg pair counts half
])
def test_auroc_matches_hand_computed(scores, y, expected):
    assert auroc(np.array(scores), np.array(y, bool)) == pytest.approx(expected)


def test_auroc_needs_both_classes():
    with pytest.raises(ValueError):
        auroc(np.array([0.1, 0.2]), np.array([True, True]))


def test_planted_direction_is_found():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 32)).astype(np.float32)
    w = rng.standard_normal(32)
    y = X @ w + 0.3 * rng.standard_normal(2000) > 0
    train, test = _split(2000, rng)
    assert fit_probe(X, y, train, test) > 0.9


def test_noise_sits_near_chance():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 20)).astype(np.float32)
    y = rng.random(2000) < 0.4
    train, test = _split(2000, rng)
    mean, std = probe_auroc(X, y, train, test, seeds=(0, 1, 2))
    assert 0.4 < mean < 0.6
    assert std >= 0


def test_seeds_change_the_result_but_a_seed_repeats():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((500, 16)).astype(np.float32)
    y = X[:, 0] + rng.standard_normal(500) > 0
    pool, test = _split(500, rng)
    assert probe_auroc(X, y, pool, test, seeds=(3,)) == probe_auroc(X, y, pool, test, seeds=(3,))
    assert probe_auroc(X, y, pool, test, seeds=(0, 1, 2))[1] > 0


def test_l2_tuned_on_val_beats_an_untuned_probe_on_a_weak_high_dim_signal():
    rng = np.random.default_rng(5)
    n, d = 5000, 1024
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = np.zeros(d)
    w[:8] = 1
    signal = X @ w
    y = signal + 2.5 * rng.standard_normal(n) > 0
    pool, test = _split(n, rng)
    # ~2k rows in 1024 isotropic dims cap any dense linear probe near 0.76 (ideal 0.86)
    untuned = fit_probe(X, y, pool, test, l2=1e-4)
    mean, _ = probe_auroc(X, y, pool, test, seeds=(0,))
    assert mean > untuned + 0.01 and mean > 0.73, (mean, untuned)


def test_sweep_finds_the_layer_with_signal(tmp_path):
    rng = np.random.default_rng(3)
    n, L, d = 1500, 4, 16
    acts = rng.standard_normal((n, L, d)).astype(np.float16)
    y = rng.random(n) < 0.5
    acts[y, 2, 0] += 2.0
    train, test = _split(n, rng)
    rows = layer_sweep(acts, y, train, test, seeds=(0, 1, 2), out_dir=tmp_path)
    assert [r["layer"] for r in rows] == list(range(L))
    assert max(rows, key=lambda r: r["auroc_mean"])["layer"] == 2
    written = [json.loads(l) for l in (tmp_path / "layer_sweep.jsonl").read_text().splitlines()]
    assert written == rows
    assert (tmp_path / "layer_sweep.png").stat().st_size > 0


def _records(rng, n):
    out = []
    for _ in range(n):
        a = int(rng.integers(1, 1000))
        b = int(rng.integers(10, 100))
        out.append({"a": a, "b": b, "op": "mul", "answer": a * b, "correct": None})
    return out


def test_baseline_features_pick_up_operand_size():
    rng = np.random.default_rng(4)
    records = _records(rng, 2000)
    y = np.array([r["a"] < 100 for r in records])  # "easy" problems are the correct ones
    X = baseline_features(records)
    assert X.shape[0] == 2000 and X.dtype == np.float32
    train, test = _split(2000, rng)
    assert fit_probe(X, y, train, test) > 0.9


def test_baseline_features_are_fixed_width_and_align_units_digits():
    X = baseline_features([{"a": 7, "b": 12, "op": "mul", "answer": 84},
                           {"a": 123, "b": 45, "op": "mul", "answer": 5535}])
    assert X.shape[0] == 2
    assert np.isfinite(X).all()
    same = baseline_features([{"a": 17, "b": 12, "op": "mul", "answer": 204},
                              {"a": 123, "b": 45, "op": "mul", "answer": 5535}])
    assert same.shape == X.shape


def test_feature_sets_fit_contrastive_bases_without_test_labels(tmp_path):
    from hypermodel.probe import feature_sets
    from hypermodel.trace import TraceSet

    rng = np.random.default_rng(6)
    n, d = 400, 16
    records = _records(rng, n)
    y = rng.random(n) < 0.5
    for r, c in zip(records, y):
        r["correct"] = bool(c)
    split = np.array(["train", "val", "test", "train", "train"] * (n // 5))
    resid = rng.standard_normal((n, 2, 3, d)).astype(np.float16)
    TraceSet.write(tmp_path, resid, resid[:, :, -1], records, split, {"layers": [3, 5]})
    ts = TraceSet.load(tmp_path)
    named = feature_sets(ts, ts.mask("train"))
    assert named["raw/L5/b"].shape == (n, d)
    assert named["contrastive/k1/L3/a"].shape == (n, 1)
    assert named["contrastive/strat-k4/all"].shape == (n, 2 * 3 * 4)
    # flipping val and test labels must not move any contrastive feature
    flipped = [dict(r, correct=(not r["correct"]) if s != "train" else r["correct"]) for r, s in zip(records, split)]
    TraceSet.write(tmp_path / "f", resid, resid[:, :, -1], flipped, split, {"layers": [3, 5]})
    named_f = feature_sets(TraceSet.load(tmp_path / "f"), ts.mask("train"))
    np.testing.assert_array_equal(named["contrastive/strat-k4/all"], named_f["contrastive/strat-k4/all"])
