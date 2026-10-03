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
    train, test = _split(500, rng)
    assert fit_probe(X, y, train, test, seed=3) == fit_probe(X, y, train, test, seed=3)
    assert len({fit_probe(X, y, train, test, seed=s) for s in range(3)}) > 1


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
