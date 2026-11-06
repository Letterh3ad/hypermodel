import json

import numpy as np
import pytest

from hypermodel.compare import load_correct, paired_bootstrap


def test_identical_runs_differ_by_exactly_zero():
    a = np.array([1, 0, 1, 1], bool)
    d = paired_bootstrap([(a, a)])
    assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0 and d["disagree"] == 0


def test_a_clear_win_has_a_ci_above_zero():
    rng = np.random.default_rng(1)
    b = rng.random(2000) < 0.4
    a = b | (rng.random(2000) < 0.2)
    d = paired_bootstrap([(a, b)])
    assert d["lo"] > 0 and d["diff"] == pytest.approx((a.astype(float) - b).mean())


def test_pooling_averages_paired_seeds():
    a1, b1 = np.ones(10, bool), np.zeros(10, bool)
    a2, b2 = np.zeros(10, bool), np.zeros(10, bool)
    assert paired_bootstrap([(a1, b1), (a2, b2)])["diff"] == pytest.approx(0.5)


def test_runs_are_paired_by_prompt_not_by_file_order(tmp_path):
    for name, rows in {"a": [("1*2=", True), ("3*4=", False)], "b": [("3*4=", True), ("1*2=", True)]}.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "test_records.jsonl").write_text(
            "".join(json.dumps({"prompt": p, "correct": c}) + "\n" for p, c in rows))
    a, b = load_correct(tmp_path / "a", tmp_path / "b")
    assert a.tolist() == [True, False] and b.tolist() == [True, True]


def test_record_files_pair_by_prompt(tmp_path):
    from hypermodel.compare import paired_records
    for name, rows in {"a.jsonl": [("1*2=", True), ("3*4=", False)], "b.jsonl": [("3*4=", True), ("1*2=", False)]}.items():
        (tmp_path / name).write_text("".join(json.dumps({"prompt": p, "correct": c}) + "\n" for p, c in rows))
    a, b = paired_records(tmp_path / "a.jsonl", tmp_path / "b.jsonl")
    assert a.tolist() == [True, False] and b.tolist() == [False, True]


def test_stratified_bootstrap_with_one_stratum_is_the_pooled_one():
    from hypermodel.compare import stratified_bootstrap
    rng = np.random.default_rng(3)
    pairs = [(rng.random(50) < 0.6, rng.random(50) < 0.5) for _ in range(3)]
    assert stratified_bootstrap([pairs], n_boot=200) == paired_bootstrap(pairs, n_boot=200)


def test_stratified_bootstrap_weights_every_stratum_equally():
    from hypermodel.compare import stratified_bootstrap
    small = [(np.ones(4, bool), np.zeros(4, bool))]
    big = [(np.zeros(100, bool), np.zeros(100, bool))]
    d = stratified_bootstrap([small, big])
    assert d["diff"] == d["lo"] == d["hi"] == pytest.approx(0.5)


def _seeded_pairs(seed_shift, n_seeds=3, n_q=400, noise=0.0, seed=5):
    """Pairs where A beats B by seed_shift[s] on each seed s, plus independent per-question noise."""
    rng = np.random.default_rng(seed)
    b = rng.random((n_seeds, n_q)) < 0.5
    p = np.clip(0.5 + np.asarray(seed_shift)[:, None] + noise * (rng.random((n_seeds, n_q)) - 0.5), 0, 1)
    a = rng.random((n_seeds, n_q)) < p
    return [(a[s], b[s]) for s in range(n_seeds)]


def test_two_way_with_one_seed_is_the_question_only_bootstrap():
    from hypermodel.compare import stratified_bootstrap
    rng = np.random.default_rng(4)
    strata = [[(rng.random(n) < 0.6, rng.random(n) < 0.5)] for n in (30, 80)]
    assert stratified_bootstrap(strata, n_boot=300, resample_seeds=True) == stratified_bootstrap(strata, n_boot=300)
    pairs = strata[0]
    assert paired_bootstrap(pairs, n_boot=300, resample_seeds=True) == paired_bootstrap(pairs, n_boot=300)


def test_two_way_ci_widens_with_seed_variance():
    pairs = [(np.tile(a, 20), np.tile(b, 20)) for a, b in _seeded_pairs([-0.3, 0.0, 0.3], n_q=50)]
    q, sq = paired_bootstrap(pairs, n_boot=1000), paired_bootstrap(pairs, n_boot=1000, resample_seeds=True)
    assert q["diff"] == sq["diff"]
    assert sq["hi"] - sq["lo"] > 3 * (q["hi"] - q["lo"])


def test_two_way_matches_question_only_without_seed_variance():
    a, b = _seeded_pairs([0.1])[0]
    pairs = [(a, b)] * 3
    q, sq = paired_bootstrap(pairs, n_boot=2000), paired_bootstrap(pairs, n_boot=2000, resample_seeds=True)
    assert sq["lo"] == pytest.approx(q["lo"], abs=0.01) and sq["hi"] == pytest.approx(q["hi"], abs=0.01)


def test_two_way_keeps_pairing_by_question():
    pairs = _seeded_pairs([0.0, 0.1, 0.2], noise=0.4)
    perm = np.random.default_rng(9).permutation(len(pairs[0][0]))
    shuffled = [(a[perm], b[perm]) for a, b in pairs]
    d, ds = (paired_bootstrap(p, n_boot=2000, resample_seeds=True) for p in (pairs, shuffled))
    assert ds["diff"] == pytest.approx(d["diff"])
    assert ds["lo"] == pytest.approx(d["lo"], abs=0.01) and ds["hi"] == pytest.approx(d["hi"], abs=0.01)
    same = paired_bootstrap([(a, a) for a, _ in pairs], n_boot=200, resample_seeds=True)
    assert same["lo"] == same["hi"] == 0


def test_two_way_needs_the_same_seeds_in_every_stratum():
    from hypermodel.compare import stratified_bootstrap
    one = (np.ones(4, bool), np.zeros(4, bool))
    with pytest.raises(ValueError):
        stratified_bootstrap([[one, one], [one]], n_boot=10, resample_seeds=True)
