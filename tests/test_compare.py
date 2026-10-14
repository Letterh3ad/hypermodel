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
