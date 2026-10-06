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
