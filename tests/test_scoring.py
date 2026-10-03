import json

import pytest

from hypermodel.arithmetic import Difficulty, few_shot_prefix, generate
from hypermodel.scoring import parse_answer, score
from hypermodel.sweep import sweep


@pytest.mark.parametrize("text,expected", [
    ("68\n12+3=15", 68),
    (" 68", 68),
    ("-7\n", -7),
    ("1,234", 1),
    ("", None),
    ("abc", None),
    ("\n68", None),  # answer must come before the first newline
])
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


@pytest.fixture(scope="module")
def small_lm():
    from hypermodel.adapter import load
    return load("EleutherAI/pythia-14m", device="cpu")


@pytest.mark.slow
def test_batched_scoring_matches_one_at_a_time(small_lm):
    d = Difficulty("add", 2)
    qs = generate(d, 6, seed=0)
    prefix = few_shot_prefix(d)
    batched = score(small_lm, qs, prefix, batch_size=6)
    single = score(small_lm, qs, prefix, batch_size=1)
    assert [r["output"] for r in batched] == [r["output"] for r in single]
    for r, q in zip(batched, qs):
        assert r["answer"] == q.answer
        assert r["correct"] == (r["pred"] == q.answer)


@pytest.mark.slow
def test_sweep_writes_summary_and_records(small_lm, tmp_path):
    settings = [Difficulty("add", 1), Difficulty("mul", 2)]
    sweep(small_lm, settings, n=5, seed=0, out_dir=tmp_path)
    summary = [json.loads(l) for l in (tmp_path / "summary.jsonl").read_text().splitlines()]
    records = [json.loads(l) for l in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert [s["setting"] for s in summary] == ["add-1", "mul-2"]
    assert all(s["n"] == 5 and 0 <= s["accuracy"] <= 1 for s in summary)
    assert len(records) == 10
    assert sum(r["correct"] for r in records if r["setting"] == "add-1") == round(summary[0]["accuracy"] * 5)
