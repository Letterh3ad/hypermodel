import time

import numpy as np
import pytest

from hypermodel.arithmetic import Difficulty, few_shot_prefix, generate
from hypermodel.trace import POSITIONS, TraceSet, operand_token_positions, record

D = Difficulty("mul", 2, digits_b=3)


@pytest.fixture(scope="module")
def small_lm():
    from hypermodel.adapter import load
    return load("EleutherAI/pythia-14m", device="cpu")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["EleutherAI/pythia-14m", "Qwen/Qwen3-0.6B-Base"])
def test_positions_point_at_last_digit_of_each_operand_and_the_final_token(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    qs = generate(D, 4, seed=0)
    prefix = few_shot_prefix(D)
    enc, pos = operand_token_positions(tok, prefix, qs)
    for i, q in enumerate(qs):
        decoded = {p: tok.decode(enc.input_ids[i, pos[i, j]]) for j, p in enumerate(POSITIONS)}
        assert decoded["a"].endswith(str(q.a)[-1])
        assert decoded["b"].endswith(str(q.b)[-1])
        assert decoded["last"].endswith("=")
        assert pos[i, POSITIONS.index("last")] == enc.input_ids.shape[1] - 1


@pytest.mark.slow
def test_trace_does_not_depend_on_batch_neighbours(small_lm, tmp_path):
    qs = generate(Difficulty("mul", 1, digits_b=4), 4, seed=0) + generate(D, 4, seed=0)
    prefix = few_shot_prefix(D)
    batched = TraceSet.load(record(small_lm, qs, prefix, [1, 3], tmp_path / "b", batch_size=8, seed=0))
    single = TraceSet.load(record(small_lm, qs, prefix, [1, 3], tmp_path / "s", batch_size=1, seed=0))
    np.testing.assert_allclose(batched.resid, single.resid, atol=2e-2)


@pytest.mark.slow
def test_round_trip(small_lm, tmp_path):
    qs = generate(D, 20, seed=0)
    out = record(small_lm, qs, few_shot_prefix(D), [0, 2, 4], tmp_path / "run", batch_size=8, seed=0)
    ts = TraceSet.load(out)
    n_layers, d = small_lm.n_layers, small_lm.d_model
    assert ts.resid.shape == (20, 3, len(POSITIONS), d)
    assert ts.last_all.shape == (20, n_layers, d)
    assert ts.meta["layers"] == [0, 2, 4]
    assert [r["prompt"] for r in ts.records] == [q.prompt for q in qs]
    assert ts.labels.dtype == bool and len(ts.labels) == 20
    # the last-token slice of the chosen layers must agree with the all-layer sweep
    np.testing.assert_array_equal(ts.resid[:, :, POSITIONS.index("last")], ts.last_all[:, [0, 2, 4]])


@pytest.mark.slow
def test_split_is_60_20_20_disjoint_and_fixed_by_seed(small_lm, tmp_path):
    qs = generate(D, 50, seed=0)
    prefix = few_shot_prefix(D)
    a = TraceSet.load(record(small_lm, qs, prefix, [1], tmp_path / "a", seed=0))
    b = TraceSet.load(record(small_lm, qs, prefix, [1], tmp_path / "b", seed=0))
    c = TraceSet.load(record(small_lm, qs, prefix, [1], tmp_path / "c", seed=1))
    assert [int(a.mask(s).sum()) for s in ("train", "val", "test")] == [30, 10, 10]
    assert not (a.mask("train") & a.mask("test")).any()
    assert (a.split == b.split).all() and not (a.split == c.split).all()


def test_reload_of_a_few_thousand_traces_is_fast(tmp_path):
    rng = np.random.default_rng(0)
    TraceSet.write(tmp_path, resid=rng.standard_normal((4000, 3, 3, 1024)).astype(np.float16),
                   last_all=rng.standard_normal((4000, 28, 1024)).astype(np.float16),
                   records=[{"correct": bool(i % 2)} for i in range(4000)],
                   split=np.array(["train"] * 4000), meta={"layers": [9, 14, 19]})
    t = time.perf_counter()
    ts = TraceSet.load(tmp_path)
    float(ts.resid[:, :, -1].astype(np.float32).mean())
    assert time.perf_counter() - t < 5


def test_mirror_questions_share_a_split():
    from hypermodel.arithmetic import Question
    from hypermodel.trace import _split
    qs = [Question(a, b, "mul", a * b) for a in range(10, 40) for b in range(10, 40)]
    split = dict(zip(qs, _split(qs, seed=0)))
    assert all(split[q] == split[Question(q.b, q.a, "mul", q.answer)] for q in qs)
    assert set(split.values()) == {"train", "val", "test"}


def test_right_padded_tokenizer_is_rejected(small_lm):
    small_lm.tokenizer.padding_side = "right"
    try:
        with pytest.raises(ValueError, match="left"):
            record(small_lm, generate(D, 2, seed=0), few_shot_prefix(D), [1], "unused")
    finally:
        small_lm.tokenizer.padding_side = "left"
