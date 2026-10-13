import json

import pytest
import torch

from hypermodel.arithmetic import Difficulty, few_shot_prefix, generate
from hypermodel.pilot import ARMS, aggregate
from hypermodel.stream import RETAIN_ARITH, TaskData


def _write(d, arm, correct):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"gate-{arm}.jsonl").write_text("".join(json.dumps({"prompt": f"{i}*7=", "correct": bool(c)}) + "\n"
                                                  for i, c in enumerate(correct)))


def _runs(tmp_path, a_better: bool):
    dirs = []
    for s in range(3):
        d = tmp_path / f"pilot-s{s}"
        g = torch.Generator().manual_seed(s)
        b = torch.rand(400, generator=g) < 0.35
        a = b | (torch.rand(400, generator=g) < 0.2) if a_better else b
        for arm, c in {"base": b, "A": a, "B": b, "C": b}.items():
            _write(d, arm, c.tolist())
        dirs.append(d)
    return dirs


def test_verdict_passes_only_when_a_beats_b_with_ci_above_zero(tmp_path):
    r = aggregate(_runs(tmp_path / "win", True))
    assert r["pass"] and r["A-B"]["lo"] > 0
    assert set(r) >= {"A-B", "A-C", "C-base", "A-base", "B-base", "pass"}
    r = aggregate(_runs(tmp_path / "tie", False))
    assert not r["pass"] and r["A-B"]["diff"] == 0


def test_seeds_must_share_the_gate_questions(tmp_path):
    dirs = _runs(tmp_path, True)
    _write(dirs[1], "A", [True] * 399)
    with pytest.raises(ValueError):
        aggregate(dirs)


@pytest.fixture(scope="module")
def small_lm():
    from hypermodel.adapter import load
    return load("EleutherAI/pythia-14m", device="cpu")


@pytest.mark.slow
def test_pilot_runs_every_arm_on_the_gate_and_never_trains_on_it(small_lm, tmp_path, monkeypatch):
    import hypermodel.edit_train as et
    import hypermodel.retain as retain
    from hypermodel.generator import Generator
    from hypermodel.observer import Observer
    from hypermodel.pilot import run_pilot

    monkeypatch.setattr(retain, "_load_wikitext", lambda split: (_ for _ in ()).throw(ConnectionError()))
    d = Difficulty.parse("mul-2x3")
    qs = generate(d, 40, 3)
    td, gate = TaskData("mul-2x3", few_shot_prefix(d), test=(), val=tuple(qs[:4]), train=tuple(qs[4:30])), qs[30:]
    seen = []
    real_train = et.train
    monkeypatch.setattr(et, "train", lambda *a, **k: seen.extend(a[3] + a[4]) or real_train(*a, **k))
    obs = Observer(in_dim=2 * 3 * small_lm.d_model, k=8, hidden=16)
    w = small_lm.residual_writers()[2]
    gen = Generator(8, 2, w.in_features, w.out_features, rank=4, basis=4, hidden=16)
    r = run_pilot(small_lm, gen, obs, [1, 3], td, list(td.train), gate, [2, 4], retain.retain_items("train", 4, 3, arithmetic=RETAIN_ARITH),
                  retain.retain_items("test", 4, 3, arithmetic=RETAIN_ARITH), tmp_path, seed=0, steps=2, batch_size=4)
    for arm in ARMS:
        rows = [json.loads(l) for l in (tmp_path / f"gate-{arm}.jsonl").read_text().splitlines()]
        assert [x["prompt"] for x in rows] == [q.prompt for q in gate]
        assert r["gate_acc"][arm] == pytest.approx(sum(x["correct"] for x in rows) / len(rows))
    assert set(r["retain_kl"]) == {"A", "B", "C"} and r["n_failing"] >= 0
    assert seen and not set(seen) & set(gate)
    # Untrained generator writes U = 0, so arm C is the base model exactly.
    assert r["gate_acc"]["C"] == r["gate_acc"]["base"]
    assert json.loads((tmp_path / "result.json").read_text())["gate_acc"] == r["gate_acc"]
