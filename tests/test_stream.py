import dataclasses
from types import SimpleNamespace

import pytest
import torch

from hypermodel.arithmetic import Difficulty, _exemplars, generate
import hypermodel.stream as stream
from hypermodel.stream import (EPISODE_N, EPISODES, MAX_DROP, MAX_KL, OWN_GAIN, STREAM, Evaluator, TaskData,
                               drawdown, episode_gate, episodes, forgetting, judge, task_data, transfer_drops)


def _sets(split):
    td, gate = split
    return {"gate": set(gate), **{k: set(getattr(td, k)) for k in ("test", "val", "train")}}


def _stream():
    return [task_data(n) for n in STREAM]


@pytest.mark.parametrize("name", STREAM)
def test_splits_are_disjoint_and_never_hold_an_exemplar(name):
    s = _sets(task_data(name))
    for a in s:
        for b in s:
            assert a == b or not s[a] & s[b], (a, b)
    assert not set().union(*s.values()) & set(_exemplars(Difficulty.parse(name)))


def test_splits_do_not_move_between_calls():
    assert task_data("mul-2x3") == task_data("mul-2x3")


def test_excluded_questions_never_reach_any_split():
    seen = frozenset(generate(Difficulty.parse("mul-2x3"), 500, 0))
    assert not set().union(*_sets(task_data("mul-2x3", exclude=seen)).values()) & seen


def test_a_large_task_gets_fresh_episodes():
    td, _ = task_data("mul-2x3")
    eps = episodes(td)
    assert len(eps) == EPISODES and all(len(e) == EPISODE_N for e in eps)
    assert len(set().union(*map(set, eps))) == EPISODES * EPISODE_N
    assert set().union(*map(set, eps)) <= set(td.train)


def test_a_small_task_cycles_its_train_pool_to_a_full_episode():
    td, gate = task_data("mod-2")  # 90 x 8 questions in all
    eps = episodes(td)
    assert len(eps) == EPISODES and all(len(e) == EPISODE_N for e in eps) and eps[0] != eps[1]
    for e in eps:
        counts = {q: e.count(q) for q in td.train}
        assert set(e) == set(td.train) and max(counts.values()) - min(counts.values()) <= 1
    assert episodes(td) == eps


def test_every_task_gets_the_same_step_budget():
    assert {len(e) for name in STREAM for e in episodes(task_data(name)[0])} == {EPISODE_N}


def test_mod_2_cut_favours_gate_and_test():
    td, gate = task_data("mod-2")
    assert (len(gate), len(td.test), len(td.val), len(td.train)) == (179, 250, 71, 216)


def test_large_tasks_keep_the_fixed_sizes():
    td, gate = task_data("mul-3")
    assert (len(gate), len(td.test), len(td.val)) == (500, 1000, 250)


def test_thresholds_are_the_spec_values():
    assert (OWN_GAIN, MAX_DROP, MAX_KL) == (0.01, 0.01, 0.05)


def test_episode_gate_needs_one_point_on_its_own_task():
    assert episode_gate("a", {"a": 0.30}, {"a": 0.31}).ok
    v = episode_gate("a", {"a": 0.30}, {"a": 0.305})
    assert not v.ok and "own" in v.reasons[0]


def test_judge_needs_one_point_on_its_own_task():
    assert judge("a", {"a": 0.30}, {"a": 0.31}, 0.0, stream=("a",)).ok
    assert not judge("a", {"a": 0.30}, {"a": 0.305}, 0.0, stream=("a",)).ok


def test_judge_allows_one_point_of_forgetting_on_every_earlier_task_and_no_more():
    before, s = {"a": 0.5, "b": 0.3, "c": 0.2}, ("a", "b", "c")
    assert judge("c", before, {"a": 0.49, "b": 0.29, "c": 0.22}, 0.0, stream=s).ok
    assert not judge("c", before, {"a": 0.5, "b": 0.28, "c": 0.22}, 0.0, stream=s).ok
    assert not judge("c", before, {"a": 0.48, "b": 0.3, "c": 0.22}, 0.0, stream=s).ok


def test_judge_caps_retain_drift():
    assert judge("a", {"a": 0.3}, {"a": 0.4}, 0.05, stream=("a",)).ok
    assert not judge("a", {"a": 0.3}, {"a": 0.4}, 0.0505, stream=("a",)).ok


def test_judge_fails_loudly_without_kl_or_earlier_scores():
    with pytest.raises(TypeError):
        judge("a", {"a": 0.3}, {"a": 0.4}, None, stream=("a",))
    with pytest.raises(TypeError):
        judge("a", {"a": 0.3}, {"a": 0.4})
    with pytest.raises(KeyError):
        judge("mod-4x1", {"mod-4x1": 0.3}, {"mod-4x1": 0.4}, 0.0)  # mul-2x3 comes first in the stream


def test_trainers_get_task_data_without_gate_sets():
    assert "gate" not in {f.name for f in dataclasses.fields(TaskData)}


def _questions(x):
    if isinstance(x, (list, tuple, set, frozenset)):
        return {q for v in x for q in _questions(v)}
    if isinstance(x, dict):
        return _questions(list(x.values()))
    if isinstance(x, TaskData):
        return _questions([x.test, x.val, x.train])
    return {x} if isinstance(x, stream.Question) else set()


def test_no_gate_question_reaches_training_or_evaluation(monkeypatch):
    import hypermodel.edit_train as et
    import hypermodel.editor as editor

    splits = _stream()
    gates = set().union(*(set(g) for _, g in splits))
    seen = set()

    def spy(ret):
        def f(*a, **k):
            seen.update(_questions([a, k]))
            return ret
        return f

    class Fake:
        shape = (1,)

        def __init__(self, *a, **k):
            seen.update(_questions([a, k]))

        def to(self, *_):
            return self

        def eval(self):
            pass

        def __call__(self, qs):
            seen.update(_questions(qs))

        def state_dict(self):
            return {}

    monkeypatch.setattr(editor, "LoRABank", Fake)
    monkeypatch.setattr(et, "ConstantMixer", Fake)
    monkeypatch.setattr(et, "train", spy([]))
    monkeypatch.setattr(et, "evaluate", spy([]))
    monkeypatch.setattr(et, "_accuracy", lambda r: 0.0)
    monkeypatch.setattr(et, "evaluate_retain", spy({"worst": 0.0}))
    stream.run_naive(SimpleNamespace(device="cpu"), [td for td, _ in splits], [], [], [0], log=lambda r: None)
    assert seen and not seen & gates


def test_evaluator_holds_only_questions_and_cannot_be_rewritten():
    splits = _stream()
    ev = Evaluator([td for td, _ in splits], {td.name: g for td, g in splits})
    with pytest.raises(AttributeError):
        ev.gates = {}
    with pytest.raises(TypeError):
        ev.gates["mul-2x3"] = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.gates["mul-2x3"][0].answer = 0
    assert not isinstance(ev, torch.nn.Module)
    for slot in Evaluator.__slots__:
        for v in getattr(ev, slot).values():
            assert isinstance(v, (str, tuple)) and not isinstance(v, torch.Tensor)
            assert isinstance(v, str) or all(isinstance(q, stream.Question) for q in v)
    assert {n: set(ev.gates[n]) for n in STREAM} == {td.name: set(g) for td, g in splits}


def test_forgetting_is_the_drop_from_each_tasks_best_earlier_score():
    # rows: after training task i; columns: accuracy on task j
    A = [[0.5, 0.1, 0.0], [0.4, 0.6, 0.0], [0.3, 0.5, 0.7]]
    f = forgetting(A)
    assert f[0] == pytest.approx(0.2) and f[1] == pytest.approx(0.1) and f[2] is None


def test_drawdown_sees_a_drop_that_later_training_repairs():
    A = [[0.42, 0.3, 0.1], [0.37, 0.6, 0.1], [0.41, 0.55, 0.7]]
    assert forgetting(A)[0] == pytest.approx(0.01)
    assert drawdown(A) == pytest.approx([0.05, 0.05, 0.0])
    d = transfer_drops(A)
    assert d[0] == [] and d[1] == pytest.approx([0.05]) and d[2] == pytest.approx([-0.04, 0.05])


def test_stream_retain_mix_is_add_4_mul_1x4_and_text():
    from hypermodel.retain import retain_items
    from hypermodel.stream import RETAIN_ARITH
    items = retain_items("train", 3, 2, arithmetic=RETAIN_ARITH)
    assert sorted({it.source for it in items}) == ["add-4", "mul-1x4", "text"]


@pytest.mark.slow
def test_naive_baseline_trains_through_the_stream_and_reports_forgetting(monkeypatch):
    from hypermodel.adapter import load
    import hypermodel.retain as retain
    from hypermodel.stream import RETAIN_ARITH, TaskData, run_naive

    monkeypatch.setattr(retain, "_load_wikitext", lambda split: (_ for _ in ()).throw(ConnectionError()))
    lm = load("EleutherAI/pythia-14m", device="cpu")
    base = {k: v.clone() for k, v in lm.model.state_dict().items()}
    tasks = []
    for name in ("add-1", "mul-1"):
        d = Difficulty.parse(name)
        qs = tuple(generate(d, 30, 0))
        tasks.append(TaskData(name, "", test=qs[4:10], val=qs[10:14], train=qs[14:]))
    r_train = retain.retain_items("train", 2, 2, arithmetic=RETAIN_ARITH)
    r_test = retain.retain_items("test", 2, 2, arithmetic=RETAIN_ARITH)
    monkeypatch.setattr(stream, "EPISODE_N", 16)
    rows = []
    res = run_naive(lm, tasks, r_train, r_test, [2, 4], batch_size=8, log=rows.append)
    assert len(res["matrix"]) == 2 and all(len(r) == 2 for r in res["matrix"])
    assert res["forgetting"][1] is None and res["retain_kl"]["worst"] >= 0
    assert len(res["drawdown"]) == 2 and len(res["transfer_drops"]) == 2
    assert sum("episode" in r for r in rows) == 2 * EPISODES
    assert all(torch.equal(v, lm.model.state_dict()[k]) for k, v in base.items())  # the base model never moves


def test_excluding_another_tasks_questions_leaves_a_small_task_whole():
    seen = frozenset(generate(Difficulty.parse("mul-2x3"), 2000, 0))
    assert task_data("mod-2", exclude=seen) == task_data("mod-2")


def test_cli_exposes_beta_and_gamma_with_unchanged_defaults():
    p = stream.parser()
    a = p.parse_args(["--out", "x"])
    assert (a.beta, a.gamma) == (1.0, 0.1)
    assert (p.parse_args(["--out", "x", "--beta", "0", "--gamma", "0"]).beta, ) == (0.0,)


def test_dtype_check_scores_both_dtypes_and_applies_the_one_point_rule(monkeypatch):
    import hypermodel.adapter as adapter
    import hypermodel.scoring as scoring

    acc = {torch.float32: 0.335, torch.bfloat16: 0.3355}
    monkeypatch.setattr(adapter, "load", lambda name, device, dtype: dtype)
    monkeypatch.setattr(scoring, "score", lambda m, qs, prefix, bs=64: [{"correct": i < acc[m] * len(qs)}
                                                                         for i in range(len(qs))])
    r = stream.dtype_check("m", "cpu", 2000)
    assert r["fp32"] == pytest.approx(0.335) and r["bf16"] == pytest.approx(0.3355) and r["bf16_ok"]


@pytest.mark.parametrize("before, after, kl", [
    ({t: 0.5 for t in STREAM}, {**{t: 0.5 for t in STREAM}, "mul-3": 0.6}, float("nan")),
    ({t: 0.5 for t in STREAM}, {**{t: 0.5 for t in STREAM}, "mul-3": 0.6, "mul-2x3": float("nan")}, 0.0),
])
def test_judge_rejects_non_finite_inputs(before, after, kl):
    with pytest.raises(ValueError):
        judge("mul-3", before, after, kl)


def test_episode_gate_rejects_non_finite_scores():
    with pytest.raises(ValueError):
        episode_gate("mul-3", {"mul-3": 0.5}, {"mul-3": float("nan")})


@pytest.mark.parametrize("n", ["0", "-5"])
def test_dtype_check_needs_a_positive_count(n):
    with pytest.raises(SystemExit):
        stream.parser().parse_args(["--dtype-check", n])
