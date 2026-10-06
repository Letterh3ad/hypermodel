import dataclasses

import pytest
import torch

from hypermodel.arithmetic import Difficulty, _exemplars, generate
from hypermodel.stream import (EPISODE_N, EPISODES, MAX_DROP, MAX_KL, OWN_GAIN, STREAM, Evaluator, episodes,
                               forgetting, judge, task_data)


def _sets(td):
    return {k: set(getattr(td, k)) for k in ("gate", "test", "val", "train")}


@pytest.mark.parametrize("name", STREAM)
def test_splits_are_disjoint_and_never_hold_an_exemplar(name):
    td = task_data(name)
    s = _sets(td)
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
    td = task_data("mul-2x3")
    eps = episodes(td)
    assert len(eps) == EPISODES and all(len(e) == EPISODE_N for e in eps)
    assert len(set().union(*map(set, eps))) == EPISODES * EPISODE_N
    assert set().union(*map(set, eps)) <= set(td.train)


def test_a_small_task_reuses_its_train_pool_in_every_episode():
    td = task_data("mod-2")  # 90 x 8 questions in all
    eps = episodes(td)
    assert len(eps) == EPISODES and len(td.gate) > 50 and len(td.test) > 100
    assert all(set(e) == set(td.train) for e in eps) and eps[0] != eps[1]


def test_judge_needs_one_point_on_its_own_task():
    assert judge("a", {"a": 0.30}, {"a": 0.30 + OWN_GAIN}).ok
    v = judge("a", {"a": 0.30}, {"a": 0.305})
    assert not v.ok and "own" in v.reasons[0]


def test_judge_allows_one_point_of_forgetting_and_no_more():
    before, earlier = {"a": 0.5, "b": 0.3}, ("a",)
    assert judge("b", before, {"a": 0.5 - MAX_DROP, "b": 0.32}, earlier=earlier).ok
    assert not judge("b", before, {"a": 0.48, "b": 0.32}, earlier=earlier).ok


def test_judge_caps_retain_drift():
    assert judge("a", {"a": 0.3}, {"a": 0.4}, worst_kl=MAX_KL).ok
    assert not judge("a", {"a": 0.3}, {"a": 0.4}, worst_kl=MAX_KL * 1.01).ok


def test_evaluator_holds_no_tensors_and_cannot_be_rewritten():
    ev = Evaluator([task_data(n) for n in STREAM])
    with pytest.raises(AttributeError):
        ev.gates = {}
    with pytest.raises(TypeError):
        ev.gates["mul-2x3"] = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.gates["mul-2x3"][0].answer = 0
    assert isinstance(ev.gates["mul-2x3"], tuple)
    assert not any(isinstance(v, torch.Tensor) for v in vars(type(ev)).values())
    assert not isinstance(ev, torch.nn.Module)


def test_forgetting_is_the_drop_from_each_tasks_best_earlier_score():
    # rows: after training task i; columns: accuracy on task j
    A = [[0.5, 0.1, 0.0], [0.4, 0.6, 0.0], [0.3, 0.5, 0.7]]
    f = forgetting(A)
    assert f[0] == pytest.approx(0.2) and f[1] == pytest.approx(0.1) and f[2] is None


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
        tasks.append(TaskData(name, "", gate=qs[:4], test=qs[4:10], val=qs[10:14], train=qs[14:]))
    r_train = retain.retain_items("train", 2, 2, arithmetic=RETAIN_ARITH)
    r_test = retain.retain_items("test", 2, 2, arithmetic=RETAIN_ARITH)
    rows = []
    res = run_naive(lm, tasks, r_train, r_test, [2, 4], batch_size=8, log=rows.append)
    assert len(res["matrix"]) == 2 and all(len(r) == 2 for r in res["matrix"])
    assert res["forgetting"][1] is None and res["retain_kl"]["worst"] >= 0
    assert sum("episode" in r for r in rows) == 2 * EPISODES
    assert all(torch.equal(v, lm.model.state_dict()[k]) for k, v in base.items())  # the base model never moves


def test_excluding_another_tasks_questions_leaves_a_small_task_whole():
    seen = frozenset(generate(Difficulty.parse("mul-2x3"), 2000, 0))
    assert task_data("mod-2", exclude=seen) == task_data("mod-2")
