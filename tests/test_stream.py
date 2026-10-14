import dataclasses
from types import SimpleNamespace

import pytest
import torch

from hypermodel.arithmetic import Difficulty, _exemplars, few_shot_prefix, generate
import hypermodel.stream as stream
from hypermodel.stream import (EPISODE_N, EPISODES, MAX_DROP, MAX_KL, METHODS, OWN_GAIN, REPLAY_N, STREAM, Evaluator,
                               TaskData, drawdown, episode_gate, episodes, forgetting, joint_episodes, judge,
                               replay_buffer, task_data, transfer_drops)


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


class _Fake:
    shape = (1,)

    def __init__(self, *a, **k):
        _Fake.seen.update(_questions([a, k]))

    def to(self, *_):
        return self

    def eval(self):
        pass

    def __call__(self, qs):
        _Fake.seen.update(_questions(qs))

    def state_dict(self):
        return {}


def _harness(monkeypatch, method, splits=None):
    """run_stream on fake banks: every train call, every scored (bank, questions) and every question touched."""
    import hypermodel.edit_train as et
    import hypermodel.editor as editor

    _Fake.seen = set()
    calls, scored = [], []

    def train(adapter, bank, mixer, train_qs, val_qs, prefix, **k):
        _Fake.seen.update(_questions([train_qs, val_qs, k.get("replay")]))
        calls.append({"bank": bank, "qs": train_qs, "val": val_qs, "prefix": prefix, **k})
        return []

    def evaluate(adapter, bank, mixer, qs, prefix, *a):
        _Fake.seen.update(_questions(qs))
        scored.append((bank, tuple(qs)))
        return []

    def retain(adapter, bank, mixer, items):
        _Fake.seen.update(_questions(items))
        return {"text": 0.0, "worst": 0.0}

    monkeypatch.setattr(editor, "LoRABank", _Fake)
    monkeypatch.setattr(et, "ConstantMixer", _Fake)
    monkeypatch.setattr(et, "train", train)
    monkeypatch.setattr(et, "evaluate", evaluate)
    monkeypatch.setattr(et, "_accuracy", lambda r: 0.0)
    monkeypatch.setattr(et, "evaluate_retain", retain)
    splits = splits or _stream()
    res = stream.run_stream(SimpleNamespace(device="cpu"), [td for td, _ in splits], [], [], [0], method=method,
                            log=lambda r: None)
    return SimpleNamespace(calls=calls, scored=scored, seen=_Fake.seen, res=res, splits=splits)


@pytest.mark.parametrize("method", METHODS)
def test_no_gate_question_reaches_training_or_evaluation(monkeypatch, method):
    h = _harness(monkeypatch, method)
    gates = set().union(*(set(g) for _, g in h.splits))
    assert h.seen and not h.seen & gates


def test_every_method_gets_the_same_step_budget(monkeypatch):
    budgets = {}
    for m in METHODS:
        calls = _harness(monkeypatch, m).calls
        budgets[m] = (len(calls), sum(c["max_steps"] for c in calls), {c["batch_size"] for c in calls})
    assert len(set(map(str, budgets.values()))) == 1, budgets
    n, steps, sizes = budgets["naive"]
    assert n == len(STREAM) * EPISODES and steps == n * (EPISODE_N // 16) and sizes == {16}


def test_joint_mixes_every_task_in_equal_shares():
    splits = _stream()
    tasks = [td for td, _ in splits]
    rows, train = joint_episodes(tasks, 0), [set(t.train) for t in tasks]
    assert len(rows) == len(tasks) and all(len(r) == EPISODES for r in rows)
    for chunk in (c for r in rows for c in r):
        assert len(chunk) == EPISODE_N
        assert [sum(q in tr for q in chunk) for tr in train] == [EPISODE_N // len(tasks)] * len(tasks)
    for t, tr in zip(tasks, train):
        used = [q for r in rows for c in r for q in c if q in tr]
        assert sorted(map(repr, used)) == sorted(repr(q) for e in episodes(t) for q in e)
    assert joint_episodes(tasks, 0) == rows and joint_episodes(tasks, 1) != rows


def test_joint_trains_one_lora_on_mixed_chunks(monkeypatch):
    h = _harness(monkeypatch, "joint")
    assert len({id(c["bank"]) for c in h.calls}) == 1
    tasks = [td for td, _ in h.splits]
    train = [set(t.train) for t in tasks]
    for c in h.calls:
        assert all(any(q in tr for q in c["qs"]) for tr in train)
        assert all(c["prefix"][q] == t.prefix for t, tr in zip(tasks, train) for q in c["qs"] if q in tr)


def test_oracle_routes_by_the_true_task_id_only(monkeypatch):
    h = _harness(monkeypatch, "oracle")
    tasks = [td for td, _ in h.splits]
    owner = {}
    for c in h.calls:
        (t,) = [t for t in tasks if set(c["qs"]) <= set(t.train)]
        assert owner.setdefault(t.name, id(c["bank"])) == id(c["bank"])
        assert not c.get("replay")
    assert len(set(owner.values())) == len(tasks)
    for bank, qs in h.scored:
        (t,) = [t for t in tasks if set(qs) <= set(t.test)] or [None]
        if t:
            assert id(bank) == owner[t.name]


def test_replay_buffer_holds_only_earlier_tasks_train_questions(monkeypatch):
    h = _harness(monkeypatch, "replay")
    splits = h.splits
    for i, (td, _) in enumerate(splits):
        earlier = set().union(*(set(t.train) for t, _ in splits[:i]))
        held = set().union(*(set(g) | set(t.test) | set(t.val) for t, g in splits))
        for c in h.calls[i * EPISODES:(i + 1) * EPISODES]:
            buf = c["replay"] or []
            assert len(buf) == REPLAY_N * i and set(buf) <= earlier and not set(buf) & held
            assert c["replay_per_batch"] == 4 and set(c["qs"]) <= set(td.train)
    assert len({id(c["bank"]) for c in h.calls}) == 1


def test_replay_buffer_is_a_fixed_draw_of_the_tasks_train_questions():
    td, gate = task_data("mod-2")
    buf = replay_buffer(td, 0)
    assert len(buf) == REPLAY_N == len(set(buf)) and set(buf) <= set(td.train)
    assert replay_buffer(td, 0) == buf and replay_buffer(td, 1) != buf


def test_unknown_method_is_refused(monkeypatch):
    with pytest.raises(ValueError):
        _harness(monkeypatch, "sleep")


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
@pytest.mark.parametrize("method", METHODS)
def test_every_method_trains_through_the_stream_and_reports_forgetting(monkeypatch, method):
    from hypermodel.adapter import load
    import hypermodel.retain as retain
    from hypermodel.stream import RETAIN_ARITH, TaskData, run_stream

    monkeypatch.setattr(retain, "_load_wikitext", lambda split: (_ for _ in ()).throw(ConnectionError()))
    lm = load("EleutherAI/pythia-14m", device="cpu")
    base = {k: v.clone() for k, v in lm.model.state_dict().items()}
    tasks = []
    for name in ("add-1", "mul-1"):
        d = Difficulty.parse(name)
        qs = tuple(generate(d, 30, 0))
        tasks.append(TaskData(name, few_shot_prefix(d), test=qs[4:10], val=qs[10:14], train=qs[14:]))
    r_train = retain.retain_items("train", 2, 2, arithmetic=RETAIN_ARITH)
    r_test = retain.retain_items("test", 2, 2, arithmetic=RETAIN_ARITH)
    monkeypatch.setattr(stream, "EPISODE_N", 16)
    monkeypatch.setattr(stream, "REPLAY_N", 4)
    rows = []
    res = run_stream(lm, tasks, r_train, r_test, [2, 4], method=method, batch_size=8, log=rows.append)
    assert len(res["matrix"]) == 2 and all(len(r) == 2 for r in res["matrix"])
    assert res["forgetting"][1] is None and res["retain_kl"]["worst"] >= 0
    assert len(res["drawdown"]) == 2 and len(res["transfer_drops"]) == 2
    assert sum("episode" in r for r in rows) == 2 * EPISODES
    recs = res["test_records"]
    assert [sum(r["task"] == t.name for r in recs) for t in tasks] == [6, 6]
    for j, t in enumerate(tasks):
        mine = [r["correct"] for r in recs if r["task"] == t.name]
        assert sum(mine) / len(mine) == pytest.approx(res["matrix"][-1][j])
    experts = stream.load_experts(lm, [2, 4], [t.name for t in tasks], method, res["state"])
    again = stream.score_tasks(lm, experts, tasks)
    assert [r["correct"] for r in again] == [r["correct"] for r in recs]
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


def test_evaluator_records_are_per_gate_question_and_accuracy_agrees(monkeypatch):
    import hypermodel.edit_train as et
    td, gate = task_data("mod-2")
    ev = Evaluator([td], {td.name: gate})
    monkeypatch.setattr(et, "evaluate", lambda a, b, m, qs, p: [{"prompt": q.prompt, "correct": q.a % 2 == 0}
                                                                  for q in qs])
    recs = ev.records(None, None, None, "mod-2")
    assert [r["prompt"] for r in recs] == [q.prompt for q in gate]
    assert ev.accuracy(None, None, None)["mod-2"] == pytest.approx(sum(r["correct"] for r in recs) / len(recs))


def test_cli_method_defaults_to_naive():
    p = stream.parser()
    assert p.parse_args(["--out", "x"]).method == "naive"
    assert p.parse_args(["--out", "x", "--method", "oracle"]).method == "oracle"
    with pytest.raises(SystemExit):
        p.parse_args(["--out", "x", "--method", "sleep"])


def _fake_run(root, name, seed, final, correct, kl=0.01):
    import json
    d = root / f"{name}-s{seed}"
    d.mkdir()
    res = {"tasks": ["a", "b"], "matrix": [[0.0, 0.0], final], "final_avg": sum(final) / 2,
           "drawdown": [0.1, 0.0], "retain_kl": {"text": kl, "worst": kl}}
    (d / "result.json").write_text(json.dumps(res))
    (d / "test_records.jsonl").write_text("".join(
        json.dumps({"task": t, "prompt": f"{t}{i}", "correct": c}) + "\n"
        for t, cs in correct.items() for i, c in enumerate(cs)))
    return d


def test_table_places_each_method_beside_the_baseline(tmp_path):
    for s in range(2):
        _fake_run(tmp_path, "base", s, [0.5, 0.5], {"a": [1, 0], "b": [1, 0, 0, 1]})
        _fake_run(tmp_path, "up", s, [1.0, 0.75], {"a": [1, 1], "b": [1, 1, 0, 1]}, kl=0.02 + s)
    rows = stream.table([tmp_path / "base", tmp_path / "up"])
    assert [r["name"] for r in rows] == ["base", "up"] and [r["seeds"] for r in rows] == [2, 2]
    base, up = rows
    assert base["final_avg"] == pytest.approx(0.5) and base["vs_base"]["diff"] == 0
    assert up["final"] == pytest.approx([1.0, 0.75]) and up["worst_kl"] == pytest.approx(1.02)
    assert up["vs_base"]["diff"] == pytest.approx(0.375)  # task means: (0.5 + 0.25) / 2, not pooled 3/6
    assert up["max_drawdown"] == pytest.approx(0.1)
    text = stream.format_table(rows)
    assert "base" in text and "up" in text and "+37.5" in text


def test_table_reports_a_method_without_records_without_a_ci(tmp_path):
    _fake_run(tmp_path, "base", 0, [0.5, 0.5], {"a": [1, 0], "b": [1, 0]})
    d = _fake_run(tmp_path, "old", 0, [0.5, 0.5], {"a": [1, 0], "b": [1, 0]})
    (d / "test_records.jsonl").unlink()
    rows = stream.table([tmp_path / "base", tmp_path / "old"])
    assert rows[1]["vs_base"] is None and "n/a" in stream.format_table(rows)
