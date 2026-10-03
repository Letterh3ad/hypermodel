import dataclasses

import pytest
import torch

from hypermodel.arithmetic import Difficulty, few_shot_prefix, generate
from hypermodel.edit_train import ConstantMixer, answer_batch, evaluate, train
from hypermodel.editor import LoRABank

PLANTED = 42


@pytest.fixture(scope="module")
def small_lm():
    from hypermodel.adapter import load
    return load("EleutherAI/pythia-14m", device="cpu")


def _planted(n, seed):
    """Questions whose 'answer' is always PLANTED: learnable, and not something the base model does."""
    return [dataclasses.replace(q, answer=PLANTED) for q in generate(Difficulty("add", 2), n, seed)]


@pytest.mark.slow
def test_plain_lora_learns_a_planted_answer_and_leaves_the_base_untouched(small_lm):
    torch.manual_seed(0)
    prefix = few_shot_prefix(Difficulty("add", 2))
    qs = _planted(120, seed=0)
    train_qs, val_qs, test_qs = qs[:80], qs[80:100], qs[100:]
    base_state = {k: v.clone() for k, v in small_lm.model.state_dict().items()}
    bank = LoRABank(small_lm, [2, 4], n_experts=2, rank=4)
    mixer = ConstantMixer(bank.shape)

    assert sum(r["correct"] for r in evaluate(small_lm, bank, mixer, test_qs, prefix)) == 0
    history = train(small_lm, bank, mixer, train_qs, val_qs, prefix, max_steps=150, batch_size=16,
                    lr=3e-3, eval_every=25, patience=4)

    acc = sum(r["correct"] for r in evaluate(small_lm, bank, mixer, test_qs, prefix)) / len(test_qs)
    assert acc >= 0.9, history
    for k, v in small_lm.model.state_dict().items():
        assert torch.equal(v, base_state[k]), k
    assert all(p.grad is None for p in small_lm.model.parameters())


@pytest.mark.slow
def test_only_the_answer_and_newline_carry_labels(small_lm):
    tok = small_lm.tokenizer
    qs = generate(Difficulty("mul", 2, 3), 3, seed=0)
    batch = answer_batch(tok, "1*2=2\n", qs, "cpu")
    for row, q in zip(batch["labels"], qs):
        assert tok.decode(row[row != -100]) == f"{q.answer}\n"


@pytest.mark.slow
def test_retain_term_limits_drift_while_the_planted_answer_is_still_learned(small_lm, monkeypatch):
    import hypermodel.retain as retain
    from hypermodel.edit_train import evaluate_retain

    monkeypatch.setattr(retain, "_load_wikitext", lambda split: (_ for _ in ()).throw(ConnectionError()))
    prefix = few_shot_prefix(Difficulty("add", 2))
    qs = _planted(100, seed=1)
    retain_train, retain_test = retain.retain_items("train", 12, 8), retain.retain_items("test", 12, 8)
    results = {}
    for beta in (0.0, 5.0):
        torch.manual_seed(0)
        bank = LoRABank(small_lm, [2, 4], n_experts=2, rank=4)
        mixer = ConstantMixer(bank.shape)
        if beta == 0.0:
            untrained = evaluate_retain(small_lm, bank, mixer, retain_test)
            assert set(untrained) == {"text", "add-4", "mod-3x1", "worst"}
            assert all(v == 0.0 for v in untrained.values())
        history = train(small_lm, bank, mixer, qs[:80], qs[80:], prefix, max_steps=100, batch_size=16, lr=3e-3,
                        eval_every=50, patience=2, retain=retain_train, beta=beta, gamma=0.0)
        assert {"kl", "norm"} <= history[-1].keys()
        acc = sum(r["correct"] for r in evaluate(small_lm, bank, mixer, qs[80:], prefix)) / 20
        results[beta] = (acc, evaluate_retain(small_lm, bank, mixer, retain_test)["worst"])
    assert results[5.0][0] >= 0.9
    assert results[5.0][1] < 0.5 * results[0.0][1], results


@pytest.mark.slow
def test_per_question_edits_survive_batched_scoring(small_lm):
    from hypermodel.condition import ConditionedMixer, FeatureStore, Standardize, question_features
    torch.manual_seed(0)
    prefix = few_shot_prefix(Difficulty("add", 2))
    qs = generate(Difficulty("add", 2), 7, seed=3)
    keys, X = question_features(qs, prefix)
    X = torch.as_tensor(X)
    bank = LoRABank(small_lm, [2, 4], n_experts=2, rank=4)
    with torch.no_grad():
        bank.U.normal_(std=0.5)
    mixer = ConditionedMixer(bank.shape, Standardize(X.mean(0), X.std(0) + 1), X.shape[1], FeatureStore(keys, X),
                             key=lambda q: prefix + q.prompt)
    with torch.no_grad():
        mixer.head[-1].weight.normal_(std=3.0)
    batched = evaluate(small_lm, bank, mixer, qs, prefix, batch_size=3)
    single = evaluate(small_lm, bank, mixer, qs, prefix, batch_size=1)
    assert [r["pred"] for r in batched] == [r["pred"] for r in single]
    assert len({r["pred"] for r in single}) > 1  # the edits really differ per question


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["none", "features", "z-frozen", "z-finetune"])
def test_every_conditioning_kind_builds_and_trains(small_lm, tmp_path, monkeypatch, kind):
    import hypermodel.retain as retain
    from hypermodel.edit_train import build_mixer, evaluate_retain, split_questions
    from hypermodel.observer import Observer
    from hypermodel.trace import TraceSet, record

    monkeypatch.setattr(retain, "_load_wikitext", lambda split: (_ for _ in ()).throw(ConnectionError()))
    d = Difficulty("mul", 2, digits_b=3)
    prefix = few_shot_prefix(d)
    ts = TraceSet.load(record(small_lm, generate(d, 40, seed=0), prefix, [1, 3], tmp_path / "t", batch_size=20))
    obs_path = tmp_path / "obs.pt"
    Observer(in_dim=2 * 3 * small_lm.d_model, k=8, hidden=16).save(obs_path)
    qs = split_questions(ts.records, ts.split)
    r_train, r_test = retain.retain_items("train", 4, 3), retain.retain_items("test", 4, 3)
    torch.manual_seed(0)
    bank = LoRABank(small_lm, [2, 4], n_experts=2, rank=4)
    mixer = build_mixer(kind, bank, small_lm, ts, r_train + r_test, qs["train"] + r_train, prefix, obs_path)
    enc = getattr(mixer, "encoder", None)
    before = {k: v.clone() for k, v in enc.state_dict().items()} if enc is not None else {}
    train(small_lm, bank, mixer, qs["train"], qs["val"], prefix, max_steps=4, batch_size=4, lr=1e-2, eval_every=4,
          retain=r_train, gamma=0.1)
    assert evaluate(small_lm, bank, mixer, qs["test"], prefix, batch_size=3)
    assert set(evaluate_retain(small_lm, bank, mixer, r_test)) >= {"text", "add-4", "mod-3x1", "worst"}
    if kind.startswith("z"):
        moved = any(not torch.equal(v, before[k]) for k, v in enc.state_dict().items() if "encoder" in k)
        assert moved is (kind == "z-finetune")
