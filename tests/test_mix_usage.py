import json

import pytest
import torch

from hypermodel.mix_usage import mixing_stats


def test_one_shared_mix_has_no_spread_and_uses_every_expert():
    s = mixing_stats(torch.ones(10, 2, 4))
    assert s["spread"] == pytest.approx(0) and s["effective_experts"] == pytest.approx(4)


def test_one_expert_per_question_shows_spread_and_one_effective_expert():
    g = torch.zeros(4, 1, 4)
    g[torch.arange(4), 0, torch.arange(4)] = 1
    s = mixing_stats(g)
    assert s["effective_experts"] == pytest.approx(1) and s["spread"] > 1


@pytest.mark.slow
def test_a_saved_conditioned_run_is_rebuilt_without_the_lm(tmp_path, monkeypatch):
    from hypermodel.adapter import load
    from hypermodel.arithmetic import Difficulty, few_shot_prefix, generate
    import hypermodel.retain as retain
    from hypermodel.edit_train import build_mixer, split_questions
    from hypermodel.editor import LoRABank
    from hypermodel.mix_usage import mixing_on_test
    from hypermodel.observer import Observer
    from hypermodel.trace import TraceSet, record

    monkeypatch.setattr(retain, "_load_wikitext", lambda split: (_ for _ in ()).throw(ConnectionError()))
    lm = load("EleutherAI/pythia-14m", device="cpu")
    d = Difficulty("mul", 2, digits_b=3)
    prefix = few_shot_prefix(d)
    ts = TraceSet.load(record(lm, generate(d, 40, seed=0), prefix, [1, 3], tmp_path / "t", batch_size=20))
    Observer(in_dim=2 * 3 * lm.d_model, k=8, hidden=16).save(tmp_path / "obs.pt")
    qs = split_questions(ts.records, ts.split)
    r = retain.retain_items("train", 4, 3)
    bank = LoRABank(lm, [2, 4], n_experts=2, rank=4)
    mixer = build_mixer("z-gain", bank, lm, ts, r, qs["train"] + r, prefix, tmp_path / "obs.pt")
    with torch.no_grad():
        mixer.head[-1].weight.normal_()
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({"conditioning": "z-gain", "seed": 0}))
    torch.save({"bank": bank.state_dict(), "mixer": mixer.state_dict()}, run / "editor-s0.pt")
    want = mixer.eval()(qs["test"])
    assert torch.allclose(mixing_on_test(run, ts), want, atol=1e-4)
