import pytest
import torch

from hypermodel.adapter import ModelAdapter
from hypermodel.editor import LoRABank
from hypermodel.retain import retain_kl

LAYERS = [0, 2]


def _batch():
    ids = torch.randint(0, 64, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": mask,
            "position_ids": torch.arange(6).expand(2, -1), "kl_mask": mask.bool()}


def test_retain_kl_is_zero_for_an_untrained_bank(tiny_model):
    bank = LoRABank(ModelAdapter(tiny_model), LAYERS, n_experts=2, rank=2)
    kl = retain_kl(ModelAdapter(tiny_model), bank, torch.ones(bank.shape), _batch(), ["text"] * 2)
    assert kl.item() == 0.0


def test_retain_kl_rises_when_an_edit_is_forced(tiny_model):
    adapter = ModelAdapter(tiny_model)
    bank = LoRABank(adapter, LAYERS, n_experts=2, rank=2)
    with torch.no_grad():
        bank.U.normal_()
    assert retain_kl(adapter, bank, torch.ones(bank.shape), _batch(), ["text"] * 2).item() > 1e-3


def test_left_padding_does_not_change_a_rows_kl(tiny_model):
    adapter = ModelAdapter(tiny_model)
    bank = LoRABank(adapter, LAYERS, n_experts=2, rank=2)
    with torch.no_grad():
        bank.U.normal_()
    ids = torch.randint(1, 64, (1, 4))
    alone = {"input_ids": ids, "attention_mask": torch.ones(1, 4, dtype=torch.long),
             "position_ids": torch.arange(4)[None], "kl_mask": torch.ones(1, 4, dtype=torch.bool)}
    pad = torch.tensor([[0, 0]])
    padded = {"input_ids": torch.cat([pad, ids], 1), "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1]]),
              "position_ids": torch.tensor([[0, 0, 0, 1, 2, 3]]),
              "kl_mask": torch.tensor([[False, False, True, True, True, True]])}  # as tokenize_regions builds it
    with torch.no_grad():
        a = retain_kl(adapter, bank, torch.ones(bank.shape), alone, ["text"])
        b = retain_kl(adapter, bank, torch.ones(bank.shape), padded, ["text"])
    assert torch.allclose(a, b, atol=1e-5)


def test_edit_norm_is_zero_untrained_and_positive_when_forced(tiny_model):
    adapter = ModelAdapter(tiny_model)
    bank = LoRABank(adapter, LAYERS, n_experts=2, rank=2)
    ids = torch.randint(0, 64, (2, 5))
    with bank.apply(torch.ones(bank.shape)) as edits:
        tiny_model(ids)
    assert edits.norm().item() == 0.0
    with torch.no_grad():
        bank.U.normal_()
    with bank.apply(torch.ones(bank.shape)) as edits:
        tiny_model(ids)
    assert edits.norm().item() > 0


def _offline(monkeypatch):
    import hypermodel.retain as retain

    def fail(*_a, **_k):
        raise ConnectionError("offline")
    monkeypatch.setattr(retain, "_load_wikitext", fail)


def test_retain_items_mix_text_and_other_arithmetic_with_disjoint_splits(monkeypatch):
    from hypermodel.retain import retain_items
    _offline(monkeypatch)
    train, test = retain_items("train", 6, 4, seed=0), retain_items("test", 6, 4, seed=0)
    for items in (train, test):
        sources = [it.source for it in items]
        assert sources.count("text") == 6 and sources.count("add-4") == 4 and sources.count("mod-3x1") == 4
    assert not {it.text for it in train} & {it.text for it in test}


def test_arithmetic_items_score_only_the_question_and_answer(monkeypatch):
    from hypermodel.arithmetic import Difficulty, few_shot_prefix
    from hypermodel.retain import retain_items
    _offline(monkeypatch)
    for it in retain_items("train", 1, 3, seed=0):
        if it.source == "text":
            assert it.start == 0
        else:
            prefix = few_shot_prefix(Difficulty.parse(it.source))
            assert it.text.startswith(prefix) and it.start == len(prefix)
            assert it.text.endswith("\n") and "=" in it.text[it.start:]


@pytest.fixture(scope="module")
def small_lm():
    from hypermodel.adapter import load
    return load("EleutherAI/pythia-14m", device="cpu")


@pytest.mark.slow
def test_kl_mask_covers_exactly_each_items_region(small_lm, monkeypatch):
    from hypermodel.retain import retain_batch, retain_items
    _offline(monkeypatch)
    items = retain_items("test", 2, 2, seed=0)
    batch = retain_batch(small_lm.tokenizer, items, "cpu")
    for ids, mask, it in zip(batch["input_ids"], batch["kl_mask"], items):
        assert small_lm.tokenizer.decode(ids[mask]) == it.text[it.start:]


def test_source_balanced_kl_weights_each_source_equally_not_each_token():
    from hypermodel.retain import source_balanced
    kl = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0, 5.0])
    rows = torch.tensor([0, 0, 0, 0, 1, 2])
    # text rows 0 and 2 pool to (1+1+1+1+5)/5 = 1.8; add-4 row 1 is 3.0; balanced = (1.8 + 3.0) / 2
    assert source_balanced(kl, rows, ["text", "add-4", "text"]).item() == pytest.approx(2.4)


def test_retain_splits_stay_disjoint_when_their_sizes_differ(monkeypatch):
    from hypermodel.retain import retain_items
    _offline(monkeypatch)
    train, test = retain_items("train", 2, 500, seed=0), retain_items("test", 2, 200, seed=0)
    for source in ("add-4", "mod-3x1"):
        a = {it.text for it in train if it.source == source}
        b = {it.text for it in test if it.source == source}
        assert len(a) == 500 and len(b) == 200 and not a & b


def test_retain_test_set_is_disjoint_from_every_seeds_training_set(monkeypatch):
    from hypermodel.retain import retain_items
    _offline(monkeypatch)
    test = {it.text for it in retain_items("test", 0, 200, seed=0)}
    for seed in (0, 1, 2):
        assert not test & {it.text for it in retain_items("train", 0, 500, seed=seed)}
