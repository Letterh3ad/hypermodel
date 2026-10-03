import torch

from hypermodel.adapter import ModelAdapter
from hypermodel.editor import LoRABank

LAYERS = [0, 2]


def _logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def test_untrained_bank_is_the_identity(tiny_model):
    bank = LoRABank(ModelAdapter(tiny_model), LAYERS, n_experts=3, rank=2)
    ids = torch.randint(0, 64, (2, 5))
    g = torch.randn(2, len(LAYERS), 3)
    with bank.apply(g):
        edited = _logits(tiny_model, ids)
    assert torch.equal(edited, _logits(tiny_model, ids))


def _randomised(tiny_model, n_experts=3, rank=2):
    bank = LoRABank(ModelAdapter(tiny_model), LAYERS, n_experts=n_experts, rank=rank, alpha=4.0)
    with torch.no_grad():
        bank.U.normal_()
    return bank


def test_writer_output_follows_the_edit_formula(tiny_model):
    bank = _randomised(tiny_model)
    writer = ModelAdapter(tiny_model).residual_writers()[2]
    h = torch.randn(2, 5, writer.in_features)
    g = torch.randn(2, len(LAYERS), 3)
    with torch.no_grad(), bank.apply(g):
        edited = writer(h)
        plain = torch.nn.functional.linear(h, writer.weight, writer.bias)
    expected = plain.clone()
    for b in range(2):
        for e in range(3):
            # LAYERS[1] is layer 2; scale = alpha / rank = 2
            expected[b] += g[b, 1, e] * 2.0 * (h[b] @ bank.V[1, e]) @ bank.U[1, e].T
    assert torch.allclose(edited, expected, atol=1e-5)


def test_shared_mixing_matches_per_example_copies(tiny_model):
    bank = _randomised(tiny_model)
    ids = torch.randint(0, 64, (2, 5))
    g = torch.randn(len(LAYERS), 3)
    with bank.apply(g):
        shared = _logits(tiny_model, ids)
    with bank.apply(g.expand(2, -1, -1)):
        per_example = _logits(tiny_model, ids)
    assert torch.allclose(shared, per_example, atol=1e-6)


def test_hooks_are_removed_on_exit(tiny_model):
    bank = _randomised(tiny_model)
    ids = torch.randint(0, 64, (2, 5))
    before = _logits(tiny_model, ids)
    with bank.apply(torch.ones(len(LAYERS), 3)):
        assert not torch.allclose(_logits(tiny_model, ids), before)
    assert torch.equal(_logits(tiny_model, ids), before)
