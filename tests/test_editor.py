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


def test_an_untrained_router_is_plain_lora_with_g0(tiny_model):
    from hypermodel.condition import RouterMixer
    ids = torch.randint(0, 64, (2, 5))
    unedited = _logits(tiny_model, ids)
    bank = LoRABank(ModelAdapter(tiny_model), LAYERS, n_experts=3, rank=2)
    with bank.apply(RouterMixer(bank.shape, bank.d_in)(None)):
        assert torch.equal(_logits(tiny_model, ids), unedited)
    bank = _randomised(tiny_model)
    with bank.apply(RouterMixer(bank.shape, bank.d_in)(None)):
        routed = _logits(tiny_model, ids)
    with bank.apply(torch.ones(len(LAYERS), 3)):
        plain = _logits(tiny_model, ids)
    assert torch.equal(routed, plain) and not torch.allclose(routed, unedited)


def _routed(tiny_model):
    from hypermodel.condition import RouterMixer
    bank = _randomised(tiny_model)
    mixer = RouterMixer(bank.shape, bank.d_in)
    with torch.no_grad():
        for p in mixer.parameters():
            p.normal_()
    return bank, mixer


def test_routed_writer_output_follows_the_per_token_formula(tiny_model):
    bank, mixer = _routed(tiny_model)
    writer = ModelAdapter(tiny_model).residual_writers()[2]
    h = torch.randn(2, 5, writer.in_features)
    with torch.no_grad(), bank.apply(mixer(None)):
        edited = writer(h)
        plain = torch.nn.functional.linear(h, writer.weight, writer.bias)
    expected = plain.clone()
    with torch.no_grad():
        for b in range(2):
            for t in range(5):
                for e in range(3):
                    # LAYERS[1] is layer 2; scale = alpha / rank = 2
                    g = mixer.g0[1, e] + torch.dot(h[b, t], mixer.W[1, :, e]) + mixer.c[1, e]
                    expected[b, t] += g * 2.0 * (h[b, t] @ bank.V[1, e]) @ bank.U[1, e].T
    assert torch.allclose(edited, expected, atol=1e-4)


def test_router_mixes_per_token_and_trains_only_bank_and_router(tiny_model):
    bank, mixer = _routed(tiny_model)
    tiny_model.requires_grad_(False)
    h = torch.randn(1, 4, bank.d_in)
    g = mixer(None)(0, h)
    assert g.shape == (1, 4, 3) and not torch.allclose(g[0, 0], g[0, 1])
    with bank.apply(mixer(None)):
        tiny_model(torch.randint(0, 64, (2, 5))).logits.pow(2).mean().backward()
    for p in (mixer.g0, mixer.W, mixer.c, bank.U, bank.V):
        assert p.grad is not None and p.grad.abs().sum() > 0
    assert all(p.grad is None for p in tiny_model.parameters())
