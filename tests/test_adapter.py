import pytest
import torch
from torch import nn

from hypermodel.adapter import ModelAdapter
from tests.conftest import D_MODEL, N_LAYERS


def test_reports_shape(tiny_model):
    a = ModelAdapter(tiny_model)
    assert a.n_layers == N_LAYERS
    assert a.d_model == D_MODEL


def test_residual_writers_are_mlp_out_projections_into_the_stream(tiny_model):
    writers = ModelAdapter(tiny_model).residual_writers()
    assert len(writers) == N_LAYERS
    for w in writers:
        assert isinstance(w, nn.Linear)
        assert w.out_features == D_MODEL
        assert w.in_features != D_MODEL  # the up/gate projections write to the MLP, not the stream


def test_hooking_a_block_captures_its_residual_output(tiny_model):
    a = ModelAdapter(tiny_model)
    seen = {}
    handle = a.blocks()[1].register_forward_hook(
        lambda _m, _i, out: seen.setdefault("h", out[0] if isinstance(out, tuple) else out))
    ids = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        hidden = tiny_model(ids, output_hidden_states=True).hidden_states
    handle.remove()
    # hidden_states[0] is the embedding, so block l's output is hidden_states[l + 1]
    assert seen["h"].shape == (2, 5, D_MODEL)
    assert torch.allclose(seen["h"], hidden[2])


def test_unknown_architecture_fails_loudly():
    class Fake(nn.Module):
        class config:
            model_type = "nope"
    with pytest.raises(ValueError, match="nope"):
        ModelAdapter(Fake())
