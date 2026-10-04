import numpy as np
import pytest
import torch

from hypermodel.arithmetic import Difficulty, Question, few_shot_prefix, generate
from hypermodel.condition import obs_input, trace_features
from hypermodel.conflict import conflict_features, digit_ids
from hypermodel.trace import capture

D = Difficulty("mul", 2, digits_b=3)


@pytest.fixture(scope="module")
def small_lm():
    from hypermodel.adapter import load
    return load("EleutherAI/pythia-14m", device="cpu")


@pytest.mark.slow
def test_digits_are_single_tokens(small_lm):
    ids = digit_ids(small_lm.tokenizer)
    assert len(set(ids)) == 10
    assert [small_lm.tokenizer.decode([i]) for i in ids] == [str(d) for d in range(10)]


@pytest.mark.slow
def test_final_layer_features_match_the_models_own_next_digit_distribution(small_lm):
    prefix = few_shot_prefix(D)
    texts = [obs_input(q, prefix).text for q in generate(D, 5, seed=1)]
    _, last_all = capture(small_lm, texts, [None] * len(texts), [1], batch_size=5)
    F = conflict_features(small_lm, last_all)
    L = small_lm.n_layers
    assert F.shape == (5, 2 * (L - 1) + 3)
    enc = small_lm.tokenizer(texts, return_tensors="pt", padding=True)
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        logits = small_lm.model(**enc, position_ids=pos).logits[:, -1].float()
    p = torch.softmax(logits[:, digit_ids(small_lm.tokenizer)], -1)
    entropy = -(p * p.log()).sum(-1)
    np.testing.assert_allclose(F[:, 2 * (L - 1)], entropy.numpy(), atol=2e-2)
    mass = torch.logsumexp(logits[:, digit_ids(small_lm.tokenizer)], -1) - torch.logsumexp(logits, -1)
    np.testing.assert_allclose(F[:, -1], mass.numpy(), atol=2e-2)
    assert np.isfinite(F).all()


@pytest.mark.slow
def test_conflict_features_never_see_the_answer(small_lm):
    prefix = few_shot_prefix(D)
    qs = [Question(47, 815, "mul", 38305), Question(47, 815, "mul", 1)]
    X = trace_features(small_lm, [obs_input(q, prefix) for q in qs], [1], conflict=True)
    np.testing.assert_array_equal(X[0], X[1])
    assert X.shape[1] == 3 * small_lm.d_model + 2 * (small_lm.n_layers - 1) + 3
