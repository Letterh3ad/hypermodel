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
