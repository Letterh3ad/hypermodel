import numpy as np
import pytest

from hypermodel.arithmetic import Difficulty, Question, few_shot_prefix, generate
from hypermodel.condition import ObsInput, obs_input, trace_features
from hypermodel.retain import RetainItem
from hypermodel.trace import TraceSet, record

D = Difficulty("mul", 2, digits_b=3)


def test_a_question_is_observed_without_its_answer_at_its_operands():
    prefix = "1*2=2\n"
    o = obs_input(Question(47, 815, "mul", 38305), prefix)
    assert o.text == "1*2=2\n47*815="
    assert o.text[o.a_char] == "7" and o.text[o.b_char] == "5"


def test_a_retain_question_is_observed_up_to_its_equals_sign():
    o = obs_input(RetainItem("9+9=18\n1234+567=1801\n", 7, "add-4"), prefix="")
    assert o.text == "9+9=18\n1234+567="
    assert o.text[o.a_char] == "4" and o.text[o.b_char] == "7"


def test_generic_text_is_observed_whole_with_no_operands():
    o = obs_input(RetainItem("The river rises in the hills.", 0, "text"), prefix="")
    assert o.text == "The river rises in the hills." and o.a_char is None and o.b_char is None


@pytest.fixture(scope="module")
def small_lm():
    from hypermodel.adapter import load
    return load("EleutherAI/pythia-14m", device="cpu")


@pytest.mark.slow
def test_features_on_the_fly_match_a_recorded_trace_run(small_lm, tmp_path):
    qs = generate(D, 6, seed=0)
    prefix = few_shot_prefix(D)
    ts = TraceSet.load(record(small_lm, qs, prefix, [1, 3], tmp_path, batch_size=4, seed=0))
    X = trace_features(small_lm, [obs_input(q, prefix) for q in qs], [1, 3], batch_size=3)
    stored = np.asarray(ts.resid, np.float32).reshape(len(qs), -1)
    np.testing.assert_allclose(X, stored, atol=2e-2)


@pytest.mark.slow
def test_text_without_operands_reads_the_last_token_in_every_slot(small_lm):
    X = trace_features(small_lm, [obs_input(RetainItem("The river rises in the hills.", 0, "text"), "")],
                       [1], batch_size=1)
    a, b, last = X.reshape(3, -1)
    np.testing.assert_array_equal(a, last)
    np.testing.assert_array_equal(b, last)


def _store(n=4, dim=6):
    import torch
    from hypermodel.condition import FeatureStore
    keys = [f"q{i}" for i in range(n)]
    return FeatureStore(keys, torch.randn(n, dim)), keys


def _items(keys):
    return [ObsInput(k) for k in keys]


def test_an_untrained_conditioned_mixer_is_plain_lora():
    import torch
    from hypermodel.condition import ConditionedMixer
    store, keys = _store()
    mixer = ConditionedMixer((2, 3), torch.nn.Linear(6, 5), 5, store, key=lambda it: it.text)
    assert torch.equal(mixer(_items(keys)), torch.ones(4, 2, 3))


def test_different_conditioning_gives_different_edits():
    import torch
    from hypermodel.condition import ConditionedMixer
    store, keys = _store()
    mixer = ConditionedMixer((2, 3), torch.nn.Linear(6, 5), 5, store, key=lambda it: it.text)
    with torch.no_grad():
        mixer.head[-1].weight.normal_()
    g = mixer(_items(keys))
    assert g.shape == (4, 2, 3) and not torch.allclose(g[0], g[1])
    assert torch.allclose(mixer(_items([keys[2]]))[0], g[2], atol=1e-6)


@pytest.mark.parametrize("frozen", [True, False])
def test_only_a_fine_tuned_observer_receives_gradient(frozen):
    import torch
    from hypermodel.condition import ConditionedMixer
    store, keys = _store()
    enc = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(6, 5))
    mixer = ConditionedMixer((2, 3), enc, 5, store, key=lambda it: it.text, freeze_encoder=frozen)
    with torch.no_grad():
        mixer.head[-1].weight.normal_()
    mixer.train()
    assert enc.training is not frozen  # a frozen observer keeps dropout off
    mixer(_items(keys)).sum().backward()
    assert all((p.grad is None) is frozen for p in enc.parameters())
    assert mixer.head[0].weight.grad is not None
    groups = mixer.param_groups(lr=1e-3, encoder_lr=1e-4)
    assert sum(len(g["params"]) for g in groups) == sum(p.requires_grad for p in mixer.parameters())


def test_question_features_share_one_schema_across_tasks_and_flag_text():
    from hypermodel.condition import question_features
    items = [Question(47, 815, "mul", 38305), Question(12, 345, "mul", 4140),
             RetainItem("9+9=18\n1234+5678=6912\n", 7, "add-4"), RetainItem("Plain prose.", 0, "text")]
    keys, X = question_features(items, prefix="")
    assert keys == ["47*815=", "12*345=", "9+9=18\n1234+5678=", "Plain prose."]
    assert X.shape[0] == 4 and not np.allclose(X[0], X[1])
    assert X[3, -1] == 1 and np.all(X[3, :-1] == 0) and np.all(X[:3, -1] == 0)


def test_question_features_carry_nothing_derived_from_the_answer():
    from hypermodel.condition import question_features
    _, X = question_features([Question(47, 815, "mul", 38305), Question(47, 815, "mul", 1),
                              RetainItem("1234+5678=6912\n", 0, "add-4"), RetainItem("1234+5678=1\n", 0, "add-4")], "")
    np.testing.assert_array_equal(X[0], X[1])
    np.testing.assert_array_equal(X[2], X[3])
