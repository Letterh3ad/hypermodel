import json

import pytest
import torch

from hypermodel.adapter import ModelAdapter
from hypermodel.arithmetic import Difficulty, _pairs, generate
from hypermodel.editor import ExpertView, LoRABank
from hypermodel.generator import META, Generator, failure_context, meta_pool
from hypermodel.stream import RETAIN_ARITH, STREAM, task_data

L, D_IN, D_OUT, R, K = 2, 64, 32, 4, 6  # tiny_model: d_model 32, intermediate 64


def _gen(**kw):
    torch.manual_seed(0)
    return Generator(z_dim=K, n_layers=L, d_in=D_IN, d_out=D_OUT, rank=R, basis=8, hidden=16, sketch=3, **kw)


def _bank(E, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(L, E, D_OUT, R, generator=g) * 0.1, torch.randn(L, E, D_IN, R, generator=g) * 0.1,
            torch.rand(E, generator=g))


def _ctx():
    return torch.randn(2 * K + 1)


def test_meta_settings_are_the_preregistered_ones():
    assert META == ("add-8", "mul-2", "mod-3", "mod-5x1")


def _ranges(name):
    d = Difficulty.parse(name)
    a, b = _pairs(d)
    return d.op, set(a), set(b)


@pytest.mark.parametrize("meta", META)
def test_meta_operand_spaces_cannot_meet_the_stream_or_retain(meta):
    # Holds for any pool drawn from these settings, whatever is excluded, commuted mirrors included.
    op, a, b = _ranges(meta)
    for other in STREAM + RETAIN_ARITH:
        op2, a2, b2 = _ranges(other)
        if op == op2:
            assert not (a & a2 and b & b2), other
            if op in ("add", "mul"):
                assert not (a & b2 and b & a2), other


def test_meta_pools_are_question_disjoint_from_stream_splits_and_retain_pools():
    from hypermodel.retain import ARITH_POOL
    meta = {q for n in META for split in meta_pool(n).values() for q in split}
    mirror = {(q.op, q.b, q.a) for q in meta if q.op in ("add", "mul")}
    for name in STREAM:
        td, gate = task_data(name)
        for q in set(gate) | set(td.test) | set(td.val) | set(td.train):
            assert q not in meta and (q.op, q.a, q.b) not in mirror
    for name in RETAIN_ARITH:
        for q in generate(Difficulty.parse(name), ARITH_POOL, 0):
            assert q not in meta and (q.op, q.a, q.b) not in mirror


def test_meta_pool_splits_are_disjoint_fixed_and_sized():
    p = meta_pool("mul-2", {"val_ctx": 50, "val_q": 20, "train": 100})
    assert {k: len(v) for k, v in p.items()} == {"val_ctx": 50, "val_q": 20, "train": 100}
    sets = [set(v) for v in p.values()]
    assert sum(map(len, sets)) == len(set().union(*sets))
    assert p == meta_pool("mul-2", {"val_ctx": 50, "val_q": 20, "train": 100})


def test_failure_context_reads_only_failing_questions():
    z = torch.tensor([[1.0, 0.0], [3.0, 2.0], [100.0, 100.0]])
    ok = torch.tensor([False, False, True])
    c = failure_context(z, ok)
    assert torch.allclose(c, torch.tensor([2.0, 1.0, 1.0, 1.0, 2 / 3]))
    assert torch.equal(failure_context(z, torch.ones(3, dtype=torch.bool)), torch.zeros(5))


def test_output_matches_the_bank_expert_format():
    U, V = _gen()(_ctx())
    assert U.shape == (L, 1, D_OUT, R) and V.shape == (L, 1, D_IN, R)


def test_bank_encoding_is_permutation_invariant():
    gen = _gen()
    U, V, s = _bank(5)
    perm = torch.tensor([3, 0, 4, 1, 2])
    a, b = gen.encode_bank(U, V, s), gen.encode_bank(U[:, perm], V[:, perm], s[perm])
    assert torch.allclose(a, b, atol=1e-6)
    assert not torch.allclose(a, gen.encode_bank(U, V, s.flip(0)))  # strength is read


def test_bank_encoding_ignores_a_rotation_inside_the_rank():
    gen = _gen()
    U, V, s = _bank(2)
    Q, _ = torch.linalg.qr(torch.randn(R, R))
    assert torch.allclose(gen.encode_bank(U, V, s), gen.encode_bank(U @ Q, V @ Q, s), atol=1e-5)


def test_empty_bank_works_and_differs_from_a_full_one():
    gen = _gen()
    ctx = _ctx()
    U, V, s = _bank(0)
    torch.testing.assert_close(gen.encode_bank(U, V, s), gen.empty)
    a = gen(ctx, (U, V, s))
    b = gen(ctx)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    with torch.no_grad():
        gen.head_u.weight.normal_()  # the zero-initialised head would hide the bank
    assert not torch.allclose(gen(ctx)[0], gen(ctx, _bank(3))[0])


def test_untrained_generator_writes_a_zero_delta_like_a_fresh_lora(tiny_model):
    gen = _gen()
    U, V = gen(_ctx())
    assert torch.equal(U, torch.zeros_like(U)) and V.abs().sum() > 0


def test_generated_expert_plugs_into_the_bank_and_trains_the_generator(tiny_model):
    gen = _gen()
    with torch.no_grad():
        gen.head_u.bias.normal_()
    adapter = ModelAdapter(tiny_model)
    tiny_model.requires_grad_(False)
    ids = torch.randint(0, 64, (2, 5))
    U, V = gen(_ctx())
    bank = LoRABank(adapter, [0, 2], n_experts=1, rank=R)
    with torch.no_grad():
        bank.U.copy_(U), bank.V.copy_(V)
        with bank.apply(torch.ones(bank.shape)):
            loaded = tiny_model(ids).logits
    host = LoRABank(adapter, [0, 2], n_experts=1, rank=R)
    view = ExpertView(host, U, V)
    assert view.shape == bank.shape
    with view.apply(torch.ones(view.shape)):
        out = tiny_model(ids).logits
    assert torch.allclose(out, loaded, atol=1e-5)
    out.pow(2).mean().backward()
    assert gen.head_u.weight.grad.abs().sum() > 0 and gen.basis_v.grad is not None
    assert host.U.grad is None


def test_save_and_load_round_trip(tmp_path):
    gen = _gen()
    with torch.no_grad():
        gen.head_u.bias.normal_()
    gen.save(tmp_path / "g.pt", observer="obs.pt")
    again, meta = Generator.load(tmp_path / "g.pt", "cpu")
    ctx = _ctx()
    assert all(torch.equal(x, y) for x, y in zip(gen(ctx), again(ctx))) and meta["observer"] == "obs.pt"


@pytest.fixture(scope="module")
def small_lm():
    from hypermodel.adapter import load
    return load("EleutherAI/pythia-14m", device="cpu")


@pytest.mark.slow
def test_meta_training_runs_logs_checkpoints_and_reruns_safely(small_lm, tmp_path, monkeypatch):
    import hypermodel.retain as retain
    from hypermodel.generator import run
    from hypermodel.observer import Observer

    monkeypatch.setattr(retain, "_load_wikitext", lambda split: (_ for _ in ()).throw(ConnectionError()))
    obs_path = tmp_path / "obs.pt"
    Observer(in_dim=2 * 3 * small_lm.d_model, k=8, hidden=16).save(obs_path)
    retain_train = retain.retain_items("train", 4, 3, arithmetic=RETAIN_ARITH)
    kw = dict(settings=("mul-2", "mod-5x1"), sizes={"val_ctx": 12, "val_q": 6, "train": 30}, obs_layers=[1, 3],
              edit_layers=[2, 4], retain=retain_train, max_steps=4, eval_every=2, patience=5, chunk=(5, 10),
              query=4, gen_kw={"basis": 4, "hidden": 16}, rank=4)
    out = tmp_path / "gen"
    r = run(small_lm, obs_path, out, seed=0, **kw)
    assert (out / "generator.pt").exists() and r["steps"] == 4
    hist = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
    assert [h["step"] for h in hist] == [2, 4] and all("val_acc" in h for h in hist)
    gen, meta = Generator.load(out / "generator.pt", "cpu")
    U, V = gen(torch.zeros(2 * 8 + 1))
    assert U.shape == (2, 1, small_lm.model.config.hidden_size, 4)
    assert meta["observer"] == str(obs_path)
    # A finished run is left alone; a crashed one resumes from its last checkpoint.
    assert run(small_lm, obs_path, out, seed=0, **kw) == json.loads((out / "result.json").read_text())
    (out / "result.json").unlink()
    r2 = run(small_lm, obs_path, out, seed=0, **{**kw, "max_steps": 6})
    hist = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
    assert r2["steps"] == 6 and [h["step"] for h in hist] == [2, 4, 6]
