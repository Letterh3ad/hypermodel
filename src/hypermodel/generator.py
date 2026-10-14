"""Generator G_psi: the observer's view of an episode's failures plus the bank -> one new rank-r expert.
Meta-trained on arithmetic settings disjoint from the stream and retain: `python -m hypermodel.generator --help`."""

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from hypermodel.arithmetic import Difficulty, Question, few_shot_prefix, generate

META = ("add-8", "mul-2", "mod-3", "mod-5x1")  # operand widths keep every question off the stream and retain; none at ceiling
META_SEED = 4241
POOL_SIZES = {"val_ctx": 500, "val_q": 250, "train": 3250}

Bank = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # U [L, E, d_out, r], V [L, E, d_in, r], strength [E]


def meta_pool(name: str, sizes: dict[str, int] = POOL_SIZES) -> dict[str, tuple[Question, ...]]:
    """One fixed pool per setting, cut in the order of sizes."""
    pool = generate(Difficulty.parse(name), sum(sizes.values()), META_SEED)
    out, i = {}, 0
    for k, n in sizes.items():
        out[k], i = tuple(pool[i:i + n]), i + n
    return out


def failure_context(z: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """[mean z, z std, failure fraction] over the failing rows; zeros where nothing fails."""
    bad = z[~ok]
    if not len(bad):
        return torch.zeros(2 * z.shape[1] + 1)
    std = bad.std(0, unbiased=False)
    return torch.cat([bad.mean(0), std, torch.tensor([len(bad) / len(z)])])


class Generator(nn.Module):
    """Context (failure_context) + Deep Sets bank encoding -> U, V for one expert, in LoRABank's [L, 1, d, r] format.

    Factorised head: U[l] = sigma * B_U[l] @ C_U[l] with B_U [d_out, basis] learned and C_U [basis, r] read from the
    context (same for V), so the head emits L * basis * r numbers per factor instead of L * d * r. The C_U head is
    zero-initialised: an untrained generator writes U = 0, a zero delta like a fresh LoRA expert, with a V of
    LoRABank's Kaiming scale."""

    def __init__(self, z_dim: int, n_layers: int, d_in: int, d_out: int, rank: int = 8, basis: int = 64,
                 hidden: int = 512, sketch: int = 8, sigma: float = 0.03, seed: int = 0):
        super().__init__()
        self.config = {"z_dim": z_dim, "n_layers": n_layers, "d_in": d_in, "d_out": d_out, "rank": rank,
                       "basis": basis, "hidden": hidden, "sketch": sketch, "sigma": sigma, "seed": seed}
        L = n_layers
        g = torch.Generator().manual_seed(seed)
        # Fixed probes: P_out^T U V^T P_in sketches each delta W, which is blind to rotations inside the rank.
        self.register_buffer("probe_in", torch.randn(d_in, sketch, generator=g))
        self.register_buffer("probe_out", torch.randn(d_out, sketch, generator=g))
        self.expert_enc = nn.Sequential(nn.Linear(L * sketch * sketch + L + 1, hidden), nn.GELU(),
                                        nn.Linear(hidden, hidden))
        self.empty = nn.Parameter(torch.zeros(hidden))
        ctx = 2 * z_dim + 1
        self.ctx_norm = nn.LayerNorm(ctx)
        self.body = nn.Sequential(nn.Linear(ctx + hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head_u = nn.Linear(hidden, L * basis * rank)
        self.head_v = nn.Linear(hidden, L * basis * rank)
        nn.init.zeros_(self.head_u.weight)
        nn.init.zeros_(self.head_u.bias)
        nn.init.normal_(self.head_v.bias, std=1 / math.sqrt(3 * d_in * rank) / sigma)  # LoRABank's V std
        self.basis_u = nn.Parameter(torch.randn(L, d_out, basis) / math.sqrt(basis))
        self.basis_v = nn.Parameter(torch.randn(L, d_in, basis) / math.sqrt(basis))

    def encode_bank(self, U: torch.Tensor, V: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
        """Mean over experts of an MLP on (delta W sketch / its norm, log norm per layer, strength)."""
        L, E = U.shape[:2]
        if E == 0:
            return self.empty
        U, V = U.to(self.empty), V.to(self.empty)
        sk = torch.einsum("lesr,lert->lest", torch.einsum("leor,os->lesr", U, self.probe_out),
                          torch.einsum("leir,it->lert", V, self.probe_in))
        fro = (torch.einsum("leor,leoq->lerq", U, U) * torch.einsum("leir,leiq->lerq", V, V)).sum((-1, -2))
        fro = fro.clamp_min(1e-12).sqrt()
        x = torch.cat([(sk / fro[..., None, None]).transpose(0, 1).reshape(E, -1), fro.log().T,
                       strength.to(self.empty)[:, None]], 1)
        return self.expert_enc(x).mean(0)

    def forward(self, ctx: torch.Tensor, bank: Bank | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        b = self.empty if bank is None else self.encode_bank(*bank)
        h = self.body(torch.cat([self.ctx_norm(ctx.to(self.empty)), b]))
        cu = self.head_u(h).view(c["n_layers"], c["basis"], c["rank"])
        cv = self.head_v(h).view(c["n_layers"], c["basis"], c["rank"])
        U = c["sigma"] * torch.einsum("lob,lbr->lor", self.basis_u, cu)
        V = c["sigma"] * torch.einsum("lib,lbr->lir", self.basis_v, cv)
        return U.unsqueeze(1), V.unsqueeze(1)

    def save(self, path: Path, **meta) -> None:
        torch.save({"config": self.config, "state": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load(cls, path: Path, device="cuda") -> tuple["Generator", dict]:
        blob = torch.load(path, map_location=device)
        gen = cls(**blob["config"]).to(device)
        gen.load_state_dict(blob["state"])
        return gen.eval(), blob["meta"]


def observe(adapter, observer, obs_layers: list[int], questions: list[Question], prefix: str) -> torch.Tensor:
    """The frozen observer's z for each question, read from the unedited model."""
    from hypermodel.condition import obs_input, trace_features
    X = trace_features(adapter, [obs_input(q, prefix) for q in questions], obs_layers)
    return torch.as_tensor(observer.encode(X))


@dataclass
class Setting:
    name: str
    prefix: str
    train: tuple[Question, ...]
    z: torch.Tensor  # [n_train, k], base model
    ok: torch.Tensor  # [n_train] base model correct
    val_ctx: torch.Tensor  # failure_context of the val-context chunk
    val_q: tuple[Question, ...]
    base_val: float


def prepare(adapter, observer, obs_layers: list[int], name: str, sizes: dict[str, int], cache: Path) -> Setting:
    """Base correctness and z for a setting's pool, cached because the base model and observer are frozen."""
    from hypermodel.scoring import score
    pool, prefix = meta_pool(name, sizes), few_shot_prefix(Difficulty.parse(name))
    seen = pool["val_ctx"] + pool["train"]
    prompts = [q.prompt for q in seen + pool["val_q"]]
    blob = torch.load(cache, map_location="cpu") if cache.exists() else None
    if blob is None or blob["prompts"] != prompts:
        ok = torch.tensor([r["correct"] for r in score(adapter, list(seen + pool["val_q"]), prefix)])
        blob = {"prompts": prompts, "ok": ok, "z": observe(adapter, observer, obs_layers, list(seen), prefix)}
        torch.save(blob, cache)
    n_ctx, ok, z = len(pool["val_ctx"]), blob["ok"], blob["z"]
    return Setting(name, prefix, pool["train"], z[n_ctx:], ok[n_ctx:len(seen)],
                   failure_context(z[:n_ctx], ok[:n_ctx]), pool["val_q"], ok[len(seen):].float().mean().item())


def with_bank(U: torch.Tensor, V: torch.Tensor, bank: Bank | None):
    """The new expert at g = 1 beside the bank's experts at their strengths: (U, V, g [L, 1 + E])."""
    g = torch.ones(U.shape[0], 1, device=U.device)
    if bank is None:
        return U, V, g
    Ub, Vb, s = bank
    return (torch.cat([U, Ub.to(U)], 1), torch.cat([V, Vb.to(V)], 1),
            torch.cat([g, s.to(g)[None].expand(U.shape[0], -1)], 1))


@torch.no_grad()
def validate(adapter, gen: Generator, host, settings: dict[str, Setting]) -> dict[str, float]:
    from hypermodel.edit_train import _accuracy, evaluate
    from hypermodel.editor import ExpertView
    gen.eval()
    out = {}
    for name, s in settings.items():
        U, V, g = with_bank(*gen(s.val_ctx), None)
        out[name] = _accuracy(evaluate(adapter, ExpertView(host, U, V), lambda _, g=g: g, list(s.val_q), s.prefix))
    gen.train()
    return out


def meta_step(adapter, gen: Generator, host, settings: dict[str, Setting], retain, rng: random.Random,
              buffer: list, episodes: int, chunk: tuple[int, int], query: int, beta: float, gamma: float,
              bank_p: float, bank_max: int, buffer_size: int) -> dict[str, float]:
    """Accumulates gradient over `episodes` episodes, each from a different setting; returns mean loss terms."""
    from hypermodel.edit_train import answer_loss
    from hypermodel.editor import ExpertView
    from hypermodel.retain import retain_batch, retain_kl

    sums = {"loss": 0.0, "kl": 0.0, "norm": 0.0, "bank": 0.0}
    by_source: dict[str, list] = {}
    for it in retain or ():
        by_source.setdefault(it.source, []).append(it)
    for name in rng.sample(sorted(settings), episodes):
        s = settings[name]
        idx = rng.sample(range(len(s.train)), min(rng.randint(*chunk), len(s.train) - query))
        rest = sorted(set(range(len(s.train))) - set(idx))
        qs = [s.train[i] for i in rng.sample(rest, query)]
        bank = None
        if buffer and rng.random() < bank_p:
            picked = rng.sample(buffer, rng.randint(1, min(bank_max, len(buffer))))
            bank = (torch.cat([u for u, _ in picked], 1), torch.cat([v for _, v in picked], 1),
                    torch.tensor([rng.uniform(0.1, 1.0) for _ in picked]))
        U, V = gen(failure_context(s.z[idx], s.ok[idx]), bank)
        Ua, Va, g = with_bank(U, V, bank)
        view = ExpertView(host, Ua, Va)
        ce, norm = answer_loss(adapter, view, lambda _: g, s.prefix, qs)
        kl = torch.zeros((), device=ce.device)
        if retain and beta:
            chunk_items = [it for src in sorted(by_source) for it in rng.sample(by_source[src], 3)]
            kl = retain_kl(adapter, view, g, retain_batch(adapter.tokenizer, chunk_items, adapter.device),
                           [it.source for it in chunk_items])
        ((ce + beta * kl + gamma * norm) / episodes).backward()
        buffer.append((U.detach(), V.detach()))
        del buffer[:-buffer_size]
        for k, v in (("loss", ce.item()), ("kl", kl.item()), ("norm", norm.item()), ("bank", float(bank is not None))):
            sums[k] += v / episodes
    return sums


def run(adapter, observer_path: Path, out: Path, seed: int = 0, settings: tuple[str, ...] = META,
        sizes: dict[str, int] = POOL_SIZES, obs_layers: list[int] = (9, 14, 19), edit_layers: list[int] = (12, 15, 18, 21),
        retain=None, max_steps: int = 6000, eval_every: int = 250, patience: int = 8, episodes: int = 2,
        chunk: tuple[int, int] = (200, 2000), query: int = 16, lr: float = 1e-3, beta: float = 1.0,
        gamma: float = 0.1, bank_p: float = 0.5, bank_max: int = 8, buffer_size: int = 64, rank: int = 8,
        gen_kw: dict | None = None, log=print) -> dict:
    """Meta-trains a generator into out/: generator.pt (best val), last.pt (resume point), history.jsonl,
    result.json. A finished run (result.json present) is returned as is; an unfinished one resumes from last.pt."""
    from hypermodel.editor import LoRABank
    from hypermodel.observer import Observer

    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    adapter.model.requires_grad_(False)
    observer = Observer.load(observer_path, adapter.device)
    t0 = time.perf_counter()
    S = {n: prepare(adapter, observer, list(obs_layers), n, sizes, out / f"cache-{n}.pt") for n in settings}
    prep_s = time.perf_counter() - t0
    base = {n: s.base_val for n, s in S.items()}
    host = LoRABank(adapter, list(edit_layers), n_experts=1, rank=rank).requires_grad_(False)
    torch.manual_seed(seed)
    w = adapter.residual_writers()[edit_layers[0]]
    gen = Generator(observer.config["k"], len(edit_layers), w.in_features, w.out_features, rank, seed=seed,
                    **(gen_kw or {})).to(adapter.device)
    opt = torch.optim.AdamW(gen.parameters(), lr=lr)
    state = {"step": 0, "best": -1.0, "best_step": 0, "stale": 0, "buffer": []}
    if (out / "last.pt").exists():
        blob = torch.load(out / "last.pt", map_location=adapter.device)
        gen.load_state_dict(blob["gen"])
        opt.load_state_dict(blob["opt"])
        state = blob["state"]
    history = out / "history.jsonl"
    if not state["step"]:
        history.write_text("")
    meta = {"observer": str(observer_path), "obs_layers": list(obs_layers), "edit_layers": list(edit_layers)}
    acc = {"loss": 0.0, "kl": 0.0, "norm": 0.0, "bank": 0.0, "n": 0}
    t = time.perf_counter()
    step = state["step"]
    while step < max_steps and state["stale"] < patience:
        step += 1
        opt.zero_grad()
        terms = meta_step(adapter, gen, host, S, retain, random.Random(f"{seed}-{step}"), state["buffer"],
                          min(episodes, len(S)), chunk, query, beta, gamma, bank_p, bank_max, buffer_size)
        nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        for k, v in terms.items():
            acc[k] += v
        acc["n"] += 1
        if step % eval_every and step != max_steps:
            continue
        sec = (time.perf_counter() - t) / acc["n"]
        val = validate(adapter, gen, host, S)
        mean = sum(val.values()) / len(val)
        n = acc.pop("n")
        row = {"step": step, **{k: v / n for k, v in acc.items()}, "val_acc": val, "val_mean": mean,
               "sec_per_step": round(sec, 3)}
        if mean > state["best"]:
            state.update(best=mean, best_step=step, stale=0)
            gen.save(out / "generator.pt", **meta, step=step, val_acc=val)
        else:
            state["stale"] += 1
        state["step"] = step
        torch.save({"gen": gen.state_dict(), "opt": opt.state_dict(), "state": state}, out / "last.pt")
        with history.open("a") as f:
            f.write(json.dumps(row) + "\n")
        log(row)
        acc = {k: 0.0 for k in ("loss", "kl", "norm", "bank")} | {"n": 0}
        t = time.perf_counter()
    result = {"steps": state["step"], "best_step": state["best_step"], "best_val_mean": state["best"],
              "base_val": base, "base_val_mean": sum(base.values()) / len(base), "prep_s": round(prep_s),
              "train_s": round(time.perf_counter() - t0 - prep_s), "seed": seed, "settings": list(settings),
              "params": sum(p.numel() for p in gen.parameters()), **meta}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


def main():
    import argparse

    from hypermodel.adapter import load
    from hypermodel.retain import retain_items
    from hypermodel.stream import RETAIN_ARITH
    from hypermodel.trace import TraceSet

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--observer", type=Path, help="default: runs/observer-20k/raw/observer-s<seed>.pt")
    p.add_argument("--traces", type=Path, default=Path("runs/traces-mul-2x3-20k"),
                   help="the observer's trace set, for its layers")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--layers", nargs="+", type=int, default=[12, 15, 18, 21])
    p.add_argument("--settings", nargs="+", default=list(META))
    p.add_argument("--pool", type=int, nargs=3, default=list(POOL_SIZES.values()), metavar=("VAL_CTX", "VAL_Q", "TRAIN"))
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--retain-text", type=int, default=2000)
    p.add_argument("--retain-arith", type=int, default=500)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    adapter = load(args.model, args.device, torch.bfloat16)
    retain = retain_items("train", args.retain_text, args.retain_arith, args.seed, arithmetic=RETAIN_ARITH)
    obs_layers = TraceSet.load(args.traces).meta["layers"]
    r = run(adapter, args.observer or Path(f"runs/observer-20k/raw/observer-s{args.seed}.pt"), args.out, args.seed,
            tuple(args.settings), dict(zip(POOL_SIZES, args.pool)), obs_layers, args.layers, retain,
            args.max_steps, args.eval_every, args.patience,
            log=lambda row: print(row, flush=True))
    print(f"generator s{args.seed}: best val {r['best_val_mean']:.3f} at step {r['best_step']} "
          f"(base {r['base_val_mean']:.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
