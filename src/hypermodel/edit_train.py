"""Train a LoRA bank on answer cross-entropy and score it by greedy exact match: `python -m hypermodel.edit_train --help`."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hypermodel.adapter import ModelAdapter, load
from hypermodel.arithmetic import Question
from hypermodel.editor import LoRABank
from hypermodel.retain import RetainItem, retain_batch, retain_items, retain_kl, token_kl

RETAIN_TEST = (400, 200)  # held-out paragraphs, and questions per arithmetic setting, for the retain KL verdict
from hypermodel.scoring import score, tokenize_regions


class ConstantMixer(nn.Module):
    """Unconditioned mixing: one learned g for every question, which makes the bank plain LoRA."""

    def __init__(self, shape: tuple[int, int]):
        super().__init__()
        self.g = nn.Parameter(torch.ones(shape))

    def forward(self, questions: list[Question]) -> torch.Tensor:
        return self.g


def answer_batch(tok, prefix: str, questions: list[Question], device) -> dict:
    """Teacher-forced prefix + prompt + answer + newline; labels only on the answer and newline."""
    batch, answer = tokenize_regions(tok, [prefix + q.prompt + f"{q.answer}\n" for q in questions],
                                     [len(prefix + q.prompt) for q in questions], device)
    return {**batch, "labels": batch["input_ids"].masked_fill(~answer, -100)}


def answer_loss(adapter: ModelAdapter, bank: LoRABank, mixer: nn.Module, prefix: str,
                questions: list[Question]) -> tuple[torch.Tensor, torch.Tensor]:
    """Answer cross-entropy and the edit norm on the same pass."""
    batch = answer_batch(adapter.tokenizer, prefix, questions, adapter.device)
    with bank.apply(mixer(questions)) as edits:
        loss = adapter.model(**batch).loss
    return loss, edits.norm()


def evaluate(adapter: ModelAdapter, bank: LoRABank, mixer: nn.Module, questions: list[Question],
             prefix: str, batch_size: int = 64) -> list[dict]:
    """Greedy exact match, each score batch edited with its own mixing."""
    records = []
    with torch.no_grad():
        for i in range(0, len(questions), batch_size):
            chunk = questions[i:i + batch_size]
            with bank.apply(mixer(chunk)):
                records += score(adapter, chunk, prefix, batch_size)
    return records


@torch.no_grad()
def evaluate_retain(adapter: ModelAdapter, bank: LoRABank, mixer: nn.Module, items: list[RetainItem],
                    batch_size: int = 8) -> dict[str, float]:
    """Retain KL in nats per token for each source, and the worst source (what the bar applies to)."""
    per_source: dict[str, list[torch.Tensor]] = {}
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        kl, rows = token_kl(adapter, bank, mixer(chunk), retain_batch(adapter.tokenizer, chunk, adapter.device))
        for r, it in enumerate(chunk):
            per_source.setdefault(it.source, []).append(kl[rows == r])
    out = {s: torch.cat(v).mean().item() for s, v in per_source.items()}
    return {**out, "worst": max(out.values())}


def _batches(items: list, size: int, rng: random.Random):
    """Endless shuffled batches; each pass drops its remainder."""
    size = min(size, len(items))
    while True:
        order = rng.sample(items, len(items))
        for i in range(0, len(order) - size + 1, size):
            yield order[i:i + size]


def _stratified(items: list[RetainItem], per_source: int, rng: random.Random):
    """Endless batches with per_source items from every source, so each step sees all of them."""
    streams = [_batches([it for it in items if it.source == s], per_source, rng)
               for s in sorted({it.source for it in items})]
    while True:
        yield [it for st in streams for it in next(st)]


def _accuracy(records: list[dict]) -> float:
    return sum(r["correct"] for r in records) / len(records)


def train(adapter: ModelAdapter, bank: LoRABank, mixer: nn.Module, train_qs: list[Question],
          val_qs: list[Question], prefix: str, max_steps: int = 2000, batch_size: int = 16,
          lr: float = 1e-3, weight_decay: float = 0.0, eval_every: int = 100, patience: int = 5,
          seed: int = 0, retain: list[RetainItem] | None = None, beta: float = 1.0, gamma: float = 0.0,
          retain_per_source: int = 3, encoder_lr: float | None = None, log=print) -> list[dict]:
    """AdamW on the bank and mixer only, minimising answer CE + beta * retain KL + gamma * edit norm.

    The KL term weights retain sources equally; a fine-tuned observer trains at encoder_lr (default lr / 10).
    Restores the state with the best val accuracy."""
    adapter.model.requires_grad_(False)
    groups = (mixer.param_groups(lr, encoder_lr or lr / 10) if hasattr(mixer, "param_groups")
              else [{"params": list(mixer.parameters()), "lr": lr}])
    opt = torch.optim.AdamW([{"params": list(bank.parameters()), "lr": lr}, *groups], weight_decay=weight_decay)
    rng = random.Random(seed)
    questions = _batches(train_qs, batch_size, rng)
    retain_stream = _stratified(retain, retain_per_source, rng) if retain and beta else None
    best = {"acc": -1.0, "state": None}
    history, stale = [], 0
    sums = {"loss": 0.0, "kl": 0.0, "norm": 0.0, "n": 0}
    t = time.perf_counter()
    for step in range(1, max_steps + 1):
        bank.train(), mixer.train()
        ce, norm = answer_loss(adapter, bank, mixer, prefix, next(questions))
        kl = torch.zeros((), device=ce.device)
        if retain_stream:
            chunk = next(retain_stream)
            kl = retain_kl(adapter, bank, mixer(chunk), retain_batch(adapter.tokenizer, chunk, adapter.device),
                           [it.source for it in chunk])
        opt.zero_grad()
        (ce + beta * kl + gamma * norm).backward()
        opt.step()
        for k, v in (("loss", ce), ("kl", kl), ("norm", norm)):
            sums[k] += v.item()
        sums["n"] += 1
        if step % eval_every and step != max_steps:
            continue
        sec_per_step = (time.perf_counter() - t) / sums["n"]
        bank.eval(), mixer.eval()
        acc = _accuracy(evaluate(adapter, bank, mixer, val_qs, prefix))
        n = sums.pop("n")
        history.append({"step": step, **{k: v / n for k, v in sums.items()}, "val_acc": acc,
                        "sec_per_step": round(sec_per_step, 3)})
        log(history[-1])
        sums = {"loss": 0.0, "kl": 0.0, "norm": 0.0, "n": 0}
        t = time.perf_counter()
        if acc > best["acc"]:
            best = {"acc": acc, "state": copy.deepcopy((bank.state_dict(), mixer.state_dict()))}
            stale = 0
        elif (stale := stale + 1) >= patience:
            break
    bank.load_state_dict(best["state"][0])
    mixer.load_state_dict(best["state"][1])
    return history


CONDITIONING = ("none", "features", "z-frozen", "z-finetune", "z-shuffled", "z-gain", "z-gain-shuffled", "z-sparse",
                "router")


def build_mixer(kind: str, bank: LoRABank, adapter: ModelAdapter, ts, retain: list[RetainItem],
                fit_items: list, prefix: str, observer: Path | None = None, top_k: int = 2,
                conflict: np.ndarray | None = None) -> nn.Module:
    """none: plain LoRA. features: question-only features. z-*: the step 2 observer on the unedited trace,
    fine-tuned unless z-frozen. z-gain: z sets one gain per layer, not the expert mix. z-sparse: z picks top_k
    experts. *-shuffled: each input's trace swapped for another's (control for z's information).
    router: per-token routing on each writer's own input in the edited pass.

    conflict: per-task-row conflict features (hypermodel.conflict), appended to the observer input; retain items
    get theirs traced now. Conditioning inputs for every task question and retain item are computed once up front."""
    from hypermodel.condition import (ConditionedMixer, FeatureStore, RouterMixer, Standardize, obs_input,
                                      question_features, trace_features)
    if kind not in CONDITIONING:
        raise ValueError(f"unknown conditioning {kind!r}")
    if kind == "none":
        return ConstantMixer(bank.shape).to(adapter.device)
    if kind == "router":
        return RouterMixer(bank.shape, bank.d_in).to(adapter.device)
    key = lambda it: obs_input(it, prefix).text  # noqa: E731
    questions = [Question(r["a"], r["b"], r["op"], r["answer"]) for r in ts.records]
    if kind == "features":
        keys, X = question_features(questions + retain, prefix)
        X = torch.as_tensor(X)
        fit = X[[keys.index(key(it)) for it in fit_items]]
        sd = fit.std(0)
        sd[sd == 0] = 1
        return ConditionedMixer(bank.shape, Standardize(fit.mean(0), sd), X.shape[1], FeatureStore(keys, X),
                                key).to(adapter.device)
    from hypermodel.observer import Observer
    obs = Observer.load(observer, adapter.device)
    obs.head.requires_grad_(False)  # the step 2 correctness head is not part of the editor
    # Task traces were recorded by phase 1 with this model; retain items are traced now.
    task = np.array(ts.resid).reshape(len(ts.records), -1)
    if conflict is not None:
        task = np.concatenate([task, conflict.astype(task.dtype)], 1)
    store = FeatureStore([prefix + r["prompt"] for r in ts.records], torch.from_numpy(task))
    store.extend([key(it) for it in retain],
                 torch.as_tensor(trace_features(adapter, [obs_input(it, prefix) for it in retain], ts.meta["layers"],
                                                conflict=conflict is not None)))
    if kind.endswith("-shuffled"):
        # Same capacity and input distribution as z-finetune, no per-input information. Shuffled within task and
        # within retain rows, so telling arithmetic from retain text is still possible (features can do that too).
        gen = torch.Generator().manual_seed(0)
        for lo, hi in ((0, len(ts.records)), (len(ts.records), len(store.features))):
            store.features[lo:hi] = store.features[lo:hi][torch.randperm(hi - lo, generator=gen)]
    mode = {"z-gain": "gain", "z-gain-shuffled": "gain", "z-sparse": "topk"}.get(kind, "mix")
    return ConditionedMixer(bank.shape, obs, obs.config["k"], store, key, freeze_encoder=kind == "z-frozen",
                            mode=mode, top_k=top_k).to(adapter.device)


def split_questions(records: list[dict], split) -> dict[str, list[Question]]:
    qs = [Question(r["a"], r["b"], r["op"], r["answer"]) for r in records]
    return {name: [q for q, s in zip(qs, split) if s == name] for name in ("train", "val", "test")}


def main():
    from hypermodel.trace import TraceSet

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, required=True, help="trace run whose split and base answers to reuse")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layers", nargs="+", type=int, default=[12, 15, 18, 21])
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--val-n", type=int, default=1000, help="val questions scored at each early-stopping check")
    p.add_argument("--beta", type=float, default=1.0, help="retain KL weight; 0 trains on answers alone")
    p.add_argument("--gamma", type=float, default=0.1, help="edit-norm weight")
    p.add_argument("--retain-text", type=int, default=2000, help="generic paragraphs in the retain train set")
    p.add_argument("--retain-arith", type=int, default=500, help="questions per retain arithmetic setting")
    p.add_argument("--conditioning", choices=CONDITIONING, default="none")
    p.add_argument("--observer", type=Path, help="default: runs/observer-20k/raw/observer-s<seed>.pt")
    p.add_argument("--encoder-lr", type=float, help="fine-tuned observer LR; default lr / 10")
    p.add_argument("--top-k", type=int, default=2, help="experts per question for z-sparse")
    p.add_argument("--conflict", action="store_true",
                   help="observer also reads <traces>/conflict.npy; pass an observer trained with it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.conflict and not args.conditioning.startswith("z"):
        p.error("--conflict only applies to observer (z-*) conditioning")

    ts = TraceSet.load(args.traces)
    qs = split_questions(ts.records, ts.split)
    prefix = ts.meta["prefix"]
    base_acc = float(ts.labels[ts.mask("test")].mean())
    torch.manual_seed(args.seed)
    adapter = load(ts.meta["model"], args.device)
    bank = LoRABank(adapter, args.layers, args.experts, args.rank, args.alpha)
    retain_train = retain_items("train", args.retain_text, args.retain_arith, args.seed)
    retain_test = retain_items("test", *RETAIN_TEST)
    observer = args.observer or Path(f"runs/observer-20k/raw/observer-s{args.seed}.pt")
    mixer = build_mixer(args.conditioning, bank, adapter, ts, retain_train + retain_test,
                        qs["train"] + retain_train, prefix, observer, args.top_k,
                        np.load(args.traces / "conflict.npy") if args.conflict else None)
    n_params = sum(p.numel() for p in [*bank.parameters(), *mixer.parameters()] if p.requires_grad)
    print(f"{args.conditioning}: {n_params:,} trainable params, base test acc {base_acc:.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    hist_file = args.out / "history.jsonl"
    hist_file.write_text("")

    def log(h):
        print(h, flush=True)
        with open(hist_file, "a") as f:
            f.write(json.dumps(h) + "\n")

    t = time.perf_counter()
    history = train(adapter, bank, mixer, qs["train"], random.Random(args.seed).sample(qs["val"], args.val_n),
                    prefix, args.max_steps, args.batch_size, args.lr, eval_every=args.eval_every,
                    patience=args.patience, seed=args.seed,
                    retain=retain_train, beta=args.beta, gamma=args.gamma, encoder_lr=args.encoder_lr, log=log)
    records = evaluate(adapter, bank, mixer, qs["test"], prefix)
    acc = _accuracy(records)
    kl = evaluate_retain(adapter, bank, mixer, retain_test)
    print("retain KL per token: " + ", ".join(f"{k} {v:.4f}" for k, v in kl.items()), flush=True)
    result = {"conditioning": args.conditioning, "test_acc": acc, "base_test_acc": base_acc, "n_test": len(records),
              "retain_kl": kl, "params": n_params, "train_s": round(time.perf_counter() - t), "history": history,
              **vars(args)}
    (args.out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    with open(args.out / "test_records.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    torch.save({"bank": bank.state_dict(), "mixer": mixer.state_dict()}, args.out / f"editor-s{args.seed}.pt")
    print(f"{args.conditioning} test acc {acc:.3f} vs base {base_acc:.3f} ({len(records)} questions) -> {args.out}")


if __name__ == "__main__":
    main()
