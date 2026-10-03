"""Train a LoRA bank on answer cross-entropy and score it by greedy exact match: `python -m hypermodel.edit_train --help`."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from hypermodel.adapter import ModelAdapter, load
from hypermodel.arithmetic import Question
from hypermodel.editor import LoRABank
from hypermodel.scoring import require_left_padding, score


class ConstantMixer(nn.Module):
    """Unconditioned mixing: one learned g for every question, which makes the bank plain LoRA."""

    def __init__(self, shape: tuple[int, int]):
        super().__init__()
        self.g = nn.Parameter(torch.ones(shape))

    def forward(self, questions: list[Question]) -> torch.Tensor:
        return self.g


def answer_batch(tok, prefix: str, questions: list[Question], device) -> dict:
    """Teacher-forced prefix + prompt + answer + newline; labels only on the answer and newline."""
    require_left_padding(tok)
    enc = tok([prefix + q.prompt + f"{q.answer}\n" for q in questions], return_tensors="pt", padding=True,
              return_offsets_mapping=True)
    starts = enc.pop("offset_mapping")[..., 0]
    prompt_len = torch.tensor([len(prefix + q.prompt) for q in questions])[:, None]
    answer = enc.attention_mask.bool() & (starts >= prompt_len)
    labels = enc.input_ids.masked_fill(~answer, -100)
    # Left padding: count positions from each row's first real token, as generate() does.
    position_ids = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    return {k: v.to(device) for k, v in
            {**enc, "position_ids": position_ids, "labels": labels}.items()}


def answer_loss(adapter: ModelAdapter, bank: LoRABank, mixer: nn.Module, prefix: str,
                questions: list[Question]) -> torch.Tensor:
    batch = answer_batch(adapter.tokenizer, prefix, questions, adapter.device)
    with bank.apply(mixer(questions)):
        return adapter.model(**batch).loss


def evaluate(adapter: ModelAdapter, bank: LoRABank, mixer: nn.Module, questions: list[Question],
             prefix: str, batch_size: int = 64) -> list[dict]:
    with torch.no_grad(), bank.apply(mixer(questions)):
        return score(adapter, questions, prefix, batch_size)


def _accuracy(records: list[dict]) -> float:
    return sum(r["correct"] for r in records) / len(records)


def train(adapter: ModelAdapter, bank: LoRABank, mixer: nn.Module, train_qs: list[Question],
          val_qs: list[Question], prefix: str, max_steps: int = 2000, batch_size: int = 16,
          lr: float = 1e-3, weight_decay: float = 0.0, eval_every: int = 100, patience: int = 5,
          seed: int = 0, log=print) -> list[dict]:
    """AdamW on the bank and mixer only; restores the state with the best val accuracy."""
    adapter.model.requires_grad_(False)
    params = [*bank.parameters(), *mixer.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    order: list[Question] = []
    best = {"acc": -1.0, "state": None}
    history, losses, stale = [], [], 0
    for step in range(1, max_steps + 1):
        if len(order) < batch_size:
            order = rng.sample(train_qs, len(train_qs))
        batch, order = order[:batch_size], order[batch_size:]
        bank.train(), mixer.train()
        loss = answer_loss(adapter, bank, mixer, prefix, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % eval_every and step != max_steps:
            continue
        bank.eval(), mixer.eval()
        acc = _accuracy(evaluate(adapter, bank, mixer, val_qs, prefix))
        history.append({"step": step, "loss": sum(losses) / len(losses), "val_acc": acc})
        log(history[-1])
        losses = []
        if acc > best["acc"]:
            best = {"acc": acc, "state": copy.deepcopy((bank.state_dict(), mixer.state_dict()))}
            stale = 0
        elif (stale := stale + 1) >= patience:
            break
    bank.load_state_dict(best["state"][0])
    mixer.load_state_dict(best["state"][1])
    return history


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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ts = TraceSet.load(args.traces)
    qs = split_questions(ts.records, ts.split)
    prefix = ts.meta["prefix"]
    base_acc = float(ts.labels[ts.mask("test")].mean())
    torch.manual_seed(args.seed)
    adapter = load(ts.meta["model"], args.device)
    bank = LoRABank(adapter, args.layers, args.experts, args.rank, args.alpha)
    mixer = ConstantMixer(bank.shape).to(adapter.device)
    n_params = sum(p.numel() for p in [*bank.parameters(), *mixer.parameters()])
    print(f"plain LoRA: {n_params:,} params, base test acc {base_acc:.3f}", flush=True)

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
                    patience=args.patience, seed=args.seed, log=log)
    records = evaluate(adapter, bank, mixer, qs["test"], prefix)
    acc = _accuracy(records)
    result = {"conditioning": "none", "test_acc": acc, "base_test_acc": base_acc, "n_test": len(records),
              "params": n_params, "train_s": round(time.perf_counter() - t), "history": history, **vars(args)}
    (args.out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    with open(args.out / "test_records.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    torch.save({"bank": bank.state_dict(), "mixer": mixer.state_dict()}, args.out / f"editor-s{args.seed}.pt")
    print(f"plain LoRA test acc {acc:.3f} vs base {base_acc:.3f} ({len(records)} questions) -> {args.out}")


if __name__ == "__main__":
    main()
