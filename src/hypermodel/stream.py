"""Step 4 task stream, sealed evaluator J and the naive sequential LoRA baseline: `python -m hypermodel.stream --help`."""

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from hypermodel.arithmetic import Difficulty, Question, few_shot_prefix, generate

STREAM = ("mul-2x3", "mod-4x1", "mul-3", "mod-2")
RETAIN_ARITH = ("add-4", "mul-1x4")  # not mod-3x1, which would fight mod-4x1
EPISODES, EPISODE_N = 4, 2000
SIZES = {"gate": 500, "test": 1000, "val": 250}
# mod-2 has only 716 questions in all, a hard limit of its 2-digit % 1-digit space. Gate and test get most of it
# (179 / 250) because J's 1 pt rule needs them; episodes cycle the 216 train questions.
SMALL_FRACTIONS = {"gate": 0.25, "test": 0.35, "val": 0.1}
STREAM_SEED = 7919  # distinct from the seeds the step 1-3 traces were drawn with
OWN_GAIN, MAX_DROP, MAX_KL = 0.01, 0.01, 0.05
_EPS = 1e-9


Gate = tuple[Question, ...]


@dataclass(frozen=True)
class TaskData:
    """What trainers see of a task. The gate set is returned beside it by task_data and goes only to the Evaluator."""
    name: str
    prefix: str
    test: tuple[Question, ...]
    val: tuple[Question, ...]
    train: tuple[Question, ...]


def task_data(name: str, exclude: frozenset[Question] = frozenset()) -> tuple[TaskData, Gate]:
    """One fixed pool per task for every seed, cut into gate, test, val and train in that order.

    exclude: questions some trainable part has already seen (the observer's trace set), kept out of every split."""
    d = Difficulty.parse(name)
    want = sum(SIZES.values()) + EPISODES * EPISODE_N
    pool = generate(d, want, STREAM_SEED, exclude=exclude)
    sizes = SIZES if len(pool) == want else {k: int(f * len(pool)) for k, f in SMALL_FRACTIONS.items()}
    cuts, i = {}, 0
    for k in ("gate", "test", "val"):
        cuts[k], i = tuple(pool[i:i + sizes[k]]), i + sizes[k]
    if not all(cuts.values()) or i >= len(pool):
        raise ValueError(f"{name}: only {len(pool)} questions left after exclusions")
    gate = cuts.pop("gate")
    return TaskData(name, few_shot_prefix(d), train=tuple(pool[i:]), **cuts), gate


def episodes(td: TaskData) -> list[tuple[Question, ...]]:
    """EPISODES chunks of EPISODE_N: fresh when the train pool allows, else reshuffled passes over the whole pool, so
    every task gets the same step budget. Only per-question counts carry over; training reshuffles each episode."""
    if len(td.train) >= EPISODES * EPISODE_N:
        return [td.train[k * EPISODE_N:(k + 1) * EPISODE_N] for k in range(EPISODES)]
    rng = random.Random(f"episodes-{td.name}")

    def draw():
        out = []
        while len(out) < EPISODE_N:
            out += rng.sample(td.train, len(td.train))
        return tuple(out[:EPISODE_N])
    return [draw() for _ in range(EPISODES)]


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reasons: tuple[str, ...]


def _finite(*scores: dict[str, float] | float) -> None:
    # NaN compares False against every threshold, so a diverged run would otherwise pass the gate.
    vals = [v for x in scores for v in (x.values() if isinstance(x, dict) else (x,))]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"gate inputs must be finite: {scores}")


def _own_gain(own: str, before: dict[str, float], after: dict[str, float]) -> list[str]:
    gain = after[own] - before[own]
    return [f"own task {own} {100 * gain:+.1f} pts, needs +{100 * OWN_GAIN:.0f}"] if gain < OWN_GAIN - _EPS else []


def episode_gate(own: str, before: dict[str, float], after: dict[str, float]) -> Verdict:
    """A new expert enters the bank only if its own task's gate accuracy rises by OWN_GAIN."""
    _finite(before, after)
    reasons = _own_gain(own, before, after)
    return Verdict(not reasons, tuple(reasons))


def judge(own: str, before: dict[str, float], after: dict[str, float], worst_kl: float,
          stream: tuple[str, ...] = STREAM) -> Verdict:
    """The consolidation gate: own task +OWN_GAIN, every task before own in the stream down at most MAX_DROP, worst
    retain KL <= MAX_KL. Earlier tasks come from the stream order, so a missing score raises KeyError."""
    if worst_kl is None:
        raise TypeError("judge needs the worst retain KL")
    _finite(before, after, worst_kl)
    reasons = _own_gain(own, before, after)
    for t in stream[:stream.index(own)]:
        if (drop := before[t] - after[t]) > MAX_DROP + _EPS:
            reasons.append(f"earlier task {t} -{100 * drop:.1f} pts, allows -{100 * MAX_DROP:.0f}")
    if worst_kl > MAX_KL:
        reasons.append(f"worst retain KL {worst_kl:.4f} > {MAX_KL}")
    return Verdict(not reasons, tuple(reasons))


class Evaluator:
    """J's data: the sealed gate sets. Not an nn.Module and holds no tensors, so no optimiser or generated edit can
    reach it; it only reads the model through evaluate()."""

    __slots__ = ("gates", "prefixes")

    def __init__(self, tasks: list[TaskData], gates: dict[str, Gate]):
        object.__setattr__(self, "gates", MappingProxyType({t.name: tuple(gates[t.name]) for t in tasks}))
        object.__setattr__(self, "prefixes", MappingProxyType({t.name: t.prefix for t in tasks}))

    def __setattr__(self, *_):
        raise AttributeError("the evaluator is sealed")

    def accuracy(self, adapter, bank, mixer, names: tuple[str, ...] | None = None) -> dict[str, float]:
        from hypermodel.edit_train import _accuracy, evaluate
        return {n: _accuracy(evaluate(adapter, bank, mixer, list(self.gates[n]), self.prefixes[n]))
                for n in (names or tuple(self.gates))}


def forgetting(matrix: list[list[float]]) -> list[float | None]:
    """matrix[i][j]: accuracy on task j after training through task i. Per task, best score before the last task
    minus the final score; None for the last task."""
    last = len(matrix) - 1
    return [max(matrix[i][j] for i in range(j, last)) - matrix[last][j] if j < last else None
            for j in range(len(matrix[0]))]


def drawdown(matrix: list[list[float]]) -> list[float]:
    """Per task, the worst drop below its running best at any row from its own training on, final row included."""
    return [max(max(matrix[i][j] for i in range(j, k + 1)) - matrix[k][j] for k in range(j, len(matrix)))
            for j in range(len(matrix[0]))]


def transfer_drops(matrix: list[list[float]]) -> list[list[float]]:
    """Row i: how much training task i dropped each earlier task j < i (negative = backward transfer)."""
    return [[matrix[i - 1][j] - matrix[i][j] for j in range(i)] if i else [] for i in range(len(matrix))]


def run_naive(adapter, tasks: list[TaskData], retain_train, retain_test, layers: list[int], seed: int = 0,
              batch_size: int = 16, lr: float = 1e-3, beta: float = 1.0, gamma: float = 0.1, log=print) -> dict:
    """Plain LoRA (step 3's `none`) trained on each episode in stream order, one pass per episode, nothing gated."""
    from hypermodel.edit_train import ConstantMixer, _accuracy, evaluate, evaluate_retain, train
    from hypermodel.editor import LoRABank

    torch.manual_seed(seed)
    bank = LoRABank(adapter, layers)
    mixer = ConstantMixer(bank.shape).to(adapter.device)

    def test_acc() -> list[float]:
        bank.eval(), mixer.eval()
        return [_accuracy(evaluate(adapter, bank, mixer, list(t.test), t.prefix)) for t in tasks]

    base, matrix, t0 = test_acc(), [], time.perf_counter()
    log({"base": dict(zip([t.name for t in tasks], base))})
    for i, t in enumerate(tasks):
        for k, ep in enumerate(episodes(t)):
            steps = max(1, EPISODE_N // batch_size)
            train(adapter, bank, mixer, list(ep), list(t.val), t.prefix, max_steps=steps, batch_size=batch_size,
                  lr=lr, eval_every=steps, patience=1, seed=1000 * seed + 10 * i + k, retain=retain_train, beta=beta,
                  gamma=gamma, log=lambda h, n=t.name, k=k: log({"task": n, "episode": k, **h}))
        matrix.append(test_acc())
        log({"after": t.name, "test_acc": dict(zip([t.name for t in tasks], matrix[-1]))})
    bank.eval(), mixer.eval()
    return {"tasks": [t.name for t in tasks], "base": base, "matrix": matrix, "forgetting": forgetting(matrix),
            "drawdown": drawdown(matrix), "transfer_drops": transfer_drops(matrix),
            "final_avg": sum(matrix[-1]) / len(tasks), "retain_kl": evaluate_retain(adapter, bank, mixer, retain_test),
            "train_s": round(time.perf_counter() - t0), "state": {"bank": bank.state_dict(), "mixer": mixer.state_dict()}}


def dtype_check(model: str, device: str, n: int = 4000, exclude: frozenset[Question] = frozenset()) -> dict:
    """Base mul-2x3 accuracy in fp32 and bf16 on n non-gate questions; bf16 is allowed within MAX_DROP of fp32."""
    from hypermodel.adapter import load
    from hypermodel.edit_train import _accuracy
    from hypermodel.scoring import score

    td, _ = task_data("mul-2x3", exclude)
    qs = list(td.test + td.train)[:n]
    r = {name: _accuracy(score(load(model, device, dt), qs, td.prefix))
         for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16))}
    return {**r, "n": len(qs), "bf16_ok": abs(r["bf16"] - r["fp32"]) <= MAX_DROP + _EPS}


def _positive(s: str) -> int:
    import argparse
    if (n := int(s)) < 1:
        raise argparse.ArgumentTypeError(f"needs a positive count, got {n}")
    return n


def parser():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, help="run directory (required unless --dtype-check)")
    p.add_argument("--traces", type=Path, default=Path("runs/traces-mul-2x3-20k"),
                   help="the observer's trace set; its questions are kept out of the stream")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16", help="bf16 per the ticket 01 check")
    p.add_argument("--layers", nargs="+", type=int, default=[12, 15, 18, 21])
    p.add_argument("--retain-text", type=int, default=2000)
    p.add_argument("--retain-arith", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--beta", type=float, default=1.0, help="retain KL weight; 0 with --gamma 0 is the free control")
    p.add_argument("--gamma", type=float, default=0.1, help="edit norm weight")
    p.add_argument("--dtype-check", type=_positive, metavar="N",
                   help="only print base mul-2x3 accuracy in fp32 vs bf16 on N questions (the ticket 01 check)")
    return p


def main():
    from hypermodel.adapter import load
    from hypermodel.edit_train import RETAIN_TEST
    from hypermodel.retain import retain_items
    from hypermodel.trace import TraceSet

    args = parser().parse_args()
    seen = TraceSet.load(args.traces)
    exclude = frozenset(Question(r["a"], r["b"], r["op"], r["answer"]) for r in seen.records)
    if args.dtype_check is not None:
        print(json.dumps(dtype_check(args.model, args.device, args.dtype_check, exclude)))
        return
    if args.out is None:
        parser().error("--out is required")
    tasks = [task_data(n, exclude)[0] for n in STREAM]
    retain_train = retain_items("train", args.retain_text, args.retain_arith, args.seed, arithmetic=RETAIN_ARITH)
    retain_test = retain_items("test", *RETAIN_TEST, arithmetic=RETAIN_ARITH)
    adapter = load(args.model, args.device, torch.bfloat16 if args.dtype == "bf16" else torch.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    hist = (args.out / "history.jsonl").open("w")

    def log(row):
        print(row, flush=True)
        hist.write(json.dumps(row) + "\n")
        hist.flush()

    result = run_naive(adapter, tasks, retain_train, retain_test, args.layers, args.seed, beta=args.beta,
                       gamma=args.gamma, log=log)
    torch.save(result.pop("state"), args.out / f"naive-s{args.seed}.pt")
    result.update({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.out / "result.json").write_text(json.dumps(result, indent=1))
    print(f"naive s{args.seed}: final avg {result['final_avg']:.3f}, forgetting {result['forgetting']}, "
          f"worst retain KL {result['retain_kl']['worst']:.4f}")


if __name__ == "__main__":
    main()
