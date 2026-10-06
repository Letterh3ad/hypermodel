"""Step 4 task stream, sealed evaluator J and the naive sequential LoRA baseline: `python -m hypermodel.stream --help`."""

import json
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
SMALL_FRACTIONS = {"gate": 0.15, "test": 0.3, "val": 0.1}  # mod-2 has only 716 questions in all
STREAM_SEED = 7919  # distinct from the seeds the step 1-3 traces were drawn with
OWN_GAIN, MAX_DROP, MAX_KL = 0.01, 0.01, 0.05
_EPS = 1e-9


@dataclass(frozen=True)
class TaskData:
    name: str
    prefix: str
    gate: tuple[Question, ...]
    test: tuple[Question, ...]
    val: tuple[Question, ...]
    train: tuple[Question, ...]


def task_data(name: str, exclude: frozenset[Question] = frozenset()) -> TaskData:
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
    return TaskData(name, few_shot_prefix(d), train=tuple(pool[i:]), **cuts)


def episodes(td: TaskData) -> list[tuple[Question, ...]]:
    """EPISODES fresh chunks of EPISODE_N when the train pool allows, else reshuffled draws from the whole pool."""
    if len(td.train) >= EPISODES * EPISODE_N:
        return [td.train[k * EPISODE_N:(k + 1) * EPISODE_N] for k in range(EPISODES)]
    rng = random.Random(f"episodes-{td.name}")
    return [tuple(rng.sample(td.train, min(EPISODE_N, len(td.train)))) for _ in range(EPISODES)]


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reasons: tuple[str, ...]


def judge(own: str, before: dict[str, float], after: dict[str, float], worst_kl: float | None = None,
          earlier: tuple[str, ...] = ()) -> Verdict:
    """J's rule: own task gate accuracy +OWN_GAIN, each earlier task down at most MAX_DROP, worst retain KL <= MAX_KL."""
    reasons = []
    if (gain := after[own] - before[own]) < OWN_GAIN - _EPS:
        reasons.append(f"own task {own} {100 * gain:+.1f} pts, needs +{100 * OWN_GAIN:.0f}")
    for t in earlier:
        if (drop := before[t] - after[t]) > MAX_DROP + _EPS:
            reasons.append(f"earlier task {t} -{100 * drop:.1f} pts, allows -{100 * MAX_DROP:.0f}")
    if worst_kl is not None and worst_kl > MAX_KL:
        reasons.append(f"worst retain KL {worst_kl:.4f} > {MAX_KL}")
    return Verdict(not reasons, tuple(reasons))


class Evaluator:
    """J's data: the sealed gate sets. Not an nn.Module and holds no tensors, so no optimiser or generated edit can
    reach it; it only reads the model through evaluate()."""

    __slots__ = ("gates", "prefixes")

    def __init__(self, tasks: list[TaskData]):
        object.__setattr__(self, "gates", MappingProxyType({t.name: t.gate for t in tasks}))
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
            steps = max(1, len(ep) // batch_size)
            train(adapter, bank, mixer, list(ep), list(t.val), t.prefix, max_steps=steps, batch_size=batch_size,
                  lr=lr, eval_every=steps, patience=1, seed=1000 * seed + 10 * i + k, retain=retain_train, beta=beta,
                  gamma=gamma, log=lambda h, n=t.name, k=k: log({"task": n, "episode": k, **h}))
        matrix.append(test_acc())
        log({"after": t.name, "test_acc": dict(zip([t.name for t in tasks], matrix[-1]))})
    bank.eval(), mixer.eval()
    return {"tasks": [t.name for t in tasks], "base": base, "matrix": matrix, "forgetting": forgetting(matrix),
            "final_avg": sum(matrix[-1]) / len(tasks), "retain_kl": evaluate_retain(adapter, bank, mixer, retain_test),
            "train_s": round(time.perf_counter() - t0), "state": {"bank": bank.state_dict(), "mixer": mixer.state_dict()}}


def main():
    import argparse

    from hypermodel.adapter import load
    from hypermodel.edit_train import RETAIN_TEST
    from hypermodel.retain import retain_items
    from hypermodel.trace import TraceSet

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--traces", type=Path, default=Path("runs/traces-mul-2x3-20k"),
                   help="the observer's trace set; its questions are kept out of the stream")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16", help="bf16 per the ticket 01 check")
    p.add_argument("--layers", nargs="+", type=int, default=[12, 15, 18, 21])
    p.add_argument("--retain-text", type=int, default=2000)
    p.add_argument("--retain-arith", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    seen = TraceSet.load(args.traces)
    exclude = frozenset(Question(r["a"], r["b"], r["op"], r["answer"]) for r in seen.records)
    tasks = [task_data(n, exclude) for n in STREAM]
    retain_train = retain_items("train", args.retain_text, args.retain_arith, args.seed, arithmetic=RETAIN_ARITH)
    retain_test = retain_items("test", *RETAIN_TEST, arithmetic=RETAIN_ARITH)
    adapter = load(args.model, args.device, torch.bfloat16 if args.dtype == "bf16" else torch.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    hist = (args.out / "history.jsonl").open("w")

    def log(row):
        print(row, flush=True)
        hist.write(json.dumps(row) + "\n")
        hist.flush()

    result = run_naive(adapter, tasks, retain_train, retain_test, args.layers, args.seed, log=log)
    torch.save(result.pop("state"), args.out / f"naive-s{args.seed}.pt")
    result.update({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.out / "result.json").write_text(json.dumps(result, indent=1))
    print(f"naive s{args.seed}: final avg {result['final_avg']:.3f}, forgetting {result['forgetting']}, "
          f"worst retain KL {result['retain_kl']['worst']:.4f}")


if __name__ == "__main__":
    main()
