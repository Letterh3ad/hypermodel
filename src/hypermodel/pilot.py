"""Generator pilot on mul-2x3 (ticket 02): generated vs standard-init expert under the same short refine.
`python -m hypermodel.pilot --help`; `--aggregate DIR...` pools seeds and prints the pre-registered verdict."""

import json
import time
from pathlib import Path

import torch

from hypermodel.arithmetic import Question
from hypermodel.stream import Evaluator, TaskData

ARMS = ("base", "A", "B", "C")  # A generated + refine, B standard init + refine, C generated, no refine
COMPARISONS = (("A", "B"), ("A", "C"), ("C", "base"), ("A", "base"), ("B", "base"))


def run_pilot(adapter, gen, observer, obs_layers: list[int], td: TaskData, episode: list[Question],
              gate: list[Question], edit_layers: list[int], retain_train, retain_test, out: Path, seed: int = 0,
              steps: int = 50, batch_size: int = 16, lr: float = 1e-3, beta: float = 1.0, gamma: float = 0.1,
              log=print) -> dict:
    """Writes gate-<arm>.jsonl (per gate question) and result.json into out; the gate goes only to the Evaluator."""
    from hypermodel import edit_train
    from hypermodel.edit_train import ConstantMixer, evaluate_retain
    from hypermodel.editor import LoRABank
    from hypermodel.generator import failure_context, observe
    from hypermodel.scoring import score

    t0 = time.perf_counter()
    adapter.model.requires_grad_(False)
    ev = Evaluator([td], {td.name: gate})
    ok = torch.tensor([r["correct"] for r in score(adapter, episode, td.prefix)])
    ctx = failure_context(observe(adapter, observer, obs_layers, episode, td.prefix), ok)
    with torch.no_grad():
        U, V = gen.eval()(ctx)
    rank = U.shape[-1]

    def expert(generated: bool):
        torch.manual_seed(seed)
        bank = LoRABank(adapter, edit_layers, n_experts=1, rank=rank)
        if generated:
            with torch.no_grad():
                bank.U.copy_(U), bank.V.copy_(V)
        return bank, ConstantMixer(bank.shape).to(adapter.device)

    out.mkdir(parents=True, exist_ok=True)
    acc, kl, hist = {}, {}, {}
    for arm in ARMS:
        bank, mixer = expert(arm in ("A", "C"))
        if arm in ("A", "B"):
            hist[arm] = edit_train.train(adapter, bank, mixer, list(episode), list(td.val), td.prefix, max_steps=steps,
                                         batch_size=batch_size, lr=lr, eval_every=steps, patience=1, seed=seed,
                                         retain=retain_train, beta=beta, gamma=gamma, log=log)
        bank.eval(), mixer.eval()
        recs = ev.records(adapter, bank, mixer, td.name)
        with open(out / f"gate-{arm}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
        acc[arm] = sum(r["correct"] for r in recs) / len(recs)
        if arm != "base" and retain_test:
            kl[arm] = evaluate_retain(adapter, bank, mixer, retain_test)
        log({"arm": arm, "gate_acc": acc[arm], **({"retain_kl": kl[arm]["worst"]} if arm in kl else {})})
    result = {"task": td.name, "seed": seed, "steps": steps, "n_gate": len(gate), "n_episode": len(episode),
              "n_failing": int((~ok).sum()), "gate_acc": acc, "retain_kl": kl, "refine_history": hist,
              "generated_norm": {"U": U.norm().item(), "V": V.norm().item()}, "wall_s": round(time.perf_counter() - t0)}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


def aggregate(dirs: list[Path]) -> dict:
    """Paired bootstrap per comparison, gate questions pooled across seeds. Pass: A - B lower bound > 0."""
    from hypermodel.compare import paired_bootstrap, paired_records

    prompts = {d: frozenset(json.loads(l)["prompt"] for l in (d / "gate-base.jsonl").open()) for d in dirs}
    if len(set(prompts.values())) != 1:
        raise ValueError("seeds were scored on different gate questions")
    out = {}
    for x, y in COMPARISONS:
        d = paired_bootstrap([paired_records(r / f"gate-{x}.jsonl", r / f"gate-{y}.jsonl") for r in dirs])
        out[f"{x}-{y}"] = {k: float(v) for k, v in d.items()}
    out["pass"] = out["A-B"]["lo"] > 0
    return out


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate", nargs="+", type=Path, metavar="DIR", help="pilot runs to pool; nothing else runs")
    p.add_argument("--generator", type=Path, help="meta-training run dir holding generator.pt")
    p.add_argument("--out", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--observer", type=Path, help="default: the one the generator was meta-trained with")
    p.add_argument("--traces", type=Path, default=Path("runs/traces-mul-2x3-20k"),
                   help="the observer's trace set; its questions are kept out of the stream")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--retain-text", type=int, default=2000)
    p.add_argument("--retain-arith", type=int, default=500)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.aggregate:
        r = aggregate(args.aggregate)
        for k, d in r.items():
            if k != "pass":
                print(f"{k:7s} {100 * d['diff']:+.2f} pts [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]"
                      f"  disagree {100 * d['disagree']:.0f}%")
        print(f"verdict ({len(args.aggregate)} seeds): {'PASS' if r['pass'] else 'FAIL'} "
              f"(A - B lower bound {100 * r['A-B']['lo']:+.2f} pts, needs > 0)")
        return
    if not (args.generator and args.out):
        p.error("--generator and --out are required unless --aggregate")
    if (args.out / "result.json").exists():
        print(f"{args.out} already done")
        return

    from hypermodel.adapter import load
    from hypermodel.edit_train import RETAIN_TEST
    from hypermodel.generator import Generator
    from hypermodel.observer import Observer
    from hypermodel.retain import retain_items
    from hypermodel.stream import RETAIN_ARITH, episodes, task_data
    from hypermodel.trace import TraceSet

    seen = TraceSet.load(args.traces)
    td, gate = task_data("mul-2x3", frozenset(Question(r["a"], r["b"], r["op"], r["answer"]) for r in seen.records))
    adapter = load(args.model, args.device, torch.bfloat16)
    gen, meta = Generator.load(args.generator / "generator.pt", args.device)
    observer = Observer.load(args.observer or Path(meta["observer"]), args.device)
    r = run_pilot(adapter, gen, observer, meta["obs_layers"], td, list(episodes(td)[0]), list(gate),
                  meta["edit_layers"], retain_items("train", args.retain_text, args.retain_arith, args.seed,
                                                    arithmetic=RETAIN_ARITH),
                  retain_items("test", *RETAIN_TEST, arithmetic=RETAIN_ARITH), args.out, args.seed, args.steps,
                  log=lambda row: print(row, flush=True))
    print(f"pilot s{args.seed}: gate acc {r['gate_acc']} -> {args.out}")


if __name__ == "__main__":
    main()
