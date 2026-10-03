"""Base accuracy across the difficulty dial: `python -m hypermodel.sweep --help`."""

import argparse
import json
from pathlib import Path

from hypermodel.adapter import ModelAdapter, load
from hypermodel.arithmetic import Difficulty, few_shot_prefix, generate
from hypermodel.scoring import score


def sweep(adapter: ModelAdapter, settings: list[Difficulty], n: int, seed: int, out_dir: Path,
          batch_size: int = 64) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    with open(out_dir / "records.jsonl", "w") as rec_f:
        for d in settings:
            records = score(adapter, generate(d, n, seed), few_shot_prefix(d), batch_size)
            for r in records:
                rec_f.write(json.dumps({"setting": d.name, **r}) + "\n")
            acc = sum(r["correct"] for r in records) / len(records)
            summary.append({"setting": d.name, "n": len(records), "accuracy": acc})
            print(f"{d.name:>8}  n={len(records):<5} acc={acc:.3f}", flush=True)
    (out_dir / "summary.jsonl").write_text("".join(json.dumps(s) + "\n" for s in summary))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings", nargs="+", type=Difficulty.parse,
                   help="explicit settings like mul-2x3 mod-2; overrides --ops/--digits")
    p.add_argument("--ops", nargs="+", default=["add", "mul", "mod"])
    p.add_argument("--digits", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    settings = args.settings or [Difficulty(op, d) for op in args.ops for d in args.digits]
    sweep(load(args.model, args.device), settings, args.n, args.seed, args.out, args.batch_size)


if __name__ == "__main__":
    main()
