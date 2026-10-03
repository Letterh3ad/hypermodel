"""Live training curves for one or more edit runs: `python -m hypermodel.watch <run dir or log> ...`."""

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_BASE = re.compile(r"base test acc ([\d.]+)")


@dataclass
class Run:
    name: str
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    base_acc: float | None = None
    test_acc: float | None = None

    def add(self, h: dict) -> None:
        self.steps.append(h["step"])
        self.loss.append(h["loss"])
        self.val_acc.append(h["val_acc"])


def read_run(path: Path) -> Run:
    """A run directory (history.jsonl, result.json) or a stdout log of `edit_train`."""
    path = Path(path)
    run = Run(path.parent.name if path.name == "history.jsonl" else path.stem if path.is_file() else path.name)
    if path.is_dir():
        hist, result = path / "history.jsonl", path / "result.json"
        for line in hist.read_text().splitlines() if hist.exists() else []:
            run.add(json.loads(line))
        if result.exists():
            r = json.loads(result.read_text())
            run.base_acc, run.test_acc = r.get("base_test_acc"), r.get("test_acc")
        return run
    for line in path.read_text(errors="replace").splitlines() if path.exists() else []:
        if line.startswith("{'step'"):
            h = ast.literal_eval(line)
            if {"step", "loss", "val_acc"} <= h.keys():
                run.add(h)
        elif m := _BASE.search(line):
            run.base_acc = float(m.group(1))
        elif m := re.search(r"test acc ([\d.]+) vs base", line):
            run.test_acc = float(m.group(1))
    return run


def main():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--every", type=float, default=5.0, help="refresh interval in seconds")
    args = p.parse_args()
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.canvas.manager.set_window_title("hypermodel: edit training")

    def draw(_frame):
        ax_acc.clear(), ax_loss.clear()
        base = None
        for i, run in enumerate(read_run(r) for r in args.runs):
            color = f"C{i}"
            label = run.name + (f" (test {run.test_acc:.3f})" if run.test_acc is not None else "")
            ax_acc.plot(run.steps, run.val_acc, "o-", color=color, label=label)
            ax_loss.plot(run.steps, run.loss, "o-", color=color, label=run.name)
            base = base if run.base_acc is None else run.base_acc
        if base is not None:
            ax_acc.axhline(base, color="grey", ls="--", label=f"base {base:.3f}")
        ax_acc.set(title="val accuracy (greedy exact match)", xlabel="step", ylabel="accuracy")
        ax_loss.set(title="train loss (answer tokens)", xlabel="step", ylabel="CE per token")
        for ax in (ax_acc, ax_loss):
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, interval=args.every * 1000, cache_frame_data=False)  # noqa: F841 (kept alive)
    draw(0)
    plt.show()


if __name__ == "__main__":
    main()
