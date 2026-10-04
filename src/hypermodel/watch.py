"""Live training curves for edit runs, previous and current: `python -m hypermodel.watch [run dir or log ...]`.

With no arguments it shows every runs/edit-* directory and picks up new ones as they start.
Tick boxes on the left show or hide each run."""

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_BASE = re.compile(r"base test acc ([\d.]+)")
KL_BAR = 0.05


@dataclass
class Run:
    name: str
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    kl: list[float | None] = field(default_factory=list)
    norm: list[float | None] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    base_acc: float | None = None
    test_acc: float | None = None
    retain_kl: dict | None = None

    def add(self, h: dict) -> None:
        self.steps.append(h["step"])
        self.loss.append(h["loss"])
        self.kl.append(h.get("kl"))
        self.norm.append(h.get("norm"))
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
            run.base_acc, run.test_acc, run.retain_kl = r.get("base_test_acc"), r.get("test_acc"), r.get("retain_kl")
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


def discover(root: Path) -> list[Path]:
    """Edit run directories under root, oldest first."""
    runs = [d for d in Path(root).glob("edit-*") if (d / "history.jsonl").exists()]
    return sorted(runs, key=lambda d: (d / "history.jsonl").stat().st_mtime)


def _summary(run: Run) -> str:
    acc = "running" if run.test_acc is None else f"test {run.test_acc:.3f}"
    if not run.retain_kl:
        return acc
    worst = run.retain_kl.get("worst", max(v for k, v in run.retain_kl.items() if k != "all"))
    return f"{acc}, KL max {worst:.3f} {'ok' if worst <= KL_BAR else 'over'}"


def main():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import CheckButtons

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", type=Path, help="default: every runs/edit-* directory")
    p.add_argument("--root", type=Path, default=Path("runs"))
    p.add_argument("--every", type=float, default=5.0, help="refresh interval in seconds")
    args = p.parse_args()

    fig = plt.figure(figsize=(14, 8))
    fig.canvas.manager.set_window_title("hypermodel: edit runs")
    grid = fig.add_gridspec(2, 2, left=0.27, right=0.98, top=0.94, bottom=0.08, hspace=0.35, wspace=0.22)
    ax_acc, ax_loss = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_kl, ax_norm = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    ui = {"paths": None, "checks": None, "shown": {}}

    def paths() -> list[Path]:
        return args.runs or discover(args.root)

    def rebuild_checks(names: list[str]):
        if ui["checks"]:
            ui["checks"].ax.remove()
        ax = fig.add_axes((0.01, 0.08, 0.22, 0.86))
        ax.set_title("runs (click to show / hide)", fontsize=9, loc="left")
        ax.set_axis_off()
        states = [ui["shown"].setdefault(n, True) for n in names]
        ui["checks"] = CheckButtons(ax, names, states)
        for label in ui["checks"].labels:
            label.set_fontsize(8)

        def toggle(name):
            ui["shown"][name] = not ui["shown"][name]
            draw()
        ui["checks"].on_clicked(toggle)

    def draw(_frame=None):
        current = paths()
        if current != ui["paths"]:
            ui["paths"] = current
            rebuild_checks([read_run(r).name for r in current])
        for ax in (ax_acc, ax_loss, ax_kl, ax_norm):
            ax.clear()
        base = None
        for i, run in enumerate(read_run(r) for r in current):
            base = base if run.base_acc is None else run.base_acc
            ui["checks"].labels[i].set_text(f"{run.name}\n  {_summary(run)}")
            ui["checks"].labels[i].set_color(f"C{i}")
            if not ui["shown"].get(run.name, True):
                continue
            style = dict(color=f"C{i}", marker="o", ms=3, label=run.name)
            ax_acc.plot(run.steps, run.val_acc, **style)
            ax_loss.plot(run.steps, run.loss, **style)
            for ax, ys in ((ax_kl, run.kl), (ax_norm, run.norm)):
                pts = [(s, y) for s, y in zip(run.steps, ys) if y is not None]
                if pts:
                    ax.plot(*zip(*pts), **style)
        if base is not None:
            ax_acc.axhline(base, color="grey", ls="--", label=f"base {base:.3f}")
        ax_kl.axhline(KL_BAR, color="red", ls=":", label=f"bar {KL_BAR} (per source, on test)")
        ax_acc.set(title="val accuracy (greedy exact match)", xlabel="step", ylabel="accuracy")
        ax_loss.set(title="train answer loss", xlabel="step", ylabel="CE per answer token")
        ax_kl.set(title="train retain KL (source-balanced)", xlabel="step", ylabel="nats per token")
        ax_norm.set(title="train edit norm", xlabel="step", ylabel="mean squared edit")
        for ax in (ax_acc, ax_loss, ax_kl, ax_norm):
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="best", fontsize=7)
        fig.canvas.draw_idle()

    anim = FuncAnimation(fig, draw, interval=args.every * 1000, cache_frame_data=False)  # noqa: F841 (kept alive)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
