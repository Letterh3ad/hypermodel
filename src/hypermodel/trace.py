"""Residual-stream traces at the operand and answer positions: `python -m hypermodel.trace --help`."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hypermodel.adapter import ModelAdapter, load
from hypermodel.arithmetic import Difficulty, Question, few_shot_prefix, generate
from hypermodel.scoring import require_left_padding, score

POSITIONS = ("a", "b", "last")  # last digit of each operand, and the "=" the answer follows
SPLITS = ("train", "val", "test")


def operand_chars(prefix: str, q: Question) -> tuple[int, int]:
    """Character index of the last digit of each operand in prefix + prompt."""
    a_end = len(prefix) + len(str(q.a)) - 1
    return a_end, a_end + 1 + len(str(q.b))


def token_positions(tok, texts: list[str], operands: list[tuple[int, int] | None]):
    """Tokenize texts (left padded) and locate POSITIONS; a text without operands reads its last token throughout."""
    require_left_padding(tok)
    enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")
    last = enc.input_ids.shape[1] - 1
    pos = torch.full((len(texts), len(POSITIONS)), last, dtype=torch.long)
    for i, chars in enumerate(operands):
        starts, ends = offsets[i, :, 0], offsets[i, :, 1]
        real = enc.attention_mask[i].bool()
        for j, c in enumerate(chars or ()):
            pos[i, j] = int(torch.nonzero(real & (starts <= c) & (c < ends))[0])
    return enc, pos


def operand_token_positions(tok, prefix: str, questions: list[Question]):
    """Tokenize prefix + prompt (left padded) and locate POSITIONS by character offsets."""
    return token_positions(tok, [prefix + q.prompt for q in questions], [operand_chars(prefix, q) for q in questions])


def _split(questions: list[Question], seed: int) -> np.ndarray:
    """60/20/20 by question, with a*b and b*a always on the same side."""
    keys = [(q.op, *sorted((q.a, q.b))) if q.op in ("add", "mul") else (q.op, q.a, q.b) for q in questions]
    unique = sorted(set(keys))
    perm = np.random.default_rng(seed).permutation(len(unique))
    n = len(unique)
    side = {}
    for name, idx in zip(SPLITS, np.split(perm, (int(0.6 * n), int(0.8 * n)))):
        side.update({unique[i]: name for i in idx})
    return np.array([side[k] for k in keys])


@dataclass
class TraceSet:
    resid: np.ndarray  # [N, len(meta["layers"]), len(POSITIONS), d]
    last_all: np.ndarray  # [N, n_layers, d] at the last position, for the layer sweep
    records: list[dict]
    split: np.ndarray
    meta: dict

    @property
    def labels(self) -> np.ndarray:
        return np.array([r["correct"] for r in self.records], dtype=bool)

    def mask(self, name: str) -> np.ndarray:
        return self.split == name

    @staticmethod
    def write(out_dir: Path, resid, last_all, records, split, meta) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "resid.npy", resid)
        np.save(out_dir / "last_all.npy", last_all)
        with open(out_dir / "index.jsonl", "w") as f:
            for r, s in zip(records, split):
                f.write(json.dumps({**r, "split": str(s)}) + "\n")
        (out_dir / "meta.json").write_text(json.dumps({**meta, "positions": list(POSITIONS)}, indent=1))
        return out_dir

    @classmethod
    def load(cls, out_dir: Path) -> "TraceSet":
        out_dir = Path(out_dir)
        rows = [json.loads(l) for l in (out_dir / "index.jsonl").read_text().splitlines()]
        split = np.array([r.pop("split") for r in rows])
        return cls(np.load(out_dir / "resid.npy", mmap_mode="r"),
                   np.load(out_dir / "last_all.npy", mmap_mode="r"),
                   rows, split, json.loads((out_dir / "meta.json").read_text()))


@torch.no_grad()
def capture(adapter: ModelAdapter, texts: list[str], operands: list[tuple[int, int] | None], layers,
            batch_size: int):
    """Residuals at POSITIONS for the given layers [N, L, P, d] and at the last token for every layer [N, n_layers, d]."""
    resid, last_all = [], []
    cur = {}

    def hook(layer):
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            rows = torch.arange(h.shape[0], device=h.device)
            cur.setdefault("last", [])  # filled in block order
            cur["last"].append(h[rows, -1].float().cpu())
            if layer in layers:
                cur[layer] = h[rows[:, None], cur["pos"]].float().cpu()
        return fn

    handles = [b.register_forward_hook(hook(l)) for l, b in enumerate(adapter.blocks())]
    try:
        for i in range(0, len(texts), batch_size):
            enc, pos = token_positions(adapter.tokenizer, texts[i:i + batch_size], operands[i:i + batch_size])
            enc = enc.to(adapter.device)
            cur.clear()
            cur["pos"] = pos.to(adapter.device)
            # Left padding: count positions from each row's first real token, as generate() does.
            position_ids = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
            adapter.model(**enc, position_ids=position_ids)
            resid.append(torch.stack([cur[l] for l in layers], 1))
            last_all.append(torch.stack(cur["last"], 1))
    finally:
        for h in handles:
            h.remove()
    return torch.cat(resid).half().numpy(), torch.cat(last_all).half().numpy()


def record(adapter: ModelAdapter, questions: list[Question], prefix: str, layers: list[int],
           out_dir: Path, batch_size: int = 64, seed: int = 0) -> Path:
    records = score(adapter, questions, prefix, batch_size)
    resid, last_all = capture(adapter, [prefix + q.prompt for q in questions],
                              [operand_chars(prefix, q) for q in questions], layers, batch_size)
    meta = {"model": adapter.model.config.name_or_path, "dtype": str(adapter.model.dtype), "layers": list(layers),
            "n_layers": adapter.n_layers, "d_model": adapter.d_model, "seed": seed, "prefix": prefix}
    return TraceSet.write(out_dir, resid, last_all, records, _split(questions, seed), meta)


def default_layers(n_layers: int) -> list[int]:
    return [round(n_layers / 3), n_layers // 2, round(2 * n_layers / 3)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--setting", type=Difficulty.parse, default=Difficulty.parse("mul-2x3"))
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--layers", nargs="+", type=int, help="default: about 1/3, 1/2, 2/3 depth")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    adapter = load(args.model, args.device, getattr(torch, args.dtype))
    layers = args.layers or default_layers(adapter.n_layers)
    qs = generate(args.setting, args.n, args.seed)
    t = time.perf_counter()
    out = record(adapter, qs, few_shot_prefix(args.setting), layers, args.out, args.batch_size, args.seed)
    t_rec = time.perf_counter() - t
    t = time.perf_counter()
    ts = TraceSet.load(out)
    float(np.asarray(ts.resid).astype(np.float32).mean() + np.asarray(ts.last_all).astype(np.float32).mean())
    print(f"{len(qs)} traces, layers {layers}, acc {ts.labels.mean():.3f}, "
          f"record {t_rec:.1f}s, full reload {time.perf_counter() - t:.2f}s -> {out}")


if __name__ == "__main__":
    main()
