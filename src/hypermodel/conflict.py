"""Conflict signals: how far each layer's logit-lens guess at the next digit is from the model's final one.

They depend on the model's own processing, not just the question. `python -m hypermodel.conflict --help`."""

from pathlib import Path

import numpy as np
import torch

from hypermodel.adapter import ModelAdapter


def digit_ids(tok) -> list[int]:
    ids = [tok.encode(str(d), add_special_tokens=False) for d in range(10)]
    if any(len(i) != 1 for i in ids):
        raise ValueError("digits are not single tokens in this tokenizer")
    return [i[0] for i in ids]


@torch.no_grad()
def conflict_features(adapter: ModelAdapter, last_all: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """[N, 2 * (L - 1) + 3] from last-token block outputs [N, L, d]:

    KL(final || layer l) over the 10 digits for each earlier layer, whether layer l's top digit agrees with the
    final one, then the final digit entropy, top-2 logit margin, and log probability mass on digits."""
    norm, W = adapter.final_norm(), adapter.unembedding()
    digits = digit_ids(adapter.tokenizer)
    Wd = W[digits]
    out = []
    for i in range(0, len(last_all), batch_size):
        h = torch.tensor(np.asarray(last_all[i:i + batch_size]), device=W.device).to(W.dtype)
        x = norm(h)
        logits = (x @ Wd.T).float()  # [B, L, 10]
        logp = logits.log_softmax(-1)
        final = logp[:, -1:]
        kl = (final.exp() * (final - logp[:, :-1])).sum(-1)
        agree = (logp[:, :-1].argmax(-1) == final.argmax(-1)).float()
        top2 = logits[:, -1].topk(2, -1).values
        entropy = -(final.exp() * final).sum(-1)
        mass = logits[:, -1].logsumexp(-1) - (x[:, -1] @ W.T).float().logsumexp(-1)
        out.append(torch.cat([kl, agree, entropy, (top2[:, 0] - top2[:, 1])[:, None], mass[:, None]], 1).cpu())
    return torch.cat(out).numpy()


def main():
    import argparse

    from hypermodel.adapter import load
    from hypermodel.trace import TraceSet

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--out", type=Path, help="default: <traces>/conflict.npy")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    ts = TraceSet.load(args.traces)
    adapter = load(ts.meta["model"], args.device)
    F = conflict_features(adapter, ts.last_all)
    out = args.out or args.traces / "conflict.npy"
    np.save(out, F)
    print(f"{F.shape} conflict features -> {out}")


if __name__ == "__main__":
    main()
