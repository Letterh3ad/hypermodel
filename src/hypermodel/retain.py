"""What an edit must not break: KL(base || edited) per token on generic text and other arithmetic."""

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from hypermodel.adapter import ModelAdapter
from hypermodel.arithmetic import Difficulty, few_shot_prefix, generate
from hypermodel.editor import LoRABank, Mixing
from hypermodel.scoring import tokenize_regions

ARITHMETIC = ("add-4", "mod-3x1")
ARITH_POOL = 3000  # under half of mod-3x1's 7200 questions, so generate() samples rather than enumerates
MAX_CHARS = 320  # about 70 tokens: keeps the vocab-sized logits of two passes inside 8 GB
_SAMPLE = Path(__file__).parent / "data" / "retain_sample.txt"


@dataclass(frozen=True)
class RetainItem:
    text: str
    start: int  # KL covers tokens from this character on, so few-shot prefixes are not scored
    source: str


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS].rsplit(" ", 1)[0]


def _load_wikitext(split: str) -> list[str]:
    from datasets import load_dataset
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]
    return [r for r in rows if len(r) > 200 and not r.strip().startswith("=")]


def _paragraphs(split: str) -> list[str]:
    try:
        rows = _load_wikitext(split)
    except Exception as e:  # offline or no datasets cache: fall back to the bundled sample
        print(f"wikitext unavailable ({e!r}); using {_SAMPLE.name}")
        rows = _SAMPLE.read_text(encoding="utf-8").splitlines()[split == "test"::2]
    return [_clip(r) for r in rows]


def retain_items(split: str, n_text: int, n_arith: int, seed: int = 0) -> list[RetainItem]:
    """n_text generic paragraphs plus n_arith questions per ARITHMETIC setting; train and test never overlap."""
    rng = random.Random(f"retain-{split}-{seed}")
    paras = _paragraphs(split)
    items = [RetainItem(t, 0, "text") for t in rng.sample(paras, min(n_text, len(paras)))]
    for name in ARITHMETIC:
        d = Difficulty.parse(name)
        prefix = few_shot_prefix(d)
        # One fixed pool for every seed and n, so splits never overlap: train takes the head, test the tail.
        pool = generate(d, ARITH_POOL, 0)
        if n_arith > ARITH_POOL // 2:
            raise ValueError(f"n_arith {n_arith} exceeds half the {ARITH_POOL}-question pool")
        half = pool[:n_arith] if split == "train" else pool[-n_arith:]
        items += [RetainItem(f"{prefix}{q.prompt}{q.answer}\n", len(prefix), name) for q in half]
    return items


def retain_batch(tok, items: list[RetainItem], device) -> dict:
    batch, kl_mask = tokenize_regions(tok, [it.text for it in items], [it.start for it in items], device)
    return {**batch, "kl_mask": kl_mask}


def token_kl(adapter: ModelAdapter, bank: LoRABank, g: Mixing, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """KL per predicted token inside batch["kl_mask"], and the row each came from; base pass has hooks off."""
    inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "position_ids")}
    # Position t predicts token t + 1, and must itself be real: under left padding a pad predicts the first token.
    # Selecting before the vocab-wide softmax saves memory.
    target = batch["kl_mask"][:, 1:] & batch["attention_mask"][:, :-1].bool()
    with torch.no_grad():
        base = adapter.model(**inputs).logits[:, :-1][target].float().log_softmax(-1)
    with bank.apply(g):
        edited = adapter.model(**inputs).logits[:, :-1][target].float().log_softmax(-1)
    kl = F.kl_div(edited, base, log_target=True, reduction="none").sum(-1)
    return kl, target.nonzero()[:, 0]


def source_balanced(kl: torch.Tensor, rows: torch.Tensor, sources: list[str]) -> torch.Tensor:
    """Mean over sources of each source's per-token mean: a paragraph has ~10x the tokens of a question."""
    src = torch.tensor([sorted(set(sources)).index(s) for s in sources], device=kl.device)[rows]
    return torch.stack([kl[src == i].mean() for i in src.unique()]).mean()


def retain_kl(adapter: ModelAdapter, bank: LoRABank, g: Mixing, batch: dict,
              sources: list[str]) -> torch.Tensor:
    return source_balanced(*token_kl(adapter, bank, g, batch), sources)
