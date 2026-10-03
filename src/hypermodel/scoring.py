"""Few-shot greedy answering, scored by exact match on the first integer."""

import re

import torch

from hypermodel.adapter import ModelAdapter
from hypermodel.arithmetic import Question

_INT = re.compile(r"-?\d+")


def require_left_padding(tok) -> None:
    if tok.padding_side != "left":
        raise ValueError("tokenizer must pad on the left so the last column is each prompt's final token")


def tokenize_regions(tok, texts: list[str], starts: list[int], device) -> tuple[dict, torch.Tensor]:
    """Left-padded batch with position ids, and a mask of the tokens that begin at or after each text's start."""
    require_left_padding(tok)
    enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    tok_starts = enc.pop("offset_mapping")[..., 0]
    region = enc.attention_mask.bool() & (tok_starts >= torch.tensor(starts)[:, None])
    # Left padding: count positions from each row's first real token, as generate() does.
    position_ids = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    return {k: v.to(device) for k, v in {**enc, "position_ids": position_ids}.items()}, region.to(device)


def parse_answer(completion: str) -> int | None:
    m = _INT.search(completion.split("\n", 1)[0])
    return int(m.group()) if m else None


@torch.no_grad()
def score(adapter: ModelAdapter, questions: list[Question], prefix: str,
          batch_size: int = 64) -> list[dict]:
    tok = adapter.tokenizer
    require_left_padding(tok)
    max_new = 2 + max(len(str(q.answer)) for q in questions)
    records = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        enc = tok([prefix + q.prompt for q in batch], return_tensors="pt", padding=True).to(adapter.device)
        out = adapter.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for q, text in zip(batch, texts):
            pred = parse_answer(text)
            records.append({"prompt": q.prompt, "a": q.a, "b": q.b, "op": q.op, "answer": q.answer,
                            "output": text, "pred": pred, "correct": pred == q.answer})
    return records
