"""Seeded arithmetic questions with a difficulty dial (operation x operand digits)."""

import operator
import random
from dataclasses import dataclass

_OPS = {
    "add": ("+", operator.add),
    "mul": ("*", operator.mul),
    "mod": ("%", operator.mod),
}
N_EXEMPLARS = 4


@dataclass(frozen=True)
class Difficulty:
    op: str
    digits: int
    digits_b: int | None = None  # None: the op's default width for the second operand

    def __post_init__(self):
        if self.op not in _OPS:
            raise ValueError(f"unknown op {self.op!r}, expected one of {sorted(_OPS)}")
        if self.digits_b is None:
            object.__setattr__(self, "digits_b", self._default_b())

    def _default_b(self) -> int:
        # A divisor as wide as the dividend makes most answers equal to a itself.
        return max(1, self.digits - 1) if self.op == "mod" else self.digits

    @property
    def name(self) -> str:
        suffix = "" if self.digits_b == self._default_b() else f"x{self.digits_b}"
        return f"{self.op}-{self.digits}{suffix}"

    @classmethod
    def parse(cls, name: str) -> "Difficulty":
        op, widths = name.split("-")
        a, _, b = widths.partition("x")
        return cls(op, int(a), int(b) if b else None)


@dataclass(frozen=True)
class Question:
    a: int
    b: int
    op: str
    answer: int

    @property
    def prompt(self) -> str:
        return f"{self.a}{_OPS[self.op][0]}{self.b}="


def _operand_range(digits: int) -> range:
    return range(0 if digits == 1 else 10 ** (digits - 1), 10 ** digits)


def _pairs(d: Difficulty) -> tuple[range, range]:
    b = _operand_range(d.digits_b)
    if d.op == "mod":
        b = range(max(2, b.start), b.stop)
    return _operand_range(d.digits), b


def _question(d: Difficulty, a: int, b: int) -> Question:
    return Question(a, b, d.op, _OPS[d.op][1](a, b))


def _sample(d: Difficulty, n: int, rng: random.Random, exclude: set[Question]) -> list[Question]:
    ra, rb = _pairs(d)
    space = len(ra) * len(rb) - len(exclude)
    n = min(n, space)
    if n > space // 2:  # dense: enumerate, shuffle
        pool = [q for a in ra for b in rb if (q := _question(d, a, b)) not in exclude]
        rng.shuffle(pool)
        return pool[:n]
    seen: dict[Question, None] = {}
    while len(seen) < n:
        q = _question(d, rng.choice(ra), rng.choice(rb))
        if q not in exclude:
            seen[q] = None
    return list(seen)


def _exemplars(d: Difficulty) -> list[Question]:
    return _sample(d, N_EXEMPLARS, random.Random(f"exemplars-{d.name}"), set())


def few_shot_prefix(d: Difficulty) -> str:
    return "".join(f"{q.prompt}{q.answer}\n" for q in _exemplars(d))


def generate(d: Difficulty, n: int, seed: int) -> list[Question]:
    """Up to n unique questions, never one of the few-shot exemplars."""
    return _sample(d, n, random.Random(f"{seed}-{d.name}"), set(_exemplars(d)))
