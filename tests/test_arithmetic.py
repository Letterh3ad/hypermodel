import pytest

from hypermodel.arithmetic import Difficulty, few_shot_prefix, generate


@pytest.mark.parametrize("op", ["add", "mul", "mod"])
@pytest.mark.parametrize("digits", [1, 2, 3])
def test_answers_are_correct_and_operands_have_the_right_size(op, digits):
    for q in generate(Difficulty(op, digits), n=50, seed=1):
        assert len(str(q.a)) == digits
        sym, expected = {"add": ("+", lambda: q.a + q.b), "mul": ("*", lambda: q.a * q.b),
                         "mod": ("%", lambda: q.a % q.b)}[op]
        assert q.answer == expected()
        assert q.prompt == f"{q.a}{sym}{q.b}="


def test_same_seed_same_questions_different_seed_different():
    d = Difficulty("add", 3)
    assert generate(d, 100, seed=7) == generate(d, 100, seed=7)
    assert generate(d, 100, seed=7) != generate(d, 100, seed=8)


def test_questions_are_unique_and_capped_by_the_problem_space():
    qs = generate(Difficulty("add", 1), n=10_000, seed=0)
    assert len(qs) == len(set(qs))
    assert len(qs) < 100  # 10 x 10 pairs minus the few-shot exemplars


def test_prefix_is_fixed_and_its_exemplars_never_appear_as_questions():
    d = Difficulty("mul", 1)
    prefix = few_shot_prefix(d)
    assert prefix == few_shot_prefix(d)
    assert prefix.count("\n") == 4
    shown = {line.split("=")[0] + "=" for line in prefix.splitlines()}
    assert not shown & {q.prompt for q in generate(d, 10_000, seed=3)}


def test_operands_can_have_different_widths():
    d = Difficulty("mul", 2, digits_b=3)
    assert d.name == "mul-2x3"
    assert all(len(str(q.a)) == 2 and len(str(q.b)) == 3 for q in generate(d, 50, seed=0))
    assert Difficulty.parse("mul-2x3") == d
    assert Difficulty.parse("add-4") == Difficulty("add", 4)


def test_difficulty_has_a_stable_name():
    assert Difficulty("mod", 2).name == "mod-2"
    assert Difficulty("add", 2, digits_b=2).name == "add-2"
    with pytest.raises(ValueError):
        Difficulty("div", 2)
