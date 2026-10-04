import json

from hypermodel.decide import main


def _observer(tmp_path, beats):
    f = tmp_path / "observer.jsonl"
    f.write_text(json.dumps({"seed": 0}) + "\n" + json.dumps({"pass": True, "beats_reference": beats}) + "\n")
    return f


def test_observer_rule_follows_the_preregistered_verdict(tmp_path):
    assert main(["observer", str(_observer(tmp_path, True))]) == 0
    assert main(["observer", str(_observer(tmp_path, False))]) == 1


def test_collapse_rule_reads_mix_usage(tmp_path):
    (tmp_path / "mix_usage.json").write_text(json.dumps({"spread": 0.01, "collapsed": True}))
    assert main(["collapsed", str(tmp_path)]) == 0
    (tmp_path / "mix_usage.json").write_text(json.dumps({"spread": 0.5, "collapsed": False}))
    assert main(["collapsed", str(tmp_path)]) == 1


def test_a_missing_result_is_a_no(tmp_path):
    assert main(["collapsed", str(tmp_path)]) == 1
    assert main(["observer", str(tmp_path / "nope.jsonl")]) == 1
