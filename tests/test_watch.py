import json

from hypermodel.watch import read_run


def test_reads_history_and_base_accuracy_from_a_training_log(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("Loading weights: 100%\n"
                   "plain LoRA: 1,048,608 params, base test acc 0.335\n"
                   "{'step': 250, 'loss': 0.44, 'val_acc': 0.322}\n"
                   "noise {'step': 1}\n"
                   "{'step': 500, 'loss': 0.42, 'val_acc': 0.36}\n")
    run = read_run(log)
    assert run.steps == [250, 500]
    assert run.loss == [0.44, 0.42]
    assert run.val_acc == [0.322, 0.36]
    assert run.base_acc == 0.335


def test_reads_a_run_directory_with_a_history_file(tmp_path):
    (tmp_path / "history.jsonl").write_text(
        "\n".join(json.dumps({"step": s, "loss": 1 / s, "val_acc": s / 1000}) for s in (100, 200)) + "\n")
    (tmp_path / "result.json").write_text(json.dumps({"base_test_acc": 0.335, "test_acc": 0.5}))
    run = read_run(tmp_path)
    assert run.steps == [100, 200]
    assert run.base_acc == 0.335
    assert run.test_acc == 0.5


def test_a_missing_or_empty_log_is_an_empty_run(tmp_path):
    run = read_run(tmp_path / "not-yet.log")
    assert run.steps == [] and run.base_acc is None


def test_reads_retain_terms_and_final_retain_kl(tmp_path):
    (tmp_path / "history.jsonl").write_text(
        json.dumps({"step": 250, "loss": 0.4, "kl": 0.02, "norm": 0.1, "val_acc": 0.4}) + "\n")
    (tmp_path / "result.json").write_text(json.dumps(
        {"base_test_acc": 0.335, "test_acc": 0.44, "retain_kl": {"text": 0.004, "add-4": 0.03, "worst": 0.03}}))
    run = read_run(tmp_path)
    assert run.kl == [0.02] and run.norm == [0.1]
    assert run.retain_kl["worst"] == 0.03


def test_older_runs_without_retain_terms_still_load(tmp_path):
    (tmp_path / "history.jsonl").write_text(json.dumps({"step": 250, "loss": 0.4, "val_acc": 0.4}) + "\n")
    run = read_run(tmp_path)
    assert run.kl == [None] and run.retain_kl is None


def test_discovers_edit_runs_newest_last(tmp_path):
    import os
    from hypermodel.watch import discover
    for i, name in enumerate(["edit-none-s0", "edit-none-retain-s0", "observer-20k"]):
        d = tmp_path / name
        d.mkdir()
        (d / "history.jsonl").write_text("")
        os.utime(d / "history.jsonl", (i, i))
    assert [p.name for p in discover(tmp_path)] == ["edit-none-s0", "edit-none-retain-s0"]
