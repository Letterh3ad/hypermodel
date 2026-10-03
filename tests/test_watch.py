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
