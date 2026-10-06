"""Pre-registered go/no-go rules for the unattended queue; exit 0 means go.

  python -m hypermodel.decide observer <observer.jsonl>   new observer beats the reference on every seed (ticket 05)
  python -m hypermodel.decide collapsed <run dir>         the run's mix barely varies across questions (ticket 07)"""

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    rule, path = argv[0], Path(argv[1])
    try:
        if rule == "observer":
            verdict = json.loads(path.read_text().splitlines()[-1])
            go = verdict.get("beats_reference") is True
        elif rule == "collapsed":
            go = json.loads((path / "mix_usage.json").read_text())["collapsed"] is True
        else:
            raise SystemExit(f"unknown rule {rule!r}")
    except FileNotFoundError as e:
        print(f"decide {rule}: no result ({e.filename}), no-go")
        return 1
    print(f"decide {rule} {path}: {'go' if go else 'no-go'}")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
