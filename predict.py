"""Predict from one JSON farm record using saved artifacts.

Usage:
    python predict.py --json samples/valid.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from src.config import VERIFICATION_CODE
from src.inference import ModelsNotFoundError, predict_record


def _fail(message: str, details=None) -> int:
    err = {"verification_code": VERIFICATION_CODE, "error": message}
    if details:
        err["details"] = details
    print(json.dumps(err, indent=2), file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict yield, risk and cluster for one farm.")
    parser.add_argument("--json", required=True, help="path to a JSON file with one record")
    args = parser.parse_args(argv)

    path = Path(args.json)
    if not path.exists():
        return _fail(f"File not found: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _fail(f"Invalid JSON in {path}: {exc}")
    if not isinstance(record, dict):
        return _fail("The JSON file must contain a single object (one farm record).")
    try:
        result = predict_record(record)
    except ValidationError as exc:
        details = [{"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]}
                   for e in exc.errors()]
        return _fail("Invalid input record.", details)
    except ModelsNotFoundError as exc:
        return _fail(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
