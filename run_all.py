"""Single entry point: recreate data, clean, train all tasks, regenerate every artifact.

Usage:
    python run_all.py --student-id ug25/27889@ines.ac.rw --seed 7889
"""
from __future__ import annotations

import argparse
import sys

from src.config import SEED, STUDENT_ID, VERIFICATION_CODE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Musanze Harvest Intelligence Service pipeline")
    parser.add_argument("--student-id", required=True, help="institutional username")
    parser.add_argument("--seed", type=int, required=True, help=f"must be {SEED}")
    args = parser.parse_args(argv)

    if args.seed != SEED:
        print(f"Error: seed must be {SEED} (derived from registration digits); got {args.seed}.",
              file=sys.stderr)
        return 2
    if args.student_id.strip().lower() != STUDENT_ID:
        print(f"Error: --student-id must be {STUDENT_ID}; got '{args.student_id}'.",
              file=sys.stderr)
        return 2

    from src.pipeline import run_pipeline

    print(f"[{VERIFICATION_CODE}] student: {args.student_id}  seed: {args.seed}")
    result = run_pipeline(student_id=args.student_id, seed=args.seed)
    from src.config import Paths
    p = Paths()
    print(f"[{VERIFICATION_CODE}] generated artifacts:")
    for f in sorted(p.artifacts.rglob("*")):
        if f.is_file():
            print("   ", f.relative_to(p.root).as_posix())
    for f in sorted(p.models.glob("*")):
        print("   ", f.relative_to(p.root).as_posix())
    print(f"[{VERIFICATION_CODE}] report: docs/AI_A2_2527889_REPORT.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
