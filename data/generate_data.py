"""Recreate data/AI_A2_2527889.csv from the registration-digit seed (7889).

Usage:  python data/generate_data.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SEED, VERIFICATION_CODE, Paths  # noqa: E402
from src.data_generation import generate_dataframe, write_csv, write_fingerprint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic farm dataset.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        print(f"Error: seed must be {SEED} (got {args.seed}).", file=sys.stderr)
        return 2
    paths = Paths()
    paths.make_dirs()
    df = generate_dataframe(args.seed)
    write_csv(df, paths.csv)
    digest = write_fingerprint(paths.csv, paths.artifacts / "data_fingerprint.txt", len(df))
    print(f"[{VERIFICATION_CODE}] wrote {paths.csv} ({len(df)} rows)")
    print(f"[{VERIFICATION_CODE}] SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
