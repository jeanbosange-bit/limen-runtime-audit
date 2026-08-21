#!/usr/bin/env python3
"""Validate frozen Phase 9 hashes without modifying source artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


EXPECTED = {
    "reconstructed_phase9/phase9_logit_lens.py": "5d63c66b108f34fe12d5e8ded0853a6219e5982f0ee8a234333b7cd37c24a40e",
    "reconstructed_extract_core/extract_core.py": "18033dbd8be2a3684f9b53e4c756a5e358de32bf2c3811b517935ad270d37a8c",
    "reconstructed_geometry_control/build_geometry_control.py": "e58a9e09016b12a80e02d1741aacf1f6c552f2bf4132f9221b3ea2ef15ca0670",
    "reconstructed_geometry_control/finalize_and_quality.py": "754f100f6cb46d61169d680f4e7b54f694e5fe76f1c6bbc4fe9418d5866cbbf5",
    "reconstructed_geometry_control/build_transfer_report_and_package.py": "730d5a6236f22ee011a4b3b1d8ba3ea436d77b31589f33f005dd914bad54f653",
    "reconstructed_scalar_readout/extract_features.py": "c8fa3842e991ba99f1ba5987b26d4c549437dc86e5549ece08d849e08e41a80a",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for relative, expected in EXPECTED.items():
        path = args.root / relative
        actual = digest(path) if path.exists() else ""
        rows.append({"artifact": relative, "expected_sha256": expected,
                     "actual_sha256": actual, "match": actual == expected})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if not all(row["match"] for row in rows):
        raise SystemExit("Phase 9 provenance validation failed")
    print(f"validated {len(rows)} frozen scripts")


if __name__ == "__main__":
    main()
