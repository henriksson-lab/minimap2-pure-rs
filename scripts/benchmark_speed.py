#!/usr/bin/env python3
"""Compare local Rust minimap2 speed against the vendored C minimap2."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
C_BIN = ROOT / "minimap2" / "minimap2"
RUST_BIN = ROOT / "target" / "release" / "minimap2-pure-rs"


CASES = [
    (
        "MT default PAF+cg",
        ["-c", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"],
    ),
    (
        "MT map-hifi PAF+cg",
        ["-c", "-x", "map-hifi", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"],
    ),
    (
        "chr11 single HiFi PAF+cg",
        ["-c", "-x", "map-hifi", "tests/data/chr11_bug_window.fa", "tests/data/chr11_bug_query.fq"],
    ),
    (
        "chr11 x200 HiFi PAF+cg",
        ["-c", "-x", "map-hifi", "tests/data/chr11_bug_window.fa", "/tmp/minimap2-rs-chr11-200.fq"],
    ),
]


def ensure_chr11_x200() -> None:
    out = Path("/tmp/minimap2-rs-chr11-200.fq")
    src = ROOT / "tests" / "data" / "chr11_bug_query.fq"
    if out.exists():
        return
    data = src.read_bytes()
    with out.open("wb") as fh:
        for _ in range(200):
            fh.write(data)


def run_once(argv: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(argv, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return time.perf_counter() - start


def measure(argv: list[str], reps: int) -> list[float]:
    run_once(argv)
    return [run_once(argv) for _ in range(reps)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=7, help="measured runs per binary")
    parser.add_argument("--case", action="append", help="substring filter; may be repeated")
    args = parser.parse_args()

    if not C_BIN.exists():
        print(f"missing C minimap2 binary: {C_BIN}", file=sys.stderr)
        return 1
    if not RUST_BIN.exists():
        print(f"missing Rust release binary: {RUST_BIN}", file=sys.stderr)
        return 1

    ensure_chr11_x200()
    filters = [s.lower() for s in args.case or []]
    selected = [(name, argv) for name, argv in CASES if not filters or any(f in name.lower() for f in filters)]
    if not selected:
        print("no benchmark cases selected", file=sys.stderr)
        return 1

    print("case\tc_mean_s\tc_median_s\trust_mean_s\trust_median_s\tratio_mean\tratio_median")
    for name, argv in selected:
        c_times = measure([str(C_BIN), *argv], args.reps)
        rust_times = measure([str(RUST_BIN), *argv], args.reps)
        c_mean = statistics.mean(c_times)
        c_median = statistics.median(c_times)
        rust_mean = statistics.mean(rust_times)
        rust_median = statistics.median(rust_times)
        print(
            f"{name}\t{c_mean:.4f}\t{c_median:.4f}\t"
            f"{rust_mean:.4f}\t{rust_median:.4f}\t"
            f"{rust_mean / c_mean:.2f}x\t{rust_median / c_median:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
