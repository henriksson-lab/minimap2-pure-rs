#!/usr/bin/env python3
"""Quick C-vs-Rust benchmark for the yeast RNA splice regression fixtures.

This is the fast-turnover benchmark loop for splice performance work.
It targets the checked-in external-small yeast RNA datasets that we already
use for parity validation:

  - 200 reads   : quick sanity / output checks
  - 5000 reads  : primary optimization loop
  - 50000 reads : milestone-scale validation

Examples:

  scripts/benchmark_yeast_splice.py --reads 5000
  scripts/benchmark_yeast_splice.py --reads 200 --sam --check-output
  scripts/benchmark_yeast_splice.py --reads 5000 --reps 5 --warmups 1
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C_BIN = ROOT / "minimap2" / "minimap2"
DEFAULT_RUST_BIN = ROOT / "target" / "release" / "minimap2-pure-rs"
YEAST_DIR = ROOT / "data" / "conformance" / "external_small" / "yeast_rna"
REF = YEAST_DIR / "yeast.fa"
JUNC = YEAST_DIR / "junctions.bed"

READSETS = {
    200: YEAST_DIR / "rna.200.fq",
    5000: YEAST_DIR / "rna.5000.fq",
    50000: YEAST_DIR / "rna.50000.fq",
}


def run_once(argv: list[str], capture_output: bool) -> tuple[float, float, str | None]:
    start = time.perf_counter()
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    stdout = proc.stdout.read() if capture_output and proc.stdout is not None else None
    _, status, rusage = os.wait4(proc.pid, 0)
    if status != 0:
        raise subprocess.CalledProcessError(status, argv, output=stdout)
    wall = time.perf_counter() - start
    cpu = rusage.ru_utime + rusage.ru_stime
    return wall, cpu, stdout


def normalize_sam(text: str) -> str:
    return "".join(line for line in text.splitlines(True) if not line.startswith("@PG\t"))


def build_args(reads: int, sam: bool, threads: int) -> list[str]:
    mode = "-a" if sam else "-c"
    return [
        "-t",
        str(threads),
        "-x",
        "splice",
        mode,
        "--junc-bed",
        str(JUNC),
        str(REF),
        str(READSETS[reads]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reads", type=int, choices=sorted(READSETS), default=5000)
    parser.add_argument("--sam", action="store_true", help="benchmark SAM output instead of PAF+cg")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3, help="measured runs per binary")
    parser.add_argument("--warmups", type=int, default=1, help="warmup runs per binary")
    parser.add_argument(
        "--metric",
        choices=("wall", "cpu"),
        default="wall",
        help="comparison metric to report; use cpu when host wall time is noisy",
    )
    parser.add_argument("--check-output", action="store_true", help="also verify Rust output matches C")
    parser.add_argument("--c-bin", default=str(DEFAULT_C_BIN))
    parser.add_argument("--rust-bin", default=str(DEFAULT_RUST_BIN))
    args = parser.parse_args()

    c_bin = Path(args.c_bin)
    rust_bin = Path(args.rust_bin)
    if not c_bin.exists():
        print(f"missing C minimap2 binary: {c_bin}", file=sys.stderr)
        return 1
    if not rust_bin.exists():
        print(f"missing Rust binary: {rust_bin}", file=sys.stderr)
        return 1
    if args.reps <= 0 or args.warmups < 0 or args.threads <= 0:
        print("--reps and --threads must be positive; --warmups must be non-negative", file=sys.stderr)
        return 1

    common_args = build_args(args.reads, args.sam, args.threads)
    c_wall_times: list[float] = []
    rust_wall_times: list[float] = []
    c_cpu_times: list[float] = []
    rust_cpu_times: list[float] = []
    c_out: str | None = None
    rust_out: str | None = None
    capture_output = args.check_output

    for i in range(args.warmups + args.reps):
        c_wall, c_cpu, c_stdout = run_once([str(c_bin), *common_args], capture_output)
        r_wall, r_cpu, r_stdout = run_once([str(rust_bin), *common_args], capture_output)
        if i >= args.warmups:
            c_wall_times.append(c_wall)
            rust_wall_times.append(r_wall)
            c_cpu_times.append(c_cpu)
            rust_cpu_times.append(r_cpu)
        c_out = c_stdout
        rust_out = r_stdout

    if args.check_output:
        assert c_out is not None and rust_out is not None
        if args.sam:
            c_cmp = normalize_sam(c_out)
            r_cmp = normalize_sam(rust_out)
        else:
            c_cmp = c_out
            r_cmp = rust_out
        if c_cmp != r_cmp:
            print("output_mismatch=yes")
            return 2
        print("output_match=yes")

    c_mean = statistics.mean(c_wall_times)
    c_median = statistics.median(c_wall_times)
    rust_mean = statistics.mean(rust_wall_times)
    rust_median = statistics.median(rust_wall_times)
    ratio_mean = rust_mean / c_mean
    ratio_median = rust_median / c_median
    c_cpu_mean = statistics.mean(c_cpu_times)
    c_cpu_median = statistics.median(c_cpu_times)
    rust_cpu_mean = statistics.mean(rust_cpu_times)
    rust_cpu_median = statistics.median(rust_cpu_times)
    ratio_cpu_mean = rust_cpu_mean / c_cpu_mean
    ratio_cpu_median = rust_cpu_median / c_cpu_median
    mode = "sam" if args.sam else "paf"
    selected = "cpu" if args.metric == "cpu" else "wall"
    print(f"dataset=yeast_splice_{args.reads}")
    print(f"mode={mode}")
    print(f"threads={args.threads}")
    print(f"metric={selected}")
    print(f"c_mean_s={c_mean:.4f}")
    print(f"c_median_s={c_median:.4f}")
    print(f"rust_mean_s={rust_mean:.4f}")
    print(f"rust_median_s={rust_median:.4f}")
    print(f"ratio_mean={ratio_mean:.2f}x")
    print(f"ratio_median={ratio_median:.2f}x")
    print(f"c_cpu_mean_s={c_cpu_mean:.4f}")
    print(f"c_cpu_median_s={c_cpu_median:.4f}")
    print(f"rust_cpu_mean_s={rust_cpu_mean:.4f}")
    print(f"rust_cpu_median_s={rust_cpu_median:.4f}")
    print(f"ratio_cpu_mean={ratio_cpu_mean:.2f}x")
    print(f"ratio_cpu_median={ratio_cpu_median:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
