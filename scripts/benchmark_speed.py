#!/usr/bin/env python3
"""Compare local Rust minimap2 speed against the vendored C minimap2.

By default this runs a small fixture suite that is suitable for quick
regression checks. For realistic benchmarks, pass one or more TSV manifests
with --manifest. Manifest rows are:

    name<TAB>args

where args is the minimap2 argument string after the binary name, for example:

    HiFi chr11 10k PAF+cg    -x map-hifi -c data/chr11.fa data/hifi.10k.fq
    ONT chr11 10k SAM        -x map-ont -a data/chr11.fa data/ont.10k.fq
    asm5 contigs PAF+cg      -x asm5 -c data/ref.fa data/contigs.fa
"""

from __future__ import annotations

import argparse
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C_BIN = ROOT / "minimap2" / "minimap2"
DEFAULT_RUST_BIN = ROOT / "target" / "release" / "minimap2-pure-rs"
CHR11_X200 = Path("/tmp/minimap2-rs-chr11-200.fq")


@dataclass(frozen=True)
class Case:
    name: str
    args: list[str]


FIXTURE_CASES = [
    Case("MT default PAF", ["minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    Case("MT default PAF+cg", ["-c", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    Case("MT map-hifi PAF+cg", ["-c", "-x", "map-hifi", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    Case("MT map-hifi SAM", ["-a", "-x", "map-hifi", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    Case("chr11 single HiFi PAF+cg", ["-c", "-x", "map-hifi", "tests/data/chr11_bug_window.fa", "tests/data/chr11_bug_query.fq"]),
    Case("chr11 x200 HiFi PAF+cg", ["-c", "-x", "map-hifi", "tests/data/chr11_bug_window.fa", str(CHR11_X200)]),
    Case("chr11 x200 HiFi SAM", ["-a", "-x", "map-hifi", "tests/data/chr11_bug_window.fa", str(CHR11_X200)]),
]


def ensure_chr11_x200() -> None:
    src = ROOT / "tests" / "data" / "chr11_bug_query.fq"
    if CHR11_X200.exists():
        return
    data = src.read_bytes()
    with CHR11_X200.open("wb") as fh:
        for _ in range(200):
            fh.write(data)


def load_manifest(path: Path) -> list[Case]:
    cases = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" not in line:
            raise ValueError(f"{path}:{lineno}: expected 'name<TAB>args'")
        name, arg_string = line.split("\t", 1)
        cases.append(Case(name.strip(), shlex.split(arg_string)))
    return cases


def run_once(argv: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(argv, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return time.perf_counter() - start


def measure(argv: list[str], warmups: int, reps: int) -> list[float]:
    for _ in range(warmups):
        run_once(argv)
    return [run_once(argv) for _ in range(reps)]


def parse_threads(value: str) -> list[int]:
    threads = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise ValueError("--threads values must be positive")
        threads.append(n)
    if not threads:
        raise ValueError("--threads must select at least one value")
    return threads


def selected_cases(args: argparse.Namespace) -> list[Case]:
    cases = [] if args.no_fixtures else list(FIXTURE_CASES)
    for manifest in args.manifest:
        cases.extend(load_manifest(manifest))
    filters = [s.lower() for s in args.case or []]
    if filters:
        cases = [case for case in cases if any(f in case.name.lower() for f in filters)]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c-bin", default=str(DEFAULT_C_BIN))
    parser.add_argument("--rust-bin", default=str(DEFAULT_RUST_BIN))
    parser.add_argument("--reps", type=int, default=7, help="measured runs per binary")
    parser.add_argument("--warmups", type=int, default=1, help="warmup runs per binary")
    parser.add_argument("--threads", default="1", help="comma-separated thread counts, e.g. 1,8")
    parser.add_argument("--case", action="append", help="substring filter; may be repeated")
    parser.add_argument("--manifest", action="append", type=Path, default=[], help="TSV benchmark manifest")
    parser.add_argument("--no-fixtures", action="store_true", help="only run manifest cases")
    args = parser.parse_args()

    c_bin = Path(args.c_bin)
    rust_bin = Path(args.rust_bin)
    if not c_bin.exists():
        print(f"missing C minimap2 binary: {c_bin}", file=sys.stderr)
        return 1
    if not rust_bin.exists():
        print(f"missing Rust release binary: {rust_bin}", file=sys.stderr)
        return 1
    if args.reps <= 0 or args.warmups < 0:
        print("--reps must be positive and --warmups must be non-negative", file=sys.stderr)
        return 1

    ensure_chr11_x200()
    try:
        threads = parse_threads(args.threads)
        cases = selected_cases(args)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if not cases:
        print("no benchmark cases selected", file=sys.stderr)
        return 1

    print("case\tthreads\tc_mean_s\tc_median_s\trust_mean_s\trust_median_s\tratio_mean\tratio_median")
    for case in cases:
        for thread_count in threads:
            thread_args = ["-t", str(thread_count)]
            c_times = measure([str(c_bin), *thread_args, *case.args], args.warmups, args.reps)
            rust_times = measure([str(rust_bin), *thread_args, *case.args], args.warmups, args.reps)
            c_mean = statistics.mean(c_times)
            c_median = statistics.median(c_times)
            rust_mean = statistics.mean(rust_times)
            rust_median = statistics.median(rust_times)
            print(
                f"{case.name}\t{thread_count}\t{c_mean:.4f}\t{c_median:.4f}\t"
                f"{rust_mean:.4f}\t{rust_median:.4f}\t"
                f"{rust_mean / c_mean:.2f}x\t{rust_median / c_median:.2f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
