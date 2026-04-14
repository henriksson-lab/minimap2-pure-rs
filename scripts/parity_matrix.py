#!/usr/bin/env python3
"""Run C minimap2 vs minimap2-pure-rs parity checks on local fixtures."""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from pathlib import Path


PAF_CASES = [
    ("MT default PAF+cg", ["-c", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT map-ont PAF+cg", ["-c", "-x", "map-ont", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT map-hifi PAF+cg", ["-c", "-x", "map-hifi", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT asm5 PAF+cg", ["-c", "-x", "asm5", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT asm10 PAF+cg", ["-c", "-x", "asm10", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("x3s default PAF+cg", ["-c", "minimap2/test/x3s-ref.fa", "minimap2/test/x3s-qry.fa"]),
    ("t2/q2 default PAF+cg", ["-c", "minimap2/test/t2.fa", "minimap2/test/q2.fa"]),
    ("chr11 fixture HiFi PAF+cg", ["-c", "-x", "map-hifi", "tests/data/chr11_bug_window.fa", "tests/data/chr11_bug_query.fq"]),
]

SAM_TAGS = ("NM", "AS", "ms", "nn")

SAM_CORE_CASES = [
    ("MT map-hifi SAM EQX", ["-a", "-x", "map-hifi", "--eqx", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT map-ont SAM", ["-a", "-x", "map-ont", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("x3s default SAM", ["-a", "minimap2/test/x3s-ref.fa", "minimap2/test/x3s-qry.fa"]),
]


def run(program: str, args: list[str]) -> list[str]:
    proc = subprocess.run(
        [program, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{program} {' '.join(args)} exited {proc.returncode}")
    return [line for line in proc.stdout.splitlines() if line and not line.startswith("@")]


def sam_core(lines: list[str]) -> list[str]:
    normalized = []
    for line in lines:
        fields = line.split("\t")
        tags = {}
        for field in fields[11:]:
            parts = field.split(":", 2)
            if len(parts) == 3:
                tags[parts[0]] = field
        selected_tags = [tags[tag] for tag in SAM_TAGS if tag in tags]
        normalized.append("\t".join([*fields[:6], *selected_tags]))
    return normalized


def check_case(name: str, args: list[str], c_bin: str, rust_bin: str, mode: str) -> bool:
    c_lines = run(c_bin, args)
    rust_lines = run(rust_bin, args)
    if mode == "sam-core":
        c_lines = sam_core(c_lines)
        rust_lines = sam_core(rust_lines)
    if c_lines == rust_lines:
        print(f"ok  {name}")
        return True
    print(f"FAIL {name}")
    diff = difflib.unified_diff(c_lines, rust_lines, fromfile="C minimap2", tofile="Rust", lineterm="")
    for i, line in enumerate(diff):
        if i >= 80:
            print("... diff truncated ...")
            break
        print(line)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-bin", default="minimap2/minimap2")
    parser.add_argument("--rust-bin", default="target/release/minimap2-pure-rs")
    parser.add_argument("--skip-build-check", action="store_true")
    args = parser.parse_args()

    if not args.skip_build_check:
        for binary in [args.c_bin, args.rust_bin]:
            if not Path(binary).exists():
                print(f"missing binary: {binary}", file=sys.stderr)
                return 2

    ok = True
    for name, case_args in PAF_CASES:
        ok = check_case(name, case_args, args.c_bin, args.rust_bin, "paf") and ok
    for name, case_args in SAM_CORE_CASES:
        ok = check_case(name, case_args, args.c_bin, args.rust_bin, "sam-core") and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
