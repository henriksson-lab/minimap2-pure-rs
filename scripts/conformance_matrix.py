#!/usr/bin/env python3
"""Run C minimap2 vs minimap2-pure-rs conformance checks from a TSV manifest."""

from __future__ import annotations

import argparse
import difflib
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C_BIN = ROOT / "minimap2" / "minimap2"
DEFAULT_RUST_BIN = ROOT / "target" / "release" / "minimap2-pure-rs"
SAM_TAGS = ("RG", "NM", "AS", "ms", "nn", "ts")
PAF_TAGS = ("tp", "cm", "s1", "s2", "rl")


@dataclass(frozen=True)
class Case:
    category: str
    name: str
    mode: str
    args: list[str]


def load_manifest(path: Path) -> list[Case]:
    cases: list[Case] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 3)
        if len(parts) != 4:
            raise ValueError(f"{path}:{lineno}: expected category<TAB>name<TAB>mode<TAB>args")
        category, name, mode, arg_string = (part.strip() for part in parts)
        if mode not in {"exact", "paf", "paf-core", "paf-overlap-core", "sam-core", "sam-header"}:
            raise ValueError(f"{path}:{lineno}: unsupported mode {mode!r}")
        cases.append(Case(category, name, mode, shlex.split(arg_string)))
    return cases


def resolve_args(args: list[str]) -> list[str]:
    resolved = []
    for arg in args:
        if arg.startswith("-") or "://" in arg:
            resolved.append(arg)
            continue
        path = Path(arg)
        if path.is_absolute() or path.exists():
            resolved.append(arg)
            continue
        repo_path = ROOT / path
        resolved.append(str(repo_path) if repo_path.exists() else arg)
    return resolved


def run(program: Path, args: list[str]) -> list[str]:
    proc = subprocess.run(
        [str(program), *resolve_args(args)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{program} {' '.join(args)} exited {proc.returncode}")
    return [line for line in proc.stdout.splitlines() if line]


def selected_tags(fields: list[str], wanted: tuple[str, ...]) -> list[str]:
    tags = {}
    for field in fields:
        parts = field.split(":", 2)
        if len(parts) == 3:
            tags[parts[0]] = field
    return [tags[tag] for tag in wanted if tag in tags]


def normalize_sam_core(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        if line.startswith("@"):
            continue
        fields = line.split("\t")
        out.append("\t".join([*fields[:6], *selected_tags(fields[11:], SAM_TAGS)]))
    return out


def normalize_sam_header(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        if line.startswith("@PG"):
            continue
        if line.startswith("@"):
            out.append(line)
        else:
            out.extend(normalize_sam_core([line]))
    return out


def normalize_paf_core(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        fields = line.split("\t")
        out.append("\t".join([*fields[:12], *selected_tags(fields[12:], PAF_TAGS)]))
    return out


def normalize(lines: list[str], mode: str) -> list[str]:
    if mode == "exact":
        return [line for line in lines if not line.startswith("@")]
    if mode == "paf":
        return lines
    if mode == "paf-core":
        return normalize_paf_core(lines)
    if mode == "paf-overlap-core":
        return sorted(normalize_paf_core(lines))
    if mode == "sam-core":
        return normalize_sam_core(lines)
    if mode == "sam-header":
        return normalize_sam_header(lines)
    raise AssertionError(mode)


def check_case(case: Case, c_bin: Path, rust_bin: Path, diff_limit: int) -> bool:
    c_lines = normalize(run(c_bin, case.args), case.mode)
    rust_lines = normalize(run(rust_bin, case.args), case.mode)
    if c_lines == rust_lines:
        print(f"ok\t{case.category}\t{case.name}")
        return True
    print(f"FAIL\t{case.category}\t{case.name}")
    diff = difflib.unified_diff(c_lines, rust_lines, fromfile="C minimap2", tofile="Rust", lineterm="")
    for i, line in enumerate(diff):
        if i >= diff_limit:
            print("... diff truncated ...")
            break
        print(line)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument("--c-bin", type=Path, default=DEFAULT_C_BIN)
    parser.add_argument("--rust-bin", type=Path, default=DEFAULT_RUST_BIN)
    parser.add_argument("--category", action="append", help="category filter; may be repeated")
    parser.add_argument("--case", action="append", help="case name substring filter; may be repeated")
    parser.add_argument("--diff-limit", type=int, default=80)
    args = parser.parse_args()

    if not args.c_bin.exists():
        print(f"missing C minimap2 binary: {args.c_bin}", file=sys.stderr)
        return 2
    if not args.rust_bin.exists():
        print(f"missing Rust release binary: {args.rust_bin}", file=sys.stderr)
        return 2

    try:
        cases = [case for manifest in args.manifest for case in load_manifest(manifest)]
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.category:
        wanted = {category.lower() for category in args.category}
        cases = [case for case in cases if case.category.lower() in wanted]
    if args.case:
        filters = [needle.lower() for needle in args.case]
        cases = [case for case in cases if any(needle in case.name.lower() for needle in filters)]
    if not cases:
        print("no conformance cases selected", file=sys.stderr)
        return 2

    ok = True
    seen_categories = set()
    passed_categories = set()
    for case in cases:
        seen_categories.add(case.category)
        case_ok = check_case(case, args.c_bin, args.rust_bin, args.diff_limit)
        ok = case_ok and ok
        if case_ok:
            passed_categories.add(case.category)

    print(f"categories_checked\t{','.join(sorted(seen_categories))}")
    print(f"categories_with_passes\t{','.join(sorted(passed_categories))}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
