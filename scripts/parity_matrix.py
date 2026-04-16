#!/usr/bin/env python3
"""Run C minimap2 vs minimap2-pure-rs parity checks on local fixtures."""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
import tempfile
from pathlib import Path


PAF_CASES = [
    ("MT default PAF+cg", ["-c", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT custom k/w PAF+cg", ["-c", "-k", "13", "-w", "7", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT map-ont PAF+cg", ["-c", "-x", "map-ont", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT map-hifi PAF+cg", ["-c", "-x", "map-hifi", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT map-pb PAF+cg", ["-c", "-x", "map-pb", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT asm5 PAF+cg", ["-c", "-x", "asm5", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT asm10 PAF+cg", ["-c", "-x", "asm10", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT asm20 PAF+cg", ["-c", "-x", "asm20", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT custom scoring PAF+cg", ["-c", "-A", "1", "-B", "3", "-O", "5,25", "-E", "2,1", "-r", "300,12000", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT chain thresholds PAF+cg", ["-c", "-n", "2", "-m", "20", "-p", "0.7", "-N", "3", "-s", "40", "-z", "200,100", "-g", "3000", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT PAF+cg+ds", ["-c", "--ds", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT PAF+cg+cs", ["-c", "--cs", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT PAF+cg+MD", ["-c", "--MD", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT PAF+cg+sdust", ["-c", "-T", "20", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT PAF+cg+ALT", ["-c", "--alt", "tests/data/mt_alt.txt", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT PAF+cg+ALT drop", ["-c", "--alt", "tests/data/mt_alt.txt", "--alt-drop", "0.2", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT PAF+cg no secondary", ["-c", "--secondary=no", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT PAF+cg all chains", ["-c", "-P", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT PAF+cg forward only", ["-c", "--for-only", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT PAF+cg reverse only", ["-c", "--rev-only", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT PAF+cg split-prefix", ["-c", "-I", "40", "--split-prefix", "/tmp/mm2rs-parity-split", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT HPC PAF+cg", ["-c", "-H", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("x3s default PAF+cg", ["-c", "minimap2/test/x3s-ref.fa", "minimap2/test/x3s-qry.fa"]),
    ("t2/q2 default PAF+cg", ["-c", "minimap2/test/t2.fa", "minimap2/test/q2.fa"]),
    ("t2/q2 short-read PAF+cg", ["-c", "-x", "sr", "minimap2/test/t2.fa", "minimap2/test/q2.fa"]),
    ("t2 paired short-read PAF+cg", ["-c", "-x", "sr", "minimap2/test/t2.fa", "tests/data/pe_r1.fq", "tests/data/pe_r2.fq"]),
    ("t2 paired short-read fragment length PAF+cg", ["-c", "-x", "sr", "-F", "1000", "minimap2/test/t2.fa", "tests/data/pe_r1.fq", "tests/data/pe_r2.fq"]),
    ("t2 grouped short-read PAF+cg", ["-c", "-x", "sr", "minimap2/test/t2.fa", "tests/data/pe_interleaved.fq"]),
    ("t2 grouped short-read frag yes PAF+cg", ["-c", "-x", "sr", "--frag=yes", "minimap2/test/t2.fa", "tests/data/pe_interleaved.fq"]),
    ("t2 grouped short-read frag no PAF+cg", ["-c", "-x", "sr", "--frag=no", "minimap2/test/t2.fa", "tests/data/pe_interleaved.fq"]),
    ("t2 paired short-read split-prefix PAF+cg", ["-c", "-x", "sr", "-I", "40", "--split-prefix", "/tmp/mm2rs-parity-pe-split", "minimap2/test/t2.fa", "tests/data/pe_r1.fq", "tests/data/pe_r2.fq"]),
    ("t2 grouped short-read split-prefix PAF+cg", ["-c", "-x", "sr", "-I", "40", "--split-prefix", "/tmp/mm2rs-parity-frag-split", "minimap2/test/t2.fa", "tests/data/pe_interleaved.fq"]),
    ("single-intron splice PAF+cg", ["-c", "-x", "splice", "tests/data/splice_ref.fa", "tests/data/splice_query.fa"]),
    ("single-intron splice:hq PAF+cg", ["-c", "-x", "splice:hq", "tests/data/splice_ref.fa", "tests/data/splice_query.fa"]),
    ("single-intron splice:sr PAF+cg", ["-c", "-x", "splice:sr", "tests/data/splice_ref.fa", "tests/data/splice_query.fa"]),
    ("annotated splice PAF+cg", ["-c", "-x", "splice", "--junc-bed", "tests/data/splice_junc.bed", "tests/data/splice_ref.fa", "tests/data/splice_query.fa"]),
    ("annotated splice old model PAF+cg", ["-c", "-x", "splice", "-J", "0", "--junc-bed", "tests/data/splice_junc.bed", "tests/data/splice_ref.fa", "tests/data/splice_query.fa"]),
    ("annotated splice junc scores PAF+cg", ["-c", "-x", "splice", "--junc-bed", "tests/data/splice_junc.bed", "--junc-bonus", "5", "--junc-pen", "7", "tests/data/splice_ref.fa", "tests/data/splice_query.fa"]),
    ("annotated splice forward strand PAF+cg", ["-c", "-x", "splice", "-u", "f", "--junc-bed", "tests/data/splice_junc.bed", "tests/data/splice_ref.fa", "tests/data/splice_query.fa"]),
    ("annotated splice max intron PAF+cg", ["-c", "-x", "splice", "-G", "1000", "--junc-bed", "tests/data/splice_junc.bed", "tests/data/splice_ref.fa", "tests/data/splice_query.fa"]),
    ("single-intron write-junc", ["-x", "splice", "--write-junc", "tests/data/splice_ref.fa", "tests/data/splice_query.fa"]),
    ("chr11 fixture HiFi PAF+cg", ["-c", "-x", "map-hifi", "tests/data/chr11_bug_window.fa", "tests/data/chr11_bug_query.fq"]),
]

PAF_CORE_CASES = [
    ("MT qstrand PAF core", ["--qstrand", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
]

SAM_TAGS = ("RG", "NM", "AS", "ms", "nn", "ts")

SAM_CORE_CASES = [
    ("MT map-hifi SAM EQX", ["-a", "-x", "map-hifi", "--eqx", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT map-hifi SAM softclip", ["-a", "-x", "map-hifi", "-Y", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT map-hifi SAM no qual", ["-a", "-x", "map-hifi", "-Q", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT map-hifi SAM secondary seq", ["-a", "-x", "map-hifi", "--secondary-seq", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("MT map-ont SAM", ["-a", "-x", "map-ont", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"]),
    ("x3s default SAM", ["-a", "minimap2/test/x3s-ref.fa", "minimap2/test/x3s-qry.fa"]),
    ("t2/q2 short-read SAM", ["-a", "-x", "sr", "minimap2/test/t2.fa", "minimap2/test/q2.fa"]),
    ("t2 paired short-read SAM", ["-a", "-x", "sr", "minimap2/test/t2.fa", "tests/data/pe_r1.fq", "tests/data/pe_r2.fq"]),
    ("t2 grouped short-read SAM", ["-a", "-x", "sr", "minimap2/test/t2.fa", "tests/data/pe_interleaved.fq"]),
    ("annotated splice SAM", ["-a", "-x", "splice", "--junc-bed", "tests/data/splice_junc.bed", "tests/data/splice_ref.fa", "tests/data/splice_query.fa"]),
]

SAM_HEADER_CASES = [
    ("x3s SAM read group", ["-a", "-R", "@RG\\tID:foo\\tSM:bar", "minimap2/test/x3s-ref.fa", "minimap2/test/x3s-qry.fa"]),
]


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="ascii") as out:
        for name, seq in records:
            out.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i:i + 80] + "\n")


def write_fastq(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="ascii") as out:
        for name, seq in records:
            out.write(f"@{name}\n{seq}\n+\n{'I' * len(seq)}\n")


def dna(length: int, seed: int) -> str:
    x = seed
    bases = []
    alphabet = "ACGT"
    for _ in range(length):
        x = (1103515245 * x + 12345) & 0x7fffffff
        bases.append(alphabet[(x >> 16) & 3])
    return "".join(bases)


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def generated_cases(
    tmpdir: Path,
) -> tuple[
    list[tuple[str, list[str]]],
    list[tuple[str, list[str]]],
    list[tuple[str, list[str]]],
    list[tuple[str, list[str]]],
    list[tuple[str, list[str], list[str], str]],
]:
    ref = tmpdir / "generated_ref.fa"
    qry = tmpdir / "generated_qry.fa"
    nohit_qry = tmpdir / "generated_nohit.fa"
    comment_qry = tmpdir / "generated_comment_qry.fa"
    split_qry = tmpdir / "generated_split_qry.fa"
    frag = tmpdir / "generated_frag.fq"
    ava = tmpdir / "generated_ava.fa"
    splice_ref = tmpdir / "generated_splice_ref.fa"
    splice_qry = tmpdir / "generated_splice_qry.fa"
    splice_bed = tmpdir / "generated_splice.bed"

    chr_a = dna(420, 11)
    chr_b = dna(520, 29)
    write_fasta(ref, [("chrA", chr_a), ("chrB", chr_b)])
    write_fasta(qry, [
        ("chrA_exact", chr_a[40:180]),
        ("chrB_exact", chr_b[30:190]),
    ])
    write_fasta(nohit_qry, [("nohit", "N" * 80)])
    comment_qry.write_text(
        f">chrA_comment sample=alpha run=1\n{chr_a[40:180]}\n",
        encoding="ascii",
    )
    write_fasta(split_qry, [("chrB_split", chr_b[20:220])])

    r1 = chr_a[24:104]
    r2 = revcomp(chr_a[140:220])
    singleton = chr_b[60:150]
    write_fastq(frag, [("fragA", r1), ("fragA", r2), ("soloB", singleton)])

    ava_base = ("ACGTGCAATGCCGATGTTACCGTAACGGTAGCTAGGCTAATCGGATCGAACCTTGGA" * 6)[:360]
    ava_r1 = ava_base[:260]
    ava_r2 = ava_base[80:340]
    ava_r3 = ("TTGACCGATTAACCGGTTAGCATCGATGCA" * 8)[:240]
    write_fasta(ava, [("read1", ava_r1), ("read2", ava_r2), ("read3", ava_r3)])

    exon1 = "AACCGGTT" * 8
    exon2 = "TTGGAACC" * 7
    exon3 = "CCGTAAGT" * 8
    intron1 = "GT" + "A" * 28 + "AG"
    intron2 = "GT" + "C" * 35 + "AG"
    splice_seq = exon1 + intron1 + exon2 + intron2 + exon3
    write_fasta(splice_ref, [("txchr", splice_seq)])
    write_fasta(splice_qry, [("multi_intron", exon1 + exon2 + exon3)])
    block_sizes = f"{len(exon1)},{len(exon2)},{len(exon3)}"
    block_starts = f"0,{len(exon1) + len(intron1)},{len(exon1) + len(intron1) + len(exon2) + len(intron2)}"
    splice_bed.write_text(
        "\t".join([
            "txchr",
            "0",
            str(len(splice_seq)),
            "multi_intron",
            "100",
            "+",
            "0",
            str(len(splice_seq)),
            "0",
            "3",
            block_sizes,
            block_starts,
        ]) + "\n",
        encoding="ascii",
    )

    paf_cases = [
        ("generated exact multi-contig PAF+cg", ["-c", str(ref), str(qry)]),
        ("generated no-hit PAF", ["--paf-no-hit", str(ref), str(nohit_qry)]),
        ("generated small mini-batch PAF+cg", ["-c", "-K", "100", str(ref), str(qry)]),
        ("generated grouped mixed-fragment PAF+cg", ["-c", "-x", "sr", str(ref), str(frag)]),
        ("generated split-prefix multi-contig PAF+cg", ["-c", "-I", "180", "--split-prefix", str(tmpdir / "split"), str(ref), str(split_qry)]),
    ]
    paf_core_cases = [
        ("generated annotated multi-intron splice PAF core", ["-c", "-x", "splice", "--junc-bed", str(splice_bed), str(splice_ref), str(splice_qry)]),
    ]
    paf_overlap_cases = [
        ("generated ava-ont overlap PAF core", ["-x", "ava-ont", str(ava), str(ava)]),
        ("generated ava-pb overlap PAF core", ["-x", "ava-pb", str(ava), str(ava)]),
    ]
    sam_core_cases = [
        ("generated exact multi-contig SAM", ["-a", str(ref), str(qry)]),
        ("generated no-hit SAM hit-only", ["-a", "--sam-hit-only", str(ref), str(nohit_qry)]),
        ("generated grouped mixed-fragment SAM", ["-a", "-x", "sr", str(ref), str(frag)]),
    ]
    asymmetric_cases = [
        (
            "generated copy-comment SAM",
            ["-a", "-y", str(ref), str(comment_qry)],
            ["-a", "--copy-comment", str(ref), str(comment_qry)],
            "sam",
        ),
    ]
    return paf_cases, paf_core_cases, paf_overlap_cases, sam_core_cases, asymmetric_cases


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
    return [line for line in proc.stdout.splitlines() if line]


def run_ok(program: str, args: list[str]) -> None:
    proc = subprocess.run(
        [program, *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{program} {' '.join(args)} exited {proc.returncode}")


def sam_core(lines: list[str]) -> list[str]:
    normalized = []
    for line in lines:
        if line.startswith("@"):
            continue
        fields = line.split("\t")
        tags = {}
        for field in fields[11:]:
            parts = field.split(":", 2)
            if len(parts) == 3:
                tags[parts[0]] = field
        selected_tags = [tags[tag] for tag in SAM_TAGS if tag in tags]
        normalized.append("\t".join([*fields[:6], *selected_tags]))
    return normalized


def sam_header_core(lines: list[str]) -> list[str]:
    normalized = []
    for line in lines:
        if line.startswith("@PG"):
            continue
        if line.startswith("@"):
            normalized.append(line)
            continue
        normalized.extend(sam_core([line]))
    return normalized


def paf_core(lines: list[str]) -> list[str]:
    normalized = []
    for line in lines:
        fields = line.split("\t")
        tags = {}
        for field in fields[12:]:
            parts = field.split(":", 2)
            if len(parts) == 3:
                tags[parts[0]] = field
        selected_tags = [tags[tag] for tag in ("tp", "cm", "s1", "s2", "rl") if tag in tags]
        normalized.append("\t".join([*fields[:12], *selected_tags]))
    return normalized


def paf_overlap_core(lines: list[str]) -> list[str]:
    return sorted(paf_core(lines))


def check_case(name: str, args: list[str], c_bin: str, rust_bin: str, mode: str) -> bool:
    return check_case_asymmetric(name, args, args, c_bin, rust_bin, mode)


def check_case_asymmetric(
    name: str,
    c_args: list[str],
    rust_args: list[str],
    c_bin: str,
    rust_bin: str,
    mode: str,
) -> bool:
    c_lines = run(c_bin, c_args)
    rust_lines = run(rust_bin, rust_args)
    if mode == "sam-core":
        c_lines = sam_core(c_lines)
        rust_lines = sam_core(rust_lines)
    elif mode == "sam-header-core":
        c_lines = sam_header_core(c_lines)
        rust_lines = sam_header_core(rust_lines)
    elif mode == "sam":
        c_lines = [line for line in c_lines if not line.startswith("@")]
        rust_lines = [line for line in rust_lines if not line.startswith("@")]
    elif mode == "paf-core":
        c_lines = paf_core(c_lines)
        rust_lines = paf_core(rust_lines)
    elif mode == "paf-overlap-core":
        c_lines = paf_overlap_core(c_lines)
        rust_lines = paf_overlap_core(rust_lines)
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


def check_output_file_case(
    name: str,
    args: list[str],
    c_bin: str,
    rust_bin: str,
    tmpdir: Path,
    mode: str,
) -> bool:
    c_out = tmpdir / f"{name.replace(' ', '_')}.c.out"
    rust_out = tmpdir / f"{name.replace(' ', '_')}.rust.out"
    run_ok(c_bin, ["-o", str(c_out), *args])
    run_ok(rust_bin, ["-o", str(rust_out), *args])
    c_lines = [line for line in c_out.read_text(encoding="ascii").splitlines() if line and not line.startswith("@")]
    rust_lines = [line for line in rust_out.read_text(encoding="ascii").splitlines() if line and not line.startswith("@")]
    if mode == "sam-core":
        c_lines = sam_core(c_lines)
        rust_lines = sam_core(rust_lines)
    elif mode == "paf-core":
        c_lines = paf_core(c_lines)
        rust_lines = paf_core(rust_lines)
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


def check_index_roundtrip_case(
    name: str,
    dump_args: list[str],
    query_args: list[str],
    c_bin: str,
    rust_bin: str,
    tmpdir: Path,
    mode: str,
) -> bool:
    c_idx = tmpdir / f"{name.replace(' ', '_')}.c.mmi"
    rust_idx = tmpdir / f"{name.replace(' ', '_')}.rust.mmi"
    run_ok(c_bin, ["-d", str(c_idx), *dump_args])
    run_ok(rust_bin, ["-d", str(rust_idx), *dump_args])
    ok = True
    for label, idx in [("C-written index", c_idx), ("Rust-written index", rust_idx)]:
        case_name = f"{name} {label}"
        ok = check_case(case_name, [str(idx), *query_args], c_bin, rust_bin, mode) and ok
    return ok


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
    with tempfile.TemporaryDirectory(prefix="mm2rs-parity-") as tmp:
        gen_paf, gen_paf_core, gen_paf_overlap, gen_sam_core, gen_asym = generated_cases(Path(tmp))
        for name, case_args in [*PAF_CASES, *gen_paf]:
            ok = check_case(name, case_args, args.c_bin, args.rust_bin, "paf") and ok
        for name, case_args in [*PAF_CORE_CASES, *gen_paf_core]:
            ok = check_case(name, case_args, args.c_bin, args.rust_bin, "paf-core") and ok
        for name, case_args in gen_paf_overlap:
            ok = check_case(name, case_args, args.c_bin, args.rust_bin, "paf-overlap-core") and ok
        for name, case_args in [*SAM_CORE_CASES, *gen_sam_core]:
            ok = check_case(name, case_args, args.c_bin, args.rust_bin, "sam-core") and ok
        for name, case_args in SAM_HEADER_CASES:
            ok = check_case(name, case_args, args.c_bin, args.rust_bin, "sam-header-core") and ok
        for name, c_args, rust_args, mode in gen_asym:
            ok = check_case_asymmetric(name, c_args, rust_args, args.c_bin, args.rust_bin, mode) and ok
        tmpdir = Path(tmp)
        ok = check_output_file_case(
            "MT output file PAF+cg",
            ["-c", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"],
            args.c_bin,
            args.rust_bin,
            tmpdir,
            "paf",
        ) and ok
        ok = check_index_roundtrip_case(
            "x3s dumped index",
            ["minimap2/test/x3s-ref.fa"],
            ["minimap2/test/x3s-qry.fa"],
            args.c_bin,
            args.rust_bin,
            tmpdir,
            "paf",
        ) and ok
        ok = check_index_roundtrip_case(
            "x3s no-seq dumped index",
            ["--idx-no-seq", "minimap2/test/x3s-ref.fa"],
            ["minimap2/test/x3s-qry.fa"],
            args.c_bin,
            args.rust_bin,
            tmpdir,
            "paf",
        ) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
