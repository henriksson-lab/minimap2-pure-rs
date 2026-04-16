# minimap2-pure-rs

A pure Rust reimplementation of [minimap2](https://github.com/lh3/minimap2) v2.30 (commit `de3c6ec`), the versatile sequence alignment program for long and short reads.

The current implementation has strong parity with the C version on the bundled
DNA mapping fixtures. On the larger local `chr11 x200` HiFi fixture it is
currently within about 3-9% of C minimap2, depending on output mode and thread
count.
It is not yet a drop-in replacement for every minimap2 workflow; see
[Known gaps](#known-gaps).

This is a translation of the original code and not the authoritative
implementation. For supported workflows, the goal is to generate the same output
as C minimap2. Please report deviations with test data.

The aim of this project is to increase performance, especially by providing this code through a type-safe library interface.
The code can also be compiled to be used for webassembly.

**Do not use this for production yet. Some minimap2 workflows are not
exhaustively validated against the C implementation.**

## Features

- **Pure Rust** -- no C dependencies or FFI
- **Cross-compatible single-part index I/O** -- reads and writes standard `.mmi`
  index files
- **ALT contig metadata** -- `--alt`/`--alt-drop` support plus ALT flag
  round-tripping in Rust-written `.mmi` files
- **PAF/SAM parity on tested fixtures** -- checked against the vendored C
  minimap2 for representative long-read, HiFi, assembly, and split-alignment
  cases
- **CIGAR generation for supported workflows** via anchor-based gap-filling
  alignment
- **Major DNA presets**: `map-ont`, `map-pb`, `map-hifi`, `sr`, `asm5`,
  `asm10`, `asm20`, `ava-ont`, `ava-pb`, and related presets
- **Splice presets**: `splice`, `splice:hq`, `splice:sr`, and `cdna` set
  minimap2-like options, including splice-aware chaining, `N` CIGAR skips,
  transcript-strand tags, junction annotation handling, and splice-specific
  DP scoring
- **Multi-threaded** mapping with [rayon](https://github.com/rayon-rs/rayon)
- **Paired-end and grouped fragment** support is implemented for two-file
  pairs and adjacent same-name single-file fragments, with C-vs-Rust parity
  coverage on bundled fixtures

## Installation

```bash
cargo install --path .
```

Or build from source:

```bash
cargo build --release
# Binary at ./target/release/minimap2-pure-rs
```

## Usage

### Basic mapping (PAF output)

```bash
minimap2-pure-rs ref.fa query.fq
```

### With CIGAR string

```bash
minimap2-pure-rs -c ref.fa query.fq
```

### SAM output

```bash
minimap2-pure-rs -a ref.fa query.fq | samtools sort -o aligned.bam
```

### Paired-end short reads

```bash
minimap2-pure-rs -ax sr ref.fa read1.fq read2.fq
```

### Using presets

```bash
minimap2-pure-rs -x map-ont  ref.fa ont_reads.fq     # Oxford Nanopore
minimap2-pure-rs -x map-pb   ref.fa pb_reads.fq      # PacBio CLR
minimap2-pure-rs -x map-hifi ref.fa hifi_reads.fq    # PacBio HiFi
minimap2-pure-rs -x sr       ref.fa reads.fq         # Short reads
minimap2-pure-rs -x asm5     ref.fa assembly.fa       # Assembly-to-reference
minimap2-pure-rs -x splice   ref.fa rna_reads.fq     # RNA/splice-aware mapping
```

### Index management

```bash
# Build and save index
minimap2-pure-rs -d ref.mmi ref.fa

# Map using prebuilt index
minimap2-pure-rs ref.mmi query.fq
```

When `--alt` is provided while dumping an index, Rust-written `.mmi` files
persist ALT contig flags and restore them when loaded by this crate.

## Options

```
Usage: minimap2-pure-rs [OPTIONS] <TARGET> [QUERY]...

Arguments:
  <TARGET>    Reference FASTA/index file
  [QUERY]...  Query FASTA/FASTQ file(s)

Options:
  -x, --preset <PRESET>     Preset (map-ont, map-pb, map-hifi, sr, splice, asm5, asm10, asm20, ...)
  -k <KMER>                 k-mer size [default: from preset]
  -w <WINDOW>               Minimizer window size [default: from preset]
  -t, --threads <THREADS>   Number of threads [default: 3]
  -a, --sam                 Output SAM
  -c                        Output CIGAR in PAF
      --cs                  Output cs tag
      --ds                  Output ds tag
      --MD                  Output MD tag
      --eqx                 Use =/X instead of M in CIGAR
  -A <MATCH_SCORE>          Matching score
  -B <MISMATCH>             Mismatch penalty
  -f <FLOAT>                Filter top fraction of repetitive minimizers
  -O <INT[,INT]>            Gap open penalty [,second]
  -E <INT[,INT]>            Gap extension penalty [,second]
  -r <INT[,INT]>            Bandwidth [,long]
  -F <MAX_FRAG_LEN>         Max fragment length
  -N <BEST_N>               Top N secondary alignments
      --secondary <yes|no>  Output secondary alignments
  -s <MIN_DP_SCORE>         Min alignment score
  -Q                        Do not output base qualities in SAM
      --sam-hit-only        Only output SAM records with hits
      --secondary-seq       Output SEQ/QUAL for secondary alignments with hard clipping
      --paf-no-hit          Output unmapped in PAF
  -d <FILE>                 Dump index to file
  -Y                        Soft clipping
  -L                        Long CIGAR in CG tag for BAM compatibility
  -R <STR>                  Read group line
  -H                        HPC mode (homopolymer compression)
  -P                        Output all chains
      --for-only            Only map to forward strand
      --rev-only            Only map to reverse strand
      --qstrand             Preserve query-strand coordinates for reverse-strand PAF output
      --copy-comment        Copy FASTA/FASTQ comments to output
      --idx-no-seq          Build an index without target sequences
      --junc-bed <FILE>     BED12 splice junction annotations
      --write-junc          Emit junction lines from spliced alignments
      --junc-bonus <INT>    Splice junction score bonus
      --junc-pen <INT>      Splice junction penalty
  -J <INT>                  Splice model selector (0 old, 1 new)
  -u <CHAR>                 Splice strand mode (f, r, b, n)
      --alt <FILE>          ALT contig name list
      --alt-drop <FLOAT>    ALT contig score drop fraction
      --split-prefix <PFX>  Split-index mapping and merge
  -G <MAX_INTRON>           Max intron length (splice mode)
  -T, --sdust-thres <INT>   SDUST threshold (0 to disable)
  -K <NUM>                  Mini-batch size for mapping
  -I <NUM>                  Split index for every NUM input bases
  -h, --help                Print help
  -V, --version             Print version
```

## Output formats

### PAF (default)

Tab-separated format with 12 mandatory fields plus optional tags:

```
query  qlen  qs  qe  strand  target  tlen  rs  re  mlen  blen  mapq  [tags]
```

Tags include: `tp` (type), `cm` (minimizer count), `s1`/`s2` (chain scores),
`NM` (edit distance), `ms` (max DP score), `AS` (alignment score),
`de`/`dv` (divergence), `cg` (CIGAR), `cs` (cs tag), `rl` (repeat length).

### SAM (-a)

Standard SAM format compatible with samtools and downstream tools.
Includes CIGAR, clipping (hard by default, soft with `-Y`), and alignment tags.

## Library API

Use minimap2-pure-rs as a Rust library:

```rust
use minimap2::aligner::Aligner;

// Build aligner with preset
let aligner = Aligner::builder()
    .preset("map-ont")
    .index("ref.fa")
    .with_cigar()
    .build()
    .unwrap();

// Map a sequence
let hits = aligner.map(b"ACGTACGT...");
for hit in &hits {
    let name = aligner.seq_name(hit.rid as usize);
    println!("{name}:{}-{} strand={} mapq={}",
        hit.rs, hit.re,
        if hit.rev { '-' } else { '+' },
        hit.mapq);
}
```

Or use the lower-level API:

```rust
use minimap2::prelude::*;

let (io, mut mo) = preset("map-hifi").unwrap();
let mi = MmIdx::build_from_file("ref.fa", io.w as i32, io.k as i32,
    io.bucket_bits, io.flag, io.mini_batch_size, io.batch_size).unwrap().unwrap();
mapopt_update(&mut mo, &mi);

let result = map_query(&mi, &mo, "read1", b"ACGT...");
```

## Architecture

```
src/
  lib.rs              Public API + prelude
  aligner.rs          High-level Aligner builder API
  main.rs             CLI (clap)
  types.rs            Core data types (Mm128, AlignReg, Cigar, ...)
  flags.rs            Bitflags (MapFlags, IdxFlags, KswFlags)
  options.rs          Presets and option validation
  seq.rs              DNA encoding, complement, 4-bit packing
  sort.rs             Radix sort for Mm128/u64
  sketch.rs           Minimizer extraction (hash64, HPC)
  sdust.rs            Low-complexity masking
  bseq.rs             FASTA/FASTQ I/O (gzip support)
  seed.rs             Seed collection and filtering
  hit.rs              Hit filtering, MAPQ, parent/secondary
  esterr.rs           Divergence estimation
  pe.rs               Paired-end pairing
  jump.rs             Splice junction extension helpers
  map.rs              Core mapping pipeline
  pipeline.rs         Multi-threaded file mapping (SE + PE)
  index/
    mod.rs            Index construction and queries
    bucket.rs         Hash bucket with hashbrown
    io.rs             Binary .mmi format I/O
  chain/
    mod.rs            Scoring functions (comput_sc, mg_log2)
    backtrack.rs      Chain backtracking
    dp.rs             DP chaining algorithm
    rmq.rs            RMQ chaining with BTreeMap
  align/
    mod.rs            Anchor-based alignment (align_skeleton)
    ksw2.rs           Scalar KSW2 (single + dual affine gap)
    ksw2_simd.rs      SIMD scaffold (runtime dispatch)
    score.rs          Scoring matrix generation
  format/
    mod.rs            Output dispatch
    paf.rs            PAF formatting
    sam.rs            SAM formatting
    cs.rs             cs/MD tag generation
```

## Differences from C minimap2

- **Fixture-equivalent, not globally feature-complete**: coordinates, scores,
  CIGARs, and selected SAM tags match on the current regression suite; untested
  workflows may still differ
- **SIMD alignment**: SSE4.1 rotated-band DP kernels with AVX2 H-score tracking, const-generic CIGAR/score-only specialization, and runtime dispatch (SSE4.1 -> SSE2 -> scalar fallback)
- **Anchor-based gap-filling**: CIGAR is generated by aligning inter-anchor gaps individually, same approach as C minimap2
- **RMQ chaining**: implemented with a simpler `BTreeMap` range scan rather than
  C minimap2's augmented red-black tree; intended to preserve behavior, but the
  asymptotic performance differs on very large anchor sets

## Known gaps

These are the remaining gaps and validation limits relative to full C minimap2
behavior:

- **Conformance is fixture-driven, not exhaustive.** The regression suite checks
  PAF/SAM parity against vendored C minimap2 for representative ONT, HiFi,
  assembly, short-read, split-index, paired/grouped fragment, ALT, and
  splice/junction cases. It is not a full upstream minimap2 conformance suite,
  and untested datasets or option combinations may still differ.
- **Splice support is implemented but not broadly benchmarked on RNA data.**
  `splice`, `splice:hq`, `splice:sr`, BED12 junction annotations, `N` CIGAR
  skips, transcript-strand tags, annotated junction rescue, and splice-specific
  DP scoring are wired and covered by fixtures. Complex noisy transcriptome
  cases still need real-data validation.
- **Split-index mapping is implemented for tested CLI paths.** `--split-prefix`
  maps single-end, grouped fragment, and two-file paired-end reads through split
  index parts and merges per-query hits. Compatibility has been validated on
  local fixtures and the full 587,760-pair E. coli SRR13321180 short-read
  sample in both PAF and SAM with forced `-I 500k --split-prefix`, including
  C-style split-record `frag_gap` handling and post-merge paired-end MAPQ/flag
  behavior. It has not been validated at the scale of upstream production
  split-index runs.
- **Short-read paired-end real-data conformance is covered for one full E. coli
  sample.** High-occurrence re-chaining, paired-end heap-sort ordering, radix
  tie behavior, MAPQ, `cm`, `s1`/`s2`, short-read `ms:i` scoring,
  sequence-aware CIGAR normalization, z-drop split counting, and
  secondary/supplementary SAM clipping now match C minimap2 for the full
  SRR13321180 E. coli conformance sample in both PAF and SAM. Broader
  short-read datasets are still needed to make that coverage representative.
- **All-vs-all overlap presets have synthetic fixture coverage.** `ava-ont`
  and `ava-pb` are wired as presets and covered by same-file overlap
  C-vs-Rust regression cases. Broader overlap validation on real read sets is
  still needed.
- **Some CLI/output combinations are intentionally constrained.** `--qstrand`
  is supported only for PAF without CIGAR/SAM. Advanced options such as `-d`,
  `-o`, `-R`, `-K`, `--ds`, `--write-junc`, `--copy-comment`,
  `--paf-no-hit`, `--sam-hit-only`, `--secondary-seq`, `--idx-no-seq`, `-L`,
  `-f`, `-F`, `-I`, `-J`, `-u`, `-Q`, `--junc-bonus`, and `--junc-pen` are
  wired, but many combinations only have fixture coverage.

## Testing

```bash
# Unit tests
cargo test

# Integration tests against C minimap2
cargo test --test integration

# Fixture parity matrix against C minimap2
scripts/parity_matrix.py

# Real-data conformance manifest
cp scripts/conformance_manifest.example.tsv scripts/conformance_manifest.local.tsv
$EDITOR scripts/conformance_manifest.local.tsv
scripts/conformance_matrix.py scripts/conformance_manifest.local.tsv

# Minimal public paired-end dataset setup
N_PAIRS=50000 scripts/prepare_minimal_conformance_data.sh
scripts/conformance_matrix.py data/conformance/ecoli_srr13321180/conformance_manifest.tsv

# Full local E. coli short-read and forced split-index validation, if raw FASTQs are present
scripts/conformance_matrix.py data/conformance/ecoli_srr13321180/conformance_full_manifest.tsv

# All tests
cargo test --all
```

The conformance manifest runner is intended for datasets too large or
site-specific to commit to this repository. Each row selects a comparison mode
such as exact PAF, normalized PAF core fields, normalized SAM core fields, or
SAM headers plus core fields.

## Performance

Current local fixture benchmark against the vendored C minimap2:

```bash
scripts/benchmark_speed.py --threads 1 --reps 7 --warmups 1
scripts/benchmark_speed.py --threads 1,4,8 --reps 3 --warmups 1 --case chr11
```

Built with `--release`, `-C target-cpu=native`, LTO, and `codegen-units = 1`.
Ratios below are Rust wall time divided by C minimap2 wall time; values below
`1.00x` mean Rust was faster.

| Case | Threads | C median | Rust median | Rust/C median |
|------|---------|----------|-------------|---------------|
| MT default PAF | 1 | 0.0167s | 0.0149s | 0.89x |
| MT default PAF+cg | 1 | 0.0296s | 0.0252s | 0.85x |
| MT map-hifi PAF+cg | 1 | 0.0312s | 0.0327s | 1.05x |
| MT map-hifi SAM | 1 | 0.0304s | 0.0266s | 0.87x |
| chr11 single HiFi PAF+cg | 1 | 0.0550s | 0.0564s | 1.02x |
| chr11 x200 HiFi PAF+cg | 1 | 3.4860s | 3.6599s | 1.05x |
| chr11 x200 HiFi SAM | 1 | 3.4380s | 3.7170s | 1.08x |
| chr11 x200 HiFi PAF+cg | 4 | 0.9814s | 1.0420s | 1.06x |
| chr11 x200 HiFi SAM | 4 | 0.9950s | 1.0476s | 1.05x |
| chr11 x200 HiFi PAF+cg | 8 | 0.5866s | 0.5991s | 1.02x |
| chr11 x200 HiFi SAM | 8 | 0.5969s | 0.6136s | 1.03x |

The MT cases are upstream mitochondrial smoke fixtures and are too small for
performance claims; startup, indexing, cache state, and timing noise dominate
their wall time. The `chr11 x200` cases are a better local regression signal,
but still only exercise a small HiFi fixture. Broad performance claims should be
based on real-data manifests.

For real-data benchmarking, copy `scripts/benchmark_manifest.example.tsv`, fill
in local FASTA/FASTQ paths, and run:

```bash
scripts/benchmark_speed.py --threads 1,8 --reps 5 --warmups 1 \
  --manifest scripts/benchmark_manifest.local.tsv --no-fixtures
```

A useful manifest should include at least HiFi reads, ONT reads, paired short
reads, assembly contigs, and one splice/RNA case when splice performance matters.

Key factors: SSE4.1+AVX2 SIMD extension alignment, SSE2 low-level local
alignment (`ksw_ll_i16`), bounds-check elimination in hot loops, thread-local
buffer reuse to minimize allocations, and const-generic kernel specialization.
See [OPTIMIZATION.md](OPTIMIZATION.md) for details.

## License

MIT

## Acknowledgments

Based on [minimap2](https://github.com/lh3/minimap2) by Heng Li.
