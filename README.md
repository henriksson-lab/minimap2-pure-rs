# minimap2-pure-rs

A pure Rust reimplementation of [minimap2](https://github.com/lh3/minimap2) v2.30 (commit `de3c6ec`), the versatile sequence alignment program for long and short reads.

The current implementation has strong parity with the C version on the bundled
DNA mapping fixtures and is effectively at C speed on the local benchmark suite.
It is not yet a full minimap2 replacement; see [Known gaps](#known-gaps).

This is a translation of the original code and not the authoritative
implementation. For supported workflows, the goal is to generate the same output
as C minimap2. Please report deviations with test data.

The aim of this project is to increase performance, especially by providing this code through a type-safe library interface.
The code can also be compiled to be used for webassembly.

**Do not use this for production yet. Some minimap2 features are incomplete or
not wired through the CLI.**

## Features

- **Pure Rust** -- no C dependencies or FFI
- **Cross-compatible single-part index I/O** -- reads and writes standard `.mmi`
  index files
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
minimap2-pure-rs -x splice   ref.fa rna_reads.fq     # Incomplete; see Known gaps
```

### Index management

```bash
# Build and save index
minimap2-pure-rs -d ref.mmi ref.fa

# Map using prebuilt index
minimap2-pure-rs ref.mmi query.fq
```

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
      --junc-bed <FILE>     Parsed, but junction annotations are not consumed by DP
      --write-junc          Emit junction lines from spliced alignments
      --junc-bonus <INT>    Splice junction score bonus
      --junc-pen <INT>      Splice junction penalty
  -J <INT>                  Splice model selector (0 old, 1 new)
  -u <CHAR>                 Splice strand mode (f, r, b, n)
      --alt <FILE>          ALT contig name list
      --alt-drop <FLOAT>    ALT contig score drop fraction
      --split-prefix <PFX>  Split-index merge for single-end mapping
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

These are known differences or validation limits relative to full C minimap2
behavior:

- **Splice and junction-aware alignment is implemented for the tracked Rust
  translation gaps.** `--splice` applies splice preset parameters and
  splice-aware chaining, converts splice-sized reference skips to `N`, emits
  transcript-strand tags, handles BED12 junction annotations, supports exact
  clipped-junction extension including ambiguous annotated candidates, and uses
  splice-specific DP scoring. Coverage is still fixture-based rather than a
  full upstream minimap2 conformance suite.
- **Multi-part index mapping is implemented for the tested CLI paths.**
  `--split-prefix` maps single-end, grouped fragment, and two-file paired-end
  reads through split index parts and merges per-query hits. Strict interleaved
  split helpers are also present.
- **Short-read high-occurrence re-chaining is implemented for the tested
  single-segment and paired-end paths.** Broader short-read datasets may still
  expose differences in secondary ordering or pairing heuristics.
- **ALT contig metadata follows C minimap2's mapping-time model.** `--alt` and
  `--alt-drop` are supported, and ALT lists are applied when mapping. `.mmi`
  files do not carry ALT state; provide `--alt` again when loading an index.
- **Some advanced CLI/output modes are partial.** `--qstrand` is currently
  supported only for PAF without CIGAR/SAM. `--ds`, `--write-junc`,
  `--copy-comment`, `--sam-hit-only`, `--secondary-seq`, `--idx-no-seq`, `-L`,
  `-f`, `-F`, `-I`, `-J`, `-u`, `-Q`, `--alt`, `--alt-drop`, `--junc-bonus`,
  and `--junc-pen` are wired, with fixture coverage for the most important
  output paths.

## Testing

```bash
# Unit tests
cargo test

# Integration tests against C minimap2
cargo test --test integration

# All tests
cargo test --all
```

## Performance

Current local fixture benchmark against the vendored C minimap2:

```bash
scripts/benchmark_speed.py --reps 5 --threads 1,3
```

Built with `--release`, `-C target-cpu=native`, LTO, and `codegen-units = 1`.
Ratios below are Rust wall time divided by C minimap2 wall time; values below
`1.00x` mean Rust was faster.

| Case | Threads | C mean | Rust mean | Rust/C mean | Rust/C median |
|------|---------|--------|-----------|-------------|---------------|
| MT default PAF | 1 | 0.0167s | 0.0151s | 0.90x | 0.91x |
| MT default PAF | 3 | 0.0162s | 0.0149s | 0.92x | 0.95x |
| MT default PAF+cg | 1 | 0.0283s | 0.0279s | 0.99x | 1.06x |
| MT default PAF+cg | 3 | 0.0300s | 0.0274s | 0.91x | 0.94x |
| MT map-hifi PAF+cg | 1 | 0.0319s | 0.0296s | 0.93x | 0.89x |
| MT map-hifi PAF+cg | 3 | 0.0314s | 0.0328s | 1.05x | 1.01x |
| MT map-hifi SAM | 1 | 0.0324s | 0.0296s | 0.91x | 0.97x |
| MT map-hifi SAM | 3 | 0.0317s | 0.0327s | 1.03x | 1.00x |
| chr11 single HiFi PAF+cg | 1 | 0.0564s | 0.0499s | 0.88x | 0.86x |
| chr11 single HiFi PAF+cg | 3 | 0.0573s | 0.0554s | 0.97x | 0.92x |
| chr11 x200 HiFi PAF+cg | 1 | 3.4601s | 3.5169s | 1.02x | 1.02x |
| chr11 x200 HiFi PAF+cg | 3 | 1.2451s | 1.2991s | 1.04x | 1.00x |
| chr11 x200 HiFi SAM | 1 | 3.5380s | 3.7272s | 1.05x | 1.04x |
| chr11 x200 HiFi SAM | 3 | 1.3200s | 1.3298s | 1.01x | 1.01x |

On these fixtures, Rust is effectively at parity with C minimap2. The MT cases
are very small and include startup/indexing noise; the `chr11 x200` cases are
the more useful local signal for alignment throughput.

Key factors: SSE4.1+AVX2 SIMD extension alignment, SSE2 low-level local
alignment (`ksw_ll_i16`), bounds-check elimination in hot loops, thread-local
buffer reuse to minimize allocations, and const-generic kernel specialization.
See [OPTIMIZATION.md](OPTIMIZATION.md) for details.

## License

MIT

## Acknowledgments

Based on [minimap2](https://github.com/lh3/minimap2) by Heng Li.
