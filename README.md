# minimap2-pure-rs

A pure Rust reimplementation of [minimap2](https://github.com/lh3/minimap2) v2.30 (commit `de3c6ec`), the versatile sequence alignment program for long and short reads.

It is not yet validated as a drop-in replacement for every minimap2 workflow; see [Validation scope](#validation-scope).

** More issues found in an audit 2026-04-18. Not ready for production **


## This is an LLM-mediated faithful (hopefully) translation, not the original code!

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this manifesto](https://rewrites.bio/) but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back into the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information
* **If you are the author of the original code and wish to move to Rust, you can obtain ownership of this repository and crate**. Until then, our commitment is to offer an as-faithful-as-possible translation of a snapshot of your code. If we find serious bugs, we will report them to you. Otherwise we will just replicate them, to ensure comparability across studies that claim to use package XYZ v.666. Think of this like a fancy Ubuntu .deb-package of your software - that is how we treat it

This blurb might be out of date. Go to [this page](https://github.com/henriksson-lab/rustification) for the latest information and further information about how we approach translation


## Features

- **Pure Rust** -- no C dependencies or FFI
- **Cross-compatible single-part index I/O** -- reads and writes standard `.mmi`
  index files
- **ALT contig metadata** -- `--alt`/`--alt-drop` support plus ALT flag
  round-tripping in Rust-written `.mmi` files
- **PAF/SAM parity on tested fixtures and datasets** -- checked against the
  vendored C minimap2 for representative ONT, HiFi, assembly, short-read,
  split-index, ALT, splice/RNA, paired/grouped fragment, and overlap cases
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
- **RMQ chaining**: implemented with compressed-key segment-tree range minima
  plus a `BTreeMap` for C-style reverse iteration in the inner fallback scan;
  intended to preserve C minimap2 behavior while avoiding linear scans for the
  main RMQ query

## Validation scope

The completed TODO items cover the known CLI/output parity gaps that were
tracked for this Rust translation. The remaining caveats are validation scope
and deliberate project scope, not missing implementations tracked for the
tested CLI workflows.

- **Conformance is fixture-driven, not exhaustive.** The regression suite checks
  PAF/SAM parity against vendored C minimap2 for representative ONT, HiFi,
  assembly, short-read, split-index, paired/grouped fragment, ALT, overlap,
  and splice/junction cases. It is not a full upstream minimap2 conformance suite,
  and untested datasets or option combinations may still differ.
- **The small external strict matrix now passes for the tested real-data
  categories.** The external paired E. coli short-read subset, HiFi 10k subset,
  ONT subset, UTI89 `asm10` assembly case, ALT metadata case, forced split-index
  long-read cases, real ONT all-vs-all overlap case, and yeast direct-RNA
  splice case pass strict conformance against C minimap2.
- **Splice support is implemented for the tested RNA workflows.** `splice`,
  `splice:hq`, `splice:sr`, BED12 junction annotations, `N` CIGAR skips,
  transcript-strand tags, annotated junction rescue, and splice-specific DP
  scoring are wired and covered by fixtures. The external yeast direct-RNA
  200-read PAF/SAM strict matrix now matches C minimap2, including the noisy
  annotated intron normalization case `SRR30335018.261583`.
- **Split-index mapping is implemented for tested CLI paths.** `--split-prefix`
  maps single-end, grouped fragment, and two-file paired-end reads through split
  index parts and merges per-query hits. Compatibility has been validated on
  local fixtures and the full 587,760-pair E. coli SRR13321180 short-read
  sample in both PAF and SAM with forced `-I 500k --split-prefix`, including
  C-style split-record `frag_gap` handling and post-merge paired-end MAPQ/flag
  behavior. Broader references and multi-part production workloads should still
  be validated before relying on untested split-index configurations.
- **Short-read paired-end real-data conformance is covered for one full E. coli
  sample.** High-occurrence re-chaining, paired-end heap-sort ordering, radix
  tie behavior, MAPQ, `cm`, `s1`/`s2`, short-read `ms:i` scoring,
  sequence-aware CIGAR normalization, z-drop split counting, and
  secondary/supplementary SAM clipping now match C minimap2 for the full
  SRR13321180 E. coli conformance sample in both PAF and SAM. Broader
  short-read datasets are still needed to make that coverage representative.
- **All-vs-all overlap presets are covered by fixtures and one real ONT
  dataset.** `ava-ont` and `ava-pb` are wired as presets and covered by
  same-file overlap C-vs-Rust regression cases. The external strict matrix also
  includes a real 1000-read ONT `ava-ont` all-vs-all PAF-core conformance case.
  Broader overlap validation on additional real read sets is still needed.
- **Some CLI/output combinations are intentionally constrained.** `--qstrand`
  is supported only for PAF without CIGAR/SAM. Advanced options such as `-d`,
  `-o`, `-R`, `-K`, `--ds`, `--write-junc`, `--copy-comment`,
  `--paf-no-hit`, `--sam-hit-only`, `--secondary-seq`, `--idx-no-seq`, `-L`,
  `-f`, `-F`, `-I`, `-J`, `-u`, `-Q`, `--junc-bonus`, and `--junc-pen` are
  wired, but many combinations only have fixture coverage.
- **C minimap2 ecosystem APIs are out of scope.** This crate targets the
  minimap2 command-line behavior and Rust-native library use. It does not
  implement C ABI compatibility, `libminimap2` embedding APIs, `mappy`, or
  downstream ecosystem interfaces that depend on minimap2's C internals.

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

# Small external HiFi, ONT, assembly, ALT, paired short-read, split-index, overlap, and RNA/splice data
scripts/prepare_external_small_conformance_data.sh

# Fast splice performance loop on the yeast RNA regression fixtures
scripts/benchmark_yeast_splice.py --reads 5000 --check-output

# Currently passing strict external categories
scripts/conformance_matrix.py data/conformance/external_small/conformance_manifest.tsv --category HiFi --category ONT --category Assembly --category ShortRead --category SplitIndex --category ALT --category Overlap --category RNA --diff-limit 40

# Heavier real-world external runs from the same downloaded inputs
scripts/conformance_matrix.py data/conformance/external_small/conformance_full_manifest.tsv

# All tests
cargo test --all
```

The conformance manifest runner is intended for datasets too large or
site-specific to commit to this repository. Each row selects a comparison mode
such as exact PAF, normalized PAF core fields, normalized SAM core fields, or
SAM headers plus core fields.

## Performance

### Whole-genome HiFi benchmark (hg38, `map-hifi`, 8 threads)

Real hg38 primary assembly (3.1 GB FASTA, 194 contigs) × **20,370 unique
HG002 HiFi reads** (374 MB FASTQ, mean 9.6 kb, downloaded from GIAB CCS 10kb
SMRT cell `m54238_180628_014238`). 3 interleaved rounds per scenario, warm
page cache, same NFS mount. Both binaries built with matched aggressive
optimization (`-O3 -march=native -flto` for C, `opt-level=3` + LTO +
`codegen-units=1` + `-C target-cpu=native` for Rust), both invoked with
`-t 8`. Median wall time:

| Scenario | C (2.30-r1290) | Rust (2.30-rs) | Δ |
|----------|---------------:|---------------:|------:|
| Index build + write (FASTA → `.mmi`) | 48.87 s | **42.09 s** | **-14 %** |
| Load `.mmi` + map 20,370 reads | 20.29 s | **16.26 s** | **-20 %** |

Stage breakdown of the load+map scenario (from tool-internal checkpoints):

| Stage | C | Rust | Δ |
|-------|--:|-----:|------:|
| `.mmi` load | 12.8 s | **10.3 s** | -19 % |
| Mapping (20,370 reads done) | 7.6 s | **5.3 s** | **-30 %** |

**Where Rust is actually faster:**

- **Index build (-14 %)** comes from the three-stage reader / sketcher /
  dispatcher pipeline + `par_iter_mut` bucket post-process running with less
  scheduling overhead than C's `kt_pipeline` + `kt_for` (Rust averages 175 %
  CPU vs C's 148 % on 8 threads during indexing).
- **`.mmi` load (-19 %)** comes from bulk-byte reads and hashbrown
  deserialization vs C's khash.
- **Mapping (-30 %)** on diverse real reads, with Rust at 234 % CPU vs C's
  179 %. An earlier, less realistic benchmark that replicated 679 reads
  30× showed the stages essentially tied — the homogeneous data was hiding
  the real picture by saturating cache in both tools. Diverse reads exercise
  branch prediction, cache, and parallelism more realistically.

**Output parity on real data:** byte-identical PAF on both the diverse
20,370-read file (24,978 lines) and the 843-line 679-read file. Known parity
items now include: FASTQ truncation rejection, singleton-in-hash `.mmi`
layout, `chn_pen_gap` / `chn_pen_skip` f64-intermediate rounding, and stable
x-only radix sort of chain anchors before the RMQ re-chain rescue path
(matching `radix_sort_128x` semantics vs Rust's `(x, y)` default `Ord`).
The last three were caught by the tracehash integration — see the notes on
optional parity tracing under `--features tracehash`.

Rust `.mmi` files are **byte-size-identical to C's** (3.50 GB) and
cross-readable: C minimap2 loads Rust-produced indexes and produces its own
native PAF output from them.

**Benchmark fairness caveats:**

- Reads are all from one SMRT cell. Broader cross-library diversity would be
  needed for production claims.
- NFS and CPU frequency scaling add timing noise; interleaved C/Rust rounds
  mitigate drift but do not eliminate it.

### Fixture benchmarks

```bash
scripts/benchmark_speed.py --threads 1 --reps 7 --warmups 1
scripts/benchmark_speed.py --threads 1,4,8 --reps 3 --warmups 1 --case chr11
```

Both binaries built with matched aggressive optimization: C with
`-O3 -march=native -flto`, Rust with `--release`, `-C target-cpu=native`,
LTO, and `codegen-units = 1`. Ratios below are Rust wall time divided by C
minimap2 wall time; values below `1.00x` mean Rust was faster.

| Case | Threads | C median | Rust median | Rust/C median |
|------|---------|----------|-------------|---------------|
| MT default PAF | 1 | 0.0163s | 0.0224s | 1.37x |
| MT default PAF+cg | 1 | 0.0289s | 0.0383s | 1.32x |
| MT map-hifi PAF+cg | 1 | 0.0306s | 0.0409s | 1.34x |
| MT map-hifi SAM | 1 | 0.0343s | 0.0415s | 1.21x |
| chr11 single HiFi PAF+cg | 1 | 0.0560s | 0.0612s | 1.09x |
| chr11 single HiFi PAF+cg | 4 | 0.0633s | 0.0654s | 1.03x |
| chr11 single HiFi PAF+cg | 8 | 0.0631s | 0.0684s | 1.08x |
| chr11 x200 HiFi PAF+cg | 1 | 3.8396s | 3.7036s | **0.96x** |
| chr11 x200 HiFi SAM | 1 | 4.0683s | 3.8622s | **0.95x** |
| chr11 x200 HiFi PAF+cg | 4 | 1.1511s | 1.1629s | 1.01x |
| chr11 x200 HiFi SAM | 4 | 1.1373s | 1.0781s | **0.95x** |
| chr11 x200 HiFi PAF+cg | 8 | 0.6473s | 0.6168s | **0.95x** |
| chr11 x200 HiFi SAM | 8 | 0.6566s | 0.6442s | **0.98x** |

The MT cases are upstream mitochondrial smoke fixtures (~20–40 ms total)
and are too small for performance claims; startup (rayon thread-pool init,
allocator setup, argv parsing) dominates and shows Rust's constant
overhead. The `chr11 single` cases (~60 ms) are still partly startup-bound.
The `chr11 x200` cases (3–4 s single-threaded) are where per-read
throughput dominates and Rust matches or slightly beats C on matched
compile flags. Broad performance claims should be based on real-data
manifests — see the whole-genome HiFi benchmark above.

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
