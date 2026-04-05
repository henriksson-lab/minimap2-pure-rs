# minimap2-rs

A pure Rust reimplementation of [minimap2](https://github.com/lh3/minimap2), the versatile sequence alignment program for long and short reads.

## Features

- **Pure Rust** -- no C dependencies or FFI
- **Cross-compatible index I/O** -- reads and writes `.mmi` index files interchangeable with C minimap2
- **PAF output matches C minimap2** on all test cases
- **Full CIGAR generation** via anchor-based gap-filling alignment
- **All major presets**: `map-ont`, `map-pb`, `map-hifi`, `sr`, `splice`, `asm5`, `asm10`, `asm20`, `ava-ont`, `ava-pb`, and more
- **Multi-threaded** mapping with [rayon](https://github.com/rayon-rs/rayon)
- **Paired-end** support (two-file input with proper SAM flags)

## Installation

```bash
cargo install --path .
```

Or build from source:

```bash
cargo build --release
# Binary at ./target/release/minimap2-rs
```

## Usage

### Basic mapping (PAF output)

```bash
minimap2-rs ref.fa query.fq
```

### With CIGAR string

```bash
minimap2-rs -c ref.fa query.fq
```

### SAM output

```bash
minimap2-rs -a ref.fa query.fq | samtools sort -o aligned.bam
```

### Paired-end short reads

```bash
minimap2-rs -ax sr ref.fa read1.fq read2.fq
```

### Using presets

```bash
minimap2-rs -x map-ont  ref.fa ont_reads.fq     # Oxford Nanopore
minimap2-rs -x map-pb   ref.fa pb_reads.fq      # PacBio CLR
minimap2-rs -x map-hifi ref.fa hifi_reads.fq    # PacBio HiFi
minimap2-rs -x sr       ref.fa reads.fq         # Short reads
minimap2-rs -x asm5     ref.fa assembly.fa       # Assembly-to-reference
minimap2-rs -x splice   ref.fa rna_reads.fq     # Spliced alignment
```

### Index management

```bash
# Build and save index
minimap2-rs -d ref.mmi ref.fa

# Map using prebuilt index
minimap2-rs ref.mmi query.fq
```

## Options

```
Usage: minimap2-rs [OPTIONS] <TARGET> [QUERY]...

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
      --MD                  Output MD tag
      --eqx                 Use =/X instead of M in CIGAR
  -A <MATCH_SCORE>          Matching score
  -B <MISMATCH>             Mismatch penalty
  -O <INT[,INT]>            Gap open penalty [,second]
  -E <INT[,INT]>            Gap extension penalty [,second]
  -r <INT[,INT]>            Bandwidth [,long]
  -N <BEST_N>               Top N secondary alignments
      --secondary <yes|no>  Output secondary alignments
  -s <MIN_DP_SCORE>         Min alignment score
      --paf-no-hit          Output unmapped in PAF
  -d <FILE>                 Dump index to file
  -Y                        Soft clipping
  -R <STR>                  Read group line
  -H                        HPC mode (homopolymer compression)
  -P                        Output all chains
      --for-only            Only map to forward strand
      --rev-only            Only map to reverse strand
  -G <MAX_INTRON>           Max intron length (splice mode)
      --sdust-thres <INT>   SDUST threshold (0 to disable)
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

Use minimap2-rs as a Rust library:

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
  jump.rs             Splice junction extension (stub)
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

- **Functionally equivalent**, not bit-exact: coordinates and scores match on all test cases; minor floating-point differences in divergence tags
- **Scalar DP alignment**: the Smith-Waterman kernel uses a scalar implementation rather than SSE2/SSE4.1 SIMD. PAF mode (no CIGAR) runs at the same speed; CIGAR mode is ~3.5x slower on long divergent alignments
- **Anchor-based gap-filling**: CIGAR is generated by aligning inter-anchor gaps individually, following the same approach as C but without the rotated-band optimization
- **Splice junctions**: `jump.rs` is a stub; splice-aware alignment via `--splice` uses the standard DP

## Testing

```bash
# Unit tests (104)
cargo test

# Integration tests against C minimap2 test data (15)
cargo test --test integration

# All tests
cargo test --all
```

## Performance

On the MT genome test (single thread):

| Mode | Rust | C minimap2 | Ratio |
|------|------|-----------|-------|
| PAF  | 15ms | 15ms | 1.0x |
| CIGAR | 98ms | 28ms | 3.5x |

The CIGAR mode gap is due to scalar vs SIMD DP alignment. The SIMD scaffold (`ksw2_simd.rs`) is in place for future SSE2 acceleration.

## License

MIT

## Acknowledgments

Based on [minimap2](https://github.com/lh3/minimap2) by Heng Li.
