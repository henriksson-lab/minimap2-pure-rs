# Performance Optimizations vs C minimap2

This documents the key changes that make the Rust minimap2 faster than the C version.
Starting point: 27% slower. Final result: **6-7% faster** in CPU cycles.

## Build Configuration

- `-C target-cpu=native` via `.cargo/config.toml` — largest single win (~17% instruction reduction). Enables AVX2/AVX-512 codegen for all functions, better instruction scheduling for the actual CPU.
- `lto = true`, `codegen-units = 1`, `opt-level = 3` in `Cargo.toml` release profile.

## SIMD Alignment Kernel (`src/align/ksw2_simd.rs`)

### SSE2 low-level local alignment (`ksw_ll_i16`)
The C implementation uses a striped SSE2 `ksw_ll_i16()` for inversion/z-drop checks. A scalar Rust translation dominated the reduced chr11 HiFi regression fixture before optimization (~73% of Rust cycles on the amplified chr11 x200 benchmark), making Rust ~2.8x slower than C on that workload.

`src/align/ksw2.rs` now dispatches `ksw_ll_i16()` to a striped SSE2 implementation on x86_64, with the scalar implementation retained as fallback. After this change, `scripts/benchmark_speed.py --reps 5` reports the chr11 x200 case at approximately parity with C minimap2, and profiling shows both implementations are again dominated by the dual-gap extension kernel.

### SSE4.1 kernel with const-generic specialization
The C version dispatches CIGAR vs score-only at runtime with `if (with_cigar)` branches inside the inner loop. The Rust version uses `extd2_sse41<const WITH_CIGAR: bool>`, letting LLVM generate two fully specialized functions. The CIGAR version has no dead score-only code and vice versa, reducing code size and register pressure.

### AVX2 VEX encoding
`#[target_feature(enable = "sse4.1,avx2")]` on the kernel enables VEX-encoded SSE instructions, which avoid false register dependencies (no partial XMM write penalties) and use 3-operand non-destructive form, reducing `mov` count.

### Extracted score tracking (`score_track_exact`)
The H-score tracking loop was extracted into a separate `#[inline(never)]` function. This reduces the main kernel's register pressure (446 stack refs vs 536 before extraction; C has 421). The extracted function uses AVX2 256-bit operations to process 8 i32 values per iteration instead of 4.

### AVX2 H-score tracking
`_mm256_cvtepi8_epi32` widens 8 signed bytes to 8 i32s in one instruction, replacing the SSE2 pattern of 4 individual `movzbl` + `_mm_setr_epi32`. Halves the H-tracking loop iterations.

## Index Lookup (`src/index/bucket.rs`)

### Unified hash table format
C minimap2 stores singletons with `key|1` and multi-hit with `key` (two different keys). Each lookup probes the hash table twice. The Rust version stores all entries under a single key format with all positions in the `p` array. Single hash probe per lookup.

### Single lookup in seed collection
`seed_collect_all` previously called `mi.get()` then `bucket.get()` (two hash probes). Now uses one `bucket.get()` call and computes multi-hit offsets from the returned slice pointer.

## Allocation Reduction

### Thread-local buffer reuse
Several per-call Vec allocations replaced with thread-local `RefCell<Vec>` patterns:
- `KSW_SCRATCH` / `KSW_I32_SCRATCH` — SIMD DP buffers (already existed)
- `CHAIN_P/F/V/T` — chain DP arrays (4 large vecs per chain call)
- `ALIGN_TSEQ_BUF` / `ALIGN_REV_BUF` — alignment target sequence buffers
- `ALIGN_QSEQ_FWD` / `ALIGN_QSEQ_REV` — query encoding buffers

### Eliminated per-seed allocation
`expand_seeds_to_anchors` previously allocated `vec![*pos]` for every singleton seed. Now uses a stack `[u64; 1]` array and borrows `&bucket.p[offset..count]` directly for multi-hit. Also pre-computes total anchor count for `Vec::with_capacity`.

### In-place seed filtering
`collect_matches` previously cloned kept seeds into a new Vec. Now uses `seeds.retain(|s| !s.flt)` to filter in-place.

### Hoisted coverage Vec in `set_parent`
The `cov: Vec<u64>` was allocated inside the per-region inner loop. Moved outside and reused with `.clear()`.

## Bounds Check Elimination

Unsafe `get_unchecked` / raw pointer access in hot loops where indices are provably in bounds:

- **`mm_sketch`** — `SEQ_NT4_TABLE[seq[i]]` lookups and circular `buf[j]` accesses
- **`lchain_dp`** — inner DP loop accessing `a[iu]`, `f[ju]`, `p[ju]`, `t[ju]`
- **`test_zdrop`** / **`compute_cigar_stats`** — per-base CIGAR walking with matrix lookups
- **SIMD kernel** — `off_a[r]`, `off_e[r]`, backtrack `bt[idx]` accesses

## Code Structure

### Function outlining
Key functions marked `#[inline(never)]` to prevent excessive inlining into `map_query` (reduced from 8599 to 5195 instructions), improving icache behavior:
- `lchain_dp`, `collect_matches`, `expand_seeds_to_anchors`
- `gen_regs`, `set_parent`
- `score_track_exact`

### Stack-allocated sketch buffer
`mm_sketch` uses `[Mm128; 256]` on the stack instead of `vec![...; w]` on the heap.

### Pre-allocated minimizer vector
`map_query` uses `Vec::with_capacity(qseq.len() / w + 16)` instead of `Vec::new()`.

## What Did NOT Help

- **mimalloc / jemalloc** — both increased cache misses on this workload vs glibc malloc
- **Software prefetching** — hash table control bytes not reachable from the HashMap struct pointer
- **Pre-hoisting flag constants** — LLVM already loads `_mm_set1_epi8(N)` from .rodata efficiently
