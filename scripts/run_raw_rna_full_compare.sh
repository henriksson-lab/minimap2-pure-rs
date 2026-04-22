#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
OUTDIR="${1:-/husky/henriksson/for_claude/skesa/mm2rs_srr30335018_paf}"
THREADS="${THREADS:-1}"

REF="$ROOT/data/conformance/external_small/yeast_rna/yeast.fa"
JUNC="$ROOT/data/conformance/external_small/yeast_rna/junctions.bed"
READS="$ROOT/data/conformance/external_small/yeast_rna/raw/SRR30335018_1.fastq.gz"
C_BIN="$ROOT/minimap2/minimap2"
RUST_BIN="$ROOT/target/release/minimap2-pure-rs"

mkdir -p "$OUTDIR"
rm -f \
  "$OUTDIR/c.paf" \
  "$OUTDIR/rust.paf" \
  "$OUTDIR/c.time" \
  "$OUTDIR/rust.time" \
  "$OUTDIR/compare.txt"

echo "Running C minimap2..."
/usr/bin/time -f "c_real=%e\nc_user=%U\nc_sys=%S" -o "$OUTDIR/c.time" \
  "$C_BIN" -t "$THREADS" -x splice -c --junc-bed "$JUNC" "$REF" "$READS" > "$OUTDIR/c.paf"

echo "Running Rust minimap2..."
/usr/bin/time -f "rust_real=%e\nrust_user=%U\nrust_sys=%S" -o "$OUTDIR/rust.time" \
  "$RUST_BIN" -t "$THREADS" -x splice -c --junc-bed "$JUNC" "$REF" "$READS" > "$OUTDIR/rust.paf"

if cmp -s "$OUTDIR/c.paf" "$OUTDIR/rust.paf"; then
  echo "output_match=yes" > "$OUTDIR/compare.txt"
else
  echo "output_match=no" > "$OUTDIR/compare.txt"
  diff -u "$OUTDIR/c.paf" "$OUTDIR/rust.paf" | sed -n '1,200p' >> "$OUTDIR/compare.txt" || true
fi

cat "$OUTDIR/c.time"
cat "$OUTDIR/rust.time"
cat "$OUTDIR/compare.txt"
