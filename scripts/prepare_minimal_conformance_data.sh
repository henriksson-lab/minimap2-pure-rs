#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/data/conformance/ecoli_srr13321180}"
N_PAIRS="${N_PAIRS:-50000}"
SRA_RUN="${SRA_RUN:-SRR13321180}"
REF_URL="${REF_URL:-https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz}"
REF_GZ="$OUT_DIR/ref/GCF_000005845.2_ASM584v2_genomic.fna.gz"
REF_FA="$OUT_DIR/ref/ecoli_k12_mg1655.fa"
RAW_DIR="$OUT_DIR/raw"
SUBSET_DIR="$OUT_DIR/subset"
MANIFEST="$OUT_DIR/conformance_manifest.tsv"

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "missing required command: $1" >&2
        exit 1
    }
}

need curl
need gzip
need prefetch
need fasterq-dump
need seqtk

mkdir -p "$OUT_DIR/ref" "$RAW_DIR" "$SUBSET_DIR"

if [[ ! -s "$REF_FA" ]]; then
    echo "[prepare] downloading E. coli K-12 MG1655 reference"
    curl -L --fail --retry 3 -o "$REF_GZ" "$REF_URL"
    gzip -cd "$REF_GZ" > "$REF_FA"
fi

if [[ ! -s "$RAW_DIR/${SRA_RUN}_1.fastq" || ! -s "$RAW_DIR/${SRA_RUN}_2.fastq" ]]; then
    echo "[prepare] downloading $SRA_RUN with prefetch"
    prefetch "$SRA_RUN" --output-directory "$RAW_DIR"
    echo "[prepare] converting $SRA_RUN to split FASTQ"
    fasterq-dump "$RAW_DIR/$SRA_RUN" --split-files --outdir "$RAW_DIR" --threads "${THREADS:-4}"
fi

R1="$SUBSET_DIR/${SRA_RUN}.first${N_PAIRS}_1.fq"
R2="$SUBSET_DIR/${SRA_RUN}.first${N_PAIRS}_2.fq"
if [[ ! -s "$R1" || ! -s "$R2" ]]; then
    echo "[prepare] taking first $N_PAIRS read pairs"
    seqtk sample -s 11 "$RAW_DIR/${SRA_RUN}_1.fastq" "$N_PAIRS" > "$R1"
    seqtk sample -s 11 "$RAW_DIR/${SRA_RUN}_2.fastq" "$N_PAIRS" > "$R2"
fi

relpath() {
    realpath --relative-to="$ROOT" "$1"
}

REF_ARG="$(relpath "$REF_FA")"
R1_ARG="$(relpath "$R1")"
R2_ARG="$(relpath "$R2")"

cat > "$MANIFEST" <<EOF
ShortRead	E coli SRR13321180 ${N_PAIRS} pairs PAF	paf	-x sr -c $REF_ARG $R1_ARG $R2_ARG
ShortRead	E coli SRR13321180 ${N_PAIRS} pairs SAM	sam-core	-x sr -a $REF_ARG $R1_ARG $R2_ARG
SplitIndex	E coli SRR13321180 ${N_PAIRS} pairs split PAF	paf	-x sr -c -I 500k --split-prefix /tmp/mm2rs-conf-ecoli-split $REF_ARG $R1_ARG $R2_ARG
SplitIndex	E coli SRR13321180 ${N_PAIRS} pairs split SAM	sam-core	-x sr -a -I 500k --split-prefix /tmp/mm2rs-conf-ecoli-split $REF_ARG $R1_ARG $R2_ARG
EOF

echo "[prepare] wrote $MANIFEST"
echo "[prepare] run: scripts/conformance_matrix.py $MANIFEST"
