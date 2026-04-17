#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/data/conformance/external_small}"
ECOLI_REF="${ECOLI_REF:-$ROOT/data/conformance/ecoli_srr13321180/ref/ecoli_k12_mg1655.fa}"
MANIFEST="$OUT_DIR/conformance_manifest.tsv"
DOWNLOADS="$OUT_DIR/downloads"

HIFI_RUN="${HIFI_RUN:-SRR10971019}"
ONT_RUN="${ONT_RUN:-SRR22784690}"
SR_RUN="${SR_RUN:-SRR26082374}"
RNA_RUN="${RNA_RUN:-SRR30335018}"

HIFI_N="${HIFI_N:-10000}"
ONT_N="${ONT_N:-20000}"
SR_N="${SR_N:-50000}"
RNA_N="${RNA_N:-200}"

UTI89_URL="${UTI89_URL:-https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/013/265/GCA_000013265.1_ASM1326v1/GCA_000013265.1_ASM1326v1_genomic.fna.gz}"
YEAST_FASTA_URL="${YEAST_FASTA_URL:-https://ftp.ensemblgenomes.ebi.ac.uk/pub/fungi/current/fasta/saccharomyces_cerevisiae/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz}"
YEAST_GFF3_URL="${YEAST_GFF3_URL:-}"
ENA_REPORT="https://www.ebi.ac.uk/ena/portal/api/filereport"

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "missing required command: $1" >&2
        exit 1
    }
}

need curl
need gzip
need awk
need python3
need seqtk

mkdir -p "$DOWNLOADS" \
    "$OUT_DIR/ecoli_k12/hifi" \
    "$OUT_DIR/ecoli_k12/ont" \
    "$OUT_DIR/ecoli_k12/illumina" \
    "$OUT_DIR/ecoli_uti89" \
    "$OUT_DIR/yeast_rna"

relpath() { realpath --relative-to="$ROOT" "$1"; }

fetch() {
    local url="$1" dest="$2"
    [[ -s "$dest" ]] && return 0
    echo "[download] $url"
    curl -L --fail --retry 3 -o "$dest" "$url"
}

ena_fastq_urls() {
    local run="$1" report="$DOWNLOADS/${run}.files.tsv"
    if [[ ! -s "$report" ]]; then
        curl -L --fail --retry 3 \
            "${ENA_REPORT}?accession=${run}&result=read_run&fields=run_accession,fastq_ftp,fastq_bytes,read_count,base_count,library_strategy,instrument_platform&format=tsv" \
            -o "$report"
    fi
    awk -F '\t' 'NR == 2 { n=split($2,a,";"); for (i=1; i<=n; ++i) if (a[i] != "") print "https://" a[i] }' "$report"
}

download_run_fastqs() {
    local run="$1" dest_dir="$2"
    mkdir -p "$dest_dir"
    mapfile -t urls < <(ena_fastq_urls "$run")
    if [[ "${#urls[@]}" -eq 0 ]]; then
        echo "no ENA FASTQ URLs found for $run" >&2
        exit 1
    fi
    local url base
    for url in "${urls[@]}"; do
        base="$(basename "$url")"
        fetch "$url" "$dest_dir/$base"
    done
}

first_fastq() {
    find "$1" -maxdepth 1 -type f \( -name '*.fastq.gz' -o -name '*.fq.gz' -o -name '*.fastq' -o -name '*.fq' \) | sort | head -n 1
}

sample_single() {
    local src="$1" n="$2" dest="$3"
    if [[ ! -s "$dest" ]]; then
        echo "[subset] $dest ($n reads)"
        seqtk sample -s 11 "$src" "$n" > "$dest"
    fi
}

sample_pair() {
    local src1="$1" src2="$2" n="$3" dest1="$4" dest2="$5"
    if [[ ! -s "$dest1" || ! -s "$dest2" ]]; then
        echo "[subset] $dest1/$dest2 ($n read pairs)"
        seqtk sample -s 11 "$src1" "$n" > "$dest1"
        seqtk sample -s 11 "$src2" "$n" > "$dest2"
    fi
}

if [[ ! -s "$ECOLI_REF" ]]; then
    echo "missing E. coli reference: $ECOLI_REF" >&2
    echo "run scripts/prepare_minimal_conformance_data.sh first, or set ECOLI_REF" >&2
    exit 1
fi

download_run_fastqs "$HIFI_RUN" "$OUT_DIR/ecoli_k12/hifi/raw"
download_run_fastqs "$ONT_RUN" "$OUT_DIR/ecoli_k12/ont/raw"
download_run_fastqs "$SR_RUN" "$OUT_DIR/ecoli_k12/illumina/raw"

HIFI_RAW="$(first_fastq "$OUT_DIR/ecoli_k12/hifi/raw")"
ONT_RAW="$(first_fastq "$OUT_DIR/ecoli_k12/ont/raw")"
mapfile -t SR_FASTQS < <(find "$OUT_DIR/ecoli_k12/illumina/raw" -maxdepth 1 -type f \( -name '*.fastq.gz' -o -name '*.fq.gz' -o -name '*.fastq' -o -name '*.fq' \) | sort)
[[ "${#SR_FASTQS[@]}" -ge 2 ]] || { echo "expected paired FASTQs for $SR_RUN" >&2; exit 1; }

HIFI_SUB="$OUT_DIR/ecoli_k12/hifi/hifi.${HIFI_N}.fq"
ONT_SUB="$OUT_DIR/ecoli_k12/ont/ont.${ONT_N}.fq"
SR_R1_SUB="$OUT_DIR/ecoli_k12/illumina/reads.${SR_N}_1.fq"
SR_R2_SUB="$OUT_DIR/ecoli_k12/illumina/reads.${SR_N}_2.fq"
sample_single "$HIFI_RAW" "$HIFI_N" "$HIFI_SUB"
sample_single "$ONT_RAW" "$ONT_N" "$ONT_SUB"
sample_pair "${SR_FASTQS[0]}" "${SR_FASTQS[1]}" "$SR_N" "$SR_R1_SUB" "$SR_R2_SUB"

UTI89_GZ="$OUT_DIR/ecoli_uti89/uti89.fa.gz"
UTI89_FA="$OUT_DIR/ecoli_uti89/uti89.fa"
fetch "$UTI89_URL" "$UTI89_GZ"
[[ -s "$UTI89_FA" ]] || gzip -cd "$UTI89_GZ" > "$UTI89_FA"
python3 - "$UTI89_FA" "$OUT_DIR/ecoli_uti89/alt.txt" <<'PYALT'
from pathlib import Path
import sys
fa = Path(sys.argv[1])
out = Path(sys.argv[2])
names = []
for line in fa.read_text().splitlines():
    if not line.startswith(">"):
        continue
    name = line[1:].split()[0]
    desc = line.lower()
    if "plasmid" in desc:
        names.append(name)
out.write_text("".join(f"{name}\n" for name in names))
PYALT

download_run_fastqs "$RNA_RUN" "$OUT_DIR/yeast_rna/raw"
RNA_RAW="$(first_fastq "$OUT_DIR/yeast_rna/raw")"
RNA_SUB="$OUT_DIR/yeast_rna/rna.${RNA_N}.fq"
sample_single "$RNA_RAW" "$RNA_N" "$RNA_SUB"

YEAST_FA_GZ="$OUT_DIR/yeast_rna/yeast.fa.gz"
YEAST_FA="$OUT_DIR/yeast_rna/yeast.fa"
fetch "$YEAST_FASTA_URL" "$YEAST_FA_GZ"
[[ -s "$YEAST_FA" ]] || gzip -cd "$YEAST_FA_GZ" > "$YEAST_FA"

if [[ -z "$YEAST_GFF3_URL" ]]; then
    GFF3_LIST="$DOWNLOADS/yeast_gff3_listing.html"
    if [[ ! -s "$GFF3_LIST" ]]; then
        curl -L --fail --retry 3 -o "$GFF3_LIST" \
            "https://ftp.ensemblgenomes.ebi.ac.uk/pub/fungi/current/gff3/saccharomyces_cerevisiae/"
    fi
    GFF3_NAME="$(sed -n 's/.*href="\([^"]*R64-1-1[^"/]*\.gff3\.gz\)".*/\1/p' "$GFF3_LIST" | grep -v '\.chromosome\.' | head -n 1)"
    [[ -n "$GFF3_NAME" ]] || { echo "failed to discover yeast GFF3 URL; set YEAST_GFF3_URL" >&2; exit 1; }
    YEAST_GFF3_URL="https://ftp.ensemblgenomes.ebi.ac.uk/pub/fungi/current/gff3/saccharomyces_cerevisiae/$GFF3_NAME"
fi
YEAST_GFF3_GZ="$OUT_DIR/yeast_rna/yeast.full.gff3.gz"
YEAST_GFF3="$OUT_DIR/yeast_rna/yeast.full.gff3"
fetch "$YEAST_GFF3_URL" "$YEAST_GFF3_GZ"
[[ -s "$YEAST_GFF3" ]] || gzip -cd "$YEAST_GFF3_GZ" > "$YEAST_GFF3"

python3 - "$YEAST_GFF3" "$OUT_DIR/yeast_rna/junctions.bed" <<'PYJUNC'
from collections import defaultdict
from pathlib import Path
import re
import sys

gff = Path(sys.argv[1])
out = Path(sys.argv[2])
exons = defaultdict(list)
chrom = {}
strand = {}

def attrs(s):
    d = {}
    for part in s.split(';'):
        if not part or '=' not in part:
            continue
        k, v = part.split('=', 1)
        d[k] = v
    return d

for line in gff.read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    fields = line.split('\t')
    if len(fields) < 9 or fields[2] != 'exon':
        continue
    a = attrs(fields[8])
    parent = a.get('Parent') or a.get('transcript_id')
    if not parent:
        continue
    parent = parent.split(',')[0]
    start = int(fields[3]) - 1
    end = int(fields[4])
    exons[parent].append((start, end))
    chrom[parent] = fields[0]
    strand[parent] = fields[6] if fields[6] in '+-' else '+'

lines = []
for tx, xs in exons.items():
    xs = sorted(set(xs))
    if len(xs) < 2:
        continue
    start = xs[0][0]
    end = xs[-1][1]
    sizes = ','.join(str(e - s) for s, e in xs) + ','
    starts = ','.join(str(s - start) for s, _ in xs) + ','
    name = re.sub(r'[^A-Za-z0-9_.:-]', '_', tx)
    lines.append('\t'.join([chrom[tx], str(start), str(end), name, '0', strand[tx], str(start), str(end), '0', str(len(xs)), sizes, starts]))

out.write_text('\n'.join(lines) + ('\n' if lines else ''))
PYJUNC

REF_ARG="$(relpath "$ECOLI_REF")"
HIFI_ARG="$(relpath "$HIFI_SUB")"
ONT_ARG="$(relpath "$ONT_SUB")"
SR_R1_ARG="$(relpath "$SR_R1_SUB")"
SR_R2_ARG="$(relpath "$SR_R2_SUB")"
UTI89_ARG="$(relpath "$UTI89_FA")"
ALT_ARG="$(relpath "$OUT_DIR/ecoli_uti89/alt.txt")"
YEAST_REF_ARG="$(relpath "$YEAST_FA")"
RNA_ARG="$(relpath "$RNA_SUB")"
JUNC_ARG="$(relpath "$OUT_DIR/yeast_rna/junctions.bed")"

cat > "$MANIFEST" <<EOF
HiFi	E coli K12 HiFi ${HIFI_N} reads PAF	paf	-x map-hifi -c $REF_ARG $HIFI_ARG
HiFi	E coli K12 HiFi ${HIFI_N} reads SAM	sam-core	-x map-hifi -a $REF_ARG $HIFI_ARG
ONT	E coli ONT ${ONT_N} reads PAF	paf	-x map-ont -c $REF_ARG $ONT_ARG
ONT	E coli ONT ${ONT_N} reads SAM	sam-core	-x map-ont -a $REF_ARG $ONT_ARG
Assembly	E coli UTI89 asm10 PAF	paf	-x asm10 -c $REF_ARG $UTI89_ARG
Assembly	E coli UTI89 asm10 SAM	sam-core	-x asm10 -a $REF_ARG $UTI89_ARG
ShortRead	E coli paired ${SR_N} reads PAF	paf	-x sr -c $REF_ARG $SR_R1_ARG $SR_R2_ARG
ShortRead	E coli paired ${SR_N} reads SAM	sam-core	-x sr -a $REF_ARG $SR_R1_ARG $SR_R2_ARG
SplitIndex	E coli HiFi forced split PAF	paf	-x map-hifi -c -I 500k --split-prefix /tmp/mm2rs-conf-ext-hifi-split $REF_ARG $HIFI_ARG
SplitIndex	E coli ONT forced split SAM	sam-core	-x map-ont -a -I 500k --split-prefix /tmp/mm2rs-conf-ext-ont-split $REF_ARG $ONT_ARG
ALT	E coli UTI89 ALT metadata PAF	paf	-x asm10 -c --alt $ALT_ARG $REF_ARG $UTI89_ARG
RNA	Yeast direct RNA splice ${RNA_N} reads PAF	paf	-x splice -c --junc-bed $JUNC_ARG $YEAST_REF_ARG $RNA_ARG
RNA	Yeast direct RNA splice ${RNA_N} reads SAM	sam-core	-x splice -a --junc-bed $JUNC_ARG $YEAST_REF_ARG $RNA_ARG
EOF

echo "[prepare] wrote $MANIFEST"
echo "[prepare] run: scripts/conformance_matrix.py $MANIFEST"
