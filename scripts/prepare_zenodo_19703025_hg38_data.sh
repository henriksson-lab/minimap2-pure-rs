#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/.tmp/zenodo19703025}"
REF="${REF:-$DATA_DIR/hg38.fa.gz}"
THREADS="${THREADS:-10}"
CASES="${CASES:-ont,wgs}"
HIFI_SUBSET="${HIFI_SUBSET:-1000}"
ONT_SUBSET="${ONT_SUBSET:-1000}"
WGS_SUBSET="${WGS_SUBSET:-1000}"
HIC_SUBSET="${HIC_SUBSET:-1000}"
ONT_REGRESSION_READ="${ONT_REGRESSION_READ:-a6933f35-c613-481f-a18f-9ccd6b517398}"
RUN_COMPARE="${RUN_COMPARE:-0}"

ZENODO_BASE_URL="https://zenodo.org/api/records/19703025/files"
HG38_URL="${HG38_URL:-https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz}"
HG38_MD5="${HG38_MD5:-1c9dcaddfa41027f17cd8f7a82c7293b}"

usage() {
  cat <<EOF
Usage: $0

Downloads hg38 plus selected Zenodo 19703025 HG002 inputs and prepares small
reproducible subsets for C-vs-Rust minimap2 parity testing.

Environment:
  DATA_DIR             Cache/output directory. Default: $ROOT/.tmp/zenodo19703025
  REF                  Reference path to create/use. Default: \$DATA_DIR/hg38.fa.gz
  CASES                Comma-separated data to prepare: hifi,ont,wgs,hic. Default: ont,wgs
  HIFI_SUBSET          Number of HiFi reads to extract. Default: 1000
  ONT_SUBSET           Number of ONT reads to extract. Default: 1000
  WGS_SUBSET           Number of paired WGS records to extract. Default: 1000
  HIC_SUBSET           Number of paired Hi-C records to extract. Default: 1000
  ONT_REGRESSION_READ  ONT read name to extract for the single-read regression.
                       Default: a6933f35-c613-481f-a18f-9ccd6b517398
  RUN_COMPARE          If 1, run scripts/run_zenodo_19703025_compare.sh afterwards.
  THREADS              Threads for optional compare run. Default: 10
  HG38_URL             Override hg38 download URL.
  HG38_MD5             Expected hg38.fa.gz MD5. Set empty to skip MD5 check.

After preparation, useful commands are:
  REF="\$REF" THREADS=10 CASES=ont,wgs scripts/run_zenodo_19703025_compare.sh
  cargo test test_zenodo_19703025_ont_hg38_a6933_vs_c --test integration -- --nocapture
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$DATA_DIR"

download_url() {
  local url="$1"
  local dest="$2"
  if [[ -s "$dest" ]]; then
    echo "[exists] $dest"
    return
  fi
  echo "[download] $url"
  curl -L "$url" -o "$dest"
}

download_zenodo() {
  local name="$1"
  download_url "$ZENODO_BASE_URL/$name/content" "$DATA_DIR/$name"
}

has_case() {
  local needle="$1"
  case ",$CASES," in
    *",$needle,"*) return 0 ;;
    *) return 1 ;;
  esac
}

check_gzip() {
  local path="$1"
  echo "[check] gzip $path"
  gzip -t "$path"
}

check_hg38_md5() {
  if [[ -z "$HG38_MD5" ]]; then
    return
  fi
  local got
  got="$(md5sum "$REF" | awk '{print $1}')"
  if [[ "$got" != "$HG38_MD5" ]]; then
    echo "error: hg38 MD5 mismatch for $REF" >&2
    echo "  expected: $HG38_MD5" >&2
    echo "  got:      $got" >&2
    exit 1
  fi
}

extract_fasta_subset() {
  local input_gz="$1"
  local n_reads="$2"
  local output="$3"
  if [[ -s "$output" ]]; then
    echo "[exists] $output"
    return
  fi
  echo "[subset] first $n_reads FASTA records -> $output"
  gzip -dc "$input_gz" | awk -v max="$n_reads" '
    /^>/ { seen++; if (seen > max) exit }
    seen > 0 { print }
  ' > "$output"
}

extract_fastq_subset() {
  local input_gz="$1"
  local n_reads="$2"
  local output="$3"
  if [[ -s "$output" ]]; then
    echo "[exists] $output"
    return
  fi
  echo "[subset] first $n_reads FASTQ records -> $output"
  gzip -dc "$input_gz" | awk -v max_lines="$((n_reads * 4))" 'NR <= max_lines { print }' > "$output"
}

extract_fasta_read() {
  local input_gz="$1"
  local read_name="$2"
  local output="$3"
  if [[ -s "$output" ]]; then
    echo "[exists] $output"
    return
  fi
  echo "[extract] FASTA read $read_name -> $output"
  gzip -dc "$input_gz" | awk -v id="$read_name" '
    /^>/ {
      name = substr($1, 2)
      keep = (name == id)
      if (keep) found = 1
    }
    keep { print }
    END {
      if (!found) exit 1
    }
  ' > "$output"
}

download_url "$HG38_URL" "$REF"
check_gzip "$REF"
check_hg38_md5

if has_case hifi; then
  download_zenodo HG002.HiFi-10k.fa.gz
  check_gzip "$DATA_DIR/HG002.HiFi-10k.fa.gz"
  extract_fastq_subset "$DATA_DIR/HG002.HiFi-10k.fa.gz" "$HIFI_SUBSET" "$DATA_DIR/HiFi-${HIFI_SUBSET}.fq"
fi

if has_case ont; then
  download_zenodo HG002.ONT-10k.fa.gz
  check_gzip "$DATA_DIR/HG002.ONT-10k.fa.gz"
  extract_fasta_subset "$DATA_DIR/HG002.ONT-10k.fa.gz" "$ONT_SUBSET" "$DATA_DIR/ONT-${ONT_SUBSET}.fa"
  extract_fasta_read "$DATA_DIR/HG002.ONT-10k.fa.gz" "$ONT_REGRESSION_READ" "$DATA_DIR/ONT-a6933.fa"
fi

if has_case wgs; then
  download_zenodo HG002.WGS-1M_1.fq.gz
  download_zenodo HG002.WGS-1M_2.fq.gz
  check_gzip "$DATA_DIR/HG002.WGS-1M_1.fq.gz"
  check_gzip "$DATA_DIR/HG002.WGS-1M_2.fq.gz"
  extract_fastq_subset "$DATA_DIR/HG002.WGS-1M_1.fq.gz" "$WGS_SUBSET" "$DATA_DIR/WGS-${WGS_SUBSET}_1.fq"
  extract_fastq_subset "$DATA_DIR/HG002.WGS-1M_2.fq.gz" "$WGS_SUBSET" "$DATA_DIR/WGS-${WGS_SUBSET}_2.fq"
fi

if has_case hic; then
  download_zenodo HG002.HiC-1M_1.fq.gz
  download_zenodo HG002.HiC-1M_2.fq.gz
  check_gzip "$DATA_DIR/HG002.HiC-1M_1.fq.gz"
  check_gzip "$DATA_DIR/HG002.HiC-1M_2.fq.gz"
  extract_fastq_subset "$DATA_DIR/HG002.HiC-1M_1.fq.gz" "$HIC_SUBSET" "$DATA_DIR/HiC-${HIC_SUBSET}_1.fq"
  extract_fastq_subset "$DATA_DIR/HG002.HiC-1M_2.fq.gz" "$HIC_SUBSET" "$DATA_DIR/HiC-${HIC_SUBSET}_2.fq"
fi

echo "[ready] reference: $REF"
echo "[ready] data dir:  $DATA_DIR"

if [[ "$RUN_COMPARE" == "1" ]]; then
  REF="$REF" DATA_DIR="$DATA_DIR" THREADS="$THREADS" CASES="$CASES" \
    "$ROOT/scripts/run_zenodo_19703025_compare.sh"
fi
