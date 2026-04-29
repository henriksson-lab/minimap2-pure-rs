#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/.tmp/zenodo19703025}"
OUTDIR="${OUTDIR:-$DATA_DIR/compare}"
REF="${REF:-}"
THREADS="${THREADS:-1}"
C_BIN="${C_BIN:-$ROOT/minimap2/minimap2}"
RUST_BIN="${RUST_BIN:-$ROOT/target/release/minimap2-pure-rs}"
CASES="${CASES:-hifi,ont,wgs,hic}"
CS="${CS:-}"
SUBSET="${SUBSET:-0}"
HIFI_SUBSET="${HIFI_SUBSET:-1000}"
ONT_SUBSET="${ONT_SUBSET:-1000}"
WGS_SUBSET="${WGS_SUBSET:-1000}"
HIC_SUBSET="${HIC_SUBSET:-1000}"

BASE_URL="https://zenodo.org/api/records/19703025/files"

usage() {
  cat <<EOF
Usage: REF=/path/to/hs38.fa $0

Environment:
  REF       Reference FASTA/index used by Zenodo map.mak, usually hs38.fa.
  DATA_DIR Download/cache directory. Default: $ROOT/.tmp/zenodo19703025
  OUTDIR   Output directory. Default: \$DATA_DIR/compare
  THREADS  Threads for both binaries. Default: 1 for deterministic comparison
  C_BIN    C minimap2 binary. Default: $ROOT/minimap2/minimap2
  RUST_BIN Rust binary. Default: $ROOT/target/release/minimap2-pure-rs
  CASES    Comma-separated subset: hifi,ont,wgs,hic. Default: all
  CS       Optional extra cs flag, for example "--cs" or "--cs=long".
  SUBSET   If 1, use prepared subset files from prepare_zenodo_19703025_hg38_data.sh.
           Defaults to 0, which compares the full downloaded Zenodo files.
  HIFI_SUBSET/ONT_SUBSET/WGS_SUBSET/HIC_SUBSET
           Subset sizes used when SUBSET=1. Defaults: 1000.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$REF" ]]; then
  usage >&2
  echo "error: REF is required" >&2
  exit 2
fi
if [[ ! -e "$REF" ]]; then
  echo "error: REF does not exist: $REF" >&2
  exit 2
fi
if [[ ! -x "$C_BIN" ]]; then
  echo "error: C minimap2 binary not executable: $C_BIN" >&2
  exit 2
fi
if [[ ! -x "$RUST_BIN" ]]; then
  echo "error: Rust minimap2 binary not executable: $RUST_BIN" >&2
  exit 2
fi

mkdir -p "$DATA_DIR" "$OUTDIR"

download() {
  local name="$1"
  local dest="$DATA_DIR/$name"
  if [[ ! -s "$dest" ]]; then
    echo "download $name"
    curl -L "$BASE_URL/$name/content" -o "$dest"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "error: missing subset file: $path" >&2
    echo "hint: run scripts/prepare_zenodo_19703025_hg38_data.sh first" >&2
    exit 2
  fi
}

has_case() {
  local needle="$1"
  case ",$CASES," in
    *",$needle,"*) return 0 ;;
    *) return 1 ;;
  esac
}

run_case() {
  local name="$1"
  shift
  local c_out="$OUTDIR/$name.c.paf"
  local r_out="$OUTDIR/$name.rust.paf"
  local c_log="$OUTDIR/$name.c.log"
  local r_log="$OUTDIR/$name.rust.log"
  local cmp_out="$OUTDIR/$name.compare.txt"

  echo "run $name: $*"
  "$C_BIN" "$@" > "$c_out" 2> "$c_log"
  "$RUST_BIN" "$@" > "$r_out" 2> "$r_log"

  if cmp -s "$c_out" "$r_out"; then
    echo "output_match=yes" > "$cmp_out"
    echo "ok $name"
  else
    echo "output_match=no" > "$cmp_out"
    diff -u "$c_out" "$r_out" | sed -n '1,240p' >> "$cmp_out" || true
    echo "FAIL $name"
    sed -n '1,80p' "$cmp_out"
    return 1
  fi
}

status=0

if has_case hifi; then
  if [[ "$SUBSET" == "1" ]]; then
    require_file "$DATA_DIR/HiFi-${HIFI_SUBSET}.fq"
    run_case "hifi-${HIFI_SUBSET}" -c -x map-hifi -t "$THREADS" $CS "$REF" "$DATA_DIR/HiFi-${HIFI_SUBSET}.fq" || status=1
  else
    download HG002.HiFi-10k.fa.gz
    run_case hifi -c -x map-hifi -t "$THREADS" $CS "$REF" "$DATA_DIR/HG002.HiFi-10k.fa.gz" || status=1
  fi
fi

if has_case ont; then
  if [[ "$SUBSET" == "1" ]]; then
    require_file "$DATA_DIR/ONT-${ONT_SUBSET}.fa"
    run_case "ont-${ONT_SUBSET}" -c -x lr:hq -t "$THREADS" $CS "$REF" "$DATA_DIR/ONT-${ONT_SUBSET}.fa" || status=1
  else
    download HG002.ONT-10k.fa.gz
    run_case ont -c -x lr:hq -t "$THREADS" $CS "$REF" "$DATA_DIR/HG002.ONT-10k.fa.gz" || status=1
  fi
fi

if has_case wgs; then
  if [[ "$SUBSET" == "1" ]]; then
    require_file "$DATA_DIR/WGS-${WGS_SUBSET}_1.fq"
    require_file "$DATA_DIR/WGS-${WGS_SUBSET}_2.fq"
    run_case "wgs-${WGS_SUBSET}" -c -x sr -t "$THREADS" $CS "$REF" "$DATA_DIR/WGS-${WGS_SUBSET}_1.fq" "$DATA_DIR/WGS-${WGS_SUBSET}_2.fq" || status=1
  else
    download HG002.WGS-1M_1.fq.gz
    download HG002.WGS-1M_2.fq.gz
    run_case wgs -c -x sr -t "$THREADS" $CS "$REF" "$DATA_DIR/HG002.WGS-1M_1.fq.gz" "$DATA_DIR/HG002.WGS-1M_2.fq.gz" || status=1
  fi
fi

if has_case hic; then
  if [[ "$SUBSET" == "1" ]]; then
    require_file "$DATA_DIR/HiC-${HIC_SUBSET}_1.fq"
    require_file "$DATA_DIR/HiC-${HIC_SUBSET}_2.fq"
    run_case "hic-${HIC_SUBSET}" -c -x sr -t "$THREADS" $CS "$REF" "$DATA_DIR/HiC-${HIC_SUBSET}_1.fq" "$DATA_DIR/HiC-${HIC_SUBSET}_2.fq" || status=1
  else
    download HG002.HiC-1M_1.fq.gz
    download HG002.HiC-1M_2.fq.gz
    run_case hic -c -x sr -t "$THREADS" $CS "$REF" "$DATA_DIR/HG002.HiC-1M_1.fq.gz" "$DATA_DIR/HG002.HiC-1M_2.fq.gz" || status=1
  fi
fi

exit "$status"
