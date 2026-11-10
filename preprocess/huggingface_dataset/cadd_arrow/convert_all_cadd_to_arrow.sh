#!/usr/bin/env bash
# Convert the whole CADD raw TSVs to per-chrom arrow files (no BED filter).
#
# Output: <out-dir>/{cadd_snv_<chrom>,cadd_indel_<chrom>}.arrow
# Per-chrom logs:  <out-dir>/logs/<chrom>.{log,time.txt}
# Manifest:        <out-dir>/MANIFEST.md
#
# 2 parallel workers — chr1's conversion peaks at ~100 GB RSS (5× chr22's
# 21 GB); two in flight at once keeps worst-case ~200 GB, well inside the
# 831 GB box even with other tenants.
#
# Idempotent: a chrom is skipped if both its SNV+indel arrow already exist.
#
# Run in background:
#   nohup bash convert_all_cadd_to_arrow.sh --in-dir <cadd_tsv_dir> --out-dir <arrow_dir> \
#     > <arrow_dir>/orchestrator.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PARALLEL=2

# Parse arguments
IN_DIR=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --in-dir)  IN_DIR="$2";  shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$IN_DIR" || -z "$OUT_DIR" ]]; then
    echo "Usage: $0 --in-dir <cadd_tsv_dir> --out-dir <arrow_output_dir>" >&2
    exit 1
fi

SNV_TSV=$IN_DIR/whole_genome_SNVs.tsv.gz
INDEL_TSV=$IN_DIR/gnomad.genomes.r4.0.indel.tsv.gz
LOG_DIR=$OUT_DIR/logs

mkdir -p "$LOG_DIR"

# hg38 chromosomes ordered by length (biggest first) — keeps wall time short
# by front-loading the long-running jobs.
CHROMS=(
    chr1 chr2 chr3 chr4 chr5 chr6 chr7 chrX
    chr8 chr9 chr11 chr10 chr12 chr13 chr14 chr15
    chr16 chr17 chr18 chr20 chr19 chrY chr22 chr21
)

cd "$REPO_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

convert_one() {
    local chrom="$1"
    local snv_out=$OUT_DIR/cadd_snv_${chrom}.arrow
    local indel_out=$OUT_DIR/cadd_indel_${chrom}.arrow
    local logfile=$LOG_DIR/${chrom}.log
    local timefile=$LOG_DIR/${chrom}.time.txt

    if [[ -f "$snv_out" && -f "$indel_out" ]]; then
        log "[$chrom] already exists, skipping"
        return 0
    fi

    log "[$chrom] START"
    if /usr/bin/time -v uv run python "$SCRIPT_DIR/poc_a2_convert_raw_cadd.py" \
            --snv-tsv "$SNV_TSV" \
            --indel-tsv "$INDEL_TSV" \
            --chrom "$chrom" \
            --out-dir "$OUT_DIR" \
            >"$logfile" 2>"$timefile"; then
        local wall=$(grep "Elapsed (wall clock) time" "$timefile" | sed 's/.*: //')
        local rss_kb=$(grep "Maximum resident set size" "$timefile" | awk '{print $NF}')
        local rss_gb=$(awk "BEGIN{printf \"%.1f\", $rss_kb / 1024 / 1024}")
        log "[$chrom] DONE wall=$wall rss=${rss_gb}GB"
    else
        log "[$chrom] FAILED — see $logfile + $timefile"
        return 1
    fi
}

export -f convert_one log
export SCRIPT_DIR OUT_DIR LOG_DIR SNV_TSV INDEL_TSV

log "START orchestrator — ${#CHROMS[@]} chroms, $PARALLEL parallel workers"
log "output: $OUT_DIR"

# Simple bash bg-job throttle: keep at most $PARALLEL convert_one jobs alive.
running=0
for chrom in "${CHROMS[@]}"; do
    if (( running >= PARALLEL )); then
        wait -n
        running=$(( running - 1 ))
    fi
    convert_one "$chrom" &
    running=$(( running + 1 ))
done
wait

log "ALL CONVERSIONS DONE — generating manifest"

# Manifest: one row per chrom with rows + bytes + wall time.
MANIFEST=$OUT_DIR/MANIFEST.md
{
    echo "# CADD full-coverage arrow — manifest"
    echo ""
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "Source: raw CADD whole-genome TSVs at $CADD_DIR/"
    echo "  SNV: $SNV_TSV"
    echo "  Indel: $INDEL_TSV"
    echo ""
    echo "Each per-chrom arrow file holds ALL CADD records for that chrom — no BED filter."
    echo "Indel arrow's \`chrom\` column carries the BCF-style \`chrN\` value (not raw \`N\`)."
    echo ""
    echo "| Chrom | SNV rows | SNV bytes | Indel rows | Indel bytes | Wall | Peak RSS |"
    echo "|---|---:|---:|---:|---:|---|---|"
    for chrom in "${CHROMS[@]}"; do
        snv_path=$OUT_DIR/cadd_snv_${chrom}.arrow
        indel_path=$OUT_DIR/cadd_indel_${chrom}.arrow
        timefile=$LOG_DIR/${chrom}.time.txt
        snv_bytes="-"
        indel_bytes="-"
        snv_rows="-"
        indel_rows="-"
        wall="-"
        rss="-"
        if [[ -f "$snv_path" ]]; then
            snv_bytes=$(stat -c %s "$snv_path")
            snv_rows=$(uv run python -c "
import pyarrow.ipc as ipc, pyarrow as pa, sys
with pa.OSFile('$snv_path','rb') as f, ipc.open_file(f) as r:
    print(r.read_all().num_rows)
" 2>/dev/null || echo "?")
        fi
        if [[ -f "$indel_path" ]]; then
            indel_bytes=$(stat -c %s "$indel_path")
            indel_rows=$(uv run python -c "
import pyarrow.ipc as ipc, pyarrow as pa, sys
with pa.OSFile('$indel_path','rb') as f, ipc.open_file(f) as r:
    print(r.read_all().num_rows)
" 2>/dev/null || echo "?")
        fi
        if [[ -f "$timefile" ]]; then
            wall=$(grep "Elapsed (wall clock) time" "$timefile" 2>/dev/null | sed 's/.*: //' || echo "-")
            rss_kb=$(grep "Maximum resident set size" "$timefile" 2>/dev/null | awk '{print $NF}' || echo "0")
            rss=$(awk "BEGIN{printf \"%.1fGB\", $rss_kb / 1024 / 1024}")
        fi
        printf "| %s | %'d | %'d | %'d | %'d | %s | %s |\n" \
            "$chrom" "${snv_rows//?/${snv_rows:-0}}" "$snv_bytes" "${indel_rows//?/${indel_rows:-0}}" "$indel_bytes" "$wall" "$rss" \
            2>/dev/null || \
        echo "| $chrom | $snv_rows | $snv_bytes | $indel_rows | $indel_bytes | $wall | $rss |"
    done
    echo ""
    echo "**Total bytes:** $(du -sb $OUT_DIR | awk '{print $1}') ($(du -sh $OUT_DIR | awk '{print $1}'))"
} > "$MANIFEST"

log "manifest: $MANIFEST"
log "END orchestrator"
