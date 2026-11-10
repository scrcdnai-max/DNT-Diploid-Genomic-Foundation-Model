#!/usr/bin/env bash
# Build per-sample, BED-restricted, hom-ref-dropped, GT-only BCFs from
# per-chromosome multi-sample panel BCFs — in ONE unified pass per chromosome.
#
# This replaces the separate BED-intersection step (intersect_bcf_regions.sh):
# `bcftools +split -R BED -i 'GT="alt"' -k FMT/GT` simultaneously
#   * restricts to the BED regions          (-R, was the intersection step)
#   * drops hom-ref / missing per sample     (-i 'GT="alt"')
#   * strips INFO, keeping only FORMAT/GT     (-k FMT/GT)
#   * demultiplexes to one file per sample    (+split)
# then `bcftools concat` joins each sample's per-chromosome pieces into a single
# multi-contig `<sample>.bcf` that the diploid_glm encoder reads via per-sample
# layout (see _resolve_sample_bcf in get_genome_sequence.py).
#
# Output layout (per-sample, flat):
#   OUT_DIR/<sample>.bcf  (+ .csi)
#
# Requires bcftools >= 1.13. Run inside the glm_biotools image — the tag is pinned
# as GLM_BIOTOOLS_IMAGE in docker_images/glm_biotools/image.env — e.g.:
#   docker run --rm --user $(id -u):$(id -g) -v /opt/hpc:/opt/hpc \
#     --ulimit nofile=65536:65536 \
#     -e JOBS=8 \
#     "${GLM_BIOTOOLS_IMAGE}" bash THIS_SCRIPT RAW_DIR BED OUT_DIR [SAMPLES_FILE]
#
# Note: --ulimit nofile=65536:65536 is required for full 3202-sample panels
#       (bcftools +split opens one fd per sample; default Docker limit is 1024).
#       Pass JOBS via -e, not as a shell prefix (shell prefix does not enter the container).
#
# Usage:
#   make_persample_bcf.sh RAW_DIR BED OUT_DIR [SAMPLES_FILE]
# Env:
#   JOBS     parallel chromosomes / concat workers (default: 8)
#   TMP_DIR  scratch for per-chrom-per-sample pieces (default: OUT_DIR/.tmp_split)

set -euo pipefail

START=$(date +%s)
echo "[$(date '+%F %T')] START make_persample_bcf"
echo "Command: $0 $*"

RAW_DIR="${1:?usage: make_persample_bcf.sh RAW_DIR BED OUT_DIR [SAMPLES_FILE]}"
BED="${2:?missing BED}"
OUT_DIR="${3:?missing OUT_DIR}"
SAMPLES_FILE="${4:-}"
JOBS="${JOBS:-8}"
TMP_DIR="${TMP_DIR:-${OUT_DIR}/.tmp_split}"

# bcftools +split opens one output fd per sample; raise the limit so 3202-sample
# panels don't hit EMFILE.  JOBS * 3202 + headroom; needs matching --ulimit in Docker.
ulimit -n 65536 2>/dev/null || true
echo "ulimit nofile: $(ulimit -n)  (need >= $((JOBS * 3202 + 1000)) for JOBS=$JOBS × 3202 samples)"

command -v bcftools >/dev/null || { echo "ERROR: bcftools not on PATH" >&2; exit 1; }
echo "bcftools: $(bcftools --version | head -1)"
echo "RAW_DIR=$RAW_DIR  BED=$BED  OUT_DIR=$OUT_DIR  SAMPLES_FILE=${SAMPLES_FILE:-<all>}  JOBS=$JOBS"

mkdir -p "$OUT_DIR" "$TMP_DIR"

# Natural karyotype sort: chr1..chr22, chrX, chrY, chrM/chrMT, then the rest.
chrom_sort_key() {
    local c="${1#chr}"
    case "$c" in
        [0-9]*) printf '%03d\n' "$c" ;;
        X) echo 100 ;;
        Y) echo 101 ;;
        M|MT) echo 102 ;;
        *) echo "200_$c" ;;
    esac
}

# Optional -S argument for +split (sample subset, for testing).
SPLIT_S=()
[[ -n "$SAMPLES_FILE" ]] && SPLIT_S=(-S "$SAMPLES_FILE")

# ── Phase 1: per-chromosome +split (parallel across chromosomes) ──────────
split_one() {
    local bcf="$1"
    local base chrom outdir
    base=$(basename "$bcf")
    # Extract chrom token (chr1, chrX, ...) from filename.
    chrom=$(grep -oE 'chr[0-9XYMt]+' <<<"$base" | head -1)
    [[ -z "$chrom" ]] && { echo "  SKIP (no chrom token): $base" >&2; return 0; }
    outdir="${TMP_DIR}/${chrom}"
    if [[ -d "$outdir" && -n "$(ls -A "$outdir" 2>/dev/null)" ]]; then
        echo "  SKIP +split (exists): $chrom"; return 0
    fi
    mkdir -p "$outdir"
    echo "  [$(date '+%T')] +split $chrom"
    bcftools +split "$bcf" -R "$BED" -i 'GT="alt"' -k FMT/GT \
        "${SPLIT_S[@]}" -Ob -o "$outdir"

    # bcftools +split 1.23 quirk with -Ob -S: outputs <sample>bcf (no dot) instead of
    # <sample>.bcf — rename so Phase 2's 'find -name *.bcf' can enumerate them.
    for f in "$outdir"/*; do
        [[ -f "$f" ]] || continue
        base=$(basename "$f")
        if [[ "$base" == *bcf && "$base" != *.bcf ]]; then
            mv "$f" "${f%bcf}.bcf"
        fi
    done
}

# Backgrounded shell functions run in subshells that inherit the parent's
# variables and arrays, so no `export` is needed for the job-pool loops below.
echo "[$(date '+%F %T')] Phase 1: +split per chromosome (JOBS=$JOBS)"
running=0
for bcf in "$RAW_DIR"/*.bcf; do
    [[ -f "$bcf" ]] || continue
    split_one "$bcf" &
    running=$((running + 1))
    if (( running >= JOBS )); then wait -n; running=$((running - 1)); fi
done
wait

# ── Phase 2: per-sample concat across chromosomes (parallel across samples) ─
# Enumerate samples from the union of all per-chrom split outputs.
mapfile -t SAMPLES < <(find "$TMP_DIR" -name '*.bcf' -printf '%f\n' | sed 's/\.bcf$//' | sort -u)
echo "[$(date '+%F %T')] Phase 2: concat ${#SAMPLES[@]} sample(s) across chromosomes (JOBS=$JOBS)"

# Chromosome dirs in karyotype order (for a well-ordered concatenation).
mapfile -t CHROM_DIRS < <(
    for d in "$TMP_DIR"/chr*; do [[ -d "$d" ]] && echo "$(chrom_sort_key "$(basename "$d")") $d"; done \
        | sort | awk '{print $2}'
)

concat_one() {
    local s="$1"
    local out="${OUT_DIR}/${s}.bcf"
    if [[ -f "$out" && -f "${out}.csi" ]]; then echo "  SKIP concat (exists): $s"; return 0; fi
    local parts=()
    local d
    for d in "${CHROM_DIRS[@]}"; do
        [[ -f "${d}/${s}.bcf" ]] && parts+=("${d}/${s}.bcf")
    done
    [[ ${#parts[@]} -eq 0 ]] && { echo "  WARN no parts for $s" >&2; return 0; }
    bcftools concat "${parts[@]}" -Ob -o "${out}.tmp"
    mv "${out}.tmp" "$out"
    bcftools index --csi "$out"
}

running=0
for s in "${SAMPLES[@]}"; do
    concat_one "$s" &
    running=$((running + 1))
    if (( running >= JOBS )); then wait -n; running=$((running - 1)); fi
done
wait

N_OUT=$(find "$OUT_DIR" -maxdepth 1 -name '*.bcf' | wc -l)
echo "[$(date '+%F %T')] DONE — ${N_OUT} per-sample BCF(s) in ${OUT_DIR}"
echo "Tip: remove scratch when satisfied:  rm -rf '${TMP_DIR}'"
END=$(date +%s)
echo "[$(date '+%F %T')] Total time: $(( (END-START)/60 ))m $(( (END-START)%60 ))s"
