#!/usr/bin/env bash
# Intersect 1000 Genomes per-chromosome BCF files with the deCODE EUR LD-block BED
# using bcftools view -R. Creates one output directory with pre-filtered BCF + index
# per autosome. Chromosome list is driven by the BED (autosomes only; chrX is not
# in the LD-block set and is skipped).
#
# Usage:
#   bash preprocess/regions/intersect_1kg_ld_blocks.sh                # full run, all autosomes in the BED
#   bash preprocess/regions/intersect_1kg_ld_blocks.sh --chroms chr22 # single-chromosome smoke test
#
# Requires: bcftools
#
# Paths (env-overridable): DATA_ROOT (default <repo>/data/1kgenomes), BCF_DIR (default
# ${DATA_ROOT}/raw_bcf), OUT_DIR (default ${DATA_ROOT}/raw_bcf_ld_blocks_eur_intersected).
#
# Provenance: preprocess/regions/ld_blocks/raw/README.md

set -euo pipefail

START_TIME=$(date +%s)
echo "=== Started at $(date) ==="
echo "Command: $0 $*"

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO}/data/1kgenomes}"
BCF_DIR="${BCF_DIR:-${DATA_ROOT}/raw_bcf}"
BED="${SCRIPT_DIR}/ld_blocks_eur_grch38.bed"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/raw_bcf_ld_blocks_eur_intersected}"

THREADS=4          # per bcftools process
MAX_PARALLEL=4      # concurrent bcftools processes (matches the CDS/exon intersect precedent)

# --- Optional --chroms filter (space-separated list), for smoke-testing on one chromosome ---
CHROMS=()
if [[ "${1:-}" == "--chroms" ]]; then
    shift
    CHROMS=("$@")
fi

if [[ ! -f "${BED}" ]]; then
    echo "ERROR: LD-block BED not found: ${BED}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
echo "BED:      ${BED} ($(wc -l < "${BED}") blocks)"
echo "BCF dir:  ${BCF_DIR}"
echo "Out dir:  ${OUT_DIR}"

# Chromosomes present in the BED, in file order (autosomes only by construction).
mapfile -t BED_CHROMS < <(cut -f1 "${BED}" | uniq)
if [[ ${#CHROMS[@]} -eq 0 ]]; then
    CHROMS=("${BED_CHROMS[@]}")
fi
echo "Chromosomes to process: ${CHROMS[*]}"

intersect_one() {
    local chrom="$1"
    local bcf
    bcf=$(ls "${BCF_DIR}"/1kGP_high_coverage_Illumina."${chrom}".filtered.*.bcf 2>/dev/null | head -1)
    if [[ -z "${bcf}" ]]; then
        echo "  WARN: no BCF found for ${chrom}, skipping" >&2
        return 0
    fi
    local base
    base=$(basename "${bcf}" .bcf)
    local out_bcf="${OUT_DIR}/${base}.ld_blocks_eur.bcf"

    if [[ -f "${out_bcf}" && -f "${out_bcf}.csi" ]]; then
        echo "  SKIP (exists): ${out_bcf}"
        return 0
    fi

    echo "  [$(date '+%H:%M:%S')] Intersecting: $(basename "${bcf}") -> $(basename "${out_bcf}")"
    bcftools view --threads "${THREADS}" -R "${BED}" -Ob -o "${out_bcf}" "${bcf}"
    bcftools index --threads "${THREADS}" "${out_bcf}"
    echo "  [$(date '+%H:%M:%S')] Done: $(basename "${out_bcf}") ($(bcftools index -n "${out_bcf}") records)"
}

# --- Process each chromosome ---
for chrom in "${CHROMS[@]}"; do
    intersect_one "${chrom}" &
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
        wait -n
    done
done
wait

# --- Summary ---
echo ""
echo "=== Intersected BCFs ==="
ls -lh "${OUT_DIR}"/*.bcf 2>/dev/null
echo "$(ls "${OUT_DIR}"/*.bcf 2>/dev/null | wc -l) files in ${OUT_DIR}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=== Finished at $(date) ==="
echo "=== Total time: $((ELAPSED / 60))m $((ELAPSED % 60))s ==="
