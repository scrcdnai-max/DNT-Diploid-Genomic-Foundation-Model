#!/usr/bin/env bash
set -euo pipefail

# Canonical 1kG datasets: per-sample BCFs × the canonical expanded BED → generate → assemble.
# One runner for every region-unbound preset; it replaced assemble_v6.sh, assemble_v6nd.sh,
# assemble_v6_haploid.sh, assemble_v7.sh and run_v6_and_v6nd.sh (reorg step 9), each of which
# was this script with the version pinned. `--version v6` = the old assemble_v6.sh,
# `--version v6 --representation haploid` = assemble_v6_haploid.sh, and so on; the old
# orchestrator is two runs in sequence (assemble takes a machine-wide lock, so never in parallel):
#   run_canonical_dataset.sh --version v6; run_canonical_dataset.sh --version v6nd
#
# Usage: run_canonical_dataset.sh --version {v4|v5|v6|v6nd|v7|v8} [--representation diploid|haploid]
#            [--phylop-bw BW] [--cadd-arrow-dir DIR] [--workers N] [--dry-run]
#
#   v4–v6nd  no per-base annotation columns (the annotation flags are rejected).
#   v7       always both annotations; they default to the files the original v7 used.
#   v8       annotations only when given.
#   haploid  no --tokens, no direction markers; --cadd-arrow-dir is diploid-only.
#   --dry-run  print the two commands and exit; nothing is checked, created or run.
#
# Paths can be overridden from the environment: DATA_ROOT (default /storage/data/1kgenomes),
# BCF_DIR, BED, REF, TOKENS, TEST_SAMPLES, OUT_DIR (relative paths are taken from the
# caller's cwd). Each run copies this script and a run_metadata_<ts>.txt (the invocation, the
# two resolved commands, host, git commit) into OUT_DIR and tees generate_<ts>.log /
# assembled/assemble_<ts>.log. Nothing is created until every argument and input check passes.

usage() { sed -n '/^# Usage:/,/^#   --dry-run/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

INVOCATION="$(printf '%q ' "$0" "$@")"; INVOCATION="${INVOCATION% }"
# A value flag must have a value that is not another flag: `--workers --dry-run` must not
# swallow --dry-run and start a real run. (An explicit "" is a value.)
need() { [[ $# -ge 2 && "$2" != --* ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }; }
VERSION="" REPRESENTATION=diploid WORKERS=96 DRY_RUN=0
PHYLOP_BW="" CADD_ARROW_DIR="" PHYLOP_SET=0 CADD_SET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)        need "$@"; VERSION="$2"; shift 2 ;;
        --representation) need "$@"; REPRESENTATION="$2"; shift 2 ;;
        --phylop-bw)      need "$@"; PHYLOP_BW="$2"; PHYLOP_SET=1; shift 2 ;;
        --cadd-arrow-dir) need "$@"; CADD_ARROW_DIR="$2"; CADD_SET=1; shift 2 ;;
        --workers)        need "$@"; WORKERS="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 1 ;;
    esac
done

case "${VERSION}" in
    v4|v5|v6|v6nd|v7|v8) ;;
    *) echo "ERROR: --version must be one of v4 v5 v6 v6nd v7 v8 (got '${VERSION:-<none>}')" >&2
       usage >&2; exit 1 ;;
esac
case "${REPRESENTATION}" in
    diploid|haploid) ;;
    *) echo "ERROR: --representation must be diploid or haploid (got '${REPRESENTATION}')" >&2; exit 1 ;;
esac
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --workers must be a positive integer (got '${WORKERS}')" >&2; exit 1; }
case "${VERSION}" in
    v7|v8) ;;
    *) if [[ -n "${PHYLOP_BW}" || -n "${CADD_ARROW_DIR}" ]]; then
           echo "ERROR: --phylop-bw / --cadd-arrow-dir need --version v7 or v8 (got ${VERSION})" >&2; exit 1
       fi ;;
esac
if [[ "${VERSION}" == v7 ]]; then
    # The original v7 dataset carries both columns; an explicit empty value is an error, not "off".
    [[ ${PHYLOP_SET} == 1 ]] || PHYLOP_BW=/storage/data/PhyloP/hg38.phyloP100way.bw
    [[ ${CADD_SET} == 1 ]] || CADD_ARROW_DIR=/storage/data/CADD/full_arrow
    if [[ -z "${PHYLOP_BW}" || -z "${CADD_ARROW_DIR}" ]]; then
        echo "ERROR: --version v7 needs both --phylop-bw and --cadd-arrow-dir" >&2; exit 1
    fi
fi
if [[ "${REPRESENTATION}" == haploid && -n "${CADD_ARROW_DIR}" ]]; then
    echo "ERROR: --cadd-arrow-dir is diploid-only, so --representation haploid cannot take it (nor v7, which needs it)" >&2; exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# — Paths ——————————————————————————————————————————————————————————————————————
DATA_ROOT="${DATA_ROOT:-/storage/data/1kgenomes}"
BCF_DIR="${BCF_DIR:-${DATA_ROOT}/raw_bcf_bed_intersected_hg38_pretrain_canonical_expanded_per_sample}"
BED="${BED:-${CANONICAL_BED}}"
REF="${REF:-${HG38_2BIT}}"
TOKENS="${TOKENS:-${DIPLOID_TOKENS}}"
TEST_SAMPLES="${TEST_SAMPLES:-${REPO}/preprocess/regions/sample_splits/sample_split_20251209_v5/1kg_test_samples_v5.txt}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/${REPRESENTATION}_hf_${VERSION}}"
# Relative paths are the caller's; make them absolute before `cd "${REPO}"`.
for _v in BCF_DIR BED REF TOKENS TEST_SAMPLES OUT_DIR PHYLOP_BW CADD_ARROW_DIR; do
    [[ -z "${!_v}" || "${!_v}" == /* ]] || printf -v "${_v}" '%s/%s' "${PWD}" "${!_v}"
done
ASM_DIR="${OUT_DIR}/assembled"
HF_CACHE="${OUT_DIR}/hf_cache"   # big partition, not ~/.cache

TOKENS_ARGS=()
[[ "${REPRESENTATION}" == diploid ]] && TOKENS_ARGS=(--tokens "${TOKENS}")
ANNOTATION_ARGS=()
[[ -n "${PHYLOP_BW}" ]] && ANNOTATION_ARGS+=(--phylop-bw "${PHYLOP_BW}")
[[ -n "${CADD_ARROW_DIR}" ]] && ANNOTATION_ARGS+=(--cadd-arrow-dir "${CADD_ARROW_DIR}")

cd "${REPO}"
GEN_CMD=("$UV" run --no-sync python -m preprocess.huggingface_dataset.create_dataset generate
    --version "${VERSION}" --representation "${REPRESENTATION}"
    ${TOKENS_ARGS[@]+"${TOKENS_ARGS[@]}"}
    --bcf-dir "${BCF_DIR}" --bed "${BED}" --ref "${REF}" --output "${OUT_DIR}"
    --test-samples-file "${TEST_SAMPLES}"
    ${ANNOTATION_ARGS[@]+"${ANNOTATION_ARGS[@]}"}
    --workers "${WORKERS}" --compression lz4 --clean-stale --log-level ERROR)
ASM_CMD=("$UV" run --no-sync python -m preprocess.huggingface_dataset.create_dataset assemble
    --shard-dir "${OUT_DIR}/train" --test-shard-dir "${OUT_DIR}/test"
    --output "${ASM_DIR}" --hf-cache-dir "${HF_CACHE}" --max-shard-size 10GB --log-level ERROR)

if [[ ${DRY_RUN} == 1 ]]; then
    echo "=== DRY RUN — nothing checked, created or run. The run would execute, from ${REPO}:"
    print_cmd "${GEN_CMD[@]}"
    print_cmd "${ASM_CMD[@]}"
    exit 0
fi

# Sanity-check inputs early (before anything is created).
if [[ "${REPRESENTATION}" == diploid ]]; then
    [ -f "$TOKENS"     ] || { echo "ERROR: tokens TSV missing: $TOKENS"             >&2; exit 1; }
fi
[ -f "$REF"            ] || { echo "ERROR: reference missing: $REF"                  >&2; exit 1; }
[ -f "$BED"            ] || { echo "ERROR: BED missing: $BED"                        >&2; exit 1; }
[ -d "$BCF_DIR"        ] || { echo "ERROR: BCF dir missing: $BCF_DIR"                >&2; exit 1; }
[ -f "$TEST_SAMPLES"   ] || { echo "ERROR: test samples missing: $TEST_SAMPLES"      >&2; exit 1; }
if [[ -n "${PHYLOP_BW}" ]]; then
    [ -f "$PHYLOP_BW"      ] || { echo "ERROR: phyloP bigWig missing: $PHYLOP_BW"        >&2; exit 1; }
fi
if [[ -n "${CADD_ARROW_DIR}" ]]; then
    [ -d "$CADD_ARROW_DIR" ] || { echo "ERROR: CADD arrow dir missing: $CADD_ARROW_DIR"  >&2; exit 1; }
fi
mkdir -p "${HF_CACHE}" "${ASM_DIR}"

# — Provenance: save the exact script + run metadata into the output dir ————
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp "${BASH_SOURCE[0]}" "$OUT_DIR/generation_script_${TIMESTAMP}.sh"
{
  echo "run_timestamp: $(date -Is)"
  echo "host: $(hostname)"
  echo "invocation: ${INVOCATION}"
  echo "generate:"; print_cmd "${GEN_CMD[@]}"
  echo "assemble:"; print_cmd "${ASM_CMD[@]}"
  echo "git_commit: $(git -C "$REPO" rev-parse HEAD)"
  echo "git_dirty_files: $(git -C "$REPO" status --porcelain | wc -l)"
} > "$OUT_DIR/run_metadata_${TIMESTAMP}.txt"
GEN_LOG="$OUT_DIR/generate_${TIMESTAMP}.log"
ASM_LOG="$ASM_DIR/assemble_${TIMESTAMP}.log"

# — Step 1 — generate per-sample shards ——————————————————————————————————————
echo "[$(date '+%F %T')] START generate" | tee -a "$GEN_LOG"
"${GEN_CMD[@]}" 2>&1 | tee -a "$GEN_LOG"
echo "[$(date '+%F %T')] END generate" | tee -a "$GEN_LOG"

# — Step 2 — assemble train + test into a single HF DatasetDict ——————————————
echo "[$(date '+%F %T')] START assemble" | tee -a "$ASM_LOG"
"${ASM_CMD[@]}" 2>&1 | tee -a "$ASM_LOG"
echo "[$(date '+%F %T')] END assemble" | tee -a "$ASM_LOG"
