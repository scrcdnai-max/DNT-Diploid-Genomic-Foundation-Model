#!/usr/bin/env bash
set -euo pipefail

# Build an LD-block dataset (v4-ldb / v6-ldb / v7-ldb): generate per-sample
# shards, then assemble into a HuggingFace DatasetDict.
#
# These versions are the v4/v6/v7 ENCODINGS over a different REGION UNIVERSE —
# the deCODE EUR LD blocks (1,361 gap-free blocks, 2.806 Gbp) instead of the
# canonical hg38_pretrain_canonical_expanded.bed (140,099 regions, 0.943 Gbp).
# See preprocess/regions/ld_blocks/PROVENANCE.md and huggingface_dataset/versions.py.
#
# Usage:
#   run_ldb_dataset.sh v7-ldb                    # generate + assemble
#   run_ldb_dataset.sh v6-ldb --convert-from v7-ldb
#                                                # skip generate: assemble the
#                                                # v7-ldb shards into v6-ldb
#   run_ldb_dataset.sh v7-ldb --dry-run          # print the generate/assemble
#                                                # commands, touch nothing, exit 0
#
# The --convert-from form exists because v7-ldb → v6-ldb is a pure
# DROP_ANNOTATIONS conversion (identical seq), so it saves a second ~5 TB
# generate. It does NOT work for v4-ldb (the anchor-SNP axis differs, so
# compat.py correctly refuses) — v4-ldb must be generated in full.
#
# Paths are env-overridable so this runs on either the data node or the large
# target machine without editing the script:
#   REPO (default: the checkout this script lives in), DATA_ROOT, BCF_DIR, TEST_SAMPLES, WORKERS
#   ALLOW_MULTISAMPLE_BCF=1   accept the ~2.5x slower multi-sample BCF layout

START_TIME=$(date +%s)
echo "=== Started at $(date '+%F %T') ==="
echo "Command: $0 $*"

# — Arguments ————————————————————————————————————————————————————————————————
VERSION="${1:-}"
CONVERT_FROM=""
case "${VERSION}" in
    v4-ldb|v6-ldb|v7-ldb) ;;
    *)
        echo "ERROR: version must be one of v4-ldb, v6-ldb, v7-ldb (got '${VERSION:-<none>}')" >&2
        echo "Usage: $0 {v4-ldb|v6-ldb|v7-ldb} [--convert-from v7-ldb] [--dry-run]" >&2
        exit 1
        ;;
esac
shift
# --dry-run may appear anywhere after the version; the rest is parsed as before.
DRY_RUN=0
_args=()
for _a in "$@"; do
    if [[ "${_a}" == "--dry-run" ]]; then DRY_RUN=1; else _args+=("${_a}"); fi
done
set -- ${_args[@]+"${_args[@]}"}  # (bash <4.4 + set -u: empty array)
if [[ $# -gt 0 ]]; then
    if [[ "$1" == "--convert-from" && -n "${2:-}" ]]; then
        CONVERT_FROM="$2"
        if [[ "${CONVERT_FROM}" == "${VERSION}" ]]; then
            echo "ERROR: --convert-from ${CONVERT_FROM} is the same as the target version" >&2
            exit 1
        fi
    else
        echo "ERROR: unrecognized argument(s): $*" >&2
        exit 1
    fi
fi

# — Paths ————————————————————————————————————————————————————————————————————
# REPO (this checkout unless exported), the data paths and GLM_BIOTOOLS_IMAGE:
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# The glm_biotools tag lives in ONE place; DOCKER_IMAGE still overrides per run.
DOCKER_IMAGE="${DOCKER_IMAGE:-${GLM_BIOTOOLS_IMAGE}}"
# DATA_ROOT comes from _common.sh
BED="${LDB_BED}"
REF="${HG38_2BIT}"
TOKENS="${DIPLOID_TOKENS}"
BCF_DIR="${BCF_DIR:-${DATA_ROOT}/raw_bcf_ld_blocks_eur_per_sample}"
TEST_SAMPLES="${TEST_SAMPLES:-${REPO}/preprocess/regions/sample_splits/sample_split_20251209_v5/1kg_test_samples_v5.txt}"
WORKERS="${WORKERS:-60}"

OUT_DIR="${DATA_ROOT}/diploid_hf_${VERSION}"
ASM_DIR="${OUT_DIR}/assembled"
HF_CACHE="${OUT_DIR}/hf_cache"   # big partition, not ~/.cache

# Shards to assemble from: our own when generating, the source version's when
# converting.
SRC_DIR="${OUT_DIR}"
[[ -n "${CONVERT_FROM}" ]] && SRC_DIR="${DATA_ROOT}/diploid_hf_${CONVERT_FROM}"

cd "${REPO}"

GEN_CMD=(
    "${PYTHON}" -m preprocess.huggingface_dataset.create_dataset generate
        --version           "${VERSION}"
        --representation    diploid
        --tokens            "${TOKENS}"
        --bcf-dir           "${BCF_DIR}"
        --bed               "${BED}"
        --ref               "${REF}"
        --output            "${OUT_DIR}"
        --test-samples-file "${TEST_SAMPLES}"
        --workers           "${WORKERS}"
        --compression       lz4
        --clean-stale
        --log-level         ERROR
)
ASSEMBLE_ARGS=(
    --shard-dir      "${SRC_DIR}/train"
    --test-shard-dir "${SRC_DIR}/test"
    --output         "${ASM_DIR}"
    --hf-cache-dir   "${HF_CACHE}"
    --max-shard-size 10GB
    --log-level      ERROR
)
# On a conversion run the shards are the SOURCE version; --target-version drives
# the on-the-fly rewrite and stamps the target's schema_version.
[[ -n "${CONVERT_FROM}" ]] && ASSEMBLE_ARGS+=(--target-version "${VERSION}")
ASM_CMD=("${PYTHON}" -m preprocess.huggingface_dataset.create_dataset assemble "${ASSEMBLE_ARGS[@]}")

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "=== DRY RUN — nothing checked or run. The run would execute, from ${REPO}:"
    if [[ -z "${CONVERT_FROM}" ]]; then print_cmd "${GEN_CMD[@]}"; else echo "  (no generate: converting from ${CONVERT_FROM} shards)"; fi
    print_cmd "${ASM_CMD[@]}"
    exit 0
fi

# — Stage 0 — preflight ——————————————————————————————————————————————————————
# Check-and-instruct: this script never launches the multi-hour Docker job that
# builds the per-sample BCFs. It tells you how, and stops.
echo "[$(date '+%F %T')] preflight"

[ -f "${REF}"    ] || { echo "ERROR: reference missing: ${REF}"   >&2; exit 1; }
[ -f "${TOKENS}" ] || { echo "ERROR: tokens TSV missing: ${TOKENS}" >&2; exit 1; }
[ -f "${BED}"    ] || { echo "ERROR: LD-block BED missing: ${BED}" >&2; exit 1; }
[ -f "${TEST_SAMPLES}" ] || { echo "ERROR: test samples missing: ${TEST_SAMPLES}" >&2; exit 1; }

# The BED must be the one the -ldb presets pin. create_dataset.py enforces this
# too, but failing here costs seconds instead of minutes of startup. The
# expected hash is read from versions.py so there is one source of truth.
EXPECTED_BED_SHA=$("${PYTHON}" -c 'from preprocess.huggingface_dataset import versions; print(versions.LDB_BED_SHA256)')
ACTUAL_BED_SHA=$(sha256sum "${BED}" | cut -d' ' -f1)
if [[ "${ACTUAL_BED_SHA}" != "${EXPECTED_BED_SHA}" ]]; then
    echo "ERROR: ${BED} does not match the region set pinned by ${VERSION}." >&2
    echo "  expected sha256: ${EXPECTED_BED_SHA}" >&2
    echo "  actual   sha256: ${ACTUAL_BED_SHA}" >&2
    echo "Rebuild it with preprocess/regions/build_ld_block_bed.py from" >&2
    echo "preprocess/regions/ld_blocks/raw/deCODE_EUR_LD_blocks.bed." >&2
    exit 1
fi
echo "  BED ......... OK (${BED}, $(wc -l < "${BED}") blocks)"

if [[ -z "${CONVERT_FROM}" ]]; then
    # generate needs the BCFs; a conversion run reads existing shards instead.
    if [[ ! -d "${BCF_DIR}" ]]; then
        cat >&2 <<EOF
ERROR: per-sample BCF dir missing: ${BCF_DIR}

create_dataset.py does NOT intersect — it consumes an already-prepared
--bcf-dir. Build the per-sample, GT-only, hom-ref-dropped BCFs first (one-time,
~140 GB). This must run inside the glm_biotools image (bcftools >= 1.13), and
needs the raised fd limit because '+split' opens one file per sample:

  docker run --rm --user \$(id -u):\$(id -g) $(docker_mounts_text)\\
    --ulimit nofile=65536:65536 -e JOBS=8 ${DOCKER_IMAGE} \\
    bash ${RUNNERS_DIR}/make_persample_bcf.sh \\
      ${DATA_ROOT}/raw_bcf_ld_blocks_eur_intersected \\
      ${BED} \\
      ${BCF_DIR}

The input above is the EXISTING 23 GB region-filtered dir: '+split -R BED' with
the same BED is idempotent, so this reuses that artifact instead of re-reading
the full raw panel.
EOF
        exit 1
    fi
    # Must be the per-sample layout. The multi-sample layout also "works" but is
    # ~2.5x slower end-to-end, which is hours at genome scale — so refuse it
    # rather than silently taking it. Reuses create_dataset's own detector so
    # the rule cannot drift.
    if ! "${PYTHON}" -c "
import pathlib, sys
from preprocess.huggingface_dataset import create_dataset
sys.exit(0 if create_dataset._is_persample_bcf_dir(pathlib.Path('${BCF_DIR}')) else 1)
"; then
        if [[ "${ALLOW_MULTISAMPLE_BCF:-0}" == "1" ]]; then
            echo "  BCF dir ..... multi-sample layout (ALLOW_MULTISAMPLE_BCF=1) — expect ~2.5x slower"
        else
            echo "ERROR: ${BCF_DIR} is not the per-sample BCF layout." >&2
            echo "It looks like per-chromosome multi-sample BCFs, which are ~2.5x" >&2
            echo "slower end-to-end. Build the per-sample dir (see make_persample_bcf.sh)," >&2
            echo "or re-run with ALLOW_MULTISAMPLE_BCF=1 to accept the slow path." >&2
            exit 1
        fi
    else
        echo "  BCF dir ..... OK (${BCF_DIR}, $(ls "${BCF_DIR}"/*.bcf 2>/dev/null | wc -l) per-sample files)"
    fi
else
    [ -d "${SRC_DIR}/train" ] || { echo "ERROR: source shards missing: ${SRC_DIR}/train" >&2; exit 1; }
    echo "  source shards OK (${SRC_DIR})"
fi

mkdir -p "${HF_CACHE}"

# — Stage 1 — generate per-sample shards —————————————————————————————————————
if [[ -z "${CONVERT_FROM}" ]]; then
    echo "[$(date '+%F %T')] START generate (${VERSION})"
    "${GEN_CMD[@]}"
    echo "[$(date '+%F %T')] END generate"
else
    echo "[$(date '+%F %T')] SKIP generate — converting from ${CONVERT_FROM} shards"
fi

# — Stage 2 — assemble train + test into a single HF DatasetDict ——————————————
echo "[$(date '+%F %T')] START assemble (${VERSION})"
"${ASM_CMD[@]}"
echo "[$(date '+%F %T')] END assemble"

# — Summary ——————————————————————————————————————————————————————————————————
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=== SUCCESS: ${VERSION} dataset at ${ASM_DIR} ==="
echo "Assembled size: $(du -sh "${ASM_DIR}" 2>/dev/null | cut -f1)"
echo "=== Finished at $(date '+%F %T') (total $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s) ==="
echo "Remember to log this run in preprocess/regions/ld_blocks/PROVENANCE.md."
