#!/usr/bin/env bash
set -euo pipefail

# Build a per-gene TSS->TES dataset (v7-tsstes / v8-tsstes): prepare the
# BED-restricted per-sample BCFs, generate per-sample shards, then assemble into
# a HuggingFace DatasetDict.
#
# These versions are the v7/v8 ENCODINGS over the `tss_tes_gencode50_tx128`
# REGION UNIVERSE: 59,572 per-gene intervals over chr1-22 + chrX from GENCODE
# v50, every one a multiple of 128 bp. There is no minimum and no maximum, so
# rows run from 1,152 bp to 2.47 Mbp -- do not assume a size band. See
# preprocess/regions/nested_genome_division/README.md.
#
#   run_tsstes_dataset.sh v7-tsstes                    # generate + assemble
#   run_tsstes_dataset.sh v7-tsstes --build-bcf        # ... and build the BCFs first
#   run_tsstes_dataset.sh v7-tsstes --convert-from v8-tsstes
#                                                      # skip generate: assemble
#                                                      # existing v8-tsstes shards
#   run_tsstes_dataset.sh v7-tsstes --dry-run          # print the generate/assemble
#                                                      # commands, touch nothing, exit 0
#
# THREE THINGS THIS REGION SET DOES THAT NO OTHER ONE IN THE REPO DOES. Each has
# a guard below; read these before changing anything here.
#
# 1. **The BED's intervals OVERLAP.** It is a per-gene division: one row per
#    gene, so a snoRNA in an intron is its own row inside its host gene's row.
#    `create_dataset.py --bed` must get the real overlapping BED (its sha256 is
#    pinned), but `bcftools -R` only ever needs the UNION, so this script
#    derives a merged copy (${UNION_BED}) for the BCF-prep step. Feeding
#    bcftools 59,572 overlapping regions is at best wasted work.
#
#    Consequence downstream: rows sum to 2.313 Gbp over 1.948 Gbp of distinct
#    sequence (1.19x). Assemble-time dedup is on the sequence hash, so the
#    HG38_REF pseudo-sample's overlapping stretches collapse -- but a
#    variant-carrying sample's two overlapping regions are two distinct rows and
#    both survive. Size the run for the 2.313 Gbp figure, not the 1.948.
#
# 2. **It includes chrX**, and a wrong --bcf-dir fails SILENTLY. For the
#    per-sample layout, create_dataset.py reads available chromosomes from the
#    BCF *header* seqnames, which retain the full source contig list. So a BCF
#    dir built against a chr1-22 BED (e.g. raw_bcf_ld_blocks_eur_per_sample)
#    still advertises chrX, no warning fires, and all 1,875 chrX regions quietly
#    produce zero variant rows -- appearing only in HG38_REF. Stage 1c below
#    queries actual chrX variants to make that impossible.
#
# 3. **It is the repo's first dataset with chrX at all.** Male chrX hemizygous
#    and half-call genotypes are mirrored to both haplotypes by
#    get_genome_sequence.py, i.e. encoded as HOMOZYGOUS; the dedicated
#    hemizygous ("single") tokens in diploid_tokens.tsv are still unused. That
#    is a modeling decision you are inheriting, not a neutral default. No guard
#    can catch it -- it is correct per the current encoder.
#
# Paths are env-overridable so this runs on either the data node or the large
# target machine without editing the script:
#   REPO (default: the checkout this script lives in), DATA_ROOT, RAW_BCF_DIR, BCF_DIR,
#   UNION_BED, TEST_SAMPLES, WORKERS
#   JOBS=8                    parallel chromosomes for the BCF build
#   DOCKER_IMAGE=<tag>            default: GLM_BIOTOOLS_IMAGE pinned in
#                                 docker_images/glm_biotools/image.env
#   ALLOW_MULTISAMPLE_BCF=1   accept the ~2.5x slower multi-sample BCF layout
#   PHYLOP_BW, CADD_ARROW_DIR   per-base annotation sources (v7/v8 support both,
#                               see versions.py allow_annotations); default to the
#                               files the original v7 dataset used (the defaults of
#                               run_canonical_dataset.sh --version v7), so tsstes
#                               gets annotation parity with it.
#                               Set either to "" to build without that annotation.

START_TIME=$(date +%s)
echo "=== Started at $(date '+%F %T') ==="
echo "Command: $0 $*"

# — Arguments ————————————————————————————————————————————————————————————————
VERSION="${1:-}"
CONVERT_FROM=""
BUILD_BCF=0
DRY_RUN=0
case "${VERSION}" in
    v7-tsstes|v8-tsstes) ;;
    *)
        echo "ERROR: version must be v7-tsstes or v8-tsstes (got '${VERSION:-<none>}')" >&2
        echo "Usage: $0 {v7-tsstes|v8-tsstes} [--build-bcf] [--convert-from v8-tsstes] [--dry-run]" >&2
        exit 1
        ;;
esac
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-bcf)
            BUILD_BCF=1; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --convert-from)
            CONVERT_FROM="${2:-}"
            [[ -n "${CONVERT_FROM}" ]] || { echo "ERROR: --convert-from needs a version" >&2; exit 1; }
            shift 2 ;;
        *)
            echo "ERROR: unrecognized argument: $1" >&2; exit 1 ;;
    esac
done

# Only v8-tsstes -> v7-tsstes is a real route (REHYDRATE_REALIGN). The reverse is
# uphill and compat.py refuses it; so does every cross-region-set pair. Catch it
# here rather than after the shards have been read.
if [[ -n "${CONVERT_FROM}" ]]; then
    if [[ "${CONVERT_FROM}" == "${VERSION}" ]]; then
        echo "ERROR: --convert-from ${CONVERT_FROM} is the same as the target version" >&2
        exit 1
    fi
    if [[ "${CONVERT_FROM}" != "v8-tsstes" || "${VERSION}" != "v7-tsstes" ]]; then
        echo "ERROR: the only supported conversion is --convert-from v8-tsstes into v7-tsstes." >&2
        echo "  ${CONVERT_FROM} -> ${VERSION} is not a downhill route; compat.py will refuse it." >&2
        exit 1
    fi
fi

# — Paths ————————————————————————————————————————————————————————————————————
# REPO (this checkout unless exported), RUNNERS_DIR, the data paths, GLM_BIOTOOLS_IMAGE:
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
DATA_ROOT="${DATA_ROOT:-/storage/data/1kgenomes}"
BED="${TSSTES_BED}"
REF="${HG38_2BIT}"
TOKENS="${DIPLOID_TOKENS}"
TEST_SAMPLES="${TEST_SAMPLES:-${REPO}/preprocess/regions/sample_splits/sample_split_20251209_v5/1kg_test_samples_v5.txt}"

RAW_BCF_DIR="${RAW_BCF_DIR:-${DATA_ROOT}/raw_bcf}"
BCF_DIR="${BCF_DIR:-${DATA_ROOT}/bcf_tss_tes_per_sample}"
UNION_BED="${UNION_BED:-${DATA_ROOT}/tss_tes_union.bed}"
WORKERS="${WORKERS:-60}"
JOBS="${JOBS:-8}"
# The glm_biotools tag lives in ONE place (image.env); DOCKER_IMAGE still overrides per run.
DOCKER_IMAGE="${DOCKER_IMAGE:-${GLM_BIOTOOLS_IMAGE}}"
PHYLOP_BW="${PHYLOP_BW-/storage/data/PhyloP/hg38.phyloP100way.bw}"
CADD_ARROW_DIR="${CADD_ARROW_DIR-/storage/data/CADD/full_arrow}"

OUT_DIR="${DATA_ROOT}/diploid_hf_${VERSION}"
ASM_DIR="${OUT_DIR}/assembled"
HF_CACHE="${OUT_DIR}/hf_cache"   # big partition, not ~/.cache
LOG_DIR="${OUT_DIR}/logs"

SRC_DIR="${OUT_DIR}"
[[ -n "${CONVERT_FROM}" ]] && SRC_DIR="${DATA_ROOT}/diploid_hf_${CONVERT_FROM}"

cd "${REPO}"

ANNOTATION_ARGS=()
[[ -n "${PHYLOP_BW}"      ]] && ANNOTATION_ARGS+=(--phylop-bw "${PHYLOP_BW}")
[[ -n "${CADD_ARROW_DIR}" ]] && ANNOTATION_ARGS+=(--cadd-arrow-dir "${CADD_ARROW_DIR}")
GEN_CMD=(
    uv run --no-sync python -m preprocess.huggingface_dataset.create_dataset generate
        --version           "${VERSION}"
        --representation    diploid
        --tokens            "${TOKENS}"
        --bcf-dir           "${BCF_DIR}"
        --bed               "${BED}"
        --ref               "${REF}"
        --output            "${OUT_DIR}"
        --test-samples-file "${TEST_SAMPLES}"
        "${ANNOTATION_ARGS[@]}"
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
ASM_CMD=(uv run --no-sync python -m preprocess.huggingface_dataset.create_dataset assemble "${ASSEMBLE_ARGS[@]}")

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "=== DRY RUN — nothing checked, built or run. The run would execute, from ${REPO}:"
    if [[ -z "${CONVERT_FROM}" ]]; then
        [[ "${BUILD_BCF}" == "1" && ! -d "${BCF_DIR}" ]] && echo "  (first the BCF build: ${RUNNERS_DIR}/make_persample_bcf.sh in ${DOCKER_IMAGE})"
        print_cmd "${GEN_CMD[@]}"
    else
        echo "  (no generate: converting from ${CONVERT_FROM} shards)"
    fi
    print_cmd "${ASM_CMD[@]}"
    exit 0
fi

mkdir -p "${LOG_DIR}" "${HF_CACHE}"

# — Stage 0 — preflight ——————————————————————————————————————————————————————
echo "[$(date '+%F %T')] preflight"

[ -f "${REF}"          ] || { echo "ERROR: reference missing: ${REF}"          >&2; exit 1; }
[ -f "${TOKENS}"       ] || { echo "ERROR: tokens TSV missing: ${TOKENS}"      >&2; exit 1; }
[ -f "${BED}"          ] || { echo "ERROR: tss-tes BED missing: ${BED}"        >&2; exit 1; }
[ -f "${TEST_SAMPLES}" ] || { echo "ERROR: test samples missing: ${TEST_SAMPLES}" >&2; exit 1; }
[[ -z "${PHYLOP_BW}"      || -f "${PHYLOP_BW}"      ]] || { echo "ERROR: phyloP bigWig missing: ${PHYLOP_BW}" >&2; exit 1; }
[[ -z "${CADD_ARROW_DIR}" || -d "${CADD_ARROW_DIR}" ]] || { echo "ERROR: CADD arrow dir missing: ${CADD_ARROW_DIR}" >&2; exit 1; }

# The BED must be the one the -tsstes presets pin. create_dataset.py enforces
# this too, but failing here costs seconds instead of minutes of startup. The
# expected hash is read from versions.py so there is one source of truth.
EXPECTED_BED_SHA=$(uv run --no-sync python -c 'from preprocess.huggingface_dataset import versions; print(versions.TSSTES_BED_SHA256)')
ACTUAL_BED_SHA=$(sha256sum "${BED}" | cut -d' ' -f1)
if [[ "${ACTUAL_BED_SHA}" != "${EXPECTED_BED_SHA}" ]]; then
    echo "ERROR: ${BED} does not match the region set pinned by ${VERSION}." >&2
    echo "  expected sha256: ${EXPECTED_BED_SHA}" >&2
    echo "  actual   sha256: ${ACTUAL_BED_SHA}" >&2
    echo "Rebuild it with:" >&2
    echo "  cd ${REPO}" >&2
    echo "  uv run --no-sync python -m preprocess.regions.nested_genome_division.build_division --out preprocess/regions/nested_genome_division/out/hg38_tss_tes.bed" >&2
    echo "and either copy it over ${BED} or re-pin TSSTES_BED_SHA256 in versions.py." >&2
    exit 1
fi
echo "  BED ......... OK ($(wc -l < "${BED}") intervals, sha ${ACTUAL_BED_SHA:0:12}…)"

# — Stage 0b — the merged union BED, for bcftools only ———————————————————————
# Derived from the pinned BED, so it cannot drift from it. Idempotent: rebuilt
# whenever it is missing or older than the BED it comes from.
if [[ ! -f "${UNION_BED}" || "${BED}" -nt "${UNION_BED}" ]]; then
    echo "  union BED ... deriving ${UNION_BED}"
    uv run --no-sync python - "${BED}" "${UNION_BED}" <<'PY'
import pathlib, sys
from preprocess.regions.nested_genome_division import sources

bed, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
spans = sources.read_bed(bed)                       # always merges -> disjoint union
tmp = out.with_suffix(out.suffix + ".tmp")
n = total = 0
with open(tmp, "w") as fh:
    for chrom in sources.DEFAULT_CHROMS:
        for a, b in spans.get(chrom, []):
            fh.write(f"{chrom}\t{a}\t{b}\n")
            n += 1
            total += b - a
tmp.replace(out)
print(f"    {n:,} merged regions, {total/1e9:.3f} Gbp")
PY
else
    echo "  union BED ... OK (${UNION_BED}, $(wc -l < "${UNION_BED}") merged regions)"
fi

# — Stage 1 — per-sample, BED-restricted BCFs ————————————————————————————————
if [[ -z "${CONVERT_FROM}" ]]; then
    if [[ ! -d "${BCF_DIR}" ]] && [[ "${BUILD_BCF}" == "1" ]]; then
        [ -d "${RAW_BCF_DIR}" ] || { echo "ERROR: raw panel missing: ${RAW_BCF_DIR}" >&2; exit 1; }
        BCF_LOG="${LOG_DIR}/make_persample_bcf_$(date '+%Y%m%d_%H%M%S').log"
        echo "[$(date '+%F %T')] START BCF build -> ${BCF_DIR}"
        echo "  log: ${BCF_LOG}   (hours, ~210 GB output + transient scratch)"
        # --ulimit is required: bcftools +split opens one fd per sample and the
        # panel has 3202. JOBS goes in via -e, not a shell prefix -- a shell
        # prefix does not enter the container.
        docker run --rm --user "$(id -u):$(id -g)" \
            -v /storage:/storage \
            --ulimit nofile=65536:65536 \
            -e JOBS="${JOBS}" \
            --name "tsstes_bcf_$$" \
            "${DOCKER_IMAGE}" \
            bash "${RUNNERS_DIR}/make_persample_bcf.sh" \
                "${RAW_BCF_DIR}" "${UNION_BED}" "${BCF_DIR}" 2>&1 | tee "${BCF_LOG}"
        echo "[$(date '+%F %T')] END BCF build"
    elif [[ ! -d "${BCF_DIR}" ]]; then
        cat >&2 <<EOF
ERROR: per-sample BCF dir missing: ${BCF_DIR}

create_dataset.py does NOT intersect — it consumes an already-prepared
--bcf-dir. Build it from the FULL panel (${RAW_BCF_DIR}, which has chrX)
restricted to the merged tss-tes union. Re-run this script with --build-bcf,
or run it yourself:

  docker run --rm --user \$(id -u):\$(id -g) -v /storage:/storage \\
    --ulimit nofile=65536:65536 -e JOBS=${JOBS} ${DOCKER_IMAGE} \\
    bash ${RUNNERS_DIR}/make_persample_bcf.sh \\
      ${RAW_BCF_DIR} \\
      ${UNION_BED} \\
      ${BCF_DIR}

Hours, ~210 GB (scaled from the 283 GB LD-block dir over 2.806 Gbp). Do NOT
substitute raw_bcf_ld_blocks_eur_per_sample or
raw_bcf_bed_intersected_hg38_pretrain_canonical_expanded_per_sample: both are
restricted to chr1-22 region sets, and stage 1c below will reject them.
EOF
        exit 1
    fi

    # 1b — layout. The multi-sample layout also "works" but is ~2.5x slower
    # end-to-end, which is hours at genome scale. Reuses create_dataset's own
    # detector so the rule cannot drift.
    if ! uv run --no-sync python -c "
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

    # 1c — chrX must carry ACTUAL VARIANTS, not just a header entry.
    # This is the guard for failure mode 2 in the header comment: a BCF dir
    # built against a chr1-22 BED still advertises chrX in its header, so
    # create_dataset.py's chromosome filter passes it and the chrX regions
    # silently yield nothing. Querying real records is the only way to tell.
    echo "  chrX check .. querying real variants…"
    uv run --no-sync python - "${BCF_DIR}" "${BED}" <<'PY'
import pathlib, sys
from cyvcf2 import VCF

bcf_dir, bed = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
samples = sorted(bcf_dir.glob("*.bcf"))[:3]
if not samples:
    sys.exit("ERROR: no *.bcf files in %s" % bcf_dir)

# The widest chrX intervals are the likeliest to carry a variant in any one
# sample; 20 of them across 3 samples is far more than enough on 1kG.
regions = []
with open(bed) as fh:
    for line in fh:
        f = line.split("\t")
        if f[0] == "chrX":
            regions.append((int(f[1]), int(f[2])))
if not regions:
    sys.exit("ERROR: %s has no chrX intervals — is it the right BED?" % bed)
regions.sort(key=lambda r: r[1] - r[0], reverse=True)
regions = regions[:20]

found = 0
for path in samples:
    with VCF(str(path)) as v:
        if "chrX" not in set(v.seqnames):
            sys.exit(
                f"ERROR: {path.name} has no chrX in its header at all.\n"
                "  The BCFs were built without chrX. Rebuild from the full panel."
            )
        for a, b in regions:
            for _ in v(f"chrX:{a + 1}-{b}"):
                found += 1
                break
            if found:
                break
    if found:
        break

if not found:
    sys.exit(
        "ERROR: chrX is in the BCF header but carries NO variants in the 20 "
        "widest chrX\n  intervals of the division, across 3 samples.\n"
        "  This is the silent-failure mode: the BCFs were region-restricted to a\n"
        "  chr1-22 BED (LD blocks / canonical), so chrX advertises itself in the\n"
        "  header but holds nothing. create_dataset.py would NOT warn — it reads\n"
        "  available chromosomes from the header. All 2,296 chrX regions would\n"
        "  produce zero variant rows.\n"
        "  Rebuild the BCFs from the full panel against the tss-tes union BED."
    )
print("    OK — chrX carries variants")
PY

else
    [ -d "${SRC_DIR}/train" ] || { echo "ERROR: source shards missing: ${SRC_DIR}/train" >&2; exit 1; }
    echo "  source shards OK (${SRC_DIR})"
fi

# — Stage 2 — generate per-sample shards —————————————————————————————————————
if [[ -z "${CONVERT_FROM}" ]]; then
    echo "[$(date '+%F %T')] START generate (${VERSION})"
    echo "  NOTE: the region set OVERLAPS — rows sum to 2.313 Gbp over 1.948 Gbp"
    echo "        of genome. Budget shard space for the larger figure."
    echo "  annotations . ${ANNOTATION_ARGS[*]:-<none>}"
    "${GEN_CMD[@]}"
    echo "[$(date '+%F %T')] END generate"
else
    echo "[$(date '+%F %T')] SKIP generate — converting from ${CONVERT_FROM} shards"
fi

# — Stage 3 — assemble train + test into a single HF DatasetDict ——————————————
echo "[$(date '+%F %T')] START assemble (${VERSION})"
"${ASM_CMD[@]}"
echo "[$(date '+%F %T')] END assemble"

# — Summary ——————————————————————————————————————————————————————————————————
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=== SUCCESS: ${VERSION} dataset at ${ASM_DIR} ==="
echo "Assembled size: $(du -sh "${ASM_DIR}" 2>/dev/null | cut -f1)"
echo "Region set:     tss_tes_gencode50_tx128 (59,572 per-gene intervals, chr1-22 + chrX)"
echo "=== Finished at $(date '+%F %T') (total $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s) ==="
echo ""
echo "Log this run in preprocess/regions/nested_genome_division/DECISION_LEDGER.md."
