#!/usr/bin/env bash
set -euo pipefail

# Build a v10 per-patient PHASED HAPLOTYPE-PAIR dataset: annotated BED → union
# BED → per-sample BCFs → `create_haplotype_dataset.py generate` (+ finalize-ref)
# → smoke verify. Modelled on run_tsstes_dataset.sh (same check-and-instruct
# style); the differences are the ones v10 forces, listed below.
#
#   run_v10_dataset.sh                    # preflight, then generate+finalize+verify
#                                         #   DETACHED under run_supervised.sh
#   run_v10_dataset.sh --build-bcf        # ... and build the per-sample BCFs first
#   run_v10_dataset.sh --dry-run          # stages 0-1 only; print the commands
#   run_v10_dataset.sh --foreground       # run every stage in this shell (what the
#                                         #   supervised child runs; also for tests)
#   TEST_SAMPLES= SAMPLES_FILE=<2 ids> MAX_SAMPLES=2 MAX_REGIONS=200 WORKERS=2 \
#     OUT_DIR=<scratch> run_v10_dataset.sh --foreground      # a smoke: flat layout
#
# SMOKE RUNS MUST CLEAR TEST_SAMPLES. `generate` inherits create_dataset.py's split
# semantics: --max-samples caps the TRAIN list, and every sample named in
# --test-samples-file is queued in full so the test split is always complete.
# A "2-sample" smoke with the BioMe split file set therefore writes 2 + 2,320
# shards (observed 2026-10-05). TEST_SAMPLES= (set but empty) disables the split.
#
# WHAT v10 IS (plan: ~/.claude/plans/i-want-to-add-vectorized-grove.md; decisions:
# ../DECISION_LEDGER.md 2026-10-05): two haploid `ACGTN^{}` strings per
# (sample × gene) straight from the BCF phasing, `sample_id` + GENCODE `gene_id`
# + strand on every row, NO dedup / shuffle / assemble, one zstd Parquet per
# patient, an HG38_REF pseudo-sample per split weighted by "samples with no row
# for this region". An unphased heterozygote FAILS that sample (exit 1). The
# marker grammar is NOT comparable to v4-v8 (a carried deletion is `{}`; `^` is
# per haplotype) — see ../README.md §"v10 — per-patient haplotype pairs".
#
# REGION SET: the SAME 59,572 intervals as v7-tsstes (tss_tes_gencode50_tx128),
# re-emitted as regions/nested_genome_division/hg38_tss_tes_v10.bed — a headered BED6+3
# (`#region_set=tss_tes_gencode50_tx128_v10`, gene_id in col 4, +gene_name
# gene_type n_transcripts) with a `.genes.tsv` sidecar. Columns 1-3 + strand
# are byte-identical to the pinned hg38_tss_tes.bed, so a BCF dir cut on the
# tss-tes union is valid v10 input — stage 0b PROVES that by comparing the
# union derived from the v10 BED against the recorded ${UNION_BED}.
#
# THE THREE tss-tes GUARDS STILL APPLY (see run_tsstes_dataset.sh for the long
# form): (1) the BED overlaps — bcftools -R only ever sees the merged union;
# (2) chrX in a BCF header does not mean chrX variants — stage 1c queries real
# records; (3) male chrX hemizygous calls are mirrored to both haplotypes.
#
# BCF PREP — alternatives considered (plan §B), NOT implemented:
#   * Region-subset multi-sample BCF per chromosome (`bcftools view -R union -Ob`
#     + index, 23 files, no fd pressure, ~1 h/chrom): the encoder reads that
#     legacy layout via set_samples, but every per-sample pass decodes ALL
#     samples' GT per record (~2.5x slower generate, ALLOW_MULTISAMPLE_BCF=1).
#     Right only for pilots / cohorts of a few hundred.
#   * Per-sample `bcftools view -S <s> -R union` straight from the raw panel:
#     N × 23 full-panel reads; infeasible at 11.6k samples.
#   * No subsetting (encoder on the raw panel): the CSI query is region-bounded
#     but each record carries every genotype; decode-bound. Rejected.
#   Chosen: reuse make_persample_bcf.sh (`bcftools +split -R union -i 'GT="alt"'
#   -k FMT/GT` per chromosome, then concat per sample) — the proven path; for
#   large cohorts the distributed tools/chunk_union_bed.py → run_chunked_split.sh →
#   concat_persample_bcf.sh route (sentinel-based resume).
#
# SUPERVISION. generate runs for hours, so stages 2-4 run DETACHED under
# run_supervised.sh (HEARTBEAT_S=300 STALL_LOG_S=1800): this script does the
# preflight (stages 0-1) in the foreground, then re-launches itself with
# --foreground under the supervisor and prints the status dir. `cat <dir>/state`
# is the check (RUNNING/STALLED/DONE/FAILED); run.log has every line. A re-run
# after a crash resumes (generate skips finished shards; finalize-ref is
# derived data and simply overwrites).
#
# PYTHON. Every Python stage runs under $PYTHON (default python3 on PATH): run this
# from the environment the repo's requirements are installed in, or export PYTHON.
#
# Env (all overridable):
#   REPO, DATA_ROOT, RAW_BCF_DIR, BCF_DIR, UNION_BED, BED, REF, TOKENS,
#   TEST_SAMPLES (empty = no split), SAMPLES_FILE, OUT_DIR, WORKERS=60, JOBS=8
#   DOCKER_IMAGE=<tag>          default: GLM_BIOTOOLS_IMAGE from image.env
#   V10_UNION_BED               where the union derived from $BED is written
#                               (default ${DATA_ROOT}/tss_tes_v10_union.bed)
#   FORMAT=parquet|tsv|both  TSV_MODE=variants|full  COMPRESSION=zstd
#   COMPRESSION_LEVEL=3  REGION_BATCH_SIZE=200  MAX_SAMPLES  MAX_REGIONS
#   ALLOW_MULTISAMPLE_BCF=1     accept the ~2.5x slower multi-sample BCF layout
#   FORCE_SINGLE_HOST_BCF=1     run make_persample_bcf.sh even for >5,000 samples
#   PYTHON (see above), SUPERVISOR (default: run_supervised.sh next to this script)
#   DOCKER_EXTRA_MOUNTS         extra dirs to bind into the glm_biotools container

START_TIME=$(date +%s)
echo "=== Started at $(date '+%F %T') on $(hostname) ==="
echo "Command: $0 $*"

# Absolute path to this script. The detached relaunch below runs AFTER
# `cd "${REPO}"`, so a relative "$0" (e.g. huggingface_dataset/runners/run_v10_dataset.sh
# from the repo root) would not resolve in the child -- it died with exit 127
# while the parent printed the success banner (review finding, 2026-10-05).
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

usage() {
    sed -n '3,/^START_TIME=/p' "$SELF" | sed '$d' | sed 's/^# \{0,1\}//'
}

# — Arguments ————————————————————————————————————————————————————————————————
BUILD_BCF=0
DRY_RUN=0
FOREGROUND=0
ORIG_ARGS=("$@")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-bcf)   BUILD_BCF=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --foreground)  FOREGROUND=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)
            echo "ERROR: unrecognized argument: $1" >&2
            echo "Usage: $0 [--build-bcf] [--dry-run] [--foreground]" >&2
            exit 1 ;;
    esac
done

# — Paths ————————————————————————————————————————————————————————————————————
# REPO: this checkout unless exported (an exported one is resolved like every path
# below). _common.sh also gives RUNNERS_DIR, DATA_ROOT, the data paths, GLM_BIOTOOLS_IMAGE, PYTHON, pyrun.
[[ -n "${REPO:-}" ]] && REPO="$(realpath -m "${REPO}")"
# shellcheck source=_common.sh
source "$(dirname "${SELF}")/_common.sh"
# The glm_biotools tag lives in ONE place (image.env); DOCKER_IMAGE still overrides per run.
# DATA_ROOT comes from _common.sh
HFD="${REPO}/preprocess/huggingface_dataset"
BED="${BED:-${TSSTES_V10_BED}}"
REF="${REF:-${HG38_2BIT}}"
TOKENS="${TOKENS:-${DIPLOID_TOKENS}}"
# TEST_SAMPLES= (set but empty) means NO train/test split: flat layout, and the run is
# limited to SAMPLES_FILE / MAX_SAMPLES. With a test file, generate always adds EVERY
# test sample to the queue (create_dataset.py semantics: --max-samples caps train only),
# so a two-sample smoke must clear it.
TEST_SAMPLES="${TEST_SAMPLES-${REPO}/preprocess/regions/sample_splits/sample_split_20251209_v5/1kg_test_samples_v5.txt}"
SAMPLES_FILE="${SAMPLES_FILE:-}"

RAW_BCF_DIR="${RAW_BCF_DIR:-${DATA_ROOT}/raw_bcf}"
BCF_DIR="${BCF_DIR:-${DATA_ROOT}/bcf_tss_tes_per_sample}"
UNION_BED="${UNION_BED:-${DATA_ROOT}/tss_tes_union.bed}"          # the one the BCFs were cut on
V10_UNION_BED="${V10_UNION_BED:-${DATA_ROOT}/tss_tes_v10_union.bed}"  # derived from $BED here
WORKERS="${WORKERS:-60}"
JOBS="${JOBS:-8}"
DOCKER_IMAGE="${DOCKER_IMAGE:-${GLM_BIOTOOLS_IMAGE}}"

OUT_DIR="${OUT_DIR:-${DATA_ROOT}/haplotype_pairs_v10}"
# Every path is made absolute HERE, before the `cd "${REPO}"` below: a relative
# REF/TOKENS/BED/... from the caller's cwd kept its meaning for the caller,
# not for a script that has changed directory. `realpath -m` does not require
# the path to exist (BCF_DIR / OUT_DIR may not yet).
for _v in DATA_ROOT BED REF TOKENS TEST_SAMPLES SAMPLES_FILE RAW_BCF_DIR BCF_DIR \
          UNION_BED V10_UNION_BED OUT_DIR; do
    [[ -n "${!_v}" ]] && printf -v "$_v" '%s' "$(realpath -m "${!_v}")"
done
LOG_DIR="${OUT_DIR}/logs"
FORMAT="${FORMAT:-parquet}"
TSV_MODE="${TSV_MODE:-variants}"
COMPRESSION="${COMPRESSION:-zstd}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-3}"
REGION_BATCH_SIZE="${REGION_BATCH_SIZE:-200}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_REGIONS="${MAX_REGIONS:-}"

# Everything resolved above must reach the supervised child unchanged.
export REPO DATA_ROOT BED REF TOKENS TEST_SAMPLES SAMPLES_FILE RAW_BCF_DIR BCF_DIR \
       UNION_BED V10_UNION_BED WORKERS JOBS DOCKER_IMAGE OUT_DIR FORMAT TSV_MODE \
       COMPRESSION COMPRESSION_LEVEL REGION_BATCH_SIZE MAX_SAMPLES MAX_REGIONS

case "${FORMAT}" in parquet|tsv|both) ;; *) echo "ERROR: FORMAT must be parquet|tsv|both (got '${FORMAT}')" >&2; exit 1 ;; esac
case "${TSV_MODE}" in variants|full) ;; *) echo "ERROR: TSV_MODE must be variants|full (got '${TSV_MODE}')" >&2; exit 1 ;; esac

# — python ——————————————————————————————————————————————————————————————————————
command -v "${PYTHON}" >/dev/null || { echo "ERROR: ${PYTHON} not found (export PYTHON)" >&2; exit 1; }
echo "python: $(command -v "${PYTHON}") ($("${PYTHON}" -V 2>&1))"
cd "${REPO}"
[[ "${DRY_RUN}" == "1" ]] || mkdir -p "${LOG_DIR}"

# — Stage 0 — the annotated BED ——————————————————————————————————————————————
echo "[$(date '+%F %T')] preflight"
[ -f "${REF}"          ] || { echo "ERROR: reference missing: ${REF}"             >&2; exit 1; }
[ -f "${TOKENS}"       ] || { echo "ERROR: tokens TSV missing: ${TOKENS}"         >&2; exit 1; }
[[ -z "${TEST_SAMPLES}" || -f "${TEST_SAMPLES}" ]] || { echo "ERROR: test samples missing: ${TEST_SAMPLES}" >&2; exit 1; }
[[ -n "${TEST_SAMPLES}" ]] || echo "  split ....... none (TEST_SAMPLES empty) — flat layout, samples limited to SAMPLES_FILE/MAX_SAMPLES"
[[ -z "${SAMPLES_FILE}" || -f "${SAMPLES_FILE}" ]] || { echo "ERROR: samples file missing: ${SAMPLES_FILE}" >&2; exit 1; }

if [[ ! -f "${BED}" ]]; then
    if [[ "${BED}" != "${TSSTES_V10_BED}" ]]; then
        echo "ERROR: BED missing: ${BED} (only the default v10 BED is built automatically)" >&2
        exit 1
    fi
    [[ "${DRY_RUN}" == "1" ]] && { echo "ERROR: ${BED} missing; a dry run does not build it" >&2; exit 1; }
    echo "  BED ......... missing — building the annotated division (~2 min, GENCODE v50)"
    pyrun -m preprocess.regions.nested_genome_division.build_division --out /dev/null --annotated-out "${BED}" --quiet
fi
BED_SHA=$(sha256sum "${BED}" | cut -d' ' -f1)
BED_REGION_SET=$(grep -m1 '^#region_set=' "${BED}" | cut -d= -f2- || true)
BED_ROWS=$(grep -vc '^#' "${BED}" || true)
echo "  BED ......... ${BED}"
echo "                ${BED_ROWS} intervals, sha256 ${BED_SHA}, #region_set=${BED_REGION_SET:-<none>}"
GENES_TSV="${BED%.bed}.genes.tsv"
if [[ -f "${GENES_TSV}" ]]; then
    echo "  genes.tsv ... OK (${GENES_TSV}; copied to <out>/region_annotations.tsv by generate)"
    EXPECT_ANNOTATIONS=1
else
    echo "  genes.tsv ... none next to the BED (region_annotations.tsv will not be written)"
    EXPECT_ANNOTATIONS=0
fi
BED_HAS_CHRX=$(grep -v '^#' "${BED}" | cut -f1 | grep -cx 'chrX' || true)

# — Stage 0b — the merged union BED, for bcftools only ———————————————————————
# Always re-derived from $BED (seconds) so it cannot drift from it, then
# reconciled with the RECORDED union the existing BCF dir was cut on.
echo "  union BED ... deriving from ${BED}"
mkdir -p "$(dirname "${V10_UNION_BED}")"     # a fresh DATA_ROOT has no dir yet
pyrun - "${BED}" "${V10_UNION_BED}" <<'PY'
import pathlib, sys
from preprocess.regions.nested_genome_division import sources

bed, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
spans = sources.read_bed(bed)                       # skips '#' lines; merges -> disjoint union
tmp = out.with_suffix(out.suffix + ".tmp")
n = total = 0
order = list(sources.DEFAULT_CHROMS) + sorted(set(spans) - set(sources.DEFAULT_CHROMS))
with open(tmp, "w") as fh:
    for chrom in order:
        for a, b in spans.get(chrom, []):
            fh.write(f"{chrom}\t{a}\t{b}\n")
            n += 1
            total += b - a
if out.exists() and out.read_bytes() == tmp.read_bytes():
    tmp.unlink()
else:
    tmp.replace(out)
print(f"    {n:,} merged regions, {total/1e9:.3f} Gbp -> {out}")
PY
V10_UNION_SHA=$(sha256sum "${V10_UNION_BED}" | cut -d' ' -f1)
if [[ -f "${UNION_BED}" && "${UNION_BED}" != "${V10_UNION_BED}" ]]; then
    REC_UNION_SHA=$(sha256sum "${UNION_BED}" | cut -d' ' -f1)
    if [[ "${REC_UNION_SHA}" == "${V10_UNION_SHA}" ]]; then
        echo "  union BED ... MATCHES the recorded ${UNION_BED} (sha ${REC_UNION_SHA:0:12}…)"
        echo "                -> a BCF dir cut on it is valid v10 input; no new BCF build needed"
    elif [[ -d "${BCF_DIR}" ]]; then
        echo "ERROR: the union derived from ${BED}" >&2
        echo "  (sha256 ${V10_UNION_SHA}) differs from the recorded ${UNION_BED}" >&2
        echo "  (sha256 ${REC_UNION_SHA}) that ${BCF_DIR} was cut on." >&2
        echo "  The existing BCFs cover a DIFFERENT region universe; generate would silently" >&2
        echo "  miss variants. Point BCF_DIR at a dir built on the v10 union, or build one:" >&2
        echo "    BCF_DIR=<new dir> UNION_BED=${V10_UNION_BED} $0 --build-bcf" >&2
        exit 1
    else
        echo "  union BED ... differs from the recorded ${UNION_BED}; using the v10 union for the build"
        UNION_BED="${V10_UNION_BED}"
    fi
else
    UNION_BED="${V10_UNION_BED}"
    echo "  union BED ... no recorded union; using ${UNION_BED}"
fi
export UNION_BED

# — Stage 1 — per-sample, BED-restricted BCFs ————————————————————————————————
BCF_INSTRUCT() {
    cat >&2 <<EOF
ERROR: per-sample BCF dir missing: ${BCF_DIR}

create_haplotype_dataset.py does NOT intersect — it consumes an already-prepared
--bcf-dir of <sample>.bcf files restricted to the merged union (${UNION_BED}).
Re-run this script with --build-bcf, or run it yourself:

  docker run --rm --user \$(id -u):\$(id -g) $(docker_mounts_text)\\
    --ulimit nofile=65536:65536 -e JOBS=${JOBS} ${DOCKER_IMAGE} \\
    bash ${RUNNERS_DIR}/make_persample_bcf.sh \\
      ${RAW_BCF_DIR} \\
      ${UNION_BED} \\
      ${BCF_DIR}${SAMPLES_FILE:+ \\
      ${SAMPLES_FILE}}

Hours, ~210 GB for 1kG (165 GB for the 11,586-sample BioMe cohort). For large
cohorts use the distributed route instead (sentinel-based resume):
  tools/chunk_union_bed.py -> run_chunked_split.sh (per host) -> stage_parts_to_storage.sh
  -> concat_persample_bcf.sh
Do NOT substitute a chr1-22 dir (raw_bcf_ld_blocks_eur_per_sample, …canonical…):
stage 1c below rejects them.
EOF
}

if [[ ! -d "${BCF_DIR}" && "${BUILD_BCF}" == "1" && "${DRY_RUN}" != "1" ]]; then
    [ -d "${RAW_BCF_DIR}" ] || { echo "ERROR: raw panel missing: ${RAW_BCF_DIR}" >&2; exit 1; }
    # Size gate. bcftools +split holds one fd per sample per parallel chromosome
    # (JOBS × N + headroom < 65536), and above a few thousand samples the
    # single-host script's unsound resume test (output non-empty) makes an
    # interrupted run unrecoverable — the distributed route was built for the
    # 11,586-sample BioMe cohort for exactly that reason. The 3,202-sample 1kG
    # panel is the proven single-host case.
    if [[ -n "${SAMPLES_FILE}" ]]; then
        N_SAMPLES=$(grep -cve '^\s*$' "${SAMPLES_FILE}")
    else
        FIRST_RAW=$(ls "${RAW_BCF_DIR}"/*.bcf 2>/dev/null | head -1)
        [ -n "${FIRST_RAW}" ] || { echo "ERROR: no *.bcf in ${RAW_BCF_DIR}" >&2; exit 1; }
        docker_mount_args
        N_SAMPLES=$(docker run --rm --user "$(id -u):$(id -g)" "${DOCKER_MOUNT_ARGS[@]}" "${DOCKER_IMAGE}" \
                        bcftools query -l "${FIRST_RAW}" | wc -l)
    fi
    echo "  BCF build ... ${N_SAMPLES} samples, JOBS=${JOBS} -> $((JOBS * N_SAMPLES + 1000)) fds (limit 65536)"
    if (( JOBS * N_SAMPLES + 1000 >= 65536 )); then
        echo "ERROR: JOBS=${JOBS} × ${N_SAMPLES} samples exceeds the 65,536 fd ulimit; lower JOBS" >&2
        echo "  (BioMe uses JOBS=5 -> 58,930) or use the distributed route (tools/chunk_union_bed.py …)." >&2
        exit 1
    fi
    if (( N_SAMPLES > 5000 )) && [[ "${FORCE_SINGLE_HOST_BCF:-0}" != "1" ]]; then
        echo "ERROR: ${N_SAMPLES} samples — use the distributed BCF route (tools/chunk_union_bed.py ->" >&2
        echo "  run_chunked_split.sh -> stage_parts_to_storage.sh -> concat_persample_bcf.sh); the" >&2
        echo "  single-host script cannot resume an interrupted chromosome. FORCE_SINGLE_HOST_BCF=1 overrides." >&2
        exit 1
    fi
    BCF_LOG="${LOG_DIR}/make_persample_bcf_$(date '+%Y%m%d_%H%M%S').log"
    echo "[$(date '+%F %T')] START BCF build -> ${BCF_DIR}"
    echo "  log: ${BCF_LOG}   (hours; output + transient scratch)"
    # --ulimit is required: bcftools +split opens one fd per sample. JOBS goes
    # in via -e, not a shell prefix -- a shell prefix does not enter the container.
    docker_mount_args
    docker run --rm --user "$(id -u):$(id -g)" \
        "${DOCKER_MOUNT_ARGS[@]}" \
        --ulimit nofile=65536:65536 \
        -e JOBS="${JOBS}" \
        --name "v10_bcf_$$" \
        "${DOCKER_IMAGE}" \
        bash "${RUNNERS_DIR}/make_persample_bcf.sh" \
            "${RAW_BCF_DIR}" "${UNION_BED}" "${BCF_DIR}" ${SAMPLES_FILE:+"${SAMPLES_FILE}"} 2>&1 | tee "${BCF_LOG}"
    echo "[$(date '+%F %T')] END BCF build"
elif [[ ! -d "${BCF_DIR}" ]]; then
    BCF_INSTRUCT
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  BCF dir ..... MISSING (dry run continues; see the instructions above)"
    else
        exit 1
    fi
fi

if [[ -d "${BCF_DIR}" ]]; then
    # 1b — layout. The multi-sample layout also "works" but is ~2.5x slower
    # end-to-end. Reuses create_dataset's own detector so the rule cannot drift.
    if ! pyrun -c "
import pathlib, sys
from preprocess.huggingface_dataset import create_dataset
sys.exit(0 if create_dataset._is_persample_bcf_dir(pathlib.Path('${BCF_DIR}')) else 1)
"; then
        if [[ "${ALLOW_MULTISAMPLE_BCF:-0}" == "1" ]]; then
            echo "  BCF dir ..... multi-sample layout (ALLOW_MULTISAMPLE_BCF=1) — expect ~2.5x slower"
        elif [[ "${DRY_RUN}" == "1" ]]; then
            echo "  BCF dir ..... NOT the per-sample layout (dry run continues; a real run needs ALLOW_MULTISAMPLE_BCF=1 or a per-sample dir)"
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

    # 1c — chrX must carry ACTUAL VARIANTS, not just a header entry (copied from
    # run_tsstes_dataset.sh). A BCF dir built against a chr1-22 BED still
    # advertises chrX in its header, so the chromosome filter passes it and the
    # chrX regions silently yield nothing. Querying real records is the only way
    # to tell. Only meaningful when $BED has chrX rows (the v10 BED does).
    if [[ "${BED_HAS_CHRX}" -gt 0 ]]; then
        echo "  chrX check .. querying real variants…"
        pyrun - "${BCF_DIR}" "${BED}" <<'PY'
import pathlib, sys
from cyvcf2 import VCF

bcf_dir, bed = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
samples = sorted(bcf_dir.glob("*.bcf"))[:3]
if not samples:
    sys.exit("ERROR: no *.bcf files in %s" % bcf_dir)

# The widest chrX intervals are the likeliest to carry a variant in any one
# sample; 20 of them across 3 samples is far more than enough.
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
        "  header but holds nothing. The chromosome filter would NOT warn — it\n"
        "  reads available chromosomes from the header. Every chrX region would\n"
        "  produce zero rows and inflate the HG38_REF weights.\n"
        "  Rebuild the BCFs from the full panel against the tss-tes union BED."
    )
print("    OK — chrX carries variants")
PY
    else
        echo "  chrX check .. skipped (${BED} has no chrX intervals)"
    fi
fi

# — The generate / finalize-ref commands ——————————————————————————————————————
GEN_ARGS=(
    generate
    --bcf-dir           "${BCF_DIR}"
    --bed               "${BED}"
    --ref               "${REF}"
    --tokens            "${TOKENS}"
    --output            "${OUT_DIR}"
    --workers           "${WORKERS}"
    --compression       "${COMPRESSION}"
    --compression-level "${COMPRESSION_LEVEL}"
    --region-batch-size "${REGION_BATCH_SIZE}"
    --format            "${FORMAT}"
    --tsv-mode          "${TSV_MODE}"
    --clean-stale
    --log-level         ERROR
)
[[ -n "${TEST_SAMPLES}" ]] && GEN_ARGS+=(--test-samples-file "${TEST_SAMPLES}")
[[ -n "${SAMPLES_FILE}" ]] && GEN_ARGS+=(--samples-file "${SAMPLES_FILE}")
[[ -n "${MAX_SAMPLES}"  ]] && GEN_ARGS+=(--max-samples  "${MAX_SAMPLES}")
[[ -n "${MAX_REGIONS}"  ]] && GEN_ARGS+=(--max-regions  "${MAX_REGIONS}")
FIN_ARGS=(
    finalize-ref
    --output            "${OUT_DIR}"
    --bed               "${BED}"
    --ref               "${REF}"
    --tokens            "${TOKENS}"
    --compression       "${COMPRESSION}"
    --compression-level "${COMPRESSION_LEVEL}"
    --log-level         ERROR
)

if [[ "${DRY_RUN}" == "1" ]]; then
    echo ""
    echo "=== DRY RUN — nothing generated. The run would execute, under run_supervised.sh:"
    print_cmd "${PYTHON}" -m preprocess.huggingface_dataset.create_haplotype_dataset "${GEN_ARGS[@]}"
    echo "then, if generate did not write HG38_REF itself:"
    print_cmd "${PYTHON}" -m preprocess.huggingface_dataset.create_haplotype_dataset "${FIN_ARGS[@]}"
    echo "then the smoke verify (alphabet, HG38_REF weights, region_annotations.tsv)."
    echo "=== Finished dry run at $(date '+%F %T') ==="
    exit 0
fi

# — Detach: stages 2-4 under the supervisor ————————————————————————————————————
if [[ "${FOREGROUND}" != "1" ]]; then
    # SUPERVISOR can point at a host-local copy: a script on a shared filesystem is
    # unsafe to replace while a remote host is lazily reading it.
    SUP="${SUPERVISOR:-${RUNNERS_DIR}/run_supervised.sh}"
    [ -x "$SUP" ] || SUP="${RUNNERS_DIR}/run_supervised.sh"
    SD="${LOG_DIR}/generate_$(hostname)_$(date +%Y%m%d_%H%M%S)"
    echo "[$(date '+%F %T')] launching stages 2-4 detached under ${SUP}"
    HEARTBEAT_S=300 STALL_LOG_S=1800 "$SUP" "$SD" -- bash "$SELF" --foreground
    cat <<EOF

=== Supervised run started: ${SD}
  state ....... cat ${SD}/state        (RUNNING / STALLED / DONE / FAILED)
  log ......... tail -f ${SD}/run.log
  heartbeat ... tail ${SD}/heartbeat.tsv
  output ...... ${OUT_DIR}
Re-running this script after a crash resumes (finished shards are skipped).
Log the run in ${HFD}/DECISION_LEDGER.md.
EOF
    exit 0
fi

# — Stage 2 — generate per-sample haplotype-pair shards ————————————————————————
echo "[$(date '+%F %T')] START generate (v10, format=${FORMAT}, ${COMPRESSION}-${COMPRESSION_LEVEL}, workers=${WORKERS})"
echo "  NOTE: the region set OVERLAPS (1.19x) and nothing is deduplicated — rows sum"
echo "        to 2.313 Gbp of reference per patient at most; size the output for that."
GEN_RC=0
pyrun -m preprocess.huggingface_dataset.create_haplotype_dataset "${GEN_ARGS[@]}" || GEN_RC=$?
if [[ "${GEN_RC}" != "0" ]]; then
    echo "[$(date '+%F %T')] generate exited ${GEN_RC} — some samples FAILED / were claimed / are missing." >&2
    echo "  Fix the cause (an unphased heterozygote FAILS its sample by design), re-run to fill" >&2
    echo "  the holes, and finalize-ref will run on the next clean pass." >&2
    exit "${GEN_RC}"
fi
echo "[$(date '+%F %T')] END generate"

# — Stage 3 — finalize-ref (if generate did not) ————————————————————————————————
if [[ "${FORMAT}" == "tsv" ]]; then
    echo "[$(date '+%F %T')] SKIP finalize-ref — FORMAT=tsv writes no Parquet, so there is no HG38_REF"
else
    NEED_REF=0
    if [[ -d "${OUT_DIR}/train" ]]; then
        for split in train test; do
            [[ -d "${OUT_DIR}/${split}" && ! -f "${OUT_DIR}/${split}/HG38_REF.parquet" ]] && NEED_REF=1
        done
    else
        [[ -f "${OUT_DIR}/HG38_REF.parquet" ]] || NEED_REF=1
    fi
    if [[ "${NEED_REF}" == "1" ]]; then
        echo "[$(date '+%F %T')] START finalize-ref (generate left no HG38_REF)"
        pyrun -m preprocess.huggingface_dataset.create_haplotype_dataset "${FIN_ARGS[@]}"
        echo "[$(date '+%F %T')] END finalize-ref"
    else
        echo "[$(date '+%F %T')] finalize-ref already done by generate (HG38_REF.parquet present)"
    fi
fi

# — Stage 4 — smoke verify ———————————————————————————————————————————————————
echo "[$(date '+%F %T')] START verify"
pyrun - "${OUT_DIR}" "${FORMAT}" "${EXPECT_ANNOTATIONS}" <<'PY'
import collections, json, random, sys
from pathlib import Path
import pyarrow.compute as pc
import pyarrow.parquet as pq
from preprocess.huggingface_dataset import create_haplotype_dataset as H

out, fmt, expect_ann = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
ALLOWED = set("ACGTN^{}")
PLAIN = set("ACGTN")
rng = random.Random(0)
problems = []

if expect_ann:
    p = out / H.REGION_ANNOTATIONS_FILENAME
    if p.is_file():
        print(f"    {p.name}: OK ({sum(1 for _ in open(p)) - 1:,} gene rows)")
    else:
        problems.append(f"{p} missing (the BED has a genes.tsv sidecar)")

for d in H._split_dirs(out):
    samples = H.read_split_samples(d)
    if fmt == "tsv":
        tsvs = sorted(d.glob("*.tsv.gz"))
        print(f"  {d.name}: {len(tsvs)} TSV shard(s) for {len(samples)} samples (no Parquet, no HG38_REF by design)")
        missing = [s for s in samples if not (d / f"{s}.tsv.gz").exists()]
        if missing:
            problems.append(f"{d.name}: {len(missing)} TSV shard(s) missing")
        continue
    ds = H.HaplotypePairDataset(d)            # validates sidecar + every footer's v10 metadata
    patient = [f for f in ds.files if f.stem != H._REF_SAMPLE_ID]
    ref = d / f"{H._REF_SAMPLE_ID}.parquet"
    print(f"  {d.name}: {len(patient)} patient shard(s), {len(samples)} samples listed, "
          f"{len(ds)} rows, region columns {ds.region_columns}")
    if not ref.is_file():
        problems.append(f"{d.name}: {ref.name} missing")
        continue
    if not patient:
        problems.append(f"{d.name}: no patient shards")
        continue
    missing = [s for s in samples if not (d / f"{s}.parquet").exists()]
    if missing:
        problems.append(f"{d.name}: {len(missing)} listed sample(s) without a shard: {missing[:5]}")

    # integrity counters: silent resolutions the encoder made, from every shard's metadata
    integrity = collections.Counter()
    for f in patient:
        integrity.update(H.read_v10_metadata(f).get("integrity", {}))
    print("    integrity over all shards: "
          + ", ".join(f"{k}={integrity[k]}" for k in H.INTEGRITY_COUNTER_KEYS))

    # (a) one patient shard: alphabet, weights, het ⇒ haps differ
    shard = rng.choice(patient)
    n = bad_alpha = bad_w = bad_het = 0
    for row in H.HaplotypePairDataset(shard):
        n += 1
        if not (set(row["hap0"]) <= ALLOWED and set(row["hap1"]) <= ALLOWED):
            bad_alpha += 1
        if row["sample_weight"] != 1:
            bad_w += 1
        if row["n_het"] > 0 and row["hap0"] == row["hap1"]:
            bad_het += 1
    print(f"    {shard.name}: {n:,} rows; alphabet violations {bad_alpha}, "
          f"weight!=1 {bad_w}, n_het>0 with hap0==hap1 {bad_het}")
    if bad_alpha or bad_w or bad_het or n == 0:
        problems.append(f"{d.name}/{shard.name}: alphabet={bad_alpha} weight={bad_w} het={bad_het} rows={n}")

    # (b) HG38_REF: plain reference, no markers, weights within range
    reft = pq.read_table(ref, columns=["chrom", "start", "end", "hap0", "hap1",
                                       "sample_weight", "n_variants"])
    rbad = 0
    for h0, h1 in zip(reft["hap0"].to_pylist(), reft["hap1"].to_pylist()):
        if h0 != h1 or not set(h0) <= PLAIN:
            rbad += 1
    w = reft["sample_weight"].to_pylist()
    if rbad or any(x < 0 or x > len(samples) for x in w) or any(reft["n_variants"].to_pylist()):
        problems.append(f"{d.name}/{ref.name}: marked/unequal rows={rbad}, weights out of range")
    print(f"    {ref.name}: {reft.num_rows:,} regions, weights {min(w)}..{max(w)} over {len(samples)} samples")

    # (c) sum(sample_weight) == n_samples on 100 random regions
    keys = list(zip(reft["chrom"].to_pylist(), reft["start"].to_pylist(), reft["end"].to_pylist()))
    refw = dict(zip(keys, w))
    pick = set(rng.sample(keys, min(100, len(keys))))
    counts = collections.Counter()
    for f in patient:
        t = pq.read_table(f, columns=["chrom", "start", "end"])
        for k in zip(t["chrom"].to_pylist(), t["start"].to_pylist(), t["end"].to_pylist()):
            if k in pick:
                counts[k] += 1
    off = [(k, counts[k], refw[k]) for k in pick if counts[k] + refw[k] != len(samples)]
    print(f"    weights: {len(pick)} random regions, sum(sample_weight) == {len(samples)} on "
          f"{len(pick) - len(off)}/{len(pick)}")
    if off:
        problems.append(f"{d.name}: weight mismatch on {len(off)} region(s), e.g. {off[:3]}")

if problems:
    print("VERIFY FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
    sys.exit(1)
print("    verify OK")
PY
echo "[$(date '+%F %T')] END verify"

# — Summary ——————————————————————————————————————————————————————————————————
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=== SUCCESS: v10 haplotype-pair dataset at ${OUT_DIR} ==="
echo "Size:        $(du -sh "${OUT_DIR}" 2>/dev/null | cut -f1)"
echo "Region set:  ${BED_REGION_SET:-<unlabelled>} (${BED_ROWS} intervals from ${BED})"
echo "Format:      ${FORMAT} (${COMPRESSION}-${COMPRESSION_LEVEL}); no dedup / shuffle / assemble by design"
echo "=== Finished at $(date '+%F %T') (total $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s) ==="
echo ""
echo "Log this run in ${HFD}/DECISION_LEDGER.md."
