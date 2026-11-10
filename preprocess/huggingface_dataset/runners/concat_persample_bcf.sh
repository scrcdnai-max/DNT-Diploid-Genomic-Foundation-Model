#!/usr/bin/env bash
set -uo pipefail

# Phase 2 for a Phase 1 that ran across two hosts: verify every part is complete,
# then concatenate each sample's parts into the single multi-contig
# OUT_DIR/<sample>.bcf that the diploid_glm encoder reads.
#
# WHY THIS REPLACES make_persample_bcf.sh's Phase 2. That one starts concatenating
# the moment its OWN Phase 1 loop ends, and its loop treats any non-empty output
# directory as a finished chromosome. `bcftools +split` creates all N_SAMPLES
# output files at startup, so a chromosome another host is still writing looks
# finished, gets skipped, and is concatenated mid-write — producing short
# per-sample BCFs with no error anywhere. This script has no such loop: it
# concatenates nothing until every expected part carries a `.done` sentinel, and
# it exits non-zero if any is missing.
#
# The output is identical to a single-machine run because the part order is
# derived from the data, not from which host produced it: karyotype order across
# chromosomes (chr1..chr22, chrX), then chunk index within a chromosome, which is
# genomic order because tools/chunk_union_bed.py cuts each chromosome into contiguous
# coordinate ranges.
#
# Usage (inside the glm_biotools image pinned in docker_images/glm_biotools/image.env):
#   concat_persample_bcf.sh OUT_DIR SAMPLES_FILE PART_ROOT [PART_ROOT ...]
# Env:
#   JOBS          concurrent concats (default: 32)
#   EXPECT_CHROMS space/comma list the parts must cover exactly
#                 (default: chr1..chr22 chrX)
#   VERIFY_N      files per part to structurally check (default: 40, 0 disables)

START=$(date +%s)
echo "[$(date '+%F %T')] START concat_persample_bcf"
echo "Command: $0 $*"

OUT_DIR="${1:?usage: concat_persample_bcf.sh OUT_DIR SAMPLES_FILE PART_ROOT...}"
SAMPLES_FILE="${2:?missing SAMPLES_FILE}"
shift 2
[[ $# -ge 1 ]] || { echo "ERROR: need at least one PART_ROOT" >&2; exit 1; }
PART_ROOTS=("$@")

JOBS="${JOBS:-32}"
VERIFY_N="${VERIFY_N:-40}"
DEFAULT_CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX"
# Accept either separator; read -a splits on whitespace only, so a comma-
# separated EXPECT_CHROMS would otherwise arrive as one unusable element.
EXPECT_CHROMS="${EXPECT_CHROMS:-}"
read -r -a EXPECT <<< "${EXPECT_CHROMS//,/ }"
[[ ${#EXPECT[@]} -gt 0 ]] || read -r -a EXPECT <<< "$DEFAULT_CHROMS"

command -v bcftools >/dev/null || { echo "ERROR: bcftools not on PATH" >&2; exit 1; }
[ -f "$SAMPLES_FILE" ] || { echo "ERROR: no SAMPLES_FILE $SAMPLES_FILE" >&2; exit 1; }
N_SAMPLES=$(grep -c '[^[:space:]]' "$SAMPLES_FILE")
ulimit -n 65536 2>/dev/null || true
echo "bcftools: $(bcftools --version | head -1)"
echo "OUT_DIR=$OUT_DIR  samples=$N_SAMPLES  JOBS=$JOBS  VERIFY_N=$VERIFY_N"
echo "PART_ROOTS: ${PART_ROOTS[*]}"

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

# ── Gather parts ──────────────────────────────────────────────────────────
# A part is a directory <chrom> or <chrom>.<idx> with a sibling <name>.done.
declare -a PART_DIRS=() PART_KEYS=()
declare -A SEEN=()
for root in "${PART_ROOTS[@]}"; do
    [ -d "$root" ] || { echo "ERROR: no PART_ROOT $root" >&2; exit 1; }
    for d in "$root"/chr*/; do
        [[ -d "$d" ]] || continue
        d="${d%/}"
        name=$(basename "$d")
        [[ -f "${d}.done" ]] || { echo "  no sentinel, ignoring: $d"; continue; }
        chrom="${name%%.*}"
        idx="000"
        [[ "$name" == *.* ]] && idx="${name#*.}"
        if [[ -n "${SEEN[$name]:-}" ]]; then
            echo "ERROR: part '$name' appears in two roots: ${SEEN[$name]} and $d" >&2
            exit 1
        fi
        SEEN[$name]="$d"
        PART_DIRS+=("$d")
        PART_KEYS+=("$(chrom_sort_key "$chrom") $idx $d")
    done
done
[[ ${#PART_DIRS[@]} -gt 0 ]] || { echo "ERROR: no completed parts found" >&2; exit 1; }

mapfile -t ORDERED < <(printf '%s\n' "${PART_KEYS[@]}" | sort -k1,1 -k2,2 | awk '{print $3}')

# ── Verify coverage ───────────────────────────────────────────────────────
echo "[$(date '+%F %T')] verifying ${#ORDERED[@]} part(s)"
declare -A GOT=()
for d in "${ORDERED[@]}"; do
    name=$(basename "$d")
    GOT["${name%%.*}"]+=" ${name#*.}"
done

fail=0
for c in "${EXPECT[@]}"; do
    [[ -z "$c" ]] && continue
    if [[ -z "${GOT[$c]:-}" ]]; then
        echo "  MISSING chromosome: $c" >&2; fail=1; continue
    fi
    # Chunk indices must run 0..K-1 with no hole, or the concat silently drops a
    # coordinate range. An unchunked part is named <chrom> and keyed 000.
    read -r -a idxs <<< "$(tr ' ' '\n' <<< "${GOT[$c]}" | grep -v '^$' | sort -u | tr '\n' ' ')"
    expect_i=0
    for i in "${idxs[@]}"; do
        if [[ "$i" == "$c" ]]; then i="000"; fi   # unchunked: <chrom> has no dot
        if (( 10#$i != expect_i )); then
            echo "  $c: chunk index gap — have [${idxs[*]}], expected $(printf '%03d' $expect_i)" >&2
            fail=1
        fi
        expect_i=$((expect_i + 1))
    done
done
for c in "${!GOT[@]}"; do
    printf '%s\n' "${EXPECT[@]}" | grep -qx "$c" || { echo "  UNEXPECTED chromosome: $c" >&2; fail=1; }
done
(( fail )) && { echo "ERROR: part coverage is incomplete — refusing to concat." >&2; exit 1; }

# ── Verify each part's contents ───────────────────────────────────────────
# The BGZF end-of-file block. A file lacking it was truncated: the writer died
# or the reader raced it. This is the check that would have caught a concat
# fired against a chromosome another host was still writing.
BGZF_EOF='1f8b08040000000000ff0600424302001b0003000000000000000000'
for d in "${ORDERED[@]}"; do
    n=$(find "$d" -maxdepth 1 -name '*.bcf' | wc -l)
    if [[ "$n" -ne "$N_SAMPLES" ]]; then
        echo "  $d: $n *.bcf, expected $N_SAMPLES" >&2; fail=1; continue
    fi
    if (( VERIFY_N > 0 )); then
        bad=0
        while read -r f; do
            [[ "$(tail -c 28 "$f" | od -An -v -tx1 | tr -d ' \n')" == "$BGZF_EOF" ]] || { bad=$((bad+1)); echo "    truncated: $f" >&2; }
        done < <(find "$d" -maxdepth 1 -name '*.bcf' | head -"$VERIFY_N")
        if (( bad )); then echo "  $d: $bad truncated file(s)" >&2; fail=1; continue; fi
    fi
    echo "  OK $(basename "$d") ($n files)"
done
(( fail )) && { echo "ERROR: part verification failed — refusing to concat." >&2; exit 1; }
echo "[$(date '+%F %T')] all parts verified"

# ── Concat ────────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"
mapfile -t SAMPLES < <(grep '[^[:space:]]' "$SAMPLES_FILE" | tr -d '\r')
echo "[$(date '+%F %T')] concat ${#SAMPLES[@]} sample(s) x ${#ORDERED[@]} parts (JOBS=$JOBS)"

concat_one() {
    local s="$1"
    local out="${OUT_DIR}/${s}.bcf"
    [[ -f "$out" && -f "${out}.csi" ]] && { return 0; }
    local parts=() d
    for d in "${ORDERED[@]}"; do
        [[ -f "${d}/${s}.bcf" ]] && parts+=("${d}/${s}.bcf")
    done
    if [[ ${#parts[@]} -ne ${#ORDERED[@]} ]]; then
        echo "  FAIL $s — ${#parts[@]} of ${#ORDERED[@]} parts present" >&2; return 1
    fi
    bcftools concat "${parts[@]}" -Ob -o "${out}.tmp" 2>/dev/null || {
        echo "  FAIL $s — bcftools concat" >&2; rm -f "${out}.tmp"; return 1; }
    bcftools index --csi "${out}.tmp" 2>/dev/null || {
        echo "  FAIL $s — bcftools index" >&2; rm -f "${out}.tmp" "${out}.tmp.csi"; return 1; }
    mv "${out}.tmp.csi" "${out}.csi"
    mv "${out}.tmp" "$out"
}

FAILED=0
running=0
done_n=0
for s in "${SAMPLES[@]}"; do
    concat_one "$s" &
    running=$((running + 1))
    if (( running >= JOBS )); then
        if ! wait -n; then FAILED=$((FAILED + 1)); fi
        running=$((running - 1)); done_n=$((done_n + 1))
        (( done_n % 500 == 0 )) && echo "  [$(date '+%T')] $done_n/${#SAMPLES[@]} concatenated"
    fi
done
while (( running > 0 )); do
    if ! wait -n; then FAILED=$((FAILED + 1)); fi
    running=$((running - 1))
done

N_OUT=$(find "$OUT_DIR" -maxdepth 1 -name '*.bcf' | wc -l)
END=$(date +%s)
echo "[$(date '+%F %T')] ${N_OUT} per-sample BCF(s) in ${OUT_DIR}, ${FAILED} failure(s)"
echo "[$(date '+%F %T')] Total time: $(( (END-START)/3600 ))h $(( ((END-START)%3600)/60 ))m"
if (( FAILED > 0 || N_OUT != N_SAMPLES )); then
    echo "INCOMPLETE — re-run to fill the gaps (finished samples are skipped)." >&2
    exit 1
fi
echo "DONE — ${N_OUT} samples, concatenated in order: $(basename "${ORDERED[0]}") .. $(basename "${ORDERED[-1]}")"
