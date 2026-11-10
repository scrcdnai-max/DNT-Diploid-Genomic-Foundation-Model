#!/usr/bin/env bash
set -uo pipefail

# Distributed Phase 1: run one `bcftools +split` per (chromosome, chunk) BED
# produced by tools/chunk_union_bed.py, into one output directory per chunk.
#
# This exists because `bcftools +split` is single-threaded and has no --threads
# (checked: its only knobs are -e/-G/-i/-k/-o/-O/-r/-R/--regions-overlap/-S), so
# a whole chromosome pins exactly one core at ~0.07 MB/s of compressed BCF. With
# 18 chromosomes that leaves ~90 of this host's 112 cores idle and puts chr2
# ~113 h away. One process per chunk uses them.
#
# TWO THINGS THIS DOES THAT make_persample_bcf.sh's Phase 1 DOES NOT:
#
# 1. **A chunk is done only when a `.done` sentinel says so.** The upstream
#    resume test is "output dir exists and is non-empty", which is wrong in both
#    directions: `bcftools +split` creates all N_SAMPLES output files at startup,
#    so an in-flight split and a split that died at 60% both look identical to a
#    finished one, and are then skipped forever. The sentinel is written only
#    after bcftools exits 0 AND the file count matches. A chunk without one is
#    wiped and redone.
# 2. **It writes nothing to shared storage.** Output goes to STAGE (local disk),
#    so N_SAMPLES x WORKERS open files never become NFSv4 open state on the
#    filer. concat_persample_bcf.sh reads STAGE and writes the final per-sample
#    BCFs to the shared tree.
#
# The bcftools invocation is otherwise byte-identical to split_one() in
# make_persample_bcf.sh — same -R semantics (default --regions-overlap 1), same
# -i 'GT="alt"', same -k FMT/GT, same -Ob, same -S, same <sample>bcf rename
# quirk — so a chunk's output is exactly the slice of that chromosome's
# single-machine output covering the chunk's regions. See tools/chunk_union_bed.py for
# why the cut points cannot duplicate a record across two chunks.
#
# Usage (inside the glm_biotools image pinned in docker_images/glm_biotools/image.env,
# which is where a bcftools >= 1.13 lives):
#   run_chunked_split.sh RAW_DIR CHUNK_DIR STAGE SAMPLES_FILE
# Env:
#   WORKERS  concurrent bcftools processes (default: 48)

START=$(date +%s)
echo "[$(date '+%F %T')] START run_chunked_split"
echo "Command: $0 $*"

RAW_DIR="${1:?usage: run_chunked_split.sh RAW_DIR CHUNK_DIR STAGE SAMPLES_FILE}"
CHUNK_DIR="${2:?missing CHUNK_DIR (output of tools/chunk_union_bed.py)}"
STAGE="${3:?missing STAGE (local scratch root)}"
SAMPLES_FILE="${4:?missing SAMPLES_FILE}"
WORKERS="${WORKERS:-48}"

command -v bcftools >/dev/null || { echo "ERROR: bcftools not on PATH" >&2; exit 1; }
[ -d "$RAW_DIR"   ] || { echo "ERROR: no RAW_DIR $RAW_DIR" >&2; exit 1; }
[ -d "$CHUNK_DIR" ] || { echo "ERROR: no CHUNK_DIR $CHUNK_DIR" >&2; exit 1; }
[ -f "$SAMPLES_FILE" ] || { echo "ERROR: no SAMPLES_FILE $SAMPLES_FILE" >&2; exit 1; }

N_SAMPLES=$(grep -c '[^[:space:]]' "$SAMPLES_FILE")
# +split opens one fd per sample per process; the fd limit is per-process, so
# this only has to cover a single split, not all WORKERS of them.
ulimit -n 65536 2>/dev/null || true
echo "bcftools: $(bcftools --version | head -1)"
echo "RAW_DIR=$RAW_DIR"
echo "CHUNK_DIR=$CHUNK_DIR  STAGE=$STAGE"
echo "SAMPLES_FILE=$SAMPLES_FILE ($N_SAMPLES samples)  WORKERS=$WORKERS  ulimit -n=$(ulimit -n)"

mkdir -p "$STAGE" "$STAGE/logs"

# Resolve a chunk name (chr3.002) to its panel BCF via the chromosome token.
raw_for() {
    local chrom="${1%%.*}"
    local hits=()
    local f
    for f in "$RAW_DIR"/*.bcf; do
        [[ -f "$f" ]] || continue
        [[ "$(basename "$f")" =~ (^|[^0-9A-Za-z])${chrom}([^0-9]|$) ]] && hits+=("$f")
    done
    [[ ${#hits[@]} -eq 1 ]] || { echo "ERROR: ${#hits[@]} panel BCFs match $chrom" >&2; return 1; }
    printf '%s\n' "${hits[0]}"
}

split_chunk() {
    local name="$1"
    local bed="${CHUNK_DIR}/${name}.bed"
    local outdir="${STAGE}/${name}"
    local done_marker="${outdir}.done"
    local log="${STAGE}/logs/${name}.log"

    if [[ -f "$done_marker" ]]; then
        echo "  SKIP (done): $name"; return 0
    fi
    # No sentinel means unknown state — in flight when it died, or never
    # started. Either way the partial output cannot be trusted, so start clean.
    if [[ -d "$outdir" ]]; then
        echo "  REDO (no sentinel, wiping partial): $name"
        rm -rf "$outdir"
    fi
    mkdir -p "$outdir"

    local bcf
    bcf=$(raw_for "$name") || return 1

    echo "  [$(date '+%T')] START $name ($(wc -l < "$bed") regions)"
    if ! bcftools +split "$bcf" -R "$bed" -i 'GT="alt"' -k FMT/GT \
            -S "$SAMPLES_FILE" -Ob -o "$outdir" >"$log" 2>&1; then
        echo "  [$(date '+%T')] FAIL $name (bcftools exit $?) — see $log" >&2
        return 1
    fi

    # bcftools +split 1.23 with -Ob -S can emit <sample>bcf (no dot); normalise
    # so the concat step can enumerate them. A no-op on builds that do not.
    local f base
    for f in "$outdir"/*; do
        [[ -f "$f" ]] || continue
        base=$(basename "$f")
        if [[ "$base" == *bcf && "$base" != *.bcf ]]; then
            mv "$f" "${f%bcf}.bcf"
        fi
    done

    local n
    n=$(find "$outdir" -maxdepth 1 -name '*.bcf' | wc -l)
    if [[ "$n" -ne "$N_SAMPLES" ]]; then
        echo "  [$(date '+%T')] FAIL $name — $n output files, expected $N_SAMPLES" >&2
        return 1
    fi

    printf 'chunk=%s\nbed=%s\nregions=%s\nfiles=%s\nbcftools=%s\nfinished=%s\n' \
        "$name" "$bed" "$(wc -l < "$bed")" "$n" \
        "$(bcftools --version | head -1)" "$(date '+%F %T%z')" > "$done_marker"
    echo "  [$(date '+%T')] DONE $name ($n files)"
}

mapfile -t CHUNKS < <(cd "$CHUNK_DIR" && ls -1 *.bed 2>/dev/null | sed 's/\.bed$//' | sort)
[[ ${#CHUNKS[@]} -gt 0 ]] || { echo "ERROR: no *.bed in $CHUNK_DIR" >&2; exit 1; }
echo "[$(date '+%F %T')] ${#CHUNKS[@]} chunk(s), $WORKERS at a time"

# Heaviest chunks first: with a fixed pool the makespan is set by whichever
# long chunk starts last, so never leave one for the tail.
mapfile -t ORDER < <(
    for c in "${CHUNKS[@]}"; do
        echo "$(stat -c %s "$(raw_for "$c")" 2>/dev/null || echo 0) $(wc -l < "${CHUNK_DIR}/${c}.bed") $c"
    done | sort -k1,1nr -k2,2nr | awk '{print $3}'
)

FAILED=0
running=0
declare -A PIDNAME=()
for c in "${ORDER[@]}"; do
    split_chunk "$c" &
    PIDNAME[$!]="$c"
    running=$((running + 1))
    if (( running >= WORKERS )); then
        # Unlike a bare `wait`, this surfaces a non-zero child status.
        if ! wait -n; then FAILED=$((FAILED + 1)); fi
        running=$((running - 1))
    fi
done
while (( running > 0 )); do
    if ! wait -n; then FAILED=$((FAILED + 1)); fi
    running=$((running - 1))
done

N_DONE=$(find "$STAGE" -maxdepth 1 -name '*.done' | wc -l)
END=$(date +%s)
echo "[$(date '+%F %T')] ${N_DONE}/${#CHUNKS[@]} chunk(s) complete, ${FAILED} failure(s)"
echo "[$(date '+%F %T')] Total time: $(( (END-START)/3600 ))h $(( ((END-START)%3600)/60 ))m"
if (( FAILED > 0 || N_DONE != ${#CHUNKS[@]} )); then
    echo "INCOMPLETE — re-run this script to redo the chunks without a .done sentinel." >&2
    exit 1
fi
echo "ALL CHUNKS COMPLETE"
