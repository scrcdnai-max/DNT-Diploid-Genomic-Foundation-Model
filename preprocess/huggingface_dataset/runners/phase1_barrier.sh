#!/usr/bin/env bash
set -uo pipefail

# Stop a running make_persample_bcf.sh container between Phase 1 and Phase 2,
# then certify its chromosomes with `.done` sentinels.
#
# WHY. make_persample_bcf.sh runs Phase 1 and Phase 2 in one process, and its
# Phase 1 resume test is "output dir exists and is non-empty". `bcftools +split`
# creates all N_SAMPLES output files at startup, so every chromosome another host
# is currently splitting already looks finished. The moment this container's own
# chromosomes are done, its loop will skip all of them and start concatenating
# work that is still being written — silently producing short per-sample BCFs.
#
# The script cannot be fixed in place while it runs: bash reads a script lazily
# through the same fd, so editing it mid-run makes the live shell execute
# garbage. So the barrier is interposed from outside instead. Stopping between
# the two phases costs nothing — Phase 1's output is complete, closed files, and
# concat_persample_bcf.sh does Phase 2 properly, gated on every part's sentinel.
#
# Usage:
#   phase1_barrier.sh CONTAINER TMP_DIR SAMPLES_FILE CHROM [CHROM ...]
# Env:
#   POLL      seconds between checks (default: 5)
#   GRACE     seconds to let the rename pass finish after the last bcftools
#             exits, before stopping the container (default: 45)
#   VERIFY_N  files per chromosome to structurally check (default: 40)

CONTAINER="${1:?usage: phase1_barrier.sh CONTAINER TMP_DIR SAMPLES_FILE CHROM...}"
TMP_DIR="${2:?missing TMP_DIR}"
SAMPLES_FILE="${3:?missing SAMPLES_FILE}"
shift 3
[[ $# -ge 1 ]] || { echo "ERROR: name at least one CHROM this container owns" >&2; exit 1; }
CHROMS=("$@")

POLL="${POLL:-5}"
GRACE="${GRACE:-45}"
VERIFY_N="${VERIFY_N:-40}"
N_SAMPLES=$(grep -c '[^[:space:]]' "$SAMPLES_FILE")

echo "[$(date '+%F %T')] START phase1_barrier"
echo "  container=$CONTAINER  chroms=${CHROMS[*]}  samples=$N_SAMPLES"
echo "  will stop the container as soon as its Phase 1 ends, before Phase 2 concat"

# `docker top` needs an explicit pid column: with `-eo args` alone the daemon
# rejects the format ("Couldn't find PID field in ps output") and prints nothing,
# which a naive counter would read as "no bcftools left" and stop the container
# on the first poll. TOP_OUT therefore fails loudly instead of returning empty.
TOP_OUT=""
top_refresh() {
    TOP_OUT=$(docker top "$CONTAINER" -eo pid,args 2>/dev/null)
    # A live container always shows its own entrypoint; anything less means the
    # query failed rather than the work finishing.
    grep -q 'make_persample_bcf\.sh' <<< "$TOP_OUT"
}
count_in_top() { grep -c -- "$1" <<< "$TOP_OUT"; }
alive() { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" == "true" ]]; }

alive || { echo "ERROR: container $CONTAINER is not running" >&2; exit 1; }
top_refresh || { echo "ERROR: 'docker top $CONTAINER -eo pid,args' did not list make_persample_bcf.sh — refusing to arm a barrier that cannot see the work" >&2; exit 1; }
echo "  detection OK: $(count_in_top 'bcftools +split') +split process(es) visible"

# ── Wait for Phase 1 to end ───────────────────────────────────────────────
last_report=0
consecutive_blind=0
while alive; do
    if ! top_refresh; then
        # Transient docker hiccup, or the container is on its way out. Never
        # treat an unreadable process list as "Phase 1 finished".
        consecutive_blind=$((consecutive_blind + 1))
        if (( consecutive_blind >= 12 )); then
            echo "[$(date '+%F %T')] cannot read the container's process list ($consecutive_blind tries) — giving up rather than guessing" >&2
            exit 1
        fi
        sleep "$POLL"
        continue
    fi
    consecutive_blind=0
    n_split=$(count_in_top 'bcftools +split')
    n_concat=$(count_in_top 'bcftools concat')
    if (( n_concat > 0 )); then
        echo "[$(date '+%F %T')] Phase 2 already started — stopping NOW" >&2
        docker stop -t 5 "$CONTAINER" >/dev/null
        break
    fi
    if (( n_split == 0 )); then
        echo "[$(date '+%F %T')] Phase 1 done (no +split left); ${GRACE}s grace for the rename pass"
        for ((i = 0; i < GRACE; i += POLL)); do
            sleep "$POLL"
            if top_refresh && (( $(count_in_top 'bcftools concat') > 0 )); then
                echo "  concat appeared during grace — stopping immediately"
                break
            fi
        done
        echo "[$(date '+%F %T')] stopping $CONTAINER"
        docker stop -t 10 "$CONTAINER" >/dev/null
        break
    fi
    now=$(date +%s)
    if (( now - last_report >= 1800 )); then
        echo "[$(date '+%F %T')] still splitting: $n_split chromosome(s) in flight"
        last_report=$now
    fi
    sleep "$POLL"
done
echo "[$(date '+%F %T')] container stopped (exit $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER" 2>/dev/null))"

# Anything Phase 2 managed to emit before the stop is untrustworthy: it was
# concatenated against chromosomes the other host had not finished. Report it
# rather than deleting — nothing here is expected to exist, so if it does, that
# is a fact worth a human look before the real Phase 2 runs.
OUT_DIR=$(dirname "$TMP_DIR")
mapfile -t STRAY < <(find "$OUT_DIR" -maxdepth 1 \( -name '*.bcf' -o -name '*.bcf.csi' -o -name '*.bcf.tmp*' \) 2>/dev/null)
if (( ${#STRAY[@]} > 0 )); then
    echo "[$(date '+%F %T')] WARNING: ${#STRAY[@]} premature Phase 2 output file(s) in $OUT_DIR" >&2
    printf '    %s\n' "${STRAY[@]:0:5}" >&2
    echo "    These were concatenated before the remote chunks finished and MUST be removed" >&2
    echo "    before running concat_persample_bcf.sh, which skips samples that already exist." >&2
fi

# ── Certify this container's chromosomes ──────────────────────────────────
BGZF_EOF='1f8b08040000000000ff0600424302001b0003000000000000000000'
fail=0
for c in "${CHROMS[@]}"; do
    d="${TMP_DIR}/${c}"
    [[ -d "$d" ]] || { echo "  $c: MISSING $d" >&2; fail=1; continue; }
    # The rename pass may have been cut short; finish it here. A no-op on
    # bcftools builds that already write <sample>.bcf.
    for f in "$d"/*; do
        [[ -f "$f" ]] || continue
        b=$(basename "$f")
        [[ "$b" == *bcf && "$b" != *.bcf ]] && mv "$f" "${f%bcf}.bcf"
    done
    n=$(find "$d" -maxdepth 1 -name '*.bcf' | wc -l)
    if [[ "$n" -ne "$N_SAMPLES" ]]; then
        echo "  $c: $n *.bcf, expected $N_SAMPLES — NOT certifying" >&2; fail=1; continue
    fi
    bad=0
    while read -r f; do
        [[ "$(tail -c 28 "$f" | od -An -v -tx1 | tr -d ' \n')" == "$BGZF_EOF" ]] || { bad=$((bad+1)); echo "    truncated: $f" >&2; }
    done < <(find "$d" -maxdepth 1 -name '*.bcf' | head -"$VERIFY_N")
    if (( bad )); then echo "  $c: $bad truncated file(s) — NOT certifying" >&2; fail=1; continue; fi
    printf 'chunk=%s\nfiles=%s\nsource=%s\nfinished=%s\n' \
        "$c" "$n" "make_persample_bcf.sh in $CONTAINER" "$(date '+%F %T%z')" > "${d}.done"
    echo "  certified $c ($n files)"
done

if (( fail )); then
    echo "[$(date '+%F %T')] BARRIER INCOMPLETE — some chromosomes were not certified." >&2
    exit 1
fi
echo "[$(date '+%F %T')] BARRIER OK — ${#CHROMS[@]} chromosome(s) certified; Phase 2 is now safe once the remote chunks finish."
