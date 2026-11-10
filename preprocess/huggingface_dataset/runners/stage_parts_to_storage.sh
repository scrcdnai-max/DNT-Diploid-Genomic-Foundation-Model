#!/usr/bin/env bash
set -uo pipefail

# Move a host-local `run_chunked_split.sh` staging root onto shared storage so
# `concat_persample_bcf.sh` can see every chromosome's parts from one host.
#
#   stage_parts_to_storage.sh SRC_ROOT DST_ROOT [PARALLEL]
#
# SRC_ROOT holds `<name>/` part directories each with a sibling `<name>.done`
# sentinel, exactly as run_chunked_split.sh leaves them.
#
# THE INVARIANT: the sentinel is copied LAST, and only after the destination's
# file count matches the source's. concat_persample_bcf.sh trusts sentinels, so
# a part directory that is present-but-incomplete must never carry one -- that
# is precisely the "non-empty dir means done" failure that produced short
# per-sample BCFs on 2026-09-01.
# Resume follows the same rule: a destination with a sentinel AND a matching
# count is skipped; anything else is re-synced.
#
# Parts are ~11,586 files of ~350 KB each, so this is metadata-bound rather than
# bandwidth-bound. Parallelism is over whole part directories.

SRC="${1:?usage: stage_parts_to_storage.sh SRC_ROOT DST_ROOT [PARALLEL]}"
DST="${2:?missing DST_ROOT}"
PAR="${3:-8}"

[ -d "$SRC" ] || { echo "ERROR: no SRC_ROOT $SRC" >&2; exit 1; }
mkdir -p "$DST" || exit 1

echo "[$(date '+%F %T')] START stage_parts_to_storage"
echo "SRC=$SRC  DST=$DST  PARALLEL=$PAR"

mapfile -t PARTS < <(find "$SRC" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
echo "found ${#PARTS[@]} part directory(ies)"

stage_one() {
    local name="$1"
    local s="$SRC/$name" d="$DST/$name"
    if [ ! -f "$SRC/$name.done" ]; then
        echo "  [$(date '+%T')] SKIP $name — no source sentinel (uncertified)"
        return 0
    fi
    local n_src; n_src=$(find "$s" -type f | wc -l)
    if [ -f "$DST/$name.done" ]; then
        local n_dst; n_dst=$(find "$d" -type f 2>/dev/null | wc -l)
        if [ "$n_dst" -eq "$n_src" ]; then
            echo "  [$(date '+%T')] SKIP $name — already staged ($n_dst files)"
            return 0
        fi
        echo "  [$(date '+%T')] REDO $name — sentinel present but $n_dst != $n_src"
        rm -f "$DST/$name.done"
    fi
    echo "  [$(date '+%T')] START $name ($n_src files)"
    if ! rsync -a --no-compress "$s/" "$d/"; then
        echo "  [$(date '+%T')] FAIL $name — rsync exit $?" >&2
        return 1
    fi
    local n_dst; n_dst=$(find "$d" -type f | wc -l)
    if [ "$n_dst" -ne "$n_src" ]; then
        echo "  [$(date '+%T')] FAIL $name — copied $n_dst of $n_src files" >&2
        return 1
    fi
    # Sentinel last, and only now.
    cp -p "$SRC/$name.done" "$DST/$name.done" || return 1
    echo "  [$(date '+%T')] DONE $name ($n_dst files)"
}
export -f stage_one
export SRC DST

printf '%s\n' "${PARTS[@]}" | xargs -P "$PAR" -I{} bash -c 'stage_one "$@"' _ {}
rc=$?

n_done=$(find "$DST" -maxdepth 1 -name '*.done' | wc -l)
n_src_done=$(find "$SRC" -maxdepth 1 -name '*.done' | wc -l)
echo "[$(date '+%F %T')] staged $n_done/$n_src_done certified part(s)"
[ "$n_done" -eq "$n_src_done" ] && [ "$rc" -eq 0 ] || { echo "INCOMPLETE" >&2; exit 1; }
echo "ALL PARTS STAGED"
