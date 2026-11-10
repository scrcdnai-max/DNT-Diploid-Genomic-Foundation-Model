#!/usr/bin/env bash
set -uo pipefail

# Detached supervisor for this repo's multi-hour stages (generate / assemble /
# concat). Runs a command, and leaves behind a status directory from which ANY
# later reader -- another session, another host, a human a day later -- can
# reconstruct what happened.
#
#   run_supervised.sh STATUS_DIR -- CMD [ARGS...]
#
# Env:
#   HEARTBEAT_S    seconds between heartbeats            (default 300)
#   STALL_LOG_S    log-silence before STALLED is possible (default 1800;
#                  assemble wants 21600 -- its Step 4 is silent for ~4h)
#   CPU_EPSILON    group CPU %% below which we call it idle (default 1.0)
#   NOTIFY_CMD     invoked as: $NOTIFY_CMD <STATE> <STATUS_DIR>   (optional)
#   FOREGROUND=1   do not re-exec detached (used by the self-test)
#
# THE CONTRACT THIS IMPLEMENTS IS preprocess/huggingface_dataset/runners/MONITOR.md.
# Read it before changing anything here; if the two disagree, MONITOR.md wins.
#
# Design points that are load-bearing (all from MONITOR.md):
#  * outcome is DURABLE ON DISK, never only in a watcher's memory (the
#    2026-09-03 assemble crash sat undetected 6.5h because the watcher died);
#  * STALLED needs TWO signals -- log silence AND ~zero group CPU delta -- on
#    two consecutive samples (the 2026-09-02 generate deadlock was alive with a
#    quiet log; a busy Step 4 is quiet but burning CPU);
#  * a heartbeat touches ONLY local metadata + <=64KB of log tail. It must never
#    stat the dataset on /storage: that is the resource the run is bottlenecked
#    on. See MONITOR.md section 5.

STATUS_DIR="${1:?usage: run_supervised.sh STATUS_DIR -- CMD [ARGS...]}"
shift
[[ "${1:-}" == "--" ]] || { echo "ERROR: expected -- before the command" >&2; exit 2; }
shift
[[ $# -ge 1 ]] || { echo "ERROR: no command given" >&2; exit 2; }

HEARTBEAT_S="${HEARTBEAT_S:-300}"
STALL_LOG_S="${STALL_LOG_S:-1800}"
CPU_EPSILON="${CPU_EPSILON:-1.0}"
NOTIFY_CMD="${NOTIFY_CMD:-}"
HEARTBEAT_SCHEMA=2

mkdir -p "$STATUS_DIR" || exit 2
STATUS_DIR="$(cd "$STATUS_DIR" && pwd)"
LOG="$STATUS_DIR/run.log"
HB="$STATUS_DIR/heartbeat.tsv"

# ── Detach ────────────────────────────────────────────────────────────────
# Re-exec under setsid so the supervisor and its child own a fresh session and
# process group, with PPID 1: it then survives the terminal/agent session that
# launched it. Guarded by an env var so the re-exec happens exactly once.
if [[ -z "${_SUPERVISED_DETACHED:-}" && -z "${FOREGROUND:-}" ]]; then
    export _SUPERVISED_DETACHED=1
    setsid "$0" "$STATUS_DIR" -- "$@" </dev/null >/dev/null 2>&1 &
    disown 2>/dev/null || true
    # Wait briefly so the caller can rely on the status dir being populated.
    for _ in $(seq 1 50); do [ -s "$STATUS_DIR/pgid" ] && break; sleep 0.1; done
    echo "supervised: $STATUS_DIR (pid $(cat "$STATUS_DIR/pid" 2>/dev/null), pgid $(cat "$STATUS_DIR/pgid" 2>/dev/null))"
    exit 0
fi

# ── Atomic single-word writes (MONITOR.md section 2) ──────────────────────
# Written via a same-directory temp + mv, so a concurrent reader never sees a
# half-written value.
put() {  # put <filename> <content>
    local f="$STATUS_DIR/$1" t
    t="$(mktemp "$STATUS_DIR/.$1.XXXXXX")" || return 1
    # mktemp is 0600; these files are read by other people and by the other
    # host, so widen before the rename (never after -- the rename is the
    # publication point).
    printf '%s\n' "$2" > "$t" && chmod 644 "$t" && mv -f "$t" "$f"
}

notify() {  # notify <STATE>
    [[ -n "$NOTIFY_CMD" ]] || return 0
    $NOTIFY_CMD "$1" "$STATUS_DIR" >>"$STATUS_DIR/notify.log" 2>&1 || true
}

# ── Launch ────────────────────────────────────────────────────────────────
printf '%q ' "$@" > "$STATUS_DIR/cmd"; printf '\n' >> "$STATUS_DIR/cmd"
START_EPOCH=$(date +%s)
put started_at "$(date '+%F %T') $START_EPOCH"
put state RUNNING

"$@" >"$LOG" 2>&1 &
CHILD=$!

# The child shares this process group, so the documented recovery gesture
# `kill -TERM -<pgid>` (MONITOR.md section 7) hits the supervisor too. Without a
# trap that leaves `state` stuck at RUNNING forever -- the exact "outcome lives
# only in a watcher" failure this script exists to prevent. Forward the signal,
# then fall through to the terminal-state write below.
trap 'kill -TERM "$CHILD" 2>/dev/null' TERM HUP INT
CHILD_PGID=$(ps -o pgid= -p "$CHILD" 2>/dev/null | tr -d ' ')
put pid "$CHILD"
put pgid "${CHILD_PGID:-$CHILD}"

if [[ ! -s "$HB" ]]; then
    printf '#version %s\n' "$HEARTBEAT_SCHEMA" > "$HB"
    printf 'ts\telapsed_s\tstate\tlog_quiet_s\tcpu_pct\tnproc\trss_gb\tlast\n' >> "$HB"
fi

# ── Group CPU/RSS sampling ────────────────────────────────────────────────
# Sums utime+stime from /proc/<pid>/stat over every process in the child's
# process group. A DELTA, not the %CPU column: that column is a lifetime
# average and would look busy forever after a fast start -- exactly the trap
# that made the 2026-09-02 deadlock hard to see.
# ONE `ps` snapshot per heartbeat, filtered by process group. Bash cannot walk
# /proc cheaply here -- measured 9s per sample on this 256-core box with ~3,000
# processes, which blows the overhead budget (MONITOR.md section 5) and starves
# the heartbeat cadence. `ps -e` does the same scan in C in ~70ms.
# `cputimes` is CPU seconds (utime+stime) for the whole process, so the delta
# across an interval is real work done -- unlike the %CPU column, which is a
# lifetime average and would look busy forever after a fast start. That is
# exactly the trap that made the 2026-09-02 deadlock hard to see.
group_sample() {  # -> "<cpu_seconds> <nproc> <rss_kb>"
    ps -eo pgid=,rss=,cputimes= 2>/dev/null | awk -v g="$1" '
        $1 == g { n++; rss += $2; cpu += $3 }
        END { printf "%d %d %d\n", cpu+0, n+0, rss+0 }'
}

# Last MEANINGFUL log line: tqdm/HF progress bars overwrite in place with \r,
# so the raw tail is a wall of carriage returns. Keep only what follows the
# final \r on each line, then take the last line that is not a bare progress bar.
last_line() {
    [ -s "$LOG" ] || { echo ""; return; }
    tail -c 65536 "$LOG" 2>/dev/null \
        | tr '\r' '\n' | grep -v '^[[:space:]]*$' \
        | grep -Ev '^[[:space:]]*[0-9]+%\|' | tail -1 | cut -c1-200
}

# ── Heartbeat loop ────────────────────────────────────────────────────────
PGID="${CHILD_PGID:-$CHILD}"
prev_cpu=$(group_sample "$PGID" | cut -d' ' -f1)
stall_hits=0
STATE=RUNNING

# Log quiescence is measured as GROWTH BETWEEN HEARTBEATS, never as
# `now - mtime`. The status dir lives on NFS and the filer's clock is not this
# host's: measured 2026-09-03, the filer runs **47 minutes ahead** of B300, so
# `now - mtime` came out at about -2,800 s forever, `log_age >= STALL_LOG_S`
# could never be true, and stall detection was silently disabled in production
# on exactly the runs it was written for. Comparing a file's own (size, mtime)
# against its previous value needs no agreement between the two clocks -- only
# that the filer's clock moves forward.
prev_sig="$(stat -c '%s:%Y' "$LOG" 2>/dev/null || echo none)"
quiet_s=0

while kill -0 "$CHILD" 2>/dev/null; do
    sleep "$HEARTBEAT_S" &
    wait $! 2>/dev/null
    kill -0 "$CHILD" 2>/dev/null || break

    now=$(date +%s)
    elapsed=$((now - START_EPOCH))
    sig="$(stat -c '%s:%Y' "$LOG" 2>/dev/null || echo none)"
    if [[ "$sig" != "$prev_sig" ]]; then
        quiet_s=0; prev_sig="$sig"
    else
        quiet_s=$((quiet_s + HEARTBEAT_S))
    fi

    read -r cpu_s nproc rss_kb <<< "$(group_sample "$PGID")"
    # Clamp: a process leaving the group makes the delta negative, which would
    # read as "idle" and could trip a false STALLED.
    d_cpu=$(( cpu_s - prev_cpu )); [ "$d_cpu" -lt 0 ] && d_cpu=0
    prev_cpu=$cpu_s
    cpu_pct=$(awk -v d="$d_cpu" -v s="$HEARTBEAT_S" \
                  'BEGIN{ printf "%.1f", d/s*100 }')
    rss_gb=$(awk -v k="$rss_kb" 'BEGIN{ printf "%.1f", k/1048576 }')

    # Two signals, two consecutive samples (MONITOR.md section 4).
    idle=$(awk -v c="$cpu_pct" -v e="$CPU_EPSILON" 'BEGIN{print (c<e)?1:0}')
    if [[ "$quiet_s" -ge "$STALL_LOG_S" && "$idle" == "1" ]]; then
        stall_hits=$((stall_hits + 1))
    else
        stall_hits=0
    fi

    new_state=RUNNING
    [[ "$stall_hits" -ge 2 ]] && new_state=STALLED
    if [[ "$new_state" != "$STATE" ]]; then
        STATE="$new_state"; put state "$STATE"
        [[ "$STATE" == "STALLED" ]] && notify STALLED
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date '+%F %T')" "$elapsed" "$STATE" "$quiet_s" "$cpu_pct" \
        "$nproc" "$rss_gb" "$(last_line)" >> "$HB"
done

# ── Terminal state ────────────────────────────────────────────────────────
wait "$CHILD"; rc=$?
# exit_code BEFORE state: a reader that sees a terminal state may rely on
# exit_code existing (MONITOR.md section 2).
put exit_code "$rc"
if [[ "$rc" -eq 0 ]]; then STATE=DONE; else STATE=FAILED; fi
put state "$STATE"
printf '%s\t%s\t%s\t-\t-\t0\t-\t%s (exit %s)\n' \
    "$(date '+%F %T')" "$(( $(date +%s) - START_EPOCH ))" "$STATE" \
    "$(last_line)" "$rc" >> "$HB"
notify "$STATE"
exit "$rc"
