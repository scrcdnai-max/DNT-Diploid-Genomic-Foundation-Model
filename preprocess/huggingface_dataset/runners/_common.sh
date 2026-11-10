# shellcheck shell=bash
# Sourced (never executed) by the runners in this directory that drive repo code:
#
#   REPO            the checkout this file lives in (three levels up); an exported REPO wins
#   RUNNERS_DIR     this directory (sibling scripts: make_persample_bcf.sh, run_supervised.sh, ...)
#   DIPLOID_TOKENS HG38_2BIT CANONICAL_BED LDB_BED TSSTES_BED TSSTES_V10_BED
#                   from preprocess/paths.py — the one place those paths are spelled
#   GLM_BIOTOOLS_IMAGE (+ _TARBALL) from docker_images/glm_biotools/image.env
#   print_cmd       print one command, shell-quoted (the --dry-run output)
#   UV, uvrun()     uv binary (ssh non-interactive PATH may lack ~/.local/bin) and
#                   `uvrun ARGS` = `$UV run --no-sync python ARGS`
#   require_host_local_venv   the launch_biome_* venv contract, for runners that call it
#
# It never runs uv: on a host that is not the primary one, a bare `uv run` DELETES the
# shared /storage .venv (preprocess/CLAUDE.md, 2026-09-03), so the paths come from the
# system python3 running paths.py (stdlib only) and each runner keeps its own venv check
# before its first uv call. Scripts that run inside the glm_biotools container
# (make_persample_bcf.sh, run_chunked_split.sh, concat_persample_bcf.sh, ...) do not source it.

RUNNERS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
REPO="${REPO:-$(cd -- "${RUNNERS_DIR}/../../.." >/dev/null && pwd -P)}"

# (assignment first: `eval "$(cmd)"` would return 0 even when cmd fails)
_paths_sh="$(python3 -I "${REPO}/preprocess/paths.py" --sh)" || {
    echo "ERROR: python3 -I ${REPO}/preprocess/paths.py --sh failed" >&2; exit 1; }
eval "${_paths_sh}"
unset _paths_sh

# shellcheck source=../../../docker_images/glm_biotools/image.env
source "${REPO}/docker_images/glm_biotools/image.env"

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
uvrun() { "$UV" run --no-sync python "$@"; }

# One shell-quoted command line, indented: what a --dry-run prints.
print_cmd() { printf '  '; printf '%q ' "$@"; echo; }

# HOST-LOCAL VENV, MANDATORY off the primary host. The repo lives on /storage and so
# does its default `.venv`, whose pinned interpreter path does not exist on the other
# host -- so a bare `uv run` there DELETES that venv and recreates it empty. On
# 2026-09-03 that killed B300's 4h26m-old assemble mid-Step-5 (`ModuleNotFoundError:
# No module named 'multiprocess.queues'`, a lazy import inside multiprocess.Pool).
# PRIMARY_HOST=1 means "this host owns the /storage .venv and uv may resolve it".
# Anywhere else, a host-local venv is mandatory and asserted.
require_host_local_venv() {
    if [[ "${PRIMARY_HOST:-0}" == "1" ]]; then
        echo "venv: shared /storage .venv (PRIMARY_HOST=1)"
    else
        export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/hpc/talz/venvs/diploid_glm}"
        [ -x "$UV_PROJECT_ENVIRONMENT/bin/python3" ] || {
            echo "ERROR: no host-local venv at $UV_PROJECT_ENVIRONMENT — build it with" >&2
            echo "  UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT uv sync --frozen" >&2
            echo "A bare uv run here would DELETE the shared /storage venv." >&2
            exit 1; }
        echo "venv: $UV_PROJECT_ENVIRONMENT ($($UV_PROJECT_ENVIRONMENT/bin/python3 -V))"
    fi
}
