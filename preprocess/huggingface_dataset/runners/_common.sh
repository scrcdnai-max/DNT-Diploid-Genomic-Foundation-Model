# shellcheck shell=bash
# Sourced (never executed) by the runners in this directory that drive repo code:
#
#   REPO            the checkout this file lives in (three levels up); an exported REPO wins
#   RUNNERS_DIR     this directory (sibling scripts: make_persample_bcf.sh, run_supervised.sh, ...)
#   DATA_ROOT       where the 1000 Genomes BCFs live and the datasets are written;
#                   default ${REPO}/data/1kgenomes, export DATA_ROOT to put it elsewhere
#   DIPLOID_TOKENS HG38_2BIT CANONICAL_BED LDB_BED TSSTES_BED TSSTES_V10_BED
#                   from preprocess/paths.py — the one place those paths are spelled
#   GLM_BIOTOOLS_IMAGE from docker_images/glm_biotools/image.env
#   PYTHON, pyrun() the interpreter of the environment the repo's requirements are installed in
#                   (default: python3 on PATH) and `pyrun ARGS` = `$PYTHON ARGS`
#   docker_mount_args / docker_mounts_text
#                   the bind mounts a glm_biotools container needs: REPO, DATA_ROOT and every
#                   directory in DOCKER_EXTRA_MOUNTS (space-separated; for inputs or outputs
#                   that live outside DATA_ROOT)
#   print_cmd       print one command, shell-quoted (the --dry-run output)
#
# The paths come from the system python3 running paths.py (stdlib only). Scripts that run
# inside the glm_biotools container (make_persample_bcf.sh, run_chunked_split.sh,
# concat_persample_bcf.sh, ...) do not source it.

RUNNERS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
REPO="${REPO:-$(cd -- "${RUNNERS_DIR}/../../.." >/dev/null && pwd -P)}"

# (assignment first: `eval "$(cmd)"` would return 0 even when cmd fails)
_paths_sh="$(python3 -I "${REPO}/preprocess/paths.py" --sh)" || {
    echo "ERROR: python3 -I ${REPO}/preprocess/paths.py --sh failed" >&2; exit 1; }
eval "${_paths_sh}"
unset _paths_sh

# shellcheck source=../../../docker_images/glm_biotools/image.env
source "${REPO}/docker_images/glm_biotools/image.env"

DATA_ROOT="${DATA_ROOT:-${REPO}/data/1kgenomes}"

PYTHON="${PYTHON:-python3}"
pyrun() { "$PYTHON" "$@"; }

# Fills the array DOCKER_MOUNT_ARGS with `-v DIR:DIR` for REPO, DATA_ROOT and
# DOCKER_EXTRA_MOUNTS. Call it right before `docker run`, after the paths are final.
docker_mount_args() {
    DOCKER_MOUNT_ARGS=()
    local d
    # shellcheck disable=SC2086  # DOCKER_EXTRA_MOUNTS is a space-separated list
    for d in "${REPO}" "${DATA_ROOT}" ${DOCKER_EXTRA_MOUNTS:-}; do
        DOCKER_MOUNT_ARGS+=(-v "${d}:${d}")
    done
}
# The same mounts as text, for the copy-paste commands printed in error messages.
docker_mounts_text() {
    docker_mount_args
    printf '%q ' "${DOCKER_MOUNT_ARGS[@]}"
}

# One shell-quoted command line, indented: what a --dry-run prints.
print_cmd() { printf '  '; printf '%q ' "$@"; echo; }
