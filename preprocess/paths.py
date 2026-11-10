"""The single home for repo-relative data paths used as defaults across ``preprocess/``.

Code that needs the token table, the reference or a committed region BED imports it from
here instead of building a ``Path(__file__)``-relative path, so moving a module never
silently changes which file it reads. Every path is absolute; nothing here checks that the
file exists (``hg38.2bit`` is gitignored and absent on some hosts).

``versions.LDB_BED_HINT`` / ``TSSTES_BED_HINT`` stay repo-relative strings (they are
recorded in provenance and the schema docs); ``tests/test_versions.py`` asserts they name
the same files as ``LDB_BED`` / ``TSSTES_BED`` below.
"""

from pathlib import Path

PREPROCESS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PREPROCESS_ROOT.parent

DIPLOID_TOKENS = PREPROCESS_ROOT / "diploid_tokens.tsv"
TOKEN_COUNTS = PREPROCESS_ROOT / "token_counts.tsv"
HG38_2BIT = PREPROCESS_ROOT / "hg38.2bit"

REGIONS_DIR = PREPROCESS_ROOT / "regions"
CANONICAL_BED = REGIONS_DIR / "hg38_pretrain_canonical_expanded.bed"
LDB_BED = REGIONS_DIR / "ld_blocks_eur_grch38.bed"
TSSTES_BED = REGIONS_DIR / "nested_genome_division" / "hg38_tss_tes.bed"
TSSTES_V10_BED = REGIONS_DIR / "nested_genome_division" / "hg38_tss_tes_v10.bed"

# The paths the shell runners need (`huggingface_dataset/runners/_common.sh` evals this).
SH_EXPORTS = ("DIPLOID_TOKENS", "HG38_2BIT", "CANONICAL_BED", "LDB_BED", "TSSTES_BED", "TSSTES_V10_BED")


def sh_exports() -> str:
    """``export NAME="${REPO}/<repo-relative path>"`` lines for :data:`SH_EXPORTS`.

    Relative to ``${REPO}`` rather than absolute, so the shell gets exactly the string
    it used to spell out by hand (``$REPO/preprocess/...``) whatever ``REPO`` is.
    """
    return "\n".join(
        f'export {name}="${{REPO}}/{globals()[name].relative_to(REPO_ROOT).as_posix()}"'
        for name in SH_EXPORTS
    )


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--sh"]:
        sys.exit("usage: python3 preprocess/paths.py --sh")
    print(sh_exports())
