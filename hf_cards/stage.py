"""Build staging/<repo>/ for every repo: new repos get the full model dir, existing ones only the files that change.

    python stage.py
"""
import json, os, shutil
from pathlib import Path

S = Path(__file__).parent
STAGE = S / "staging"
# Directory holding the trained checkpoints, one sub-directory per training run.
MODEL_ROOT = Path(os.environ.get("DNT_MODEL_ROOT", "models"))
NEW = {  # repo -> source model dir (weights + remote code + tokenizer)
    "DNT-650M_POST-CPL-SMP": MODEL_ROOT / "650m_phase1_smpcpl/final",
    "DNT-100M_POST-CPL-SMP": MODEL_ROOT / "100m_post_v7_4k_RTFIX/genome-200",
    "DNT-8M-MLM": MODEL_ROOT / "8m_mlmonly_2e4_v6_RTFIX/genome-1000",
    "DNT-8M-CPL-2e4": MODEL_ROOT / "8m_cpl_fullepoch_2e4_v6/final",
    "DNT-100M_PRE-CPL-2e4": MODEL_ROOT / "100m_cpl_fullepoch_2e4_v6/final",
}
EXISTING = ["DNT-8M-CPL", "DNT-8M-CPL-SMP", "DNT-100M_PRE-CPL", "DNT-100M_PRE-CPL-SMP",
            "Directional_Tokens_plus_Unordered_8M", "Phased_Nucleotide_Tokenization_8M"]
KEEP = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
BANNED = ("aux_heads",)


def common(d: Path, repo: str):
    shutil.copy(S / "cards" / f"{repo}.md", d / "README.md")
    shutil.copy(S / "LICENSE", d / "LICENSE")
    shutil.copy(S.parent / "preprocess" / "diploid_tokens.tsv", d / "diploid_tokens.tsv")
    shutil.copy(S / "diploid_encode.py", d / "diploid_encode.py")


if __name__ == "__main__":
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for repo, src in NEW.items():
        d = STAGE / repo; d.mkdir(parents=True)
        src = Path(src)
        for f in src.iterdir():
            if f.name in KEEP or f.name.endswith(".py"):
                assert not f.name.startswith(BANNED), f
                shutil.copy(f, d / f.name)
        cfg = json.load(open(d / "config.json"))
        assert cfg.get("aux_tasks") in ([], None), (repo, cfg.get("aux_tasks"))
        if isinstance(cfg.get("rare_token_recipe"), dict):
            cfg["rare_token_recipe"].pop("derived_from", None)   # internal path
        json.dump(cfg, open(d / "config.json", "w"), indent=2)
        txt = (d / "config.json").read_text()
        assert not any(s in txt for s in ("/home/", "/mnt/", "/opt/", "/storage", "wandb")), repo  # no local paths
        common(d, repo)
        print(repo, sorted(p.name for p in d.iterdir()))
    for repo in EXISTING:
        d = STAGE / repo; d.mkdir(parents=True)
        common(d, repo)
        print(repo, sorted(p.name for p in d.iterdir()))
