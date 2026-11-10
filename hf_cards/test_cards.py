"""Every python block in a card must run, and print what its comments claim.

    python test_cards.py <repo> <model_dir>
Runs the card's code blocks in order against a local model dir (REPO is rewritten to the dir;
hf_hub_download is rewritten to a local lookup). Fails on any exception or on a wrong expected value.
"""
import re, sys, io, contextlib
from pathlib import Path

repo, mdir = sys.argv[1], Path(sys.argv[2])
card = (Path(__file__).with_name("cards") / f"{repo}.md").read_text(encoding="utf-8")
blocks = re.findall(r"```python\n(.*?)```", card, re.S)
assert blocks, "no python blocks"

prelude = f"""
import os
from huggingface_hub import hf_hub_download as _hfd
def hf_hub_download(repo, fn, **kw): return os.path.join({str(mdir)!r}, fn)
"""
ns = {}
exec(prelude, ns)
src = "\n".join(b for b in blocks if "-> str |" not in b)   # every block except the bare signature
src = src.replace(f'REPO = "scrc-dnai/{repo}"', f'REPO = {str(mdir)!r}').replace('from huggingface_hub import hf_hub_download', '')
# blocks that only define a helper need `de`; the quickstart defines it first, so order matters and is kept.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(src, ns)
out = buf.getvalue()
print(out)

# assertions: what the card's comments promise
seq = ns["seq"]; per_token = ns["per_token"]; offsets = ns["offsets"]
d = ns["model"].config.embed_dim
if "to_ordered" in src:
    assert seq[:12] == "ACGTA^ƇGTACA" and len(seq) == 512, seq[:12]
    assert per_token.shape == (512, d) and offsets[5] == 6 and seq[6] == "Ƈ"
else:
    assert seq[:12] == "ACGTA^‹ĈGTAC" and len(seq) == 512, seq[:12]
    assert per_token.shape == (512, d), per_token.shape
    assert offsets[5] == 7 and seq[7] == "Ĉ"
assert ns["coarse"].shape[0] == len(seq) // 128, ns["coarse"].shape
assert ns["pooled"].shape == (d,)
assert float(ns["score"]) < 0
# encode_to_multiple helper: awkward length trims to a multiple of 128
s2, o2, hi = ns["encode_to_multiple"](ns["hap0"][:700], ns["hap1"][:700], ns["tokens_tsv"])
assert len(s2) % 128 == 0 and hi <= 700, (len(s2), hi)
print("CARD OK", repo)
