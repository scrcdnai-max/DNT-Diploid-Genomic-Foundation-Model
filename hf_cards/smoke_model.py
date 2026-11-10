"""Smoke a local DNT checkpoint dir the way the model card Quickstart does.

usage: python smoke_model.py <model_dir> [pre|post]
Prints the facts the card must state; asserts the ones that are load-bearing.
"""
import importlib.util, json, os, sys, time
import torch

d = sys.argv[1]
kind = sys.argv[2] if len(sys.argv) > 2 else ("post" if "posttrained" in open(f"{d}/config.json").read() else "pre")
tokens_tsv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocess", "diploid_tokens.tsv")
enc = f"{os.path.dirname(os.path.abspath(__file__))}/diploid_encode.py"
spec = importlib.util.spec_from_file_location("de", enc); de = importlib.util.module_from_spec(spec); spec.loader.exec_module(de)

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
cfg = json.load(open(f"{d}/config.json"))
assert cfg.get("aux_tasks") in ([], None), cfg.get("aux_tasks")
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
t0 = time.time()
Loader = AutoModel if kind == "post" else AutoModelForMaskedLM
model = Loader.from_pretrained(d, trust_remote_code=True).eval()
print(f"loaded in {time.time()-t0:.0f}s; params {sum(p.numel() for p in model.parameters())/1e6:.2f}M; kind={kind}")
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(dev)

hap0 = "ACGTACGTAC" * 51
hap1 = hap0[:5] + "G" + hap0[6:]
seq, offsets = de.encode_diploid(hap0, hap1, tokens_tsv=tokens_tsv, return_offsets=True)
assert seq[:12] == "ACGTA^‹ĈGTAC", seq[:12]
assert len(seq) == 512 and offsets[5] == 7
batch = tok(seq, return_tensors="pt").to(dev)
assert batch["input_ids"].shape == (1, 512) and batch["input_ids"][0, 0].item() == tok.convert_tokens_to_ids("A")
kw = {}
if kind == "post":
    sp = model.encode_species("human")
    assert sp.item() == 27
    print("encode_species device:", sp.device, "| model device:", next(model.parameters()).device)
    kw["species_ids"] = sp.to(dev)
with torch.no_grad():
    out = model(**batch, output_hidden_states=True, **kw)
hs = out.hidden_states
print("n hidden_states", len(hs), "shapes", [tuple(h.shape) for h in hs])
H = hs[-1][0]
print("per_token", tuple(H.shape), "has logits", hasattr(out, "logits") and out.logits is not None, "logits shape", tuple(out.logits.shape))
# attention_mask ignored?
with torch.no_grad():
    out2 = model(input_ids=batch["input_ids"], attention_mask=torch.zeros_like(batch["input_ids"]), output_hidden_states=True, **kw)
print("attention_mask ignored:", torch.equal(out2.hidden_states[-1], hs[-1]))
# non-multiple-of-128 length
try:
    with torch.no_grad():
        model(input_ids=batch["input_ids"][:, :500], **kw)
    print("len 500: OK (no error)")
except Exception as e:
    print("len 500:", type(e).__name__, str(e)[:120])
# masked scoring
ids = batch["input_ids"].clone(); i = offsets[5]; ids[0, i] = tok.mask_token_id
with torch.no_grad():
    lg = model(input_ids=ids, **kw).logits
lp = lg[0, i].log_softmax(-1)
print("logP(Ĉ) at het =", round(lp[tok.convert_tokens_to_ids("Ĉ")].item(), 3), "argmax token", tok.convert_ids_to_tokens(lp.argmax().item()))
# bf16 cast
try:
    m2 = model.to(torch.bfloat16)
    with torch.no_grad():
        m2(**batch, **kw)
    print("bf16: OK")
except Exception as e:
    print("bf16:", type(e).__name__, str(e)[:100])
print("SMOKE OK", d)
