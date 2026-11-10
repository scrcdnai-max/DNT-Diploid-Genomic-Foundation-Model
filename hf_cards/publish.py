"""Push staged repos to the Hub and fix up settings + collection.

    python publish.py plan      # print what would happen
    python publish.py new       # create + upload the five new repos, set gated=auto
    python publish.py existing  # upload README/LICENSE/encoder/tsv to the six existing repos, delete aux_heads files
    python publish.py aux       # make DNT-100M_POST-CPL-SMP-AUX private and drop it from the collection
    python publish.py collection
    python publish.py verify
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi

ORG = "scrc-dnai"
COLL = "scrc-dnai/diploid-genomic-llms-6a70744dcf889708759fb215"
STAGE = Path(__file__).with_name("staging")
NEW = ["DNT-650M_POST-CPL-SMP", "DNT-100M_POST-CPL-SMP", "DNT-8M-MLM", "DNT-8M-CPL-2e4", "DNT-100M_PRE-CPL-2e4"]
EXISTING = ["DNT-8M-CPL", "DNT-8M-CPL-SMP", "DNT-100M_PRE-CPL", "DNT-100M_PRE-CPL-SMP",
            "Directional_Tokens_plus_Unordered_8M", "Phased_Nucleotide_Tokenization_8M"]
AUX = "DNT-100M_POST-CPL-SMP-AUX"
ORDER = ["DNT-650M_POST-CPL-SMP", "DNT-100M_PRE-CPL-2e4", "DNT-8M-CPL-2e4", "DNT-100M_POST-CPL-SMP",
         "DNT-100M_PRE-CPL-SMP", "DNT-8M-CPL-SMP", "DNT-100M_PRE-CPL", "DNT-8M-CPL", "DNT-8M-MLM",
         "Directional_Tokens_plus_Unordered_8M", "Phased_Nucleotide_Tokenization_8M"]
api = HfApi()


def rid(r): return f"{ORG}/{r}"


def do_new():
    for r in NEW:
        api.create_repo(rid(r), repo_type="model", exist_ok=True, private=False)
        api.upload_folder(repo_id=rid(r), folder_path=STAGE / r, commit_message=f"Add {r}: weights, remote code, tokenizer, encoder, card")
        api.update_repo_settings(rid(r), gated="auto")
        print("pushed", rid(r))


def do_existing():
    for r in EXISTING:
        files = set(api.list_repo_files(rid(r)))
        ops = [f for f in ("aux_heads.safetensors", "aux_heads_config.json") if f in files]
        api.upload_folder(repo_id=rid(r), folder_path=STAGE / r, delete_patterns=ops or None,
                          commit_message="Rewrite model card; add LICENSE, diploid_encode.py, diploid_tokens.tsv" + ("; drop training-only aux head files" if ops else ""))
        print("updated", rid(r), "deleted:", ops)


def do_aux():
    api.update_repo_settings(rid(AUX), private=True)
    coll = api.get_collection(COLL)
    for it in coll.items:
        if it.item_id == rid(AUX):
            api.delete_collection_item(COLL, it.item_object_id)
            print("removed from collection", it.item_id)
    print("private:", api.model_info(rid(AUX)).private)


def do_collection():
    coll = api.get_collection(COLL)
    have = {it.item_id: it for it in coll.items}
    for r in ORDER:
        if rid(r) not in have:
            api.add_collection_item(COLL, item_id=rid(r), item_type="model", exists_ok=True)
            print("added", r)
    coll = api.get_collection(COLL)
    have = {it.item_id: it for it in coll.items}
    for pos, r in enumerate(ORDER):
        api.update_collection_item(COLL, have[rid(r)].item_object_id, position=pos)
    api.update_collection_metadata(COLL, description="Diploid genomic language models (DNT): NTv3 backbones continually pretrained on reference-aligned diploid token streams from phased 1000 Genomes data. Paper: https://doi.org/10.64898/2026.09.05.749576")
    print([it.item_id for it in api.get_collection(COLL).items])


def do_verify():
    ok = True
    for r in NEW + EXISTING:
        i = api.model_info(rid(r), files_metadata=False)
        cd = i.card_data.to_dict() if i.card_data else {}
        files = set(s.rfilename for s in i.siblings)
        row = dict(private=i.private, gated=i.gated, license=cd.get("license"), name=cd.get("license_name"),
                   link=cd.get("license_link"), base=cd.get("base_model"),
                   LICENSE="LICENSE" in files, enc="diploid_encode.py" in files, tsv="diploid_tokens.tsv" in files,
                   aux=any(f.startswith("aux_heads") for f in files))
        good = (not row["private"] and row["gated"] == "auto" and row["license"] == "other" and row["name"] == "research-only"
                and row["link"] == "LICENSE" and row["LICENSE"] and row["enc"] and row["tsv"] and not row["aux"])
        ok &= good
        print(("OK  " if good else "BAD "), r, row)
    print("AUX private:", api.model_info(rid(AUX)).private)
    print("collection:", [it.item_id.split("/")[1] for it in api.get_collection(COLL).items])
    print("ALL OK" if ok else "PROBLEMS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "plan":
        print("new:", NEW, "\nexisting:", EXISTING, "\naux -> private:", AUX, "\ncollection order:", ORDER)
    else:
        {"new": do_new, "existing": do_existing, "aux": do_aux, "collection": do_collection, "verify": do_verify}[cmd]()
