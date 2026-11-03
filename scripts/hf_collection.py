"""Keeps the Hugging Face collection of Krafla models in step with
scripts/hf_models.json. Run by .github/workflows/hf-collection.yml (after
every publishing run of hf-models.yml, or by hand).

    python3 scripts/hf_collection.py              # needs HF_TOKEN with write access
    python3 scripts/hf_collection.py --owner jonpark0 --dry-run   # no token: show what it would do

The collection (the catalog's "collection" title and description, created if
missing) holds every published <owner>/<source name>-krafla repository of the
catalog, in catalog order (smallest model first), each with a note: the
Krafla version its files need, the --chat template and the size of the Q4_0
file. Repositories outside the catalog stay where they are, after these.
"""
import argparse
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hf_publish  # noqa: E402

REPO_URL = hf_publish.REPO_URL


def repo_of(owner, model):
    return f"{owner}/{model['source'].split('/')[1]}-krafla"


def note_for(api, repo, model):
    """'Krafla 0.9.0+ · --chat qwen3.5-nothink · Q4_0 406 MiB' (GiB from 1 GiB up), from the card and files."""
    from huggingface_hub import hf_hub_download

    base = model["source"].split("/")[1]
    info = api.model_info(repo, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings or []}
    parts = []
    try:
        card = open(hf_hub_download(repo, "README.md"), encoding="utf-8").read()
        m = re.search(r"version ([0-9]+\.[0-9]+\.[0-9]+) or later", card)
        if m:
            parts.append(f"Krafla {m.group(1)}+")
    except Exception:   # a card that cannot be read leaves the version out
        pass
    parts.append(f"--chat {model['chat']}")
    q4 = sizes.get(f"{base}-Q4_0.mlb")
    if q4:
        parts.append(f"Q4_0 {q4 / 2**30:.1f} GiB" if q4 >= 2**30 else f"Q4_0 {q4 / 2**20:.0f} MiB")
    return " · ".join(parts)


def find_collection(api, owner, title):
    for c in api.list_collections(owner=owner, limit=100):
        if c.title == title:
            return api.get_collection(c.slug)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", help="Hugging Face account (default: the token's)")
    ap.add_argument("--dry-run", action="store_true", help="show the changes without making them (no token needed)")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None
    if not token and not a.dry_run:
        sys.exit("HF_TOKEN (write access) is needed; --dry-run shows the changes without one")
    api = HfApi(token=token)
    owner = a.owner or api.whoami()["name"]
    catalog = hf_publish.load_catalog()
    meta = catalog["collection"]

    want = []   # (repo, note) in catalog order, published ones only
    for model in catalog["models"]:
        repo = repo_of(owner, model)
        if api.repo_exists(repo):
            want.append((repo, note_for(api, repo, model)))
    print(f"{len(want)} published models in the catalog", flush=True)

    coll = find_collection(api, owner, meta["title"])
    if coll is None:
        print(f"create collection {meta['title']!r}", flush=True)
        if a.dry_run:
            for i, (repo, note) in enumerate(want):
                print(f"  {i:2d} {repo}  [{note}]")
            return 0
        coll = api.create_collection(meta["title"], namespace=owner, description=meta["description"])
    elif coll.description != meta["description"]:
        print("update the description", flush=True)
        if not a.dry_run:
            api.update_collection_metadata(coll.slug, description=meta["description"])

    have = {it.item_id: it for it in coll.items}
    for repo, note in want:
        if repo not in have:
            print(f"add {repo}  [{note}]", flush=True)
            if not a.dry_run:
                api.add_collection_item(coll.slug, repo, "model", note=note, exists_ok=True)
        elif (have[repo].note or "") != note:
            print(f"note {repo}: [{have[repo].note}] -> [{note}]", flush=True)
            if not a.dry_run:
                api.update_collection_item(coll.slug, have[repo].item_object_id, note=note)
    if not a.dry_run:
        coll = api.get_collection(coll.slug)
        have = {it.item_id: it for it in coll.items}
    # catalog order first; anything else keeps its relative order after them
    for pos, (repo, _) in enumerate(want):
        item = have.get(repo)
        if item is not None and item.position != pos:
            print(f"move {repo} to {pos}", flush=True)
            if not a.dry_run:
                api.update_collection_item(coll.slug, item.item_object_id, position=pos)
    url = f"https://huggingface.co/collections/{coll.slug}"
    print(f"collection: {url}", flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(f"Collection: {url} ({len(want)} models)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
