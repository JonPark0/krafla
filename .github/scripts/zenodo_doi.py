"""Finds the Zenodo DOI of one release of this repository.

    python3 .github/scripts/zenodo_doi.py v0.6.1 [--timeout 1800]

Zenodo's GitHub integration archives each published release and mints a
DOI for it a few minutes later. This script reads the concept DOI (all
versions) from CITATION.cff, then asks Zenodo's public API, which needs no
token, for the version whose `version` field matches the tag (with or without
a leading "v"). It retries until --timeout and prints one JSON line:
{"doi": ..., "url": ..., "concept_doi": ...}. Exit status 2: not found.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
API = "https://zenodo.org/api"


def concept_from_citation(path):
    with open(path, encoding="utf-8") as f:
        m = re.search(r"^doi:\s*\"?(10\.5281/zenodo\.(\d+))\"?\s*$", f.read(), re.M)
    if not m:
        sys.exit(f"no Zenodo doi in {path}")
    return m.group(1), m.group(2)


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "krafla-release"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def norm(version):
    return str(version or "").strip().lstrip("vV")


def record_doi(rec):
    """DOI of a record in the legacy or the InvenioRDM serialization."""
    doi = rec.get("doi") or ((rec.get("pids") or {}).get("doi") or {}).get("identifier")
    if not doi and rec.get("id"):
        doi = f"10.5281/zenodo.{rec['id']}"
    return doi


def records(concept_id):
    """Every version of the concept, newest first, as far as the API shows them."""
    out = []
    try:   # the concept id resolves to the latest version
        latest = get(f"{API}/records/{concept_id}")
        out.append(latest)
        if latest.get("id"):
            hits = get(f"{API}/records/{latest['id']}/versions?allversions=true&size=50")
            out += hits.get("hits", {}).get("hits", [])
    except (urllib.error.URLError, ValueError, OSError):
        pass
    try:
        hits = get(f"{API}/records?q=conceptrecid:{concept_id}&all_versions=true&sort=mostrecent&size=50")
        out += hits.get("hits", {}).get("hits", [])
    except (urllib.error.URLError, ValueError, OSError):
        pass
    return out


def find(tag, concept_id):
    for rec in records(concept_id):
        if norm((rec.get("metadata") or {}).get("version")) == norm(tag):
            doi = record_doi(rec)
            if doi:
                return doi, f"https://doi.org/{doi}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", help="release tag, e.g. v0.6.1")
    ap.add_argument("--citation", default=os.path.join(ROOT, "CITATION.cff"))
    ap.add_argument("--timeout", type=int, default=0, help="seconds to keep retrying (0: one try)")
    ap.add_argument("--interval", type=int, default=30)
    a = ap.parse_args()
    concept_doi, concept_id = concept_from_citation(a.citation)
    deadline = time.time() + a.timeout
    while True:
        found = find(a.tag, concept_id)
        if found:
            print(json.dumps({"doi": found[0], "url": found[1], "concept_doi": concept_doi}))
            return
        if time.time() >= deadline:
            print(f"no Zenodo record for {a.tag} under {concept_doi}", file=sys.stderr)
            sys.exit(2)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
