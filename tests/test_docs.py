"""The docs' generated files: every chart and figure renders as well-formed SVG
with a title and a description, and each figure's facts still match their
sources (scripts/sync_docs.py --check, run in CI, checks they are current)."""
import os
import sys
import warnings
import xml.dom.minidom

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import docfigs  # noqa: E402
import sync_docs  # noqa: E402


def test_targets_render():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        targets = sync_docs.targets()
    svgs = {p: t for p, t in targets.items() if p.endswith(".svg")}
    assert len(svgs) == 4 + len(docfigs.FIGURES)
    for path, text in svgs.items():
        doc = xml.dom.minidom.parseString(text.encode("utf-8"))
        root = doc.documentElement
        assert root.tagName == "svg", path
        assert root.getElementsByTagName("title")[0].firstChild.data.strip(), path
        assert root.getElementsByTagName("desc")[0].firstChild.data.strip(), path


def test_rule_figure_is_mlb():
    """The candidates the figure draws are the ones tools/mlb.py chooses among."""
    import numpy as np
    x = docfigs.RULE_BLOCK
    lf, df = docfigs.mlb._q4_search(np.asarray(x, np.float32)[None])
    best = min(docfigs.fit_candidates(x), key=lambda c: docfigs.weighted_error(x, c[1], c[0]))
    assert best[0] == float(df[0]) and best[1] == lf[0].tolist()
