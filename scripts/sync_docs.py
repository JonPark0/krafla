"""Writes the docs files that are derived from other files, or checks that
they are current (--check, run in CI):

  docs/assets/decode-ratio.svg   the decode ratios of docs/benchmarks.md's
                                 x86-64 and AArch64 tables, one thread
  docs/assets/prefill-ratio.svg  the prefill ratios of the same rows
  docs/assets/q4-kld.svg         the KL divergence of the fit and flat Q4
                                 rules against rounding, docs/q4-scales.md
  docs/assets/q4-mixed.svg       Llama 3.2 1B's size and perplexity with
                                 --q8 classes, docs/user-guide.md
  docs/changelog.md              CHANGELOG.md, as a page of the docs site

    python3 scripts/sync_docs.py            # rewrite
    python3 scripts/sync_docs.py --check    # exit 1 if a file is stale
"""
import argparse
import math
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BENCH = os.path.join(ROOT, "docs", "benchmarks.md")
Q4_NOTE = os.path.join(ROOT, "docs", "q4-scales.md")
GUIDE = os.path.join(ROOT, "docs", "user-guide.md")
CHART = os.path.join(ROOT, "docs", "assets", "decode-ratio.svg")
PREFILL_CHART = os.path.join(ROOT, "docs", "assets", "prefill-ratio.svg")
Q4_CHART = os.path.join(ROOT, "docs", "assets", "q4-kld.svg")
MIXED_CHART = os.path.join(ROOT, "docs", "assets", "q4-mixed.svg")
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")
CHANGELOG_PAGE = os.path.join(ROOT, "docs", "changelog.md")
CHANGELOG_FRONT = """---
title: "Changelog: what changed in each Krafla release"
description: "Release notes of Krafla: new models, kernels, formats, speed-ups and fixes in each version."
nav_order: 10
nav_title: "Changelog"
---
<!-- Generated from CHANGELOG.md by scripts/sync_docs.py; edit CHANGELOG.md instead. -->

"""


def md_tables(path):
    """{section heading: [table]} of a Markdown file, each table a list of row
    dicts keyed by the header cells."""
    out, section, header = {}, None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"^#{2,3} (.+)$", line)
            if m:
                section, header = m.group(1).strip(), None
                continue
            if not line.startswith("|"):
                header = None
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if header is None:
                header = cells
                out.setdefault(section, []).append([])
            elif not set(line) <= set("|-: "):
                out[section][-1].append(dict(zip(header, cells)))
    return out


def table_with(path, section, column):
    """The first table of `section` that has `column`."""
    for t in md_tables(path).get(section, []):
        if t and column in t[0]:
            return t
    sys.exit(f"no table with a '{column}' column under '{section}' in {path}")


def num(cell):
    """'1252 MiB', '1.52×', '−3.4%', '**21.73**' -> float."""
    m = re.search(r"[-−]?\d+(?:\.\d+)?", cell.replace(",", ""))
    if not m:
        sys.exit(f"not a number: {cell!r}")
    return float(m.group(0).replace("−", "-"))


# Charts: one style for all. Blue and orange are the first two categorical
# slots of a palette checked for color-vision deficiency (adjacent CVD ΔE 25
# light, 27 dark) and for 3:1 contrast on both backgrounds. They follow the
# reader's system setting, as images cannot see the site's switch.
STYLE = (".bg{fill:#fff;stroke:#e5e7eb}.t{fill:#1f2937}.m{fill:#6b7280}.g{stroke:#eceef1}"
         ".ax{stroke:#9ca3af}.s1{fill:#2a78d6}.s2{fill:#eb6834}.lb{fill:#fff}.ring{stroke:#fff}"
         ".cn{stroke:#c9ccd1}.h{paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}\n"
         "@media (prefers-color-scheme:dark){.bg{fill:#1f2023;stroke:#3a3b40}.t{fill:#e5e7eb}"
         ".m{fill:#9ca3af}.g{stroke:#2e3035}.ax{stroke:#6b7280}.s1{fill:#3987e5}.s2{fill:#d95926}"
         ".lb{fill:#1f2023}.ring{stroke:#1f2023}.cn{stroke:#4b4f57}.h{stroke:#1f2023}}")


def svg_head(cid, W, H, title, subtitle, desc, legend=()):
    """Opening of a chart: card, title, subtitle and a legend of (class, label, mark)
    at the top right, mark "bar" or "dot"."""
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-labelledby="{cid}-title {cid}-desc" font-family="system-ui, -apple-system, '
        'Segoe UI, Roboto, Helvetica, Arial, sans-serif">',
        f'<title id="{cid}-title">{title}</title>',
        f'<desc id="{cid}-desc">{desc}</desc>',
        f"<style>\n{STYLE}\n</style>",
        f'<rect class="bg" x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="8"/>',
        f'<text class="t" x="20" y="32" font-size="17" font-weight="600">{title}</text>',
        f'<text class="m" x="20" y="54" font-size="13">{subtitle}</text>',
    ]
    x = W - 20
    for cls, label, mark in reversed(legend):
        x -= 7.2 * len(label) + 30
        if mark == "dot":
            out.append(f'<circle class="{cls}" cx="{x + 6}" cy="74" r="5"/>')
        else:
            out.append(f'<rect class="{cls}" x="{x}" y="68" width="12" height="12" rx="3"/>')
        out.append(f'<text class="t" x="{x + 17}" y="79" font-size="13">{label}</text>')
    return out


def hbar(cls, x0, y, w, h, r=4):
    """A bar from the baseline x0, square there, rounded at its data end."""
    r = min(r, w / 2, h / 2)
    return (f'<path class="{cls}" d="M{x0:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}'
            f'a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.1f}z"/>')


def ratio_groups(column):
    """[(group label, [(ISA, {weights: Krafla / llama.cpp})])] of the one-thread
    rows: column "Decode ratio", or "prefill" for Krafla prefill / llama.cpp prefill."""
    groups = []
    for section, label in (("Speed", "x86-64 · Intel Xeon (Emerald Rapids), 4 vCPUs"),
                           ("AArch64", "AArch64 · GitHub Arm runner, 4 vCPUs")):
        rows = {}
        for r in table_with(BENCH, section, "Decode ratio"):
            if r["Threads"] != "1":
                continue
            isa = r["ISA level"].replace(" (native)", "")
            v = (num(r["Krafla prefill"]) / num(r["llama.cpp prefill"]) if column == "prefill"
                 else num(r[column]))
            rows.setdefault(isa, {})[r["Weights"]] = v
        if not rows:
            sys.exit(f"no one-thread rows in the '{section}' table of {BENCH}")
        groups.append((label, list(rows.items())))
    return groups


def ratio_chart(cid, groups, title, subtitle):
    W, left, right = 760, 170, 40
    row_h, bar_h, gap, group_h = 36, 12, 2, 30
    top = 86
    n_rows = sum(len(rows) for _, rows in groups)
    plot_h = n_rows * row_h + len(groups) * group_h
    H = top + plot_h + 44
    vmax = max(v for _, rows in groups for _, d in rows for v in d.values())
    xmax = max(4, int(vmax) + 1)
    sx = (W - left - right) / xmax
    x = lambda v: left + v * sx                                        # noqa: E731
    desc = "; ".join(f"{isa}: Q4_0 {d.get('Q4_0', 0):.2f}x, Q8_0 {d.get('Q8_0', 0):.2f}x"
                     for _, rows in groups for isa, d in rows)
    out = svg_head(cid, W, H, title, subtitle, desc, (("s1", "Q4_0", "bar"), ("s2", "Q8_0", "bar")))
    y_end = top + plot_h
    for i in range(xmax + 1):
        out.append(f'<line class="g" x1="{x(i):.1f}" y1="{top}" x2="{x(i):.1f}" y2="{y_end}"/>')
        out.append(f'<text class="m" x="{x(i):.1f}" y="{y_end + 18}" font-size="12" '
                   f'text-anchor="middle">{i}×</text>')
    y = top
    for label, rows in groups:
        # the group label sits on a strip of background over the grid lines
        out.append(f'<rect class="lb" x="12" y="{y + 6}" width="{W - 24}" height="{group_h - 10}"/>')
        out.append(f'<text class="m" x="20" y="{y + 20}" font-size="12" font-weight="600">{label}</text>')
        y += group_h
        out.append(f'<line class="ax" x1="{x(1):.1f}" y1="{y}" x2="{x(1):.1f}" y2="{y + len(rows) * row_h}" '
                   'stroke-width="1.5"/>')
        for isa, d in rows:
            out.append(f'<text class="t" x="{left - 12}" y="{y + row_h / 2 + 4:.1f}" font-size="13" '
                       f'text-anchor="end">{isa}</text>')
            by = y + (row_h - 2 * bar_h - gap) / 2
            for cls, key in (("s1", "Q4_0"), ("s2", "Q8_0")):
                v = d.get(key)
                if v is not None:
                    out.append(hbar(cls, left, by, v * sx, bar_h))
                    out.append(f'<text class="t h" x="{x(v) + 5:.1f}" y="{by + bar_h - 2:.1f}" '
                               f'font-size="11">{v:.2f}×</text>')
                by += bar_h + gap
            y += row_h
    out.append(f'<text class="m" x="{x(1) + 4:.1f}" y="{y_end + 34}" font-size="12">llama.cpp = 1×</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def q4_rows():
    """[(group label, [(model, fit change %, flat change %)])], KL divergence
    against llama.cpp's rounded Q4_0, from docs/q4-scales.md."""
    kld = {}
    for r in table_with(Q4_NOTE, "Which part of the rule matters", "Fitted, llama.cpp weights"):
        kld[r["Model"]] = (num(r["Rounded"]), num(r["Fitted, `x²`"]), num(r["Fitted, llama.cpp weights"]))
    llama = []
    for r in table_with(Q4_NOTE, "Llama architecture models", "Flat KLD"):
        kld[r["Model"]] = (num(r["Rounded KLD"]), num(r["Fitted KLD"]), num(r["Flat KLD"]))
        llama.append(r["Model"])
    change = lambda m: (m, *[(v / kld[m][0] - 1) * 100 for v in kld[m][1:]])   # noqa: E731
    qwen = [m for m in kld if m.startswith("Qwen")]
    gemma = [m for m in kld if m.startswith("Gemma")]
    if len(qwen) + len(llama) + len(gemma) != len(kld):
        sys.exit(f"{Q4_NOTE}: a model in the scale tables is neither Qwen, Gemma nor in the Llama table")
    return [("Qwen · default: fit", [change(m) for m in qwen]),
            ("Llama architecture · default: flat", [change(m) for m in llama]),
            ("Gemma 3 · default: rounded (0%)", [change(m) for m in gemma])]


def q4_chart(groups):
    W, left, right = 760, 200, 36
    row_h, group_h, top = 24, 30, 86
    n_rows = sum(len(rows) for _, rows in groups)
    plot_h = n_rows * row_h + len(groups) * group_h
    H = top + plot_h + 52
    vals = [v for _, rows in groups for _, *vs in rows for v in vs]
    lo, hi = min(-5, 5 * math.floor(min(vals) / 5)), max(5, 5 * math.ceil(max(vals) / 5))
    sx = (W - left - right) / (hi - lo)
    x = lambda v: left + (v - lo) * sx                                 # noqa: E731
    desc = "; ".join(f"{m}: fit {f:+.1f}%, flat {g:+.1f}%" for _, rows in groups for m, f, g in rows)
    out = svg_head("q4", W, H, "Q4_0 KL divergence against llama.cpp's rounding",
                   "Change from llama.cpp's Q4_0, no importance matrix; left of 0 is closer to the "
                   "unquantized model", desc, (("s1", "fit (x²)", "dot"), ("s2", "flat", "dot")))
    y_end = top + plot_h
    for v in range(lo, hi + 1, 5):
        out.append(f'<line class="g" x1="{x(v):.1f}" y1="{top}" x2="{x(v):.1f}" y2="{y_end}"/>')
        tick = "0%" if v == 0 else f"{v:+d}%".replace("-", "−")
        out.append(f'<text class="m" x="{x(v):.1f}" y="{y_end + 18}" font-size="12" '
                   f'text-anchor="middle">{tick}</text>')
    y = top
    for label, rows in groups:
        out.append(f'<rect class="lb" x="12" y="{y + 6}" width="{W - 24}" height="{group_h - 10}"/>')
        out.append(f'<text class="m" x="20" y="{y + 20}" font-size="12" font-weight="600">{label}</text>')
        y += group_h
        out.append(f'<line class="ax" x1="{x(0):.1f}" y1="{y}" x2="{x(0):.1f}" y2="{y + len(rows) * row_h}" '
                   'stroke-width="1.5"/>')
        for m, f, g in rows:
            cy = y + row_h / 2
            out.append(f'<text class="t" x="{left - 12}" y="{cy + 4:.1f}" font-size="13" text-anchor="end">{m}</text>')
            out.append(f'<line class="cn" x1="{x(f):.1f}" y1="{cy:.1f}" x2="{x(g):.1f}" y2="{cy:.1f}" '
                       'stroke-width="2"/>')
            for cls, v in (("s1", f), ("s2", g)):
                out.append(f'<circle class="{cls} ring" cx="{x(v):.1f}" cy="{cy:.1f}" r="5" stroke-width="2"/>')
            y += row_h
    out.append(f'<text class="m" x="{x(0):.1f}" y="{y_end + 36}" font-size="12" text-anchor="middle">'
               "llama.cpp's Q4_0 = 0</text>")
    out.append(f'<text class="m" x="{left}" y="{y_end + 36}" font-size="12">← closer</text>')
    out.append(f'<text class="m" x="{W - right}" y="{y_end + 36}" font-size="12" text-anchor="end">further →</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# label placement for the size/perplexity chart: left of the point instead of right
LEFT_LABELS = {"Q4 + Q8 `v,down`"}


def mixed_points():
    """[(label, MiB, perplexity)] of Llama 3.2 1B in the user guide's first table."""
    return [(r["Weights"], num(r["Size"]), num(r["Perplexity"]))
            for r in table_with(GUIDE, "Q8, Q4, or mixed?", "Perplexity")]


def mixed_chart(points):
    W, H, left, right, top, bottom = 760, 380, 64, 40, 86, 52
    xs, ys = [p[1] for p in points], [p[2] for p in points]
    x0, x1 = 100 * math.floor(min(xs) / 100), 100 * math.ceil(max(xs) / 100)
    y0, y1 = math.floor(min(ys)), math.ceil(max(ys))
    pw, ph = W - left - right, H - top - bottom
    x = lambda v: left + (v - x0) / (x1 - x0) * pw                     # noqa: E731
    y = lambda v: top + (y1 - v) / (y1 - y0) * ph                      # noqa: E731
    desc = "; ".join(f"{lab.replace('`', '')}: {mib:.0f} MiB, perplexity {p:.2f}" for lab, mib, p in points)
    out = svg_head("mix", W, H, "Llama 3.2 1B: file size against perplexity",
                   "Q4 with some matrix classes kept in Q8 (--q8); lower left is better", desc)
    for i in range(int((y1 - y0) * 2) + 1):
        v = y0 + i / 2
        out.append(f'<line class="g" x1="{left}" y1="{y(v):.1f}" x2="{W - right}" y2="{y(v):.1f}"/>')
        out.append(f'<text class="m" x="{left - 8}" y="{y(v) + 4:.1f}" font-size="12" text-anchor="end">'
                   f"{v:.1f}</text>")
    for v in range(int(x0), int(x1) + 1, 100):
        out.append(f'<text class="m" x="{x(v):.1f}" y="{top + ph + 18}" font-size="12" '
                   f'text-anchor="middle">{v}</text>')
    out.append(f'<line class="ax" x1="{left}" y1="{top + ph}" x2="{W - right}" y2="{top + ph}"/>')
    out.append(f'<text class="m" x="{left + pw / 2:.1f}" y="{top + ph + 38}" font-size="12" '
               'text-anchor="middle">file size (MiB)</text>')
    out.append(f'<text class="m" x="20" y="{top - 10}" font-size="12">perplexity</text>')
    for lab, mib, p in points:
        cx, cy = x(mib), y(p)
        out.append(f'<circle class="s1 ring" cx="{cx:.1f}" cy="{cy:.1f}" r="5" stroke-width="2"/>')
        text = lab.replace("`", "")
        anchor, tx = ("end", cx - 10) if lab in LEFT_LABELS else ("start", cx + 10)
        out.append(f'<text class="t h" x="{tx:.1f}" y="{cy + 4:.1f}" font-size="12" text-anchor="{anchor}">'
                   f"{text} · {p:.2f}</text>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def changelog_page():
    with open(CHANGELOG, encoding="utf-8") as f:
        return CHANGELOG_FRONT + f.read()


def targets():
    return {
        CHART: ratio_chart("dr", ratio_groups("Decode ratio"), "Decode speed against llama.cpp, one thread",
                           "Qwen2.5 0.5B Instruct, same Q4_0 and Q8_0 formats; above 1× Krafla is faster"),
        PREFILL_CHART: ratio_chart("pr", ratio_groups("prefill"), "Prefill speed against llama.cpp, one thread",
                                   "Qwen2.5 0.5B Instruct, 128-token prompt; above 1× Krafla is faster"),
        Q4_CHART: q4_chart(q4_rows()),
        MIXED_CHART: mixed_chart(mixed_points()),
        CHANGELOG_PAGE: changelog_page(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if a derived file is not current")
    a = ap.parse_args()
    stale = []
    for path, text in targets().items():
        try:
            with open(path, encoding="utf-8") as f:
                current = f.read()
        except FileNotFoundError:
            current = None
        if current == text:
            continue
        rel = os.path.relpath(path, ROOT)
        if a.check:
            stale.append(rel)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"wrote {rel}")
    if stale:
        sys.exit("stale: " + ", ".join(stale) + " (run python3 scripts/sync_docs.py)")


if __name__ == "__main__":
    main()
