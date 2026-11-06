"""SVG figures of the docs site: the shared chart style, and diagrams of the
concepts the pages explain. scripts/sync_docs.py writes them to docs/assets/
and checks in CI that they are current.

Every figure is drawn from data, not by hand: the Q4 rule figure quantizes its
block with tools/mlb.py, and the others take their facts from the code
(mlb.Q8_CLASSES, the release wrappers' ISA order) as noted where they do.
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mlb  # noqa: E402

# Blue, orange, aqua and yellow: the first four categorical slots of a palette
# checked for color-vision deficiency (adjacent CVD ΔE >= 8.4) in light and
# dark mode; aqua and yellow are below 3:1 on white, so they always carry a
# label. .f* are tints of the same hues for boxes behind text. The figures
# follow the reader's system setting, as images cannot see the site's switch.
STYLE = (
    ".bg{fill:#fff;stroke:#e5e7eb}.t{fill:#1f2937}.m{fill:#6b7280}.g{stroke:#eceef1}"
    ".ax{stroke:#9ca3af}.s1{fill:#2a78d6}.s2{fill:#eb6834}.s3{fill:#1baf7a}.s4{fill:#eda100}"
    ".k1{stroke:#2a78d6}.k2{stroke:#eb6834}.k3{stroke:#1baf7a}.k4{stroke:#eda100}"
    ".sx{fill:#9ca3af}.k0{stroke:#c9ccd1}.f1{fill:#e4eefb}.f2{fill:#fde9e0}.f3{fill:#dcf4ea}.f4{fill:#fdf1d6}.f0{fill:#f3f4f6}"
    ".lb{fill:#fff}.ring{stroke:#fff}.cn{stroke:#c9ccd1}.ln{stroke:#9ca3af;fill:none}"
    ".box{fill:#fff;stroke:#c9ccd1}.on{fill:#fff}.ondk{fill:#1f2937}.ah{fill:#9ca3af}"
    ".h{paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}"
    ".mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}\n"
    "@media (prefers-color-scheme:dark){.bg{fill:#1f2023;stroke:#3a3b40}.t{fill:#e5e7eb}"
    ".m{fill:#9ca3af}.g{stroke:#2e3035}.ax{stroke:#6b7280}.s1{fill:#3987e5}.s2{fill:#d95926}"
    ".sx{fill:#6b7280}.k0{stroke:#4b4f57}.s3{fill:#199e70}.s4{fill:#c98500}.k1{stroke:#3987e5}.k2{stroke:#d95926}.k3{stroke:#199e70}"
    ".k4{stroke:#c98500}.f1{fill:#1d2d45}.f2{fill:#3b2519}.f3{fill:#16362b}.f4{fill:#382d12}"
    ".f0{fill:#2a2c31}.lb{fill:#1f2023}.ring{stroke:#1f2023}.cn{stroke:#4b4f57}"
    ".ln{stroke:#6b7280}.box{fill:#1f2023;stroke:#4b4f57}.ah{fill:#6b7280}.h{stroke:#1f2023}}"
)

ARROW = ('<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path class="ah" d="M0,0L8,4L0,8z"/></marker></defs>')


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_head(cid, W, H, title, subtitle, desc, legend=()):
    """Opening of a figure: card, title, subtitle and a legend of (class, label, mark)
    at the top right, mark "bar", "dot" or "box"."""
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-labelledby="{cid}-title {cid}-desc" font-family="system-ui, -apple-system, '
        'Segoe UI, Roboto, Helvetica, Arial, sans-serif">',
        f'<title id="{cid}-title">{esc(title)}</title>',
        f'<desc id="{cid}-desc">{esc(desc)}</desc>',
        f"<style>\n{STYLE}\n</style>",
        ARROW,
        f'<rect class="bg" x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="8"/>',
        f'<text class="t" x="20" y="32" font-size="17" font-weight="600">{esc(title)}</text>',
        f'<text class="m" x="20" y="54" font-size="13">{esc(subtitle)}</text>',
    ]
    x = W - 20
    for cls, label, mark in reversed(legend):
        x -= 7.2 * len(label) + 30
        if mark == "dot":
            out.append(f'<circle class="{cls}" cx="{x + 6}" cy="74" r="5"/>')
        elif mark == "box":
            k = cls.replace("f", "k")
            out.append(f'<rect class="{cls} {k}" x="{x}" y="68" width="12" height="12" rx="3"/>')
        else:
            out.append(f'<rect class="{cls}" x="{x}" y="68" width="12" height="12" rx="3"/>')
        out.append(f'<text class="t" x="{x + 17}" y="79" font-size="13">{esc(label)}</text>')
    return out


def text(x, y, s, size=12, cls="t", anchor="start", weight=None, extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}"{a}{w}{extra}>{esc(s)}</text>'


def rect(x, y, w, h, cls, rx=4, extra=""):
    return f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"{extra}/>'


def line(x1, y1, x2, y2, cls="ln", width=1, arrow=False, extra=""):
    m = ' marker-end="url(#ah)"' if arrow else ""
    return (f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke-width="{width}"{m}{extra}/>')


def hbar(cls, x0, y, w, h, r=4):
    """A bar from the baseline x0, square there, rounded at its data end."""
    r = min(r, w / 2, h / 2)
    return (f'<path class="{cls}" d="M{x0:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}'
            f'a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.1f}z"/>')


def close(out):
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- Q4_0 block

def q4_block():
    """Byte layout of one Q4_0 block: 32 weights, 16 bytes of nibbles, one scale."""
    W, H, x0 = 760, 396, 56
    out = svg_head("q4b", W, H, "One Q4_0 block: 32 weights in 18 bytes",
                   "Byte j holds weight j in its low nibble and weight j + 16 in its high nibble",
                   "A block of 32 weights of one row. Weights 0 to 15 go to the low nibbles of bytes 0 to 15, "
                   "weights 16 to 31 to their high nibbles. One f16 scale d per block. Each weight is "
                   "d times (nibble minus 8). 16 bytes plus a 2-byte scale per 32 weights is 4.5 bits per weight.",
                   (("f1", "weights 0–15 · low nibble", "box"), ("f2", "weights 16–31 · high nibble", "box")))
    cw, cy, ch = 20, 108, 24
    out.append(text(x0, cy - 10, "32 consecutive weights of one row", 12, "m"))
    for j in range(32):
        f, k = ("f1", "k1") if j < 16 else ("f2", "k2")
        out.append(rect(x0 + j * cw + 1, cy, cw - 2, ch, f"{f} {k}", 3))
        out.append(text(x0 + j * cw + cw / 2, cy + 16, str(j), 10, "t", "middle"))
    bw, by, bh = 40, 222, 30
    for j in range(32):    # a thread from each weight to its nibble
        k = "k1" if j < 16 else "k2"
        b = j % 16
        tx = x0 + b * bw + (bw * 0.75 if j < 16 else bw * 0.25)
        out.append(line(x0 + j * cw + cw / 2, cy + ch, tx, by, k, 1, extra=' opacity="0.55"'))
    for b in range(16):
        x = x0 + b * bw
        out.append(rect(x + 1, by, bw / 2 - 1, bh, "f2 k2", 0))
        out.append(rect(x + bw / 2, by, bw / 2 - 1, bh, "f1 k1", 0))
        out.append(text(x + bw * 0.25 + 0.5, by + 19, str(b + 16), 10, "t", "middle"))
        out.append(text(x + bw * 0.75, by + 19, str(b), 10, "t", "middle"))
        out.append(text(x + bw / 2, by + bh + 14, f"byte {b}" if b in (0, 15) else str(b), 10, "m", "middle"))
    out.append(text(x0 + bw * 0.25, by + bh + 28, "high", 10, "m", "middle"))
    out.append(text(x0 + bw * 0.75, by + bh + 28, "low", 10, "m", "middle"))
    out.append(text(x0 + 16 * bw + 8, by + 13, "q array:", 11, "m"))
    out.append(text(x0 + 16 * bw + 8, by + 27, "16 bytes", 11, "m"))
    dy = 316
    out.append(rect(x0, dy, 76, 30, "f0 cn", 4, ' stroke-width="1"'))
    out.append(text(x0 + 38, dy + 20, "d (f16)", 12, "t", "middle", 600))
    out.append(text(x0 + 88, dy + 13, "d array: one scale per block,", 11, "m"))
    out.append(text(x0 + 88, dy + 27, "2 bytes (4 with --scales f32)", 11, "m"))
    out.append(text(x0 + 310, dy + 13, "weight = d × (nibble − 8)", 13, "t", weight=600))
    out.append(text(x0 + 310, dy + 30, "nibble 0…15 → −8…7", 12, "m"))
    out.append(text(20, H - 16, "16 + 2 bytes per 32 weights: 4.5 bits per weight (Q8_0: 32 + 2 bytes, 8.5 bits). "
                    "An .mlb matrix stores all its q bytes, then all its scales.", 11, "m"))
    return close(out)


# ------------------------------------------------------- Q4 rules, one block

def q4_rule_block():
    """The block of 10,000 Gaussian ones whose x²-weighted error gain is the median."""
    rng = np.random.default_rng(2026)
    B = rng.standard_normal((10000, 32)).astype(np.float32)
    lr, dr = mlb._q4_round(B)
    lf, df = mlb._q4_search(B)
    w = B * B
    wr = (w * (B - lr * dr[:, None]) ** 2).sum(1)
    wf = (w * (B - lf * df[:, None]) ** 2).sum(1)
    gain = 1 - wf / wr
    i = int(np.argsort(np.abs(gain - np.median(gain)))[0])
    return B[i], (lr[i], float(dr[i])), (lf[i], float(df[i])), float(np.median(gain))


def q4_rules():
    x, (lr, dr), (lf, df), med = q4_rule_block()
    W, H = 760, 420
    left, right = 150, 196
    M = float(np.abs(x).max()) * 1.12
    sx = (W - left - right) / (2 * M)
    X = lambda v: left + (v + M) * sx                                  # noqa: E731
    yr, yw, yf = 132, 222, 312
    wr = float((x * x * (x - lr * dr) ** 2).sum())
    wf = float((x * x * (x - lf * df) ** 2).sum())
    er = float(((x - lr * dr) ** 2).sum())
    ef = float(((x - lf * df) ** 2).sum())
    big = set(np.argsort(-np.abs(x))[:8].tolist())
    desc = (f"32 Gaussian weights. Rounded: d = {dr:.4f}, the largest weight lands exactly on a level. "
            f"Fitted: d = {df:.4f}. Error weighted by x squared: {wr:.3f} rounded, {wf:.3f} fitted "
            f"({(wf / wr - 1) * 100:+.0f}%); plain squared error {er:.3f} and {ef:.3f}.")
    out = svg_head("q4r", W, H, "Rounded against fitted scales, one block",
                   f"32 Gaussian weights: the block at the median gain of 10,000 such blocks "
                   f"(x²-weighted error {med * 100:.0f}% lower)", desc,
                   (("sx", "rounded level", "bar"), ("s1", "fitted level", "bar")))
    # level rows
    for y, d, cls, k, label, sub in ((yr, dr, "sx", "ax", "Rounded", "llama.cpp: d = largest ÷ −8"),
                                      (yf, df, "s1", "k1", "Fitted", "Krafla: d by least squares")):
        lv = [d * (q - 8) for q in range(16)]
        out.append(line(X(min(lv)), y, X(max(lv)), y, "g", 1))
        for v in lv:
            out.append(f'<rect class="{cls}" x="{X(v) - 1.5:.1f}" y="{y - 8}" width="3" height="16" rx="1.5"/>')
        out.append(text(20, y - 2, label, 13, "t", weight=600))
        out.append(text(20, y + 14, sub, 11, "m"))
    # weights, stacked where they would overlap
    placed = []
    pos = {}
    for j in np.argsort(x):
        px = X(float(x[j]))
        k = 0
        while any(abs(px - qx) < 9 and qk == k for qx, qk in placed):
            k += 1
        placed.append((px, k))
        off = ((k + 1) // 2) * 9 * (1 if k % 2 else -1)
        pos[int(j)] = (px, yw + off)
    out.append(text(20, yw + 4, "Weights", 13, "t", weight=600))
    out.append(text(20, yw + 20, "8 largest in bold", 11, "m"))
    for j, (px, py) in pos.items():
        b = j in big
        wd, op = (1.6, "1") if b else (1, "0.45")
        out.append(line(px, py, X(float(lr[j] * dr)), yr + 8, "ax", wd, extra=f' opacity="{op}"'))
        out.append(line(px, py, X(float(lf[j] * df)), yf - 8, "k1", wd, extra=f' opacity="{op}"'))
    for j, (px, py) in pos.items():
        cls = "t" if j in big else "m"
        out.append(f'<circle class="{cls} ring" cx="{px:.1f}" cy="{py:.1f}" r="{4.5 if j in big else 3.5}" '
                   'stroke-width="1.5"/>')
    # numbers on the right
    rx = W - right + 18
    out.append(text(rx, yr - 6, f"d = {dr:.4f}", 12))
    pct = lambda a, b: f"{(a / b - 1) * 100:+.0f}%".replace("-", "−")    # noqa: E731
    out.append(text(rx, yr + 10, f"x²-weighted error {wr:.3f}", 11, "m"))
    out.append(text(rx, yr + 24, f"squared error {er:.3f}", 11, "m"))
    out.append(text(rx, yf - 6, f"d = {df:.4f}", 12))
    out.append(text(rx, yf + 10, f"x²-weighted error {wf:.3f}, {pct(wf, wr)}", 11, "m"))
    out.append(text(rx, yf + 24, f"squared error {ef:.3f}, {pct(ef, er)}", 11, "m"))
    # value axis
    ya = yf + 40
    out.append(line(left, ya, W - right, ya, "ax", 1))
    for v in (-2, -1, 0, 1, 2):
        if -M <= v <= M:
            out.append(line(X(v), ya, X(v), ya + 4, "ax", 1))
            out.append(text(X(v), ya + 17, f"{v:+d}".replace("+0", "0").replace("-", "−"), 11, "m", "middle"))
    out.append(text(20, H - 30, "Each line runs from a weight to the level it is stored as; its slant is the error. "
                    "Rounding puts the largest weight exactly", 11, "m"))
    out.append(text(20, H - 15, "on a level. The fit moves the levels so the large weights, whose errors cost "
                    "most (x² weights), land closer.", 11, "m"))
    return close(out)


# ------------------------------------------------------------- weight packing

def packing():
    W, H = 760, 484
    G, S = 4, 8      # rows per group, 4-byte steps per Q8_0 block
    rows = ("s1", "s2", "s3", "s4")
    oncls = ("on", "on", "on", "ondk")    # white on blue, orange and aqua; dark on yellow
    out = svg_head("pk", W, H, "Weight packing: G rows interleaved in 4-byte steps",
                   "Q8_0, G = 4 (SSE2, SSE4.1, NEON); G is 8 for AVX2 and AVX-VNNI, 16 for AVX-512 VNNI",
                   "Row-major order stores row 0's 32 bytes, then row 1's. The packed order stores step 0 "
                   "(bytes 0 to 3) of rows 0 to 3, then step 1 of rows 0 to 3, and so on. One 16-byte load "
                   "then holds 4 bytes of each of the 4 rows, and one dot instruction against 4 activation "
                   "bytes broadcast to every lane adds to all 4 rows' sums.",
                   tuple((rows[r], f"row {r}", "bar") for r in range(G)))
    x0, cw, ch = 60, 20, 22

    def strip(y, order, label, sub):
        out.append(text(20, y - 22, label, 13, "t", weight=600))
        out.append(text(20 + 7.4 * len(label) + 12, y - 22, sub, 11, "m"))
        for i, (r, s) in enumerate(order):
            out.append(rect(x0 + i * cw + 1, y, cw - 2, ch, rows[r], 3))
            out.append(text(x0 + i * cw + cw / 2, y + 15, str(s), 10, oncls[r], "middle"))
        out.append(line(x0, y + ch + 8, x0 + len(order) * cw, y + ch + 8, "ln", 1, True))
        out.append(text(x0 + len(order) * cw + 6, y + ch + 12, "memory", 11, "m"))

    ya, yb = 116, 202
    strip(ya, [(r, s) for r in range(G) for s in range(S)], "Row-major (the file)",
          "each row's block in turn; the digit is the 4-byte step")
    strip(yb, [(r, s) for s in range(S) for r in range(G)], "Packed (in memory, at load)",
          "step s of every row in the group, then step s + 1")
    # brace over the first load
    out.append(f'<path class="ln" d="M{x0 + 1},{yb - 6}v-5h{G * cw - 2}v5" stroke-width="1.2"/>')
    out.append(text(x0 + G * cw + 8, yb - 8, "one 16-byte load: 4 bytes of each row", 11, "m"))
    # one instruction
    yc = 296
    out.append(text(20, yc - 14, "One dot instruction advances all G rows", 13, "t", weight=600))
    lx, lw = 60, 76
    for r in range(G):
        out.append(rect(lx + r * (lw + 4), yc, lw, 26, rows[r], 4))
        out.append(text(lx + r * (lw + 4) + lw / 2, yc + 17, f"row {r} · 4 B", 10, oncls[r], "middle"))
    out.append(text(lx + 2 * (lw + 4) - 2, yc + 44, "×", 16, "m", "middle"))
    for r in range(G):
        out.append(rect(lx + r * (lw + 4), yc + 54, lw, 26, "f0 cn", 4, ' stroke-width="1"'))
        out.append(text(lx + r * (lw + 4) + lw / 2, yc + 71, "x[c…c+3]", 10, "t", "middle"))
    out.append(text(lx + G * (lw + 4) + 6, yc + 71, "activation bytes, broadcast", 11, "m"))
    out.append(line(lx + G * (lw + 4) + 10, yc + 13, lx + G * (lw + 4) + 60, yc + 13, "ln", 1.2, True))
    ix = lx + G * (lw + 4) + 66
    out.append(text(ix, yc + 9, "vpdpbusd (VNNI) · sdot (NEON dotprod)", 11, "t"))
    out.append(text(ix, yc + 23, "pmaddubsw + pmaddwd (SSE4.1, AVX2)", 11, "t"))
    ys = yc + 104
    for r in range(G):
        out.append(rect(lx + r * (lw + 4), ys, lw, 26, rows[r], 4, ' opacity="0.9"'))
        out.append(text(lx + r * (lw + 4) + lw / 2, ys + 17, f"sum, row {r}", 10, oncls[r], "middle"))
    for r in range(G):
        cx = lx + r * (lw + 4) + lw / 2
        out.append(line(cx, yc + 82, cx, ys - 3, "ln", 1, True))
    out.append(text(lx + G * (lw + 4) + 6, ys + 12, "one int32 sum per row, in its own 32-bit lane;", 11, "m"))
    out.append(text(lx + G * (lw + 4) + 6, ys + 26, "the group's block scales sit together, so scaling", 11, "m"))
    out.append(text(lx + G * (lw + 4) + 6, ys + 40, "them is one vector multiply-add", 11, "m"))
    out.append(text(20, H - 15, "SSE2 has no 8-bit multiply: it splits each 16-bit lane into its two bytes and "
                    "uses pmaddwd, on the same layout.", 11, "m"))
    return close(out)


# --------------------------------------------------- one layer, matrix classes

def layer_flow():
    W, H = 760, 474
    cls_of = mlb._CLASS_OF
    q8_first = ("v", "o", "down")     # the classes the user guide keeps in Q8 first
    out = svg_head("ly", W, H, "One token through a layer: where each matrix class sits",
                   "Every matrix is an integer matvec on Q8 activations; tags are the --q8 class names",
                   "Embedding row (class emb) to the residual stream x. Attention: RMSNorm, quantize to Q8, "
                   "Wq and Wk (class qk) and Wv (class v), RoPE, attention over the KV cache, quantize, Wo "
                   "(class o), added to x. Feed-forward: RMSNorm, quantize, W1 and W3 (class up), SiLU and "
                   "multiply, quantize, W2 (class down), added to x. After the last layer: RMSNorm, quantize, "
                   "Wcls (class cls) to the logits. v, o and down are the classes kept in Q8 by --q8 v,o,down.",
                   (("f0", "matrix · --q8 class", "box"), ("f1", "kept by --q8 v,o,down", "box")))

    def op(x, y, w, label, h=30):
        out.append(rect(x, y, w, h, "box", 6, ' stroke-width="1"'))
        out.append(text(x + w / 2, y + h / 2 + 4, label, 11, "t", "middle"))
        return x + w

    def mat(x, y, w, names, h=30):
        c = cls_of[names[0]]
        f, k = ("f1", "k1") if c in q8_first else ("f0", "cn")
        label = ", ".join({"wq": "Wq", "wk": "Wk", "wv": "Wv", "wo": "Wo", "w1": "W1", "w3": "W3", "w2": "W2",
                           "wcls": "Wcls", "tok_emb": "tok_emb"}[n] for n in names)
        out.append(rect(x, y, w, h, f"{f} {k}", 6, ' stroke-width="1.2"'))
        out.append(text(x + 8, y + h / 2 + 4, label, 11, "t", weight=600))
        out.append(text(x + w - 8, y + h / 2 + 4, c, 11, "m", "end", extra=' font-style="italic"'))
        return x + w

    def arrow(x1, x2, y):
        out.append(line(x1 + 1, y, x2 - 3, y, "ln", 1.2, True))

    XL, XR, XT = 44, 700, 724      # residual stream: left start, the + node, the turn on the right

    def residual_row(y, label, steps):
        """steps: list of (kind, width, payload); kind op|mat|pair. Residual line above the path."""
        yres, ypath = y, y + 46
        out.append(line(XL, yres, XR - 8, yres, "ax", 1.5))
        out.append(line(70, yres, 70, ypath + 15, "ln", 1.2))
        out.append(text(80, yres + 20, label, 12, "m", weight=600))
        x = 70
        cy = ypath + 15
        for i, (kind, w, payload) in enumerate(steps):
            nx = x + (14 if i else 0)
            if i:
                arrow(x, nx, cy)
            if kind == "op":
                x = op(nx, ypath, w, payload)
            elif kind == "mat":
                x = mat(nx, ypath, w, payload)
            else:     # two matrices stacked, fed by the same input
                mat(nx, ypath - 19, w, payload[0])
                mat(nx, ypath + 19, w, payload[1])
                out.append(line(nx - 6, ypath - 4, nx - 6, ypath + 34, "ln", 1.2))
                x = nx + w
        # back up to the residual stream
        out.append(line(x, cy, XR, cy, "ln", 1.2))
        out.append(line(XR, cy, XR, yres + 9, "ln", 1.2, True))
        out.append(f'<circle class="box" cx="{XR}" cy="{yres}" r="8" stroke-width="1.2"/>')
        out.append(text(XR, yres + 4, "+", 13, "t", "middle", 600))

    def turn(y_from, y_to, x_to):
        """From the + node at y_from round the right side and back to x_to at y_to."""
        yb = y_to - 12
        out.append(f'<path class="ln" d="M{XR + 8},{y_from}H{XT}V{yb}H{x_to}V{y_to - 2}" stroke-width="1.5" '
                   'marker-end="url(#ah)"/>')
        out.append(text(XT - 6, y_from - 6, "x", 12, "m", "end"))

    # embedding
    ye = 88
    mat(XL, ye, 150, ["tok_emb"], 28)
    out.append(text(XL + 160, ye + 18, "the token's row becomes the residual stream x", 11, "m"))
    out.append(line(XL + 16, ye + 28, XL + 16, 148, "ln", 1.5, True))
    residual_row(150, "Attention", [
        ("op", 64, "RMSNorm"), ("op", 30, "Q8"), ("pair", 100, (["wq", "wk"], ["wv"])),
        ("op", 44, "RoPE"), ("op", 124, "attention · KV cache"), ("op", 30, "Q8"), ("mat", 70, ["wo"])])
    turn(150, 272, XL + 16)
    residual_row(272, "Feed-forward", [
        ("op", 64, "RMSNorm"), ("op", 30, "Q8"), ("pair", 100, (["w1"], ["w3"])),
        ("op", 92, "SiLU(W1)·W3"), ("op", 30, "Q8"), ("mat", 70, ["w2"])])
    out.append('<path class="ln" d="M24,144v228" stroke-width="1"/>')
    out.append(text(14, 258, "× n_layers", 11, "m", "middle", extra=' transform="rotate(-90 14 258)"'))
    # classifier
    yc = 400
    turn(272, yc, XL + 16)
    x = op(XL, yc, 64, "RMSNorm")
    arrow(x, x + 14, yc + 15)
    x = op(x + 14, yc, 30, "Q8")
    arrow(x, x + 14, yc + 15)
    x = mat(x + 14, yc, 90, ["wcls"])
    arrow(x, x + 14, yc + 15)
    out.append(text(x + 18, yc + 19, "logits", 12, "t", weight=600))
    out.append(text(20, H - 16, "A matvec multiplies int8 activations by int8 or 4-bit weights, sums each 32-weight "
                    "block exactly in int32, then scales it by d_w · d_x.", 11, "m"))
    return close(out)


# ------------------------------------------------------------------ ISA ladder

X86 = [("avx512vnni", "avx512_vnni", "Intel Cascade Lake, Ice Lake, Sapphire Rapids;", "AMD Zen 4 and later"),
       ("avxvnni", "avx_vnni", "Intel Alder Lake and later, AMD Zen 5", ""),
       ("avx2", "avx2", "Intel Haswell and later, AMD Zen and later", ""),
       ("sse41", "sse4_1", "Intel Penryn/Nehalem and later,", "AMD Bulldozer and later"),
       ("sse2", "sse2", "every x86-64 CPU", "")]
ARM = [("neon_i8mm", "i8mm", "Cortex-A510/A710/X2, Neoverse N2/V1,", "Apple M2, Graviton 3 and later"),
       ("sve", "sve", "Neoverse V1/V2/N2, A64FX, Graviton 3", "(uses the dotprod kernels)"),
       ("neon_dotprod", "asimddp", "Cortex-A55/A76, Apple M1, Graviton 2", "and later"),
       ("neon", "asimd", "any AArch64; on ARMv7 boards with NEON (flag neon),",
        "the armv7hf archive: neon, then scalar")]


def wrapper_order():
    """The ISA lists scripts/package_release.sh gives the Linux wrappers."""
    import re
    with open(os.path.join(ROOT, "scripts", "package_release.sh"), encoding="utf-8") as f:
        s = f.read()
    x86 = re.search(r'^linux linux-x86_64 "" "([^"]+)"', s, re.M).group(1).split()
    arm = re.search(r'^linux linux-aarch64 \S+ "([^"]+)"', s, re.M).group(1).split()
    return x86, arm


def isa_ladder():
    x86_order, arm_order = wrapper_order()
    if [r[0] for r in X86] != x86_order or [r[0] for r in ARM] != arm_order:
        sys.exit(f"docfigs.py: the ISA ladder does not follow scripts/package_release.sh "
                 f"({x86_order}, {arm_order})")
    W, H = 760, 566
    out = svg_head("isa", W, H, "Which build runs: from the fastest down",
                   "The release's ml_run, ml_server and ml_bench start the first build the CPU supports",
                   "x86-64, fastest first: " + ", ".join(f"{b} ({f})" for b, f, *_ in X86)
                   + ". AArch64: " + ", ".join(f"{b} ({f})" for b, f, *_ in ARM)
                   + ". A build on a CPU without its features exits with code 3 and names the missing one.")
    ramp = ("#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4")    # blue, darker = faster

    def ladder(x, rungs, title, flag_note):
        out.append(text(x, 92, title, 14, "t", weight=600))
        out.append(text(x, 108, flag_note, 11, "m"))
        y, rh, gap = 122, 58, 18
        for i, (b, flag, l1, l2) in enumerate(rungs):
            out.append(rect(x, y, 340, rh, "box", 6, ' stroke-width="1"'))
            out.append(f'<rect x="{x}" y="{y}" width="6" height="{rh}" rx="3" fill="{ramp[i]}"/>')
            out.append(text(x + 16, y + 22, b, 14, "t mono", weight=600))
            out.append(text(x + 330, y + 22, flag, 11, "m mono", "end"))
            out.append(text(x + 16, y + 39, l1, 11, "m"))
            if l2:
                out.append(text(x + 16, y + 52, l2, 11, "m"))
            if i < len(rungs) - 1:
                out.append(line(x + 40, y + rh + 1, x + 40, y + rh + gap - 2, "ln", 1.2, True))
                out.append(text(x + 48, y + rh + 13, "not supported", 10, "m"))
            y += rh + gap
        return y

    ladder(20, X86, "x86-64 (Linux and Windows archives)", "right: the /proc/cpuinfo flag it needs")
    ladder(400, ARM, "AArch64 (linux-aarch64 archive)", "right: the /proc/cpuinfo flag it needs")
    out.append(text(20, H - 30, "KRAFLA_ISA=avx2 (for example) picks a build by hand. Started on a CPU without "
                    "its features, a build exits with code 3", 11, "m"))
    out.append(text(20, H - 15, "and names the missing one instead of crashing. Building from source: "
                    "make ISA=… with any of these, or scalar.", 11, "m"))
    return close(out)


FIGURES = {
    "q4-block.svg": q4_block,
    "q4-rules.svg": q4_rules,
    "packing.svg": packing,
    "layer-flow.svg": layer_flow,
    "isa-ladder.svg": isa_ladder,
}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "assets")
    for name, fn in FIGURES.items():
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            f.write(fn())
        print("wrote", name)
