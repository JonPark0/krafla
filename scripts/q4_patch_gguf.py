"""Rewrite the Q4_0 tensors of a GGUF file with Krafla's Q4 quantizer, so that
llama.cpp's tools can score Krafla's rule on the same tensors as its own.

    scripts/q4_patch_gguf.py REF.gguf TARGET.gguf [--imatrix IMAT.gguf]
        [--rule fit|round] [--check]

REF holds the unquantized weights (F16, BF16 or F32); TARGET is a Q4_0 GGUF of
the same model made with `llama-quantize --pure`. Each of TARGET's Q4_0
tensors is quantized again from REF by tools/mlb.py's quantize_q4 (f16
scales) and written in place: Krafla's Q4 block is GGML's block_q4_0 (f16
scale, then 16 bytes holding element j in the low nibble and j + 16 in the
high one, value d * (q - 8)). --imatrix takes a llama-imatrix GGUF and gives
each matrix its mean squared inputs, as llama-quantize does; an entry whose
count is 0 gives every channel the same importance. --check compares instead
of writing: with --rule round the blocks must equal llama-quantize's Q4_0.

Needs llama.cpp's gguf-py: set GGUF_PY to its directory (llama.cpp/gguf-py)
or install the gguf package. docs/q4-scales.md describes the measurements.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402
import mlb  # noqa: E402


def read_imatrix(path):
    """llama-imatrix GGUF -> {tensor name: mean squared input per channel}."""
    t = {x.name: x for x in gguf.GGUFReader(path).tensors}
    out = {}
    for name, x in t.items():
        if not name.endswith(".in_sum2"):
            continue
        base = name[: -len(".in_sum2")]
        sums = np.asarray(x.data, np.float32).reshape(-1)
        counts = np.asarray(t[base + ".counts"].data, np.float32).reshape(-1)
        if len(counts) != 1:
            continue   # one count per expert: mixtures of experts are not handled
        out[base] = sums / counts[0] if counts[0] > 0 else np.ones_like(sums)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref")
    ap.add_argument("target")
    ap.add_argument("--imatrix")
    ap.add_argument("--rule", default="fit", choices=("fit", "round"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    ref = {x.name: x for x in gguf.GGUFReader(a.ref).tensors}
    tgt = gguf.GGUFReader(a.target)
    imat = read_imatrix(a.imatrix) if a.imatrix else {}
    mm = np.memmap(a.target, mode="r" if a.check else "r+")
    n = same = blocks = weighted = 0
    t0 = time.time()
    for x in tgt.tensors:
        if x.tensor_type != gguf.GGMLQuantizationType.Q4_0:
            continue
        r = ref[x.name]
        cols = int(x.shape[0])   # ne0, the row length
        w = dequantize(r.data, r.tensor_type).astype(np.float32).reshape(-1, cols)
        imp = imat.get(x.name)
        weighted += imp is not None
        q, d = mlb.quantize_q4(w, imp=imp, f16=True, rule=a.rule)
        nb = cols // 32
        q = np.asarray(q, np.uint8).reshape(-1, nb, 16)
        d = np.asarray(d, np.float16).reshape(-1, nb, 1).view(np.uint8)
        new = np.concatenate([d, q], axis=-1).reshape(-1)
        off = int(x.data_offset)
        old = np.asarray(mm[off: off + new.size])
        if old.size != new.size:
            sys.exit(f"{x.name}: {old.size} bytes in {a.target}, {new.size} quantized")
        if a.check:
            eq = (old.reshape(-1, 18) == new.reshape(-1, 18)).all(axis=1)
            same += int(eq.sum())
            blocks += eq.size
        else:
            mm[off: off + new.size] = new
        n += 1
    if not a.check:
        mm.flush()
    msg = f"{n} Q4_0 tensors, rule {a.rule}, importance for {weighted}, {time.time() - t0:.0f} s"
    if a.check:
        msg += f"; identical blocks {same}/{blocks}"
    print(msg)
    if a.check and same != blocks:
        sys.exit(1)


if __name__ == "__main__":
    main()
