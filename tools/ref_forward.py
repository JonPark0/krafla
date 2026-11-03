"""Numpy reference forward pass with the runtime's exact quantized semantics.

Validates the scalar C build: activations are re-quantized to Q8_0 before every
matvec, block dot products are integer, scales are applied in float.
Written for clarity, not speed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlb  # noqa: E402

QK = mlb.QK


def _int_weights(t):
    """QT -> int32 array [rows, nblocks, 32]."""
    rows, cols = t.shape
    if t.type == mlb.WTYPES["q8"]:
        return t.q.reshape(rows, cols // QK, QK).astype(np.int32)
    p = t.q.reshape(rows, cols // QK, QK // 2).astype(np.int32)
    return np.concatenate([(p & 0xF) - 8, (p >> 4) - 8], axis=-1)


class Model:
    def __init__(self, cfg, t):
        self.cfg, self.t = cfg, t
        self._cache = {}

    def matvec(self, w, x):
        """w: QT, x: float32[cols] -> float32[rows]."""
        key = id(w)
        if key not in self._cache:
            self._cache[key] = (_int_weights(w), w.d.astype(np.float64))
        wi, wd = self._cache[key]
        q, d = mlb.quantize_q8(x[None, :])
        xi = q[0].reshape(-1, QK).astype(np.int32)
        sums = np.einsum("rbj,bj->rb", wi, xi)
        return (sums * wd * d[0].astype(np.float64)).sum(axis=1).astype(np.float32)


def quantize_rows(x, n):
    """x[..., k] float32 -> (int8 q, float32 d) per row of n, the quantize_q8 rule
    with one block of n (the runtime's Q8 KV cache and quantized query heads)."""
    x = np.asarray(x, np.float32)
    b = x.reshape(-1, n)
    amax = np.abs(b).max(axis=-1)
    amax = np.where(amax < mlb.TINY, np.float32(0), amax).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(amax != 0, np.float32(127) / amax, np.float32(0)).astype(np.float32)
    d = (amax / np.float32(127)).astype(np.float32)
    q = np.clip(mlb._round_half_away((b * idv[:, None]).astype(np.float32)), -127, 127)
    return q.astype(np.int8), d


def rmsnorm(x, w, eps):
    x64 = x.astype(np.float64)
    return (x64 / np.sqrt(np.mean(x64 * x64) + eps) * w).astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def rope_tables(cfg, local=False):
    if local:   # the local RoPE base, without scaling (Gemma 3's sliding layers)
        cfg = dict(cfg, rope_theta=cfg["rope_local_theta"], rope_scaling=None)
    freq, af = mlb.rope_inv_freq(cfg), mlb.rope_attention_factor(cfg)
    ang = np.arange(cfg["max_seq"], dtype=np.float64)[:, None] * freq[None, :]
    return (np.cos(ang) * af).astype(np.float32), (np.sin(ang) * af).astype(np.float32)


def rope(v, c, s, hd, rd=None):
    """Rotates pairs (2i, 2i+1) of the first rd (default hd) dimensions of each head."""
    rd = rd or hd
    v = np.asarray(v, np.float32).reshape(-1, hd).copy()
    r = v[:, :rd].reshape(-1, rd // 2, 2).astype(np.float64)
    a, b = r[..., 0], r[..., 1]
    v[:, :rd] = np.stack([a * c - b * s, a * s + b * c], axis=-1).reshape(-1, rd).astype(np.float32)
    return v.reshape(-1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def silu(x):
    x = np.asarray(x, np.float64)
    return x * sigmoid(x)


def linear_attention(m, cfg, L, h, st):
    """One position of a gated-delta-rule layer (Qwen3.5): h is the normalized
    input; st holds the layer's conv window and recurrent state, updated."""
    nk, nv, dk, dv, kc, conv, vw = mlb.lin_dims(cfg)
    eps = float(cfg["norm_eps"])
    qkv = m.matvec(L["lqkv"], h)
    z = m.matvec(L["lz"], h).astype(np.float64)
    b = m.matvec(L["lb"], h)
    a = m.matvec(L["la"], h)
    win = np.concatenate([st["conv"], qkv.astype(np.float64)[:, None]], axis=1)   # [conv, kc]
    st["conv"] = win[:, 1:]
    co = silu((win * np.asarray(L["lconv"], np.float64)).sum(axis=1))
    q, k, v = co[:nk * dk].reshape(nk, dk), co[nk * dk:2 * nk * dk].reshape(nk, dk), co[2 * nk * dk:].reshape(nv, dv)
    beta = sigmoid(b)
    g = -np.exp(np.asarray(L["lalog"], np.float64)) * np.logaddexp(0.0, a.astype(np.float64) + L["ldt"])
    o = np.zeros(vw, np.float32)
    for hh in range(nv):
        kh = hh // (nv // nk)
        qn = q[kh] / np.sqrt((q[kh] * q[kh]).sum() + 1e-6) / np.sqrt(dk)
        kn = k[kh] / np.sqrt((k[kh] * k[kh]).sum() + 1e-6)
        S = st["S"][hh] * np.exp(g[hh])
        delta = (v[hh] - kn @ S) * beta[hh]
        S = S + np.outer(kn, delta)
        st["S"][hh] = S
        out = qn @ S
        zz = z[hh * dv:(hh + 1) * dv]
        o[hh * dv:(hh + 1) * dv] = (L["lnorm"] * (out / np.sqrt(np.mean(out * out) + eps)) * silu(zz)).astype(np.float32)
    return m.matvec(L["lout"], o)


def ffn_block(m, cfg, L, x, eps, gelu):
    """x after the attention residual -> x after the FFN residual."""
    h = rmsnorm(x, L["ffn_norm"], eps)
    g = m.matvec(L["w1"], h).astype(np.float64)
    u = m.matvec(L["w3"], h).astype(np.float64)
    if gelu:   # tanh approximation (gelu_pytorch_tanh)
        g = (0.5 * g * (1.0 + np.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))) * u).astype(np.float32)
    else:
        g = (g / (1.0 + np.exp(-g)) * u).astype(np.float32)
    f = m.matvec(L["w2"], g)
    if "post_ffn_norm" in L:
        f = rmsnorm(f, L["post_ffn_norm"], eps)
    return (x + f).astype(np.float32)


def forward(cfg, t, tokens, kv="f32"):
    """-> float32 logits [len(tokens), vocab], position p attending to 0..p
    (to the last sliding_window positions, if set).
    kv="q8" emulates the runtime's Q8 KV cache (--kv q8), kv="f16" its F16
    cache (K and V rounded to half precision)."""
    m = Model(cfg, t)
    dim, nh, nkv = cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"]
    hd, eps = mlb.head_dim(cfg), float(cfg["norm_eps"])
    local = rope_tables(cfg, True) if cfg.get("rope_local_theta") else None
    group = nh // nkv
    cos_t, sin_t = rope_tables(cfg)
    rd = mlb.rot_dim(cfg)
    nk, nv, dk, dv, kc, conv, vw = mlb.lin_dims(cfg)
    lin = [{"conv": np.zeros((conv, kc - 1)), "S": np.zeros((nv, dk, dv))} if mlb.layer_linear(cfg, li) else None
           for li in range(cfg["n_layers"])]
    emb = mlb.dequant(t["tok_emb"])
    escale = np.float32(cfg.get("embed_scale") or 1.0)
    ascale = float(cfg.get("attn_scale") or 0.0) or 1.0 / np.sqrt(hd)
    acap, fcap = float(cfg.get("attn_softcap") or 0.0), float(cfg.get("final_softcap") or 0.0)
    gelu = cfg.get("act") == "gelu_tanh"
    kcache = [[] for _ in t["layers"]]
    vcache = [[] for _ in t["layers"]]
    out = []
    for pos, tok in enumerate(tokens):
        x = (emb[tok].astype(np.float32) * escale).astype(np.float32)
        for li, L in enumerate(t["layers"]):
            win = mlb.layer_window(cfg, li)
            ct, st = local if mlb.layer_local_rope(cfg, li) else (cos_t, sin_t)
            h = rmsnorm(x, L["attn_norm"], eps)
            if lin[li] is not None:
                a = linear_attention(m, cfg, L, h, lin[li])
                if "post_attn_norm" in L:
                    a = rmsnorm(a, L["post_attn_norm"], eps)
                x = ffn_block(m, cfg, L, (x + a).astype(np.float32), eps, gelu)
                continue
            q = m.matvec(L["wq"], h)
            k = m.matvec(L["wk"], h)
            v = m.matvec(L["wv"], h)
            if "bq" in L:   # float32 adds, as the runtime does
                q, k, v = q + L["bq"], k + L["bk"], v + L["bv"]
            if "q_norm" in L:   # per head, before RoPE (Qwen3)
                q = np.concatenate([rmsnorm(q[i:i + hd], L["q_norm"], eps) for i in range(0, len(q), hd)])
                k = np.concatenate([rmsnorm(k[i:i + hd], L["k_norm"], eps) for i in range(0, len(k), hd)])
            if not mlb.layer_no_rope(cfg, li):
                q = rope(q, ct[pos], st[pos], hd, rd)
                k = rope(k, ct[pos], st[pos], hd, rd)
            if kv == "f16":
                k = np.asarray(k, np.float32).astype(np.float16).astype(np.float64)
                v = np.asarray(v, np.float32).astype(np.float16).astype(np.float64)
            if kv == "q8":   # per position and KV head: int8 quants, one scale
                kq, kd = quantize_rows(k, hd)
                vq, vd = quantize_rows(v, hd)
                k = (kq.astype(np.float64) * kd[:, None].astype(np.float64)).reshape(-1)
                v = (vq.astype(np.float64) * vd[:, None].astype(np.float64)).reshape(-1)
                qq, qd = quantize_rows(q, hd)
                q = (qq.astype(np.float64) * qd[:, None].astype(np.float64)).reshape(-1)
            kcache[li].append(k)
            vcache[li].append(v)
            first = max(0, pos + 1 - win) if win else 0
            K = np.stack(kcache[li][first:]).astype(np.float64)   # [positions attended, kv_dim]
            V = np.stack(vcache[li][first:]).astype(np.float64)
            o = np.zeros(nh * hd, np.float32)
            for hh in range(nh):
                kh = hh // group
                qh = np.asarray(q[hh * hd:(hh + 1) * hd], np.float64)
                sc = K[:, kh * hd:(kh + 1) * hd] @ qh * ascale
                if acap:   # Gemma 2: scores soft-capped at acap
                    sc = acap * np.tanh(sc / acap)
                p = softmax(sc)
                o[hh * hd:(hh + 1) * hd] = p @ V[:, kh * hd:(kh + 1) * hd]
            if "wg" in L:   # Qwen3.5: the output gated by sigmoid of a projection of the input
                o = (o * sigmoid(m.matvec(L["wg"], h))).astype(np.float32)
            a = m.matvec(L["wo"], o)
            if "post_attn_norm" in L:   # Gemma 3: the block's output is normalized before the residual
                a = rmsnorm(a, L["post_attn_norm"], eps)
            x = ffn_block(m, cfg, L, (x + a).astype(np.float32), eps, gelu)
        lg = m.matvec(t["wcls"], rmsnorm(x, t["final_norm"], eps))
        if fcap:   # Gemma 2: logits soft-capped at fcap
            lg = (fcap * np.tanh(lg.astype(np.float64) / fcap)).astype(np.float32)
        out.append(lg)
    return np.stack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--tokens", required=True, help="comma-separated token ids")
    ap.add_argument("--out", required=True, help="raw float32 [n_tokens, vocab]")
    ap.add_argument("--kv", choices=["f32", "f16", "q8"], default="f32", help="KV cache type to emulate")
    a = ap.parse_args()
    cfg, t = mlb.read_mlb(a.model)
    forward(cfg, t, [int(s) for s in a.tokens.split(",")], a.kv).tofile(a.out)


if __name__ == "__main__":
    main()
