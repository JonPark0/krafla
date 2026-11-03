"""Generates a random-weight .mlb model for tests."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlb  # noqa: E402


def generate(cfg, seed=0):
    """cfg: dim, n_layers, n_heads, n_kv_heads, hidden_dim, vocab, max_seq, rope_theta,
    norm_eps -> (cfg, weights) with float32 weights scaled by 1/sqrt(fan_in)."""
    rng = np.random.default_rng(seed)
    dim, hid, hd = cfg["dim"], cfg["hidden_dim"], mlb.head_dim(cfg)
    qd, kv = hd * cfg["n_heads"], hd * cfg["n_kv_heads"]

    def mat(rows, cols):
        return (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)

    def norm(n=dim):
        return (1.0 + 0.01 * rng.standard_normal(n)).astype(np.float32)

    w = {"tok_emb": (0.5 * rng.standard_normal((cfg["vocab"], dim))).astype(np.float32), "layers": []}
    nk, nv, dk, dv, kc, conv, vw = mlb.lin_dims(cfg)
    for li in range(cfg["n_layers"]):
        if mlb.layer_linear(cfg, li):   # gated delta rule (Qwen3.5)
            w["layers"].append({"attn_norm": norm(), "lqkv": mat(conv, dim), "lz": mat(vw, dim),
                                "lb": mat(nv, dim), "la": mat(nv, dim),
                                "lconv": (0.5 * rng.standard_normal((conv, kc))).astype(np.float32),
                                "ldt": (0.5 * rng.standard_normal(nv)).astype(np.float32),
                                "lalog": np.log(rng.uniform(0.5, 4.0, nv)).astype(np.float32),
                                "lnorm": norm(dv), "lout": mat(dim, vw), "ffn_norm": norm(),
                                "w1": mat(hid, dim), "w2": mat(dim, hid), "w3": mat(hid, dim)})
            if cfg.get("post_norms"):
                w["layers"][-1].update(post_attn_norm=norm(), post_ffn_norm=norm())
            continue
        w["layers"].append({"attn_norm": norm(), "wq": mat(qd, dim), "wk": mat(kv, dim),
                            "wv": mat(kv, dim), "wo": mat(dim, qd), "ffn_norm": norm(),
                            "w1": mat(hid, dim), "w2": mat(dim, hid), "w3": mat(hid, dim)})
        if cfg.get("attn_gate"):
            w["layers"][-1]["wg"] = mat(qd, dim)
        if cfg.get("qkv_bias"):
            w["layers"][-1].update(bq=(0.2 * rng.standard_normal(qd)).astype(np.float32),
                                   bk=(0.2 * rng.standard_normal(kv)).astype(np.float32),
                                   bv=(0.2 * rng.standard_normal(kv)).astype(np.float32))
        if cfg.get("post_norms"):
            w["layers"][-1].update(post_attn_norm=norm(), post_ffn_norm=norm())
        if cfg.get("qk_norm"):   # spread out, so that a wrong permutation shows
            w["layers"][-1].update(q_norm=(1.0 + 0.3 * rng.standard_normal(hd)).astype(np.float32),
                                   k_norm=(1.0 + 0.3 * rng.standard_normal(hd)).astype(np.float32))
    w["final_norm"] = norm()
    w["wcls"] = mat(cfg["vocab"], dim)
    cfg = dict(cfg, shared_classifier=0)
    return cfg, w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wtype", choices=("q8", "q4"), default="q8")
    ap.add_argument("--q8", default="", metavar="LIST",
                    help="q4 only: matrix classes kept in Q8, comma-separated from "
                         "emb, cls, qk, v, o, up, down (e.g. emb,cls,v,down)")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=96)
    ap.add_argument("--max-seq", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scales", choices=("f32", "f16"), default="f32", help="storage of the block scales")
    ap.add_argument("--qkv-bias", action="store_true", help="Q/K/V projection biases (Qwen2)")
    ap.add_argument("--head-dim", type=int, default=0, help="head size other than dim / heads (Qwen3)")
    ap.add_argument("--qk-norm", action="store_true", help="RMSNorm of each query and key head (Qwen3)")
    ap.add_argument("--sliding-window", type=int, default=0, help="attend to this many positions at most")
    ap.add_argument("--layer-pattern", type=int, default=0, metavar="N",
                    help="with --sliding-window: only layers other than every N-th slide (Gemma 3), "
                         "and with --rope-local-theta use that base")
    ap.add_argument("--rope-local-theta", type=float, default=0.0, help="RoPE base of the sliding layers")
    ap.add_argument("--linear-every", type=int, default=0, metavar="N",
                    help="linear-attention (gated delta rule) layers, all but every N-th (Qwen3.5: 4)")
    ap.add_argument("--lin-heads", default="2,4", metavar="K,V", help="linear attention key and value heads")
    ap.add_argument("--lin-dims", default="16,16", metavar="DK,DV", help="their head sizes")
    ap.add_argument("--lin-conv", type=int, default=4, help="conv kernel size")
    ap.add_argument("--rot-dim", type=int, default=0, help="RoPE on the first N dimensions of each head only")
    ap.add_argument("--attn-gate", action="store_true", help="attention output gated by a projection (Qwen3.5)")
    ap.add_argument("--nope-every", type=int, default=0, metavar="N",
                    help="every N-th layer leaves q and k unrotated (NoPE, SmolLM3)")
    ap.add_argument("--gelu", action="store_true", help="GELU (tanh) FFN activation instead of SiLU (Gemma)")
    ap.add_argument("--post-norms", action="store_true", help="RMSNorm of each block's output (Gemma 3)")
    ap.add_argument("--embed-scale", type=float, default=0.0, help="factor on the embedding rows (Gemma: sqrt(dim))")
    ap.add_argument("--attn-scale", type=float, default=0.0, help="attention score factor instead of 1/sqrt(head_dim)")
    ap.add_argument("--attn-softcap", type=float, default=0.0, help="soft-cap attention scores at this value (Gemma 2)")
    ap.add_argument("--final-softcap", type=float, default=0.0, help="soft-cap the logits at this value (Gemma 2)")
    ap.add_argument("--rope-scaling", default=None, metavar="F,LO,HI,ORIG",
                    help="llama3 RoPE scaling: factor, low/high freq factors, original context")
    ap.add_argument("--rope-linear", type=float, default=None, metavar="F", help="linear RoPE scaling")
    ap.add_argument("--rope-yarn", default=None, metavar="F,ORIG",
                    help="yarn RoPE scaling (beta_fast 32, beta_slow 1, default attention factor)")
    ap.add_argument("--dump-emb-row", type=int, default=None,
                    help="also write <out>.emb<N>.f32: dequantized embedding row N")
    a = ap.parse_args()
    cfg = dict(dim=a.dim, n_layers=a.layers, n_heads=a.heads, n_kv_heads=a.kv_heads,
               hidden_dim=a.hidden, vocab=a.vocab, max_seq=a.max_seq,
               rope_theta=10000.0, norm_eps=1e-5, qkv_bias=int(a.qkv_bias), head_dim=a.head_dim or None,
               qk_norm=int(a.qk_norm), sliding_window=a.sliding_window, act="gelu_tanh" if a.gelu else "silu",
               post_norms=int(a.post_norms), embed_scale=a.embed_scale, attn_scale=a.attn_scale,
               attn_softcap=a.attn_softcap, final_softcap=a.final_softcap)
    if a.layer_pattern:   # layer i slides unless (i + 1) % N == 0
        sl = [(i + 1) % a.layer_pattern != 0 for i in range(a.layers)]
        cfg["layer_flags"] = [(mlb.LAYER_SLIDING if s else 0) | (mlb.LAYER_LOCAL_ROPE if s and a.rope_local_theta else 0)
                              for s in sl]
        cfg["rope_local_theta"] = a.rope_local_theta or None
    if a.rot_dim:
        cfg["rot_dim"] = a.rot_dim
    if a.attn_gate:
        cfg["attn_gate"] = 1
    if a.linear_every:   # layer i is linear unless (i + 1) % N == 0
        nk, nv = (int(x) for x in a.lin_heads.split(","))
        dk, dv = (int(x) for x in a.lin_dims.split(","))
        cfg.update(lin_k_heads=nk, lin_v_heads=nv, lin_k_dim=dk, lin_v_dim=dv, lin_conv=a.lin_conv,
                   layer_flags=[0 if (i + 1) % a.linear_every == 0 else mlb.LAYER_LINEAR for i in range(a.layers)])
    if a.nope_every:   # layer i has no RoPE when (i + 1) % N == 0
        flags = cfg.get("layer_flags") or [0] * a.layers
        cfg["layer_flags"] = [f | (mlb.LAYER_NO_ROPE if (i + 1) % a.nope_every == 0 else 0) for i, f in enumerate(flags)]
    if a.rope_scaling:
        f, lo, hi, orig = a.rope_scaling.split(",")
        cfg["rope_scaling"] = {"type": "llama3", "factor": float(f), "low_freq_factor": float(lo),
                               "high_freq_factor": float(hi), "original_max_position_embeddings": int(orig)}
    if a.rope_linear:
        cfg["rope_scaling"] = {"type": "linear", "factor": a.rope_linear}
    if a.rope_yarn:
        f, orig = a.rope_yarn.split(",")
        cfg["rope_scaling"] = {"type": "yarn", "factor": float(f), "beta_fast": 32.0, "beta_slow": 1.0,
                               "original_max_position_embeddings": int(orig),
                               "attention_factor": 0.1 * np.log(float(f)) + 1.0}
    cfg, w = generate(cfg, a.seed)
    mlb.write_mlb(a.out, cfg, w, a.wtype, [c for c in a.q8.split(",") if c], scales=a.scales)
    if a.dump_emb_row is not None:
        _, t = mlb.read_mlb(a.out)
        row = mlb.dequant(t["tok_emb"])[a.dump_emb_row].astype(np.float32)
        row.tofile(f"{a.out}.emb{a.dump_emb_row}.f32")


if __name__ == "__main__":
    main()
