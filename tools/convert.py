"""Converts checkpoints to .mlb.

  convert.py llama2c IN.bin OUT.mlb [--wtype q8|q4] [--q8 LIST]   llama2.c legacy (v0) export
  convert.py hf DIR OUT.mlb [--wtype q8|q4] [--q8 LIST] [--max-seq N] [--scales f16|f32]
                                                      HF Llama, Mistral, Qwen2, Qwen3, Gemma 2, Gemma 3: config.json + *.safetensors
  --imatrix FILE (q4): weight the Q4 fits by an importance matrix from
                       ml_run --imatrix, run on a Q8 conversion of the same model
  --q4-rule fit|round (q4): fit each block's scale, or llama.cpp's Q4_0 rounding;
                       the default takes round for Gemma 3 (its QAT checkpoints
                       expect it) and the fit for the others
"""
import argparse
import glob
import json
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlb  # noqa: E402


def load_llama2c(path):
    """-> (cfg, float32 weights) from a llama2.c v0 checkpoint."""
    data = np.fromfile(path, np.uint8)
    dim, hid, nl, nh, nkv, vocab, seq = struct.unpack("<7i", data[:28].tobytes())
    shared = vocab > 0
    vocab = abs(vocab)
    hd = dim // nh
    kv = hd * nkv
    f = data[28:].view(np.float32)
    off = 0

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        if off + n > f.size:
            raise ValueError("checkpoint truncated")
        a = f[off:off + n].reshape(shape)
        off += n
        return a

    tok = take(vocab, dim)
    stacks = {"attn_norm": take(nl, dim), "wq": take(nl, dim, dim), "wk": take(nl, kv, dim),
              "wv": take(nl, kv, dim), "wo": take(nl, dim, dim), "ffn_norm": take(nl, dim),
              "w1": take(nl, hid, dim), "w2": take(nl, dim, hid), "w3": take(nl, hid, dim)}
    final = take(dim)
    take(seq * hd // 2)   # freq_cis_real (recomputed at load)
    take(seq * hd // 2)   # freq_cis_imag
    w = {"tok_emb": tok, "final_norm": final,
         "layers": [{k: v[i] for k, v in stacks.items()} for i in range(nl)]}
    if not shared:
        w["wcls"] = take(vocab, dim)
    cfg = dict(dim=dim, n_layers=nl, n_heads=nh, n_kv_heads=nkv, hidden_dim=hid, vocab=vocab,
               max_seq=seq, rope_theta=10000.0, norm_eps=1e-5, shared_classifier=int(shared))
    return cfg, w


_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


class Lazy:
    """An array made when it is first used (np.asarray): a converter holding
    these keeps only the tensor it is writing in memory, not the checkpoint
    in float32 (a 4B model would need 16 GB)."""

    def __init__(self, fn, shape):
        self.fn, self.shape = fn, tuple(shape)

    def __array__(self, dtype=None, copy=None):
        a = self.fn()
        return a if dtype is None else a.astype(dtype, copy=False)

    def __add__(self, v):
        return Lazy(lambda: np.asarray(self) + v, self.shape)

    def __matmul__(self, o):
        return np.asarray(self) @ o

    def __rmatmul__(self, o):
        return o @ np.asarray(self)


def lazy(fn, src):
    """Lazy array fn(src array), of src's shape."""
    return Lazy(lambda: fn(np.asarray(src)), src.shape)


def read_safetensors(path):
    """-> {name: Lazy float32 array}. Supports F32, F16, BF16."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        base = 8 + n
    raw = np.memmap(path, np.uint8, mode="r")
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] not in _DTYPES:
            raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
        a, b = info["data_offsets"]

        def get(a=a, b=b, info=info):
            arr = np.frombuffer(raw[base + a:base + b], _DTYPES[info["dtype"]]).reshape(info["shape"])
            if info["dtype"] == "BF16":
                return (arr.astype(np.uint32) << 16).view(np.float32)
            return arr.astype(np.float32)
        out[name] = Lazy(get, info["shape"])
    return out


def _unrotate(w, n_heads, rd=None):
    """HF rotate_half q/k rows -> interleaved-pair rows (llama2.c convention),
    in the first rd rows of each head (default: all of them; partial rotary
    embeddings rotate only those)."""
    rows, dim = w.shape
    hd = rows // n_heads
    rd = rd or hd
    w = np.asarray(w).reshape(n_heads, hd, dim)
    rot = w[:, :rd].reshape(n_heads, 2, rd // 2, dim).transpose(0, 2, 1, 3).reshape(n_heads, rd, dim)
    return np.concatenate([rot, w[:, rd:]], axis=1).reshape(rows, dim)


HF_TYPES = ("llama", "mistral", "qwen2", "qwen3", "qwen3_5_text", "smollm3", "gemma2", "gemma3_text")


def load_qwen35(c, pre, t, max_seq=None):
    """Qwen3.5 and 3.6 (dense): linear-attention (gated delta rule) layers
    and gated attention layers, as listed in layer_types. The vision tower and
    the multi-token prediction head (mtp.*) are left out."""
    nh, nkv = c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim") or c["hidden_size"] // nh
    rp = c.get("rope_parameters") or {}
    if (rp.get("rope_type") or "default") != "default" or c.get("rope_scaling"):
        raise ValueError("RoPE scaling is not supported for Qwen3.5")
    rd = int(hd * float(c.get("partial_rotary_factor") or rp.get("partial_rotary_factor") or 1.0))
    if c.get("hidden_act", "silu") != "silu" or c.get("attention_bias"):
        raise ValueError("Qwen3.5 with an activation other than SiLU or attention biases is not supported")
    lin = [x == "linear_attention" for x in c["layer_types"]]
    if any(x not in ("linear_attention", "full_attention") for x in c["layer_types"]):
        raise ValueError(f"layer types other than linear_attention and full_attention: {set(c['layer_types'])}")
    tied = c.get("tie_word_embeddings", False) or "lm_head.weight" not in t
    n = c["num_hidden_layers"]
    cfg = dict(dim=c["hidden_size"], n_layers=n, n_heads=nh, n_kv_heads=nkv, hidden_dim=c["intermediate_size"],
               vocab=c["vocab_size"], max_seq=min(c["max_position_embeddings"], max_seq or c["max_position_embeddings"]),
               qkv_bias=0, rope_theta=float(c.get("rope_theta") or rp.get("rope_theta") or 10000.0),
               norm_eps=float(c.get("rms_norm_eps", 1e-6)), shared_classifier=int(tied), rope_scaling=None,
               head_dim=hd, qk_norm=1, sliding_window=0, act="silu", model_type="qwen3_5_text",
               rot_dim=rd if rd != hd else None, attn_gate=int(bool(c.get("attn_output_gate", True))),
               lin_k_heads=c["linear_num_key_heads"], lin_v_heads=c["linear_num_value_heads"],
               lin_k_dim=c["linear_key_head_dim"], lin_v_dim=c["linear_value_head_dim"],
               lin_conv=c["linear_conv_kernel_dim"], layer_flags=[mlb.LAYER_LINEAR if x else 0 for x in lin])
    norm = lambda v: v + 1.0   # zero-centered RMSNorm weights: x * (1 + w)
    w = {"tok_emb": t[pre + "embed_tokens.weight"], "final_norm": norm(t[pre + "norm.weight"]), "layers": []}
    for i in range(n):
        p = f"{pre}layers.{i}."
        L = {"attn_norm": norm(t[p + "input_layernorm.weight"]), "ffn_norm": norm(t[p + "post_attention_layernorm.weight"]),
             "w1": t[p + "mlp.gate_proj.weight"], "w2": t[p + "mlp.down_proj.weight"], "w3": t[p + "mlp.up_proj.weight"]}
        if lin[i]:
            a = p + "linear_attn."
            conv = t[a + "conv1d.weight"]   # [channels, 1, kernel]
            L.update(lqkv=t[a + "in_proj_qkv.weight"], lz=t[a + "in_proj_z.weight"], lb=t[a + "in_proj_b.weight"],
                     la=t[a + "in_proj_a.weight"], lout=t[a + "out_proj.weight"], ldt=t[a + "dt_bias"],
                     lalog=t[a + "A_log"], lnorm=t[a + "norm.weight"],   # the gated norm: plain weights
                     lconv=Lazy(lambda conv=conv: np.asarray(conv).reshape(conv.shape[0], -1),
                                (conv.shape[0], conv.shape[-1])))
        else:
            s = p + "self_attn."
            qg = t[s + "q_proj.weight"]   # per head: the query's head_dim rows, then the gate's
            half = (nh * hd, qg.shape[1])
            L.update(wq=Lazy(lambda qg=qg: _unrotate(np.asarray(qg).reshape(nh, 2, hd, -1)[:, 0].reshape(half), nh, rd),
                             half),
                     wg=Lazy(lambda qg=qg: np.asarray(qg).reshape(nh, 2, hd, -1)[:, 1].reshape(half), half),
                     wk=lazy(lambda x: _unrotate(x, nkv, rd), t[s + "k_proj.weight"]),
                     wv=t[s + "v_proj.weight"], wo=t[s + "o_proj.weight"],
                     q_norm=norm(lazy(lambda x: _unrotate(x[:, None], 1, rd)[:, 0], t[s + "q_norm.weight"])),
                     k_norm=norm(lazy(lambda x: _unrotate(x[:, None], 1, rd)[:, 0], t[s + "k_norm.weight"])))
        w["layers"].append(L)
    if not tied:
        w["wcls"] = t["lm_head.weight"]
    return cfg, w


def _window(c):
    """-> (window, per-layer sliding flags or None when every layer slides); (None, None) without a window."""
    w = c.get("sliding_window")
    if not w:
        return None, None
    n = c["num_hidden_layers"]
    if c.get("layer_types"):
        sliding = [t == "sliding_attention" for t in c["layer_types"]]
    elif c.get("model_type") == "gemma3_text":   # every sliding_window_pattern-th layer attends to all
        pat = c.get("sliding_window_pattern", 6)
        sliding = [(i + 1) % pat != 0 for i in range(n)]
    elif c.get("model_type") == "gemma2":   # even layers slide, odd ones attend to all
        sliding = [i % 2 == 0 for i in range(n)]
    elif c.get("model_type") in ("qwen2", "qwen3"):   # layers from max_window_layers on, if enabled
        sliding = [bool(c.get("use_sliding_window")) and i >= c.get("max_window_layers", n) for i in range(n)]
    else:
        sliding = [True] * n
    if not any(sliding):
        return None, None
    return int(w), (None if all(sliding) else sliding)


def _rope_scaling(c, hd):
    """config.json rope_scaling -> write_mlb's rope_scaling, or None; refuses unknown types."""
    rs = c.get("rope_scaling")
    if not rs:
        return None
    kind = rs.get("rope_type", rs.get("type"))
    if rs.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("partial rotary embeddings are not supported")
    if kind == "llama3":
        return {"type": "llama3", "factor": float(rs["factor"]),
                "low_freq_factor": float(rs["low_freq_factor"]),
                "high_freq_factor": float(rs["high_freq_factor"]),
                "original_max_position_embeddings": int(rs["original_max_position_embeddings"])}
    if kind == "linear":
        return {"type": "linear", "factor": float(rs["factor"])}
    if kind == "yarn":   # as transformers' _compute_yarn_parameters
        if not rs.get("truncate", True):
            raise ValueError("yarn with truncate=false is not supported")
        orig = int(rs.get("original_max_position_embeddings") or c["max_position_embeddings"])
        f = float(rs.get("factor") or c["max_position_embeddings"] / orig)

        def mscale(scale, m=1.0):
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
        af = rs.get("attention_factor")
        if af is None:
            ms, msa = rs.get("mscale"), rs.get("mscale_all_dim")
            af = mscale(f, ms) / mscale(f, msa) if ms and msa else mscale(f)
        return {"type": "yarn", "factor": f, "beta_fast": float(rs.get("beta_fast") or 32),
                "beta_slow": float(rs.get("beta_slow") or 1), "original_max_position_embeddings": orig,
                "attention_factor": float(af)}
    raise ValueError(f"rope_scaling {rs} is not supported (llama3, linear and yarn are): "
                     "the converted model would compute wrong outputs")


def load_hf(d, max_seq=None):
    """-> (cfg, float32 weights) from a HF Llama, Mistral, Qwen2, Qwen3 or Gemma 3 (text) directory."""
    with open(os.path.join(d, "config.json")) as fh:
        c = json.load(fh)
    pre = "model."   # tensor name prefix of the text model
    if c.get("model_type") == "gemma3" and "text_config" in c:
        # Gemma 3 4B and up ship as image-text models: convert the text model
        # (language_model.*) and leave the vision tower out
        c, pre = c["text_config"], "language_model.model."
        c.setdefault("model_type", "gemma3_text")
    if c.get("model_type") == "qwen3_5" and "text_config" in c:
        # Qwen3.5 and 3.6 ship as image-text models: convert the language model
        outer, c, pre = c, dict(c["text_config"]), "model.language_model."
        c.setdefault("model_type", "qwen3_5_text")
        c.setdefault("tie_word_embeddings", outer.get("tie_word_embeddings", False))
    mtype = c.get("model_type", "llama")
    if mtype == "qwen3_5_text":
        t = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            t.update(read_safetensors(f))
        if not t:
            raise FileNotFoundError(f"no .safetensors in {d}")
        return load_qwen35(c, pre, t, max_seq)
    if mtype not in HF_TYPES:
        raise ValueError(f"model_type {mtype!r} is not supported (only {', '.join(HF_TYPES)})")
    gemma = mtype in ("gemma2", "gemma3_text")
    act = c.get("hidden_activation", c.get("hidden_act", "silu"))
    if act not in ("silu", "gelu_pytorch_tanh"):
        raise ValueError(f"activation {act!r} is not supported (silu and gelu_pytorch_tanh are)")
    if c.get("mlp_bias"):
        raise ValueError("MLP biases are not supported")
    if c.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("partial rotary embeddings are not supported")
    nh = c["num_attention_heads"]
    hd = c.get("head_dim") or c["hidden_size"] // nh
    rope_scaling = _rope_scaling(c, hd)
    t = {}
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {d}")
    for f in files:
        t.update(read_safetensors(f))
    nkv = c.get("num_key_value_heads", nh)
    tied = c.get("tie_word_embeddings", False) or "lm_head.weight" not in t
    seq = min(c["max_position_embeddings"], max_seq or c["max_position_embeddings"])
    window, sliding = _window(c)
    if window and window >= seq:   # the window never cuts anything off
        window, sliding = None, None
    qk_norm = pre + "layers.0.self_attn.q_norm.weight" in t
    if (t[pre + "layers.0.self_attn.q_proj.weight"].shape[0] != nh * hd or
            t[pre + "layers.0.self_attn.k_proj.weight"].shape[0] != nkv * hd):
        raise ValueError(f"q/k projections do not match {nh} and {nkv} heads of {hd}")
    if any(k.endswith(("o_proj.bias", "mlp.gate_proj.bias", "mlp.up_proj.bias", "mlp.down_proj.bias"))
           for k in t):
        raise ValueError("biases other than on the q/k/v projections are not supported")
    bias = pre + "layers.0.self_attn.q_proj.bias" in t
    cfg = dict(dim=c["hidden_size"], n_layers=c["num_hidden_layers"], n_heads=nh, n_kv_heads=nkv,
               hidden_dim=c["intermediate_size"], vocab=c["vocab_size"], max_seq=seq, qkv_bias=int(bias),
               rope_theta=float(c.get("rope_theta", 10000.0)), norm_eps=float(c.get("rms_norm_eps", 1e-5)),
               shared_classifier=int(tied), rope_scaling=rope_scaling, head_dim=hd, qk_norm=int(qk_norm),
               sliding_window=window or 0,
               layer_flags=[mlb.LAYER_SLIDING if sl else 0 for sl in sliding] if sliding else None,
               act="gelu_tanh" if act == "gelu_pytorch_tanh" else "silu", model_type=mtype)
    if gemma:
        # sliding layers rotate with the local base; embeddings scaled by
        # sqrt(dim); scores by query_pre_attn_scalar^-0.5; norms of each block's
        # output. Gemma's RMSNorm multiplies by (1 + w): the +1 goes into the
        # stored weights.
        # Gemma 2 soft-caps the attention scores and the logits.
        cfg.update(embed_scale=float(np.float32(math.sqrt(c["hidden_size"]))),
                   attn_scale=float(c.get("query_pre_attn_scalar", hd)) ** -0.5, post_norms=1,
                   attn_softcap=float(c.get("attn_logit_softcapping") or 0.0),
                   final_softcap=float(c.get("final_logit_softcapping") or 0.0))
        if mtype == "gemma3_text":
            n = cfg["n_layers"]
            local = ([t == "sliding_attention" for t in c["layer_types"]] if c.get("layer_types") else
                     [(i + 1) % c.get("sliding_window_pattern", 6) != 0 for i in range(n)])
            # the local layers keep their RoPE base even when the context fits the window
            cfg.update(layer_flags=[(mlb.LAYER_LOCAL_ROPE | (mlb.LAYER_SLIDING if window else 0)) if x else 0
                                    for x in local],
                       rope_local_theta=float(c.get("rope_local_base_freq", 10000.0)))
    if mtype == "smollm3":   # NoPE: no_rope_layers[i] is 0 where layer i does not rotate q and k
        n = cfg["n_layers"]
        use = c.get("no_rope_layers") or [int((i + 1) % c.get("no_rope_layer_interval", 4) != 0) for i in range(n)]
        if not all(use):
            flags = list(cfg["layer_flags"] or [0] * n)
            cfg["layer_flags"] = [f | (0 if u else mlb.LAYER_NO_ROPE) for f, u in zip(flags, use)]
    norm = (lambda v: v + 1.0) if gemma else (lambda v: v)
    w = {"tok_emb": t[pre + "embed_tokens.weight"], "final_norm": norm(t[pre + "norm.weight"]), "layers": []}
    for i in range(cfg["n_layers"]):
        p = f"{pre}layers.{i}."
        w["layers"].append({
            "attn_norm": norm(t[p + "input_layernorm.weight"]),
            "wq": lazy(lambda a: _unrotate(a, nh), t[p + "self_attn.q_proj.weight"]),
            "wk": lazy(lambda a: _unrotate(a, nkv), t[p + "self_attn.k_proj.weight"]),
            "wv": t[p + "self_attn.v_proj.weight"],
            "wo": t[p + "self_attn.o_proj.weight"],
            "ffn_norm": norm(t[p + ("pre_feedforward_layernorm" if gemma else "post_attention_layernorm") + ".weight"]),
            "w1": t[p + "mlp.gate_proj.weight"],
            "w2": t[p + "mlp.down_proj.weight"],
            "w3": t[p + "mlp.up_proj.weight"],
        })
        if bias:   # q/k biases follow their rows through the RoPE permutation
            w["layers"][-1].update(bq=lazy(lambda a: _unrotate(a[:, None], nh)[:, 0], t[p + "self_attn.q_proj.bias"]),
                                   bk=lazy(lambda a: _unrotate(a[:, None], nkv)[:, 0], t[p + "self_attn.k_proj.bias"]),
                                   bv=t[p + "self_attn.v_proj.bias"])
        if qk_norm:   # per-head weights, permuted as each head's q/k rows are
            w["layers"][-1].update(q_norm=norm(lazy(lambda a: _unrotate(a[:, None], 1)[:, 0], t[p + "self_attn.q_norm.weight"])),
                                   k_norm=norm(lazy(lambda a: _unrotate(a[:, None], 1)[:, 0], t[p + "self_attn.k_norm.weight"])))
        if gemma:   # Gemma's post_attention_layernorm normalizes the attention output
            w["layers"][-1].update(post_attn_norm=norm(t[p + "post_attention_layernorm.weight"]),
                                   post_ffn_norm=norm(t[p + "post_feedforward_layernorm.weight"]))
    if not tied:
        w["wcls"] = t["lm_head.weight"]
    return cfg, w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("llama2c", "hf"))
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--wtype", choices=("q8", "q4"), default="q8")
    ap.add_argument("--q8", default="", metavar="LIST",
                    help="q4 only: matrix classes kept in Q8, comma-separated from "
                         "emb, cls, qk, v, o, up, down (e.g. emb,cls,v,down)")
    ap.add_argument("--max-seq", type=int, default=None, help="hf: cap the context length the file allows (default: the model's; "
                    "ml_run --ctx sets the KV cache size at run time)")
    ap.add_argument("--imatrix", metavar="FILE", help="q4 only: importance matrix from ml_run --imatrix")
    ap.add_argument("--q4-rule", choices=("auto", "fit", "round"), default="auto",
                    help="q4 only: fit each block's scale, or round as llama.cpp's Q4_0 does (better for "
                         "Gemma 3, and what its quantization-aware checkpoints expect); auto: round for "
                         "Gemma 3, fit otherwise")
    ap.add_argument("--scales", choices=("f16", "f32"), default="f16",
                    help="storage of the block scales (f16: Q4 4.5 bits per weight, Q8 8.5; f32: 5 and 9)")
    a = ap.parse_args(argv)
    if a.imatrix and a.wtype != "q4":
        ap.error("--imatrix needs --wtype q4")
    if a.imatrix and a.q4_rule == "round":
        ap.error("--imatrix weights the fitted Q4 rule only, not --q4-rule round")
    cfg, w = load_llama2c(a.src) if a.kind == "llama2c" else load_hf(a.src, a.max_seq)
    imat = mlb.read_imatrix(a.imatrix, cfg) if a.imatrix else None
    rule = a.q4_rule
    if rule == "auto":   # an importance matrix asks for the fit
        rule = "round" if cfg.get("model_type") == "gemma3_text" and not imat else "fit"
    mlb.write_mlb(a.out, cfg, w, a.wtype, [c for c in a.q8.split(",") if c], imat, a.scales, rule)
    print(f"wrote {a.out}: dim={cfg['dim']} layers={cfg['n_layers']} vocab={cfg['vocab']} "
          f"max_seq={cfg['max_seq']} wtype={a.wtype}{' (q8: ' + a.q8 + ')' if a.q8 and a.wtype == 'q4' else ''} "
          f"shared_classifier={cfg['shared_classifier']} scales={a.scales}"
          f"{' q4_rule=' + rule if a.wtype == 'q4' else ''}")
    if imat:
        print(f"imatrix {a.imatrix}: {imat['tokens']} tokens")


if __name__ == "__main__":
    main()
