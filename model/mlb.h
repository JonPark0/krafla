/* .mlb model file: header + 64 B-aligned tensor arrays, mapped zero-copy. */
#ifndef ML_MLB_H
#define ML_MLB_H
#include "core/file.h"
#include "core/ml.h"

#define ML_MLB_HEADER_SIZE 256

typedef struct {
    int dim, n_layers, n_heads, n_kv_heads, hidden_dim, vocab, max_seq, head_dim, kv_dim;
    int q_dim;          /* n_heads * head_dim: dim, unless head_dim is set (offset 80; Qwen3) */
    float rope_theta, norm_eps;
    int wtype;
    unsigned q8_mask;   /* Q4 models: ML_Q8M_* classes stored in Q8 (offset 72) */
    int qkv_bias;       /* Q/K/V projections have biases (offset 76; Qwen2) */
    int qk_norm;        /* RMSNorm of each query and key head before RoPE (offset 84; Qwen3) */
    int sliding_window; /* attend to the last this many positions; 0: all (offset 92) */
    int layer_flags;    /* 1: per-layer ML_LAYER_* bytes follow the header (offset 96) */
    float rope_local_theta;   /* RoPE base of ML_LAYER_LOCAL_ROPE layers, unscaled (offset 100) */
    int act;            /* FFN activation: ML_ACT_SILU or ML_ACT_GELU_TANH (offset 104) */
    int post_norms;     /* RMSNorm of the attention and FFN outputs before the residual (108; Gemma 3) */
    float embed_scale;  /* factor on embedding rows; 1 when the header has 0 (offset 112) */
    float attn_scale;   /* attention score factor; 1/sqrt(head_dim) when the header has 0 (116) */
    int scale_f16;      /* matrix scales are stored as f16 (offset 120) */
    float attn_softcap; /* > 0: attention scores become cap * tanh(score / cap) (124; Gemma 2) */
    float final_softcap;/* > 0: the same on the logits (128) */
    /* RoPE scaling (header offsets 52..71, 88): ML_ROPE_*. For yarn the low
     * and high frequency factors hold beta_fast and beta_slow. */
    int rope_type, rope_orig_max_seq;
    float rope_factor, rope_low_freq_factor, rope_high_freq_factor;
    float rope_attn_factor;   /* yarn: factor on cos and sin (offset 88) */
    int rot_dim;        /* leading dimensions of each head that RoPE rotates; head_dim if 0 in the header (132) */
    int attn_gate;      /* attention output multiplied by sigmoid(wg x) (136; Qwen3.5) */
    /* linear attention, gated delta rule (140..159; Qwen3.5): key and value
     * heads, their sizes, the causal conv kernel; derived: conv channels
     * (2 * nk * dk + nv * dv), value width (nv * dv), linear layers */
    int lin_nk, lin_nv, lin_dk, lin_dv, lin_conv;
    int lin_channels, lin_vdim, n_linear;
} ml_config;

enum { ML_ROPE_NONE = 0, ML_ROPE_LLAMA3 = 1, ML_ROPE_LINEAR = 2, ML_ROPE_YARN = 3 };
/* Per-layer flags: the layer uses the sliding window; the local RoPE base;
 * no RoPE at all (NoPE, SmolLM3); linear attention in place of attention
 * (Qwen3.5; alone). Without per-layer flags the window applies to every layer. */
enum { ML_LAYER_SLIDING = 1, ML_LAYER_LOCAL_ROPE = 2, ML_LAYER_NO_ROPE = 4, ML_LAYER_LINEAR = 8,
       ML_LAYER_ALL = 15 };
enum { ML_ACT_SILU = 0, ML_ACT_GELU_TANH = 1 };
enum { ML_Q8M_EMB = 1, ML_Q8M_CLS = 2, ML_Q8M_QK = 4, ML_Q8M_V = 8, ML_Q8M_O = 16,
       ML_Q8M_UP = 32, ML_Q8M_DOWN = 64, ML_Q8M_ALL = 127 };

typedef struct {
    const float *attn_norm, *ffn_norm;
    ml_qtensor wq, wk, wv, wo, w1, w2, w3;
    const float *bq, *bk, *bv;   /* NULL unless cfg.qkv_bias */
    const float *q_norm, *k_norm;   /* [head_dim]; NULL unless cfg.qk_norm */
    const float *post_attn_norm, *post_ffn_norm;   /* [dim]; NULL unless cfg.post_norms */
    int window;          /* positions attended at most; 0: all */
    int local_rope;      /* uses the RoPE tables of rope_local_theta */
    int no_rope;         /* queries and keys are not rotated (NoPE) */
    ml_qtensor wg;       /* attention output gate [q_dim][dim] (cfg.attn_gate) */
    /* linear-attention layers (linear set) hold these instead of wq..wo:
     * lqkv [channels][dim], lz [vdim][dim], lb, la [nv][dim], lout [dim][vdim];
     * F32 lconv [channels][conv], ldt, lalog [nv], lnorm [dv] */
    int linear;
    ml_qtensor lqkv, lz, lb, la, lout;
    const float *lconv, *ldt, *lalog, *lnorm;
} ml_layer;

/* The quantized matrices of layer L (rows > 0), at most ML_LAYER_MATS. */
#define ML_LAYER_MATS 9
int ml_layer_mats(ml_layer *L, ml_qtensor **out);

typedef struct {
    ml_config cfg;
    ml_file file;
    ml_qtensor tok_emb, wcls;
    const float *final_norm;
    ml_layer *layers;
    void *pack_mem;   /* row-interleaved weight copies (ml_model_pack), or NULL */
    ml_file pack_file;   /* or the same, mapped from a pack cache file */
    int packed;          /* the matrices point at packed copies */
    float *scales;       /* f16-scale files: the matrices' scales as floats */
} ml_model;

ml_status ml_model_load(ml_model *m, const char *path);
void ml_model_free(ml_model *m);
/* f16-scale files: keeps the float scales only where still read (the
 * embedding table, and matrices without packed copies). */
ml_status ml_model_trim_scales(ml_model *m);

#endif
