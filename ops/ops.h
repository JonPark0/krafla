/* Operators built on the bound kernels, split across the worker pool. */
#ifndef ML_OPS_H
#define ML_OPS_H
#include "core/threads.h"
#include "model/mlb.h"

/* y[0..W->rows) = W x; rows split across threads in multiples of 4. */
void ml_matvec(ml_pool *pool, float *y, const ml_qtensor *W, const ml_qact *x);

/* Y[t*ldy + row] = W x_t for the n tokens of X (prompt prefill). */
void ml_matmul(ml_pool *pool, float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X);

/* One layer of the KV cache, rows [ctx][kv_dim]. F32 when k is set,
 * F16 when kh is set; otherwise Q8 with one scale per position and KV head:
 * kq, vq hold the quants and kd, vd the scales [ctx][n_kv_heads]. */
typedef struct {
    const float *k, *v;
    const int8_t *kq, *vq;
    const float *kd, *vd;
    int window;   /* positions attended at most (sliding window); 0: all */
    const uint16_t *kh, *vh;
    int ctx;      /* positions the cache holds; also the stride of the score rows in att_scratch */
} ml_kv_layer;

/* Quantizes x[0, n) with a single scale, by the quantize_q8 rule. */
void ml_quantize_row(int8_t *q, float *d, const float *x, int n);

/* Causal GQA attention for one position. att_scratch: n_heads * kv->ctx
 * floats; qq, qd (Q8 cache only): dim bytes and n_heads floats for the
 * quantized query. */
void ml_attention(ml_pool *pool, float *out, const float *q, const ml_kv_layer *kv, float *att_scratch,
                  int8_t *qq, float *qd, const ml_config *cfg, int pos);
/* ml_attention for positions pos0..pos0+n-1 at once (prompt prefill), the
 * cache already holding their K and V; out, q: [n][dim]. att_scratch:
 * max(n_heads, ml_pool_size * ml_attention_batch_heads) * kv->ctx floats;
 * qq, qd (Q8 cache): n times ml_attention's. Each position's output is the
 * one ml_attention gives, bit for bit, with one pass over the pool for all. */
int ml_attention_batch_heads(const ml_config *cfg, int nthreads);

/* x[r * ld + i] = cap * tanh(x[r * ld + i] / cap) for rows r < rows, i < n
 * (logit soft-capping, Gemma 2). */
void ml_softcap_rows(float *x, size_t ld, int rows, int n, float cap);
void ml_attention_batch(ml_pool *pool, float *out, const float *q, const ml_kv_layer *kv, float *att_scratch,
                        int8_t *qq, float *qd, const ml_config *cfg, int pos0, int n);

/* One linear-attention layer (gated delta rule, Qwen3.5) and its state: the
 * conv window [channels][conv - 1] and the recurrent state [nv][dk][dv]. */
typedef struct {
    const ml_config *c;
    const ml_layer *L;
    float *conv, *S;
} ml_delta_layer;
/* n tokens in order: qkv [n][channels] (the projections, overwritten), z
 * [n][vdim], b and a [n][nv] -> out [n][vdim] (before lout). scratch: 3 * dv
 * floats per pool thread. See ops/delta.c. */
void ml_delta_run(ml_pool *pool, const ml_delta_layer *d, float *qkv, const float *z, const float *b, const float *a,
                  float *out, float *scratch, int n);

#endif
