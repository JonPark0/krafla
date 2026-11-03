/* Linear attention with the gated delta rule (Qwen3.5), for a run of tokens
 * in order. Per token and value head h (key head h / (nv / nk)):
 *   conv:   x = silu(sum_j w[c][j] * window[c][j]) per channel, the window
 *           ending at this token's projection (causal, kernel `conv`)
 *   q, k:   L2-normalized per key head (eps 1e-6), q also by 1/sqrt(dk)
 *   gates:  beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
 *   state:  S = S * exp(g); S += k (beta (v - S^T k))^T; o = S^T q
 *   output: RMSNorm(o) * w * silu(z), per head
 * Channels and heads are independent, so they are split across the pool;
 * every one takes the tokens in order, with the same arithmetic as a single
 * token: results do not depend on the thread count or the batch size. */
#include <math.h>
#include <string.h>
#include "ops/ops.h"

#define DELTA_PARALLEL_MIN (1 << 16)   /* state multiply-adds below which one thread does all */

typedef struct {
    const ml_delta_layer *d;
    float *qkv;
    const float *z, *b, *a;
    float *out;
    float *scratch;   /* 3 * dv floats per thread */
    int n;
} delta_ctx;

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* Causal conv and SiLU over channels [c0, c1), then each token's q and k
 * normalized (key heads whose channels fall in range). */
static void conv_task(void *p, int tid, int nt) {
    const delta_ctx *x = p;
    const ml_config *c = x->d->c;
    const int ch = c->lin_channels, K = c->lin_conv;
    const int c0 = (int)((long long)ch * tid / nt), c1 = (int)((long long)ch * (tid + 1) / nt);
    for (int i = c0; i < c1; i++) {
        const float *w = x->d->L->lconv + (size_t)i * K;
        float *win = x->d->conv + (size_t)i * (K - 1);
        for (int t = 0; t < x->n; t++) {
            float *v = x->qkv + (size_t)t * ch + i;
            float acc = 0.0f;
            for (int j = 0; j < K - 1; j++) acc += w[j] * win[j];
            acc += w[K - 1] * *v;
            for (int j = 0; j + 1 < K - 1; j++) win[j] = win[j + 1];
            if (K > 1) win[K - 2] = *v;
            *v = silu(acc);
        }
    }
}

/* q and k of key heads [h0, h1) for every token, normalized in place. */
static void norm_task(void *p, int tid, int nt) {
    const delta_ctx *x = p;
    const ml_config *c = x->d->c;
    const int nk = c->lin_nk, dk = c->lin_dk, ch = c->lin_channels;
    const int h0 = (int)((long long)2 * nk * tid / nt), h1 = (int)((long long)2 * nk * (tid + 1) / nt);
    const float qs = 1.0f / sqrtf((float)dk);
    for (int t = 0; t < x->n; t++)
        for (int h = h0; h < h1; h++) {   /* h < nk: a query head; else a key head */
            float *v = x->qkv + (size_t)t * ch + (size_t)h * dk;
            float ss = 0.0f;
            for (int i = 0; i < dk; i++) ss += v[i] * v[i];
            const float inv = 1.0f / sqrtf(ss + 1e-6f) * (h < nk ? qs : 1.0f);
            for (int i = 0; i < dk; i++) v[i] *= inv;
        }
}

/* The state update, output and gated norm of value heads [h0, h1). */
static void head_task(void *p, int tid, int nt) {
    const delta_ctx *x = p;
    const ml_config *c = x->d->c;
    const ml_layer *L = x->d->L;
    const int nk = c->lin_nk, nv = c->lin_nv, dk = c->lin_dk, dv = c->lin_dv, ch = c->lin_channels;
    const int h0 = (int)((long long)nv * tid / nt), h1 = (int)((long long)nv * (tid + 1) / nt);
    float *kvm = x->scratch + (size_t)tid * 3 * dv, *delta = kvm + dv, *o = delta + dv;
    for (int h = h0; h < h1; h++) {
        const int kh = h / (nv / nk);
        float *S = x->d->S + (size_t)h * dk * dv;
        const float alog = -expf(L->lalog[h]);
        for (int t = 0; t < x->n; t++) {
            const float *row = x->qkv + (size_t)t * ch;
            const float *q = row + (size_t)kh * dk, *k = row + (size_t)(nk + kh) * dk;
            const float *v = row + (size_t)2 * nk * dk + (size_t)h * dv;
            const float bt = x->b[(size_t)t * nv + h], at = x->a[(size_t)t * nv + h] + L->ldt[h];
            const float beta = 1.0f / (1.0f + expf(-bt));
            const float decay = expf(alog * (at > 20.0f ? at : log1pf(expf(at))));
            for (int j = 0; j < dv; j++) kvm[j] = 0.0f;
            for (int i = 0; i < dk; i++) {
                float *s = S + (size_t)i * dv;
                for (int j = 0; j < dv; j++) {
                    s[j] *= decay;
                    kvm[j] += s[j] * k[i];
                }
            }
            for (int j = 0; j < dv; j++) {
                delta[j] = (v[j] - kvm[j]) * beta;
                o[j] = 0.0f;
            }
            for (int i = 0; i < dk; i++) {
                float *s = S + (size_t)i * dv;
                for (int j = 0; j < dv; j++) {
                    s[j] += k[i] * delta[j];
                    o[j] += s[j] * q[i];
                }
            }
            float ss = 0.0f;
            for (int j = 0; j < dv; j++) ss += o[j] * o[j];
            const float inv = 1.0f / sqrtf(ss / (float)dv + c->norm_eps);
            const float *z = x->z + (size_t)t * c->lin_vdim + (size_t)h * dv;
            float *out = x->out + (size_t)t * c->lin_vdim + (size_t)h * dv;
            for (int j = 0; j < dv; j++) out[j] = o[j] * inv * L->lnorm[j] * silu(z[j]);
        }
    }
}

void ml_delta_run(ml_pool *pool, const ml_delta_layer *d, float *qkv, const float *z, const float *b, const float *a,
                  float *out, float *scratch, int n) {
    const ml_config *c = d->c;
    delta_ctx x = { d, qkv, z, b, a, out, scratch, n };
    const double work = (double)n * c->lin_nv * c->lin_dk * c->lin_dv;
    if (work < DELTA_PARALLEL_MIN) {
        conv_task(&x, 0, 1);
        norm_task(&x, 0, 1);
        head_task(&x, 0, 1);
        return;
    }
    ml_pool_run(pool, conv_task, &x);
    ml_pool_run(pool, norm_task, &x);
    ml_pool_run(pool, head_task, &x);
}
