#include <stdlib.h>
#include <string.h>
#include "core/clock.h"
#include "server/engine.h"

ml_status srv_engine_init(srv_engine *e, ml_state *s, ml_tokenizer *tok) {
    e->s = s;
    e->tok = tok;
    e->ncached = 0;
    e->cached = malloc(sizeof(int) * (size_t)s->ctx);
    return e->cached ? ML_OK : ML_E_NOMEM;
}

void srv_engine_free(srv_engine *e) {
    free(e->cached);
    e->cached = NULL;
    e->ncached = 0;
}

/* Length of the longest suffix of s[0, n) that is a proper prefix of a stop string. */
static size_t stop_prefix(const srv_stop *t, const char *s, size_t n) {
    size_t best = 0;
    for (int i = 0; i < t->nstop; i++) {
        const size_t k = strlen(t->stop[i]);
        if (!k) continue;
        for (size_t m = (k - 1 < n ? k - 1 : n); m > best; m--)
            if (!memcmp(s + n - m, t->stop[i], m)) { best = m; break; }
    }
    return best;
}

int srv_stop_feed(srv_stop *t, const char *piece, size_t len, int flush, size_t *from, size_t *n) {
    sb_put(&t->text, piece, len);
    *from = t->out;
    *n = 0;
    if (!t->text.p) return 0;
    /* a stop string cannot start before `out`: what was released ended with no stop prefix */
    size_t cut = t->text.n;
    int hit = 0;
    for (int i = 0; i < t->nstop; i++) {
        const size_t k = strlen(t->stop[i]);
        if (!k || t->text.n - t->out < k) continue;
        for (size_t j = t->out; j + k <= t->text.n && j < cut; j++)
            if (!memcmp(t->text.p + j, t->stop[i], k)) { cut = j; hit = 1; break; }
    }
    size_t end = cut;
    if (!hit) {
        if (!flush) {
            end -= stop_prefix(t, t->text.p + t->out, end - t->out);
            end = t->out + utf8_complete(t->text.p + t->out, end - t->out);
        }
    } else {
        t->text.n = cut;
        t->text.p[cut] = '\0';
    }
    *n = end - t->out;
    t->out = end;
    return hit;
}

static int is_stop_id(const srv_request *r, int tok) {
    for (int i = 0; i < r->nstop_ids; i++)
        if (r->stop_ids[i] == tok) return 1;
    return 0;
}

ml_status srv_generate(srv_engine *e, const int *prompt, int np, const srv_request *r,
                       srv_emit_fn emit, void *ud, sbuf *reply, srv_result *res) {
    ml_state *s = e->s;
    const int ctx = s->ctx, vocab = s->m->cfg.vocab;
    memset(res, 0, sizeof *res);
    res->prompt_tokens = np;
    if (np < 1 || np >= ctx) return ML_E_SHAPE;
    /* reuse the cached prefix; the last prompt token always runs, for its logits */
    int k = 0;
    while (k < e->ncached && k < np - 1 && e->cached[k] == prompt[k]) k++;
    /* linear attention (Qwen3.5) keeps states, not a history of positions:
     * they can be continued, but not cut back to a shorter prefix */
    if (s->m->cfg.n_linear && k < e->ncached) k = 0;
    res->cached_tokens = k;
    e->ncached = k;
    double t0 = ml_now();
    ml_status st = ml_prefill(s, prompt + k, np - 1 - k, k, NULL);
    if (st != ML_OK) return st;
    memcpy(e->cached + k, prompt + k, sizeof(int) * (size_t)(np - 1 - k));
    e->ncached = np - 1;

    ml_sampler smp;
    if ((st = ml_sampler_init(&smp, vocab, &r->so)) != ML_OK) return st;
    for (int i = 0; i < np; i++) ml_sampler_accept(&smp, prompt[i]);
    srv_stop stop = { { NULL, 0, 0, 0 }, 0, r->stop, r->nstop };
    int token = prompt[np - 1], pos = np - 1, prev = r->prev, done = 0;
    res->finish = SRV_FINISH_LENGTH;
    double t1 = ml_now();
    res->t_prefill = t1 - t0;
    while (!done && pos < ctx) {
        float *logits = ml_forward(s, token, pos);
        if (!logits) { st = ML_E_SHAPE; break; }
        e->cached[pos++] = token;
        e->ncached = pos;
        const int next = ml_sample(&smp, logits, vocab);
        ml_sampler_accept(&smp, next);
        if (is_stop_id(r, next)) { res->finish = SRV_FINISH_STOP; break; }
        res->completion_tokens++;
        const char *piece = ml_decode(e->tok, prev, next);
        prev = token = next;
        const int last = (r->max_tokens > 0 && res->completion_tokens >= r->max_tokens) || pos >= ctx;
        size_t from, n;
        if (srv_stop_feed(&stop, piece, strlen(piece), last, &from, &n)) { res->finish = SRV_FINISH_STOP; done = 1; }
        if (n && emit && emit(ud, stop.text.p + from, n)) { res->finish = SRV_FINISH_ABORT; break; }
        if (last) break;
    }
    if (res->finish == SRV_FINISH_STOP && !done) {   /* stop token: release what was held back */
        size_t from, n;
        srv_stop_feed(&stop, "", 0, 1, &from, &n);
        if (n && emit && emit(ud, stop.text.p + from, n)) res->finish = SRV_FINISH_ABORT;
    }
    res->t_decode = ml_now() - t1;
    if (stop.text.oom) st = ML_E_NOMEM;
    if (reply && stop.text.p) sb_put(reply, stop.text.p, stop.out);
    sb_free(&stop.text);
    ml_sampler_free(&smp);
    return st;
}
