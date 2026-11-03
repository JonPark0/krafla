/* Generation for the HTTP server: one model state whose KV cache is reused
 * across requests that share a prompt prefix, stop strings, and text handed
 * out in pieces that never end inside a UTF-8 sequence or a possible stop
 * string. */
#ifndef ML_ENGINE_H
#define ML_ENGINE_H
#include "model/sampler.h"
#include "model/tokenizer.h"
#include "model/transformer.h"
#include "server/json.h"

#define SRV_MAX_STOP 4

typedef struct {
    ml_state *s;
    ml_tokenizer *tok;
    int *cached;    /* the tokens whose K and V are in the cache, positions 0..ncached-1 */
    int ncached;
} srv_engine;

typedef struct {
    ml_sampler_opts so;
    int max_tokens;                    /* reply tokens at most; 0: until the context is full */
    const char *stop[SRV_MAX_STOP];    /* stop strings: the reply ends before the first one */
    int nstop;
    int stop_ids[4];                   /* tokens that end the reply (not part of it) */
    int nstop_ids;
    int prev;                          /* token before the reply, for ml_decode; -1: none */
} srv_request;

enum { SRV_FINISH_STOP, SRV_FINISH_LENGTH, SRV_FINISH_ABORT };

typedef struct {
    int prompt_tokens, cached_tokens, completion_tokens;
    int finish;                        /* SRV_FINISH_* */
    double t_prefill, t_decode;        /* seconds */
} srv_result;

/* Called with each new piece of the reply; returns nonzero to stop (the
 * client went away). */
typedef int (*srv_emit_fn)(void *ud, const char *text, size_t n);

ml_status srv_engine_init(srv_engine *e, ml_state *s, ml_tokenizer *tok);
void srv_engine_free(srv_engine *e);
/* Runs the prompt (1 to ctx-1 tokens), reusing the cached prefix it shares
 * with the previous request (with linear-attention layers, only when the
 * prompt continues all of it), then generates. The reply text goes to emit
 * piece by piece (if emit is set) and, whole, to reply. The logits, and so
 * the reply for a given seed, do not depend on what was cached. */
ml_status srv_generate(srv_engine *e, const int *prompt, int np, const srv_request *r,
                       srv_emit_fn emit, void *ud, sbuf *reply, srv_result *res);

/* Stop strings over text arriving in pieces. */
typedef struct {
    sbuf text;          /* all text so far */
    size_t out;         /* bytes released */
    const char *const *stop;
    int nstop;
} srv_stop;

/* Adds a piece. Returns 1 if a stop string now occurs: the text is cut
 * before it. *from, *n: the newly released bytes of text.p (none may end
 * inside a UTF-8 sequence or a prefix of a stop string, unless flush). */
int srv_stop_feed(srv_stop *t, const char *piece, size_t len, int flush, size_t *from, size_t *n);

#endif
