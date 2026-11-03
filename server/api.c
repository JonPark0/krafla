#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model/chat.h"
#include "server/api.h"

typedef struct {
    int *p;
    int n, cap;
} ivec;

static int iv_push(ivec *v, const int *x, int n) {
    if (v->n + n > v->cap) {
        int cap = v->cap ? v->cap : 256;
        while (cap < v->n + n) cap *= 2;
        int *p = realloc(v->p, sizeof(int) * (size_t)cap);
        if (!p) return -1;
        v->p = p;
        v->cap = cap;
    }
    memcpy(v->p + v->n, x, sizeof(int) * (size_t)n);
    v->n += n;
    return 0;
}

/* Encodes text onto v; -1 if memory runs out. */
static int encode_push(ml_tokenizer *tok, const char *text, int bos, ivec *v) {
    const size_t cap = 3 * strlen(text) + 16;   /* at most one token per byte, plus BOS and a prefix */
    if (cap > (size_t)1 << 28) return -1;
    int *buf = malloc(sizeof(int) * cap);
    if (!buf) return -1;
    const int k = ml_encode(tok, text, bos, 0, buf, (int)cap);
    const int rc = k < 0 ? -1 : iv_push(v, buf, k);
    free(buf);
    return rc;
}

/* Appends the text of a message's content (a string, null, or an array of
 * {"type": "text", "text": ...} parts) to b. -1 on other content. */
static int content_text(const js_doc *d, int c, sbuf *b) {
    if (js_type(d, c) == JS_NULL) return 0;
    if (js_type(d, c) == JS_STR) { sb_put(b, js_str(d, c), (size_t)d->n[c].len); return 0; }
    if (js_type(d, c) != JS_ARR) return -1;
    for (int i = d->n[c].child; i >= 0; i = d->n[i].next) {
        const char *type = js_str(d, js_get(d, i, "type"));
        const int t = js_get(d, i, "text");
        if (!type || strcmp(type, "text") || js_type(d, t) != JS_STR) return -1;
        sb_put(b, js_str(d, t), (size_t)d->n[t].len);
    }
    return 0;
}

typedef struct {
    sbuf user, reply;
    int has_reply;
} turn;

int srv_chat_tokens(ml_tokenizer *tok, int tpl, const char *default_system, const js_doc *d, int msgs,
                    int **toks, int *n, char *err, size_t errlen) {
    *toks = NULL;
    *n = 0;
    if (js_type(d, msgs) != JS_ARR || d->n[msgs].len == 0) {
        snprintf(err, errlen, "'messages' must be a non-empty array");
        return -1;
    }
    turn *turns = calloc((size_t)d->n[msgs].len, sizeof *turns);
    sbuf sys = { 0 };
    ivec v = { 0 };
    int nt = 0, rc = -1;
    if (!turns) { snprintf(err, errlen, "out of memory"); return -1; }
    for (int m = d->n[msgs].child, idx = 0; m >= 0; m = d->n[m].next, idx++) {
        const char *role = js_str(d, js_get(d, m, "role"));
        const int content = js_get(d, m, "content");
        sbuf text = { 0 };
        if (!role) { snprintf(err, errlen, "messages[%d] has no 'role'", idx); goto out; }
        if (content_text(d, content, &text)) {
            sb_free(&text);
            snprintf(err, errlen, "messages[%d].content: only text is supported", idx);
            goto out;
        }
        const int calls = js_get(d, m, "tool_calls");
        if (js_type(d, calls) == JS_ARR && d->n[calls].len > 0) {
            sb_free(&text);
            snprintf(err, errlen, "messages[%d]: tool calls are not supported", idx);
            goto out;
        }
        sbuf *dst = NULL;
        if (!strcmp(role, "system") || !strcmp(role, "developer")) dst = &sys;
        else if (!strcmp(role, "user")) {
            if (nt == 0 || turns[nt - 1].has_reply) nt++;
            dst = &turns[nt - 1].user;
        } else if (!strcmp(role, "assistant")) {
            if (nt == 0) {
                sb_free(&text);
                snprintf(err, errlen, "messages[%d]: an assistant message must follow a user message", idx);
                goto out;
            }
            turns[nt - 1].has_reply = 1;
            dst = &turns[nt - 1].reply;
        } else {
            sb_free(&text);
            snprintf(err, errlen, "messages[%d]: role '%s' is not supported", idx, role);
            goto out;
        }
        if (dst->n && text.n) sb_puts(dst, "\n\n");   /* consecutive messages of one role are joined */
        if (text.p) sb_put(dst, text.p, text.n);
        sb_free(&text);
    }
    if (nt == 0) { snprintf(err, errlen, "'messages' has no user message"); goto out; }
    const char *system = sys.n ? sys.p : default_system;
    int stop = ml_tokenizer_find(tok, ml_chat_stop(tpl));
    if (stop < 0) stop = ml_tokenizer_eos(tok);
    for (int i = 0; i < nt; i++) {
        char *r = ml_chat_render(tpl, system, turns[i].user.p ? turns[i].user.p : "", i == 0);
        if (!r) { snprintf(err, errlen, "out of memory"); goto out; }
        sbuf text = { 0 };
        sb_puts(&text, r);
        free(r);
        if (turns[i].has_reply && turns[i].reply.p) sb_put(&text, turns[i].reply.p, turns[i].reply.n);
        const int bad = text.oom || encode_push(tok, text.p, i == 0 || ml_chat_bos_each_turn(tpl), &v);
        sb_free(&text);
        if (bad) { snprintf(err, errlen, "out of memory"); goto out; }
        if (turns[i].has_reply && i + 1 < nt && stop >= 0 && iv_push(&v, &stop, 1)) {
            snprintf(err, errlen, "out of memory");
            goto out;
        }
    }
    if (sys.oom) { snprintf(err, errlen, "out of memory"); goto out; }
    *toks = v.p;
    *n = v.n;
    v.p = NULL;
    rc = 0;
out:
    for (int i = 0; i < d->n[msgs].len; i++) { sb_free(&turns[i].user); sb_free(&turns[i].reply); }
    free(turns);
    sb_free(&sys);
    free(v.p);
    return rc;
}

/* Token ids from an array of numbers; -1 if one is not an id below vocab. */
static int ids_push(const js_doc *d, int a, int vocab, ivec *v) {
    for (int i = d->n[a].child; i >= 0; i = d->n[i].next) {
        const double x = d->n[i].num;
        if (d->n[i].type != JS_NUM || !(x >= 0 && x < vocab) || x != (double)(int)x) return -1;
        const int id = (int)x;
        if (iv_push(v, &id, 1)) return -1;
    }
    return 0;
}

int srv_text_tokens(ml_tokenizer *tok, int vocab, const js_doc *d, int p, int **toks, int *n, char *err, size_t errlen) {
    ivec v = { 0 };
    *toks = NULL;
    *n = 0;
    if (js_type(d, p) == JS_ARR && d->n[p].len == 1 && js_type(d, d->n[p].child) != JS_NUM)
        p = d->n[p].child;   /* one prompt in an array */
    int bad;
    if (js_type(d, p) == JS_STR) bad = encode_push(tok, js_str(d, p), 1, &v);
    else if (js_type(d, p) == JS_ARR) bad = ids_push(d, p, vocab, &v);
    else {
        snprintf(err, errlen, "'prompt' must be a string or an array of token ids (one prompt per request)");
        return -1;
    }
    if (bad) {
        free(v.p);
        if (js_type(d, p) == JS_ARR) snprintf(err, errlen, "'prompt': token ids must be integers below %d", vocab);
        else snprintf(err, errlen, "out of memory");
        return -1;
    }
    if (v.n == 0) { free(v.p); snprintf(err, errlen, "'prompt' is empty"); return -1; }
    *toks = v.p;
    *n = v.n;
    return 0;
}
