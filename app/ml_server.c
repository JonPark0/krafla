/* ml_server: an OpenAI-compatible HTTP server for one model. Requests are
 * served one at a time; the KV cache is kept between them, so a request
 * that repeats the previous conversation runs only its new tokens. */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "config/ml_config.h"
#include "core/clock.h"
#include "model/chat.h"
#include "model/pack_model.h"
#include "model/tokenizer.h"
#include "model/transformer.h"
#include "server/api.h"
#include "server/engine.h"
#include "server/http.h"
#include "server/json.h"
#include "server/net.h"

static const char USAGE[] =
    "usage: ml_server MODEL.mlb -z TOKENIZER [options]\n"
    "  -z FILE          tokenizer: .mlt from tools/convert_tokenizer.py, or llama2.c tokenizer.bin\n"
    "  --chat TEMPLATE  chat template for /v1/chat/completions: llama3, zephyr, llama2, chatml,\n"
    "                   qwen, smollm, gemma, qwen3.5 or qwen3.5-nothink (as ml_run --chat);\n"
    "                   without it only\n"
    "                   /v1/completions is served\n"
    "  --system TEXT    system prompt for chat requests that have none\n"
    "  --host ADDR      address to listen on (default 127.0.0.1; 0.0.0.0 for every interface)\n"
    "  --port N         port (default 8080; 0 picks a free one)\n"
    "  --api-key KEY    require 'Authorization: Bearer KEY' (default: $KRAFLA_API_KEY, if set)\n"
    "  --alias NAME     model id in responses and /v1/models (default: the file name without .mlb)\n"
    "  --cors ORIGIN    allow browser pages from ORIGIN ('*' for any) to call the server\n"
    "  --ctx N          context length, shared by the conversation and the reply (default: the\n"
    "                   model's max_seq, at most 4096)\n"
    "  -n N             reply tokens at most when a request sets no max_tokens (default 0: until\n"
    "                   the context is full)\n"
    "  -T F, -p F, --top-k N, --repeat-penalty F, --repeat-last-n N, -s N\n"
    "                   sampling for requests that do not set it (defaults as ml_run: greedy;\n"
    "                   without -s each request gets a new seed)\n"
    "  --eos ID         token that ends a reply, in place of the tokenizer's\n"
    "  -t N             threads (default 1)\n"
    "  --kv TYPE        KV cache type: f32 (default), f16 or q8\n"
    "  --batch N        prompt tokens per prefill step, 1..128 (default 128)\n"
    "  --no-pack, --pack-cache  as ml_run\n"
    "  -h, --help       show this help\n";

#define MAX_BODY (8u << 20)

static int usage(const char *msg) {
    if (msg) fprintf(stderr, "error: %s\n", msg);
    fprintf(stderr, "%skrafla %s, build ISA: %s\n", USAGE, ML_VERSION, ML_ISA_NAME);
    return 2;
}

static int parse_int(const char *s, long lo, long hi, int *out) {
    char *end;
    errno = 0;
    const long v = strtol(s, &end, 10);
    if (end == s || *end || errno == ERANGE || v < lo || v > hi) return 0;
    *out = (int)v;
    return 1;
}

static int parse_float(const char *s, float lo, float hi, float *out) {
    char *end;
    const float v = strtof(s, &end);
    if (end == s || *end || !(v >= lo && v <= hi)) return 0;
    *out = v;
    return 1;
}

typedef struct {
    ml_model m;
    ml_tokenizer tok;
    ml_state s;
    srv_engine eng;
    int tpl;                   /* chat template, or -1 */
    const char *system, *api_key, *cors;
    char model_id[256], fingerprint[64], extra[512];
    ml_sampler_opts so;        /* defaults for requests */
    int seed_given, max_tokens, eos_opt;
    long started;
    unsigned long long count;
} server;

static int send_json(server *sv, ml_sock c, int status, const sbuf *b) {
    if (b->oom) {
        static const char oom[] = "{\"error\":{\"message\":\"out of memory\",\"type\":\"server_error\",\"param\":null,\"code\":null}}";
        return http_respond(c, 500, "application/json", sv->extra, oom, sizeof oom - 1);
    }
    return http_respond(c, status, "application/json", sv->extra, b->p, b->n);
}

static void send_error(server *sv, ml_sock c, int status, const char *msg, const char *param, const char *code) {
    sbuf b = { 0 };
    sb_puts(&b, "{\"error\":{\"message\":");
    sb_json_str(&b, msg, strlen(msg));
    sb_printf(&b, ",\"type\":\"%s\",\"param\":", status == 401 ? "authentication_error"
                                                : status >= 500 ? "server_error" : "invalid_request_error");
    if (param) sb_json_str(&b, param, strlen(param));
    else sb_puts(&b, "null");
    sb_puts(&b, ",\"code\":");
    if (code) sb_json_str(&b, code, strlen(code));
    else sb_puts(&b, "null");
    sb_puts(&b, "}}");
    send_json(sv, c, status, &b);
    sb_free(&b);
}

static void model_json(const server *sv, sbuf *b) {
    sb_puts(b, "{\"id\":");
    sb_json_str(b, sv->model_id, strlen(sv->model_id));
    sb_printf(b, ",\"object\":\"model\",\"created\":%ld,\"owned_by\":\"krafla\"}", sv->started);
}

/* A number member: absent or null leaves *v. 0, or -1 with err set. */
static int get_num(const js_doc *d, int o, const char *key, double lo, double hi, int integer, double *v,
                   char *err, size_t el) {
    const int i = js_get(d, o, key);
    if (js_type(d, i) == JS_NULL) return 0;
    const double x = js_type(d, i) == JS_NUM ? d->n[i].num : NAN;
    if (!(x >= lo && x <= hi) || (integer && x != floor(x))) {
        snprintf(err, el, "'%s' must be %s from %g to %g", key, integer ? "an integer" : "a number", lo, hi);
        return -1;
    }
    *v = x;
    return 0;
}

typedef struct {
    srv_request rq;
    int stream, include_usage;
} params;

/* Reads the sampling and output fields shared by both endpoints. 0, or -1
 * with err and the offending field in *param. */
static int read_params(server *sv, const js_doc *d, int chat, params *p, char *err, size_t el, const char **param) {
    memset(p, 0, sizeof *p);
    p->rq.so = sv->so;
    if (!sv->seed_given) p->rq.so.seed = (unsigned long long)time(NULL) * 6364136223846793005ull + ++sv->count;
    p->rq.max_tokens = sv->max_tokens;
    const int o = 0;
    double v;
    static const char *const fields[] = { "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k",
                                          "repeat_penalty", "repetition_penalty", "repeat_last_n", "seed", "n" };
    static const double lo[] = { 1, 1, 0, 0, 0, 1e-3, 1e-3, 0, -9.2e18, 1 }, hi[] = { 1 << 24, 1 << 24, 1e3, 1, 1 << 24,
                                                                                       1e3, 1e3, 1 << 24, 1.8e19, 1 };
    static const int integer[] = { 1, 1, 0, 0, 1, 0, 0, 1, 1, 1 };
    for (size_t f = 0; f < sizeof fields / sizeof fields[0]; f++) {
        v = NAN;
        *param = fields[f];
        if (get_num(d, o, fields[f], lo[f], hi[f], integer[f], &v, err, el)) {
            if (f == 9) snprintf(err, el, "'n' must be 1: one reply per request");
            return -1;
        }
        if (isnan(v)) continue;
        switch (f) {
        case 0: case 1: p->rq.max_tokens = (int)v; break;
        case 2: p->rq.so.temperature = (float)v; break;
        case 3: p->rq.so.topp = (float)v; break;
        case 4: p->rq.so.topk = (int)v; break;
        case 5: case 6: p->rq.so.repeat_penalty = (float)v; break;
        case 7: p->rq.so.repeat_last_n = (int)v; break;
        case 8: p->rq.so.seed = v < 0 ? (unsigned long long)(long long)v : (unsigned long long)v; break;
        default: break;
        }
    }
    const int stream = js_get(d, o, "stream"), opts = js_get(d, o, "stream_options");
    p->stream = js_type(d, stream) == JS_TRUE;
    p->include_usage = js_type(d, js_get(d, opts, "include_usage")) == JS_TRUE;
    const int stop = js_get(d, o, "stop");
    *param = "stop";
    if (js_type(d, stop) == JS_STR) {
        if (d->n[stop].len) p->rq.stop[p->rq.nstop++] = js_str(d, stop);
    } else if (js_type(d, stop) == JS_ARR) {
        if (d->n[stop].len > SRV_MAX_STOP) { snprintf(err, el, "'stop' holds at most %d strings", SRV_MAX_STOP); return -1; }
        for (int i = d->n[stop].child; i >= 0; i = d->n[i].next) {
            if (js_type(d, i) != JS_STR) { snprintf(err, el, "'stop' must be a string or an array of strings"); return -1; }
            if (d->n[i].len) p->rq.stop[p->rq.nstop++] = js_str(d, i);
        }
    } else if (js_type(d, stop) != JS_NULL) {
        snprintf(err, el, "'stop' must be a string or an array of strings");
        return -1;
    }
    *param = "logprobs";
    if (js_type(d, js_get(d, o, "logprobs")) == JS_TRUE || js_type(d, js_get(d, o, "logprobs")) == JS_NUM) {
        snprintf(err, el, "logprobs are not supported");
        return -1;
    }
    *param = "tools";
    const int tools = js_get(d, o, "tools");
    if (js_type(d, tools) == JS_ARR && d->n[tools].len > 0) { snprintf(err, el, "tools are not supported"); return -1; }
    *param = "response_format";
    const char *fmt = js_str(d, js_get(d, js_get(d, o, "response_format"), "type"));
    if (fmt && strcmp(fmt, "text")) { snprintf(err, el, "only the 'text' response format is supported"); return -1; }
    if (!chat) {
        *param = "echo";
        if (js_type(d, js_get(d, o, "echo")) == JS_TRUE) { snprintf(err, el, "echo is not supported"); return -1; }
        *param = "suffix";
        if (js_type(d, js_get(d, o, "suffix")) == JS_STR) { snprintf(err, el, "suffix is not supported"); return -1; }
        *param = "best_of";
        if (get_num(d, o, "best_of", 1, 1, 1, &v, err, el)) return -1;
    }
    /* tokens that end the reply, as ml_run: the template's stop token and
     * EOS for chat; BOS and EOS for plain text (and the template's stop
     * token when the server has one); or --eos alone */
    const int stop_tpl = sv->tpl >= 0 ? ml_tokenizer_find(&sv->tok, ml_chat_stop(sv->tpl)) : -1;
    int ids[3] = { stop_tpl, -1, -1 };
    if (sv->eos_opt >= 0) ids[1] = sv->eos_opt;
    else {
        ids[1] = ml_tokenizer_eos(&sv->tok);
        if (!chat) ids[2] = ml_tokenizer_bos(&sv->tok);
    }
    for (int i = 0; i < 3; i++)
        if (ids[i] >= 0) p->rq.stop_ids[p->rq.nstop_ids++] = ids[i];
    *param = NULL;
    return 0;
}

typedef struct {
    server *sv;
    ml_sock c;
    int chat;
    char id[64];
    long created;
    int failed;
} stream_ctx;

static void chunk_head(const stream_ctx *x, sbuf *b) {
    sb_printf(b, "data: {\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", x->id,
              x->chat ? "chat.completion.chunk" : "text_completion", x->created);
    sb_json_str(b, x->sv->model_id, strlen(x->sv->model_id));
    sb_printf(b, ",\"system_fingerprint\":\"%s\",\"choices\":[", x->sv->fingerprint);
}

static int stream_send(stream_ctx *x, sbuf *b) {
    if (!x->failed && (b->oom || ml_net_send(x->c, b->p, b->n))) x->failed = 1;
    return x->failed;
}

static int emit_piece(void *ud, const char *text, size_t n) {
    stream_ctx *x = ud;
    sbuf b = { 0 };
    chunk_head(x, &b);
    sb_puts(&b, x->chat ? "{\"index\":0,\"delta\":{\"content\":" : "{\"index\":0,\"text\":");
    sb_json_str(&b, text, n);
    sb_puts(&b, x->chat ? "},\"logprobs\":null,\"finish_reason\":null}]}\n\n" : ",\"logprobs\":null,\"finish_reason\":null}]}\n\n");
    const int rc = stream_send(x, &b);
    sb_free(&b);
    return rc;
}

static void usage_json(sbuf *b, const srv_result *r) {
    sb_printf(b, "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
                 "\"prompt_tokens_details\":{\"cached_tokens\":%d}}",
              r->prompt_tokens, r->completion_tokens, r->prompt_tokens + r->completion_tokens, r->cached_tokens);
}

/* POST /v1/chat/completions (chat) or /v1/completions. Returns the status sent. */
static int complete(server *sv, ml_sock c, const http_req *req, int chat, srv_result *res) {
    char err[512];
    const char *param = NULL;
    js_doc d;
    memset(res, 0, sizeof *res);
    if (!req->body || js_parse(&d, req->body, (size_t)req->content_length)) {
        if (req->body) snprintf(err, sizeof err, "invalid JSON: %s at byte %lu", d.err ? d.err : "error", (unsigned long)d.err_at);
        else snprintf(err, sizeof err, "the request has no JSON body");
        if (req->body) js_free(&d);
        send_error(sv, c, 400, err, NULL, NULL);
        return 400;
    }
    int status = 400, *toks = NULL, np = 0;
    params p;
    if (js_type(&d, 0) != JS_OBJ) { send_error(sv, c, 400, "the body must be a JSON object", NULL, NULL); goto out; }
    if (chat && sv->tpl < 0) {
        send_error(sv, c, 400, "this server has no chat template: start it with --chat, or use /v1/completions", NULL, NULL);
        goto out;
    }
    if (read_params(sv, &d, chat, &p, err, sizeof err, &param)) { send_error(sv, c, 400, err, param, NULL); goto out; }
    const int rc = chat ? srv_chat_tokens(&sv->tok, sv->tpl, sv->system, &d, js_get(&d, 0, "messages"), &toks, &np, err, sizeof err)
                        : srv_text_tokens(&sv->tok, sv->m.cfg.vocab, &d, js_get(&d, 0, "prompt"), &toks, &np, err, sizeof err);
    if (rc) { send_error(sv, c, 400, err, chat ? "messages" : "prompt", NULL); goto out; }
    if (np >= sv->s.ctx) {
        snprintf(err, sizeof err, "the context holds %d tokens, but the %s %d tokens; start the server with a larger "
                 "--ctx (at most %d for this model)", sv->s.ctx, chat ? "messages take" : "prompt takes", np, sv->m.cfg.max_seq);
        send_error(sv, c, 400, err, chat ? "messages" : "prompt", "context_length_exceeded");
        res->prompt_tokens = np;
        goto out;
    }
    p.rq.prev = chat ? -1 : toks[np - 1];
    stream_ctx x = { sv, c, chat, "", (long)time(NULL), 0 };
    snprintf(x.id, sizeof x.id, "%s-%lx%llx", chat ? "chatcmpl" : "cmpl", (unsigned long)x.created, ++sv->count);
    sbuf reply = { 0 }, b = { 0 };
    if (p.stream) {
        if (http_stream_start(c, sv->extra)) goto out;
        if (chat) {   /* the role comes first, as OpenAI sends it */
            chunk_head(&x, &b);
            sb_puts(&b, "{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n");
            stream_send(&x, &b);
            sb_free(&b);
        }
    }
    const ml_status st = x.failed ? ML_OK : srv_generate(&sv->eng, toks, np, &p.rq, p.stream ? emit_piece : NULL, &x, &reply, res);
    status = 200;
    const char *finish = res->finish == SRV_FINISH_STOP ? "stop" : "length";
    if (st != ML_OK) {
        status = 500;
        if (!p.stream) send_error(sv, c, 500, ml_status_str(st), NULL, NULL);
    } else if (p.stream) {
        if (!x.failed && res->finish != SRV_FINISH_ABORT) {
            chunk_head(&x, &b);
            sb_printf(&b, chat ? "{\"index\":0,\"delta\":{},\"logprobs\":null,\"finish_reason\":\"%s\"}]}\n\n"
                               : "{\"index\":0,\"text\":\"\",\"logprobs\":null,\"finish_reason\":\"%s\"}]}\n\n", finish);
            if (p.include_usage) {
                chunk_head(&x, &b);
                sb_puts(&b, "],");
                usage_json(&b, res);
                sb_puts(&b, "}\n\n");
            }
            sb_puts(&b, "data: [DONE]\n\n");
            stream_send(&x, &b);
        }
    } else {
        sb_printf(&b, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", x.id,
                  chat ? "chat.completion" : "text_completion", x.created);
        sb_json_str(&b, sv->model_id, strlen(sv->model_id));
        sb_printf(&b, ",\"system_fingerprint\":\"%s\",\"choices\":[{\"index\":0,", sv->fingerprint);
        if (chat) {
            sb_puts(&b, "\"message\":{\"role\":\"assistant\",\"content\":");
            sb_json_str(&b, reply.p ? reply.p : "", reply.n);
            sb_puts(&b, "}");
        } else {
            sb_puts(&b, "\"text\":");
            sb_json_str(&b, reply.p ? reply.p : "", reply.n);
        }
        sb_printf(&b, ",\"logprobs\":null,\"finish_reason\":\"%s\"}],", finish);
        usage_json(&b, res);
        sb_puts(&b, "}");
        send_json(sv, c, 200, &b);
    }
    sb_free(&b);
    sb_free(&reply);
out:
    free(toks);
    js_free(&d);
    return status;
}

/* Compares in time independent of where the strings differ. */
static int same_secret(const char *a, const char *b) {
    const size_t na = strlen(a), nb = strlen(b);
    unsigned diff = (unsigned)(na ^ nb);
    for (size_t i = 0; i < na; i++) diff |= (unsigned char)a[i] ^ (unsigned char)(nb ? b[i % nb] : 0);
    return diff == 0;
}

static void handle(server *sv, ml_sock c, const char *peer) {
    http_req r;
    const double t0 = ml_now();
    int status = http_read(c, &r, MAX_BODY);
    if (status < 0) return;
    if (status > 0) {
        send_error(sv, c, status, http_reason(status), NULL, NULL);
        fprintf(stderr, "%s - %d\n", peer, status);
        return;
    }
    const char *path = r.path;
    if (!strncmp(path, "/v1/", 4)) path += 3;
    const int get = !strcmp(r.method, "GET"), post = !strcmp(r.method, "POST");
    srv_result res;
    memset(&res, 0, sizeof res);
    const char *bearer = "";   /* the scheme name is case-insensitive */
    if ((r.auth[0] == 'B' || r.auth[0] == 'b') && !strncmp(r.auth + 1, "earer ", 6)) bearer = r.auth + 7;
    else if (!strncmp(r.auth, "BEARER ", 7)) bearer = r.auth + 7;
    sbuf b = { 0 };
    int done = 0;
    if (!strcmp(r.method, "OPTIONS")) {   /* CORS preflight */
        http_respond(c, 204, "text/plain", sv->extra, NULL, 0);
        status = 204;
        done = 1;
    } else if (!strcmp(path, "/health")) {
        if (get) { sb_puts(&b, "{\"status\":\"ok\"}"); send_json(sv, c, status = 200, &b); done = 1; }
    } else if (sv->api_key && !same_secret(bearer, sv->api_key)) {
        send_error(sv, c, status = 401, "missing or incorrect API key: send 'Authorization: Bearer KEY'", NULL, "invalid_api_key");
        done = 1;
    } else if (!strcmp(path, "/models")) {
        if (get) {
            sb_puts(&b, "{\"object\":\"list\",\"data\":[");
            model_json(sv, &b);
            sb_puts(&b, "]}");
            send_json(sv, c, status = 200, &b);
            done = 1;
        }
    } else if (!strncmp(path, "/models/", 8)) {
        if (get && !strcmp(path + 8, sv->model_id)) { model_json(sv, &b); send_json(sv, c, status = 200, &b); done = 1; }
        else if (get) { send_error(sv, c, status = 404, "no such model", "model", "model_not_found"); done = 1; }
    } else if (!strcmp(path, "/chat/completions") || !strcmp(path, "/completions")) {
        if (post) { status = complete(sv, c, &r, path[1] == 'c' && path[2] == 'h', &res); done = 1; }
    } else {
        send_error(sv, c, status = 404, "no such endpoint", NULL, NULL);
        done = 1;
    }
    if (!done) send_error(sv, c, status = 405, "method not allowed", NULL, NULL);
    sb_free(&b);
    fprintf(stderr, "%s %s %s %d", peer, r.method, r.path, status);
    if (res.prompt_tokens)
        fprintf(stderr, ": prompt %d tokens (%d cached), reply %d tokens%s, prefill %.1f tok/s, decode %.2f tok/s",
                res.prompt_tokens, res.cached_tokens, res.completion_tokens,
                res.finish == SRV_FINISH_ABORT ? " (client left)" : "",
                res.t_prefill > 0 ? (res.prompt_tokens - res.cached_tokens) / res.t_prefill : 0.0,
                res.t_decode > 0 ? res.completion_tokens / res.t_decode : 0.0);
    fprintf(stderr, " (%.0f ms)\n", (ml_now() - t0) * 1e3);
    http_req_free(&r);
}

int ml_main(int argc, char **argv) {
    const char *model_path = NULL, *tok_path = NULL, *chat = NULL, *host = "127.0.0.1", *alias = NULL;
    server sv;
    memset(&sv, 0, sizeof sv);
    sv.tpl = -1;
    sv.eos_opt = -1;
    sv.so = (ml_sampler_opts){ 0.0f, 0.9f, 0, 1.0f, 64, 1 };
    int port = 8080, nthreads = 1, kv_type = ML_F32, batch = ML_BATCH_MAX, ctx = 0, no_pack = 0, pack_cache = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            printf("%skrafla %s, build ISA: %s\n", USAGE, ML_VERSION, ML_ISA_NAME);
            return 0;
        }
        if (!strcmp(a, "--no-pack")) { no_pack = 1; continue; }
        if (!strcmp(a, "--pack-cache")) { pack_cache = 1; continue; }
        if (a[0] != '-') { if (model_path) return usage("more than one model given"); model_path = a; continue; }
        if (i + 1 >= argc) return usage("option needs a value");
        const char *v = argv[++i];
        int ok = 1;
        unsigned long long seed;
        char *end;
        if (!strcmp(a, "-z")) tok_path = v;
        else if (!strcmp(a, "--chat")) ok = (sv.tpl = ml_chat_template(chat = v)) >= 0;
        else if (!strcmp(a, "--system")) sv.system = v;
        else if (!strcmp(a, "--host")) host = v;
        else if (!strcmp(a, "--port")) ok = parse_int(v, 0, 65535, &port);
        else if (!strcmp(a, "--api-key")) ok = (sv.api_key = v)[0] != '\0';
        else if (!strcmp(a, "--alias")) ok = (alias = v)[0] != '\0' && strlen(v) < sizeof sv.model_id;
        else if (!strcmp(a, "--cors")) ok = (sv.cors = v)[0] != '\0' && strlen(v) < 200 && !strpbrk(v, "\r\n");
        else if (!strcmp(a, "--ctx")) ok = parse_int(v, 1, 1 << 24, &ctx);
        else if (!strcmp(a, "-n")) ok = parse_int(v, 0, 1 << 24, &sv.max_tokens);
        else if (!strcmp(a, "-T")) ok = parse_float(v, 0.0f, 1e3f, &sv.so.temperature);
        else if (!strcmp(a, "-p")) ok = parse_float(v, 0.0f, 1.0f, &sv.so.topp);
        else if (!strcmp(a, "--top-k")) ok = parse_int(v, 0, 1 << 24, &sv.so.topk);
        else if (!strcmp(a, "--repeat-penalty")) ok = parse_float(v, 1e-3f, 1e3f, &sv.so.repeat_penalty);
        else if (!strcmp(a, "--repeat-last-n")) ok = parse_int(v, 0, 1 << 24, &sv.so.repeat_last_n);
        else if (!strcmp(a, "-s")) {
            errno = 0;
            seed = strtoull(v, &end, 10);
            ok = end != v && !*end && errno != ERANGE && v[0] != '-';
            sv.so.seed = seed;
            sv.seed_given = 1;
        }
        else if (!strcmp(a, "--eos")) ok = parse_int(v, 0, 1 << 24, &sv.eos_opt);
        else if (!strcmp(a, "-t")) ok = parse_int(v, 1, 64, &nthreads);
        else if (!strcmp(a, "--batch")) ok = parse_int(v, 1, ML_BATCH_MAX, &batch);
        else if (!strcmp(a, "--kv")) {
            if (!strcmp(v, "f32")) kv_type = ML_F32;
            else if (!strcmp(v, "q8")) kv_type = ML_Q8_0;
            else if (!strcmp(v, "f16")) kv_type = ML_F16;
            else ok = 0;
        }
        else return usage("unknown option");
        if (!ok) { fprintf(stderr, "error: bad value '%s' for %s\n", v, a); return usage(NULL); }
    }
    if (!model_path) return usage(NULL);
    if (!tok_path) return usage("ml_server needs -z tokenizer");
    if (pack_cache && no_pack) return usage("give --pack-cache or --no-pack, not both");
    if (sv.system && sv.tpl < 0) return usage("--system needs --chat");
    if (!sv.api_key && getenv("KRAFLA_API_KEY") && getenv("KRAFLA_API_KEY")[0]) sv.api_key = getenv("KRAFLA_API_KEY");
    if (alias) snprintf(sv.model_id, sizeof sv.model_id, "%s", alias);
    else {
        const char *base = model_path;
        for (const char *q = model_path; *q; q++)
            if (*q == '/' || *q == '\\') base = q + 1;
        snprintf(sv.model_id, sizeof sv.model_id, "%s", base);
        const size_t n = strlen(sv.model_id);
        if (n > 4 && !strcmp(sv.model_id + n - 4, ".mlb")) sv.model_id[n - 4] = '\0';
    }
    snprintf(sv.fingerprint, sizeof sv.fingerprint, "krafla-%s-%s", ML_VERSION, ML_ISA_NAME);
    if (sv.cors)
        snprintf(sv.extra, sizeof sv.extra, "Access-Control-Allow-Origin: %s\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                 "Access-Control-Allow-Headers: Authorization, Content-Type\r\n", sv.cors);

    ml_status st = ml_model_load(&sv.m, model_path);
    if (st != ML_OK) { fprintf(stderr, "error: %s: %s\n", model_path, ml_status_str(st)); return 1; }
    const ml_config *c = &sv.m.cfg;
    int rc = 1;
    if (sv.eos_opt >= c->vocab) { ml_model_free(&sv.m); return usage("--eos id out of range"); }
    if (ctx > c->max_seq) {
        fprintf(stderr, "error: --ctx %d is above the model's max_seq (%d)\n", ctx, c->max_seq);
        ml_model_free(&sv.m);
        return 2;
    }
    if ((st = ml_tokenizer_load(&sv.tok, tok_path, c->vocab)) != ML_OK) {
        fprintf(stderr, "error: %s: %s\n", tok_path, ml_status_str(st));
        ml_model_free(&sv.m);
        return 1;
    }
    if ((st = ml_state_init(&sv.s, &sv.m, nthreads, kv_type, ctx)) != ML_OK) {
        fprintf(stderr, "error: %s\n", ml_status_str(st));
        goto out_tok;
    }
    if (!no_pack && pack_cache && ml_pack_group()) {
        char cache[4096];
        const char *how;
        snprintf(cache, sizeof cache, "%s.pk%d", model_path, ml_pack_group());
        st = ml_model_pack_cached(&sv.m, sv.s.pool, cache, &how);
        if (st == ML_OK) fprintf(stderr, "pack cache %s: %s\n", cache, how);
    } else if (!no_pack) {
        st = ml_model_pack(&sv.m, sv.s.pool);
    }
    if (st != ML_OK) { fprintf(stderr, "error: packing weights: %s\n", ml_status_str(st)); goto out_state; }
    sv.s.batch = batch;
    if ((st = srv_engine_init(&sv.eng, &sv.s, &sv.tok)) != ML_OK) { fprintf(stderr, "error: %s\n", ml_status_str(st)); goto out_state; }
    char err[256];
    int bound = port;
    ml_sock ls = ML_SOCK_BAD;
    if (ml_net_init() == 0) ls = ml_net_listen(host, port, &bound, err, sizeof err);
    else snprintf(err, sizeof err, "cannot start the network library");
    if (ls == ML_SOCK_BAD) { fprintf(stderr, "error: %s\n", err); goto out_engine; }
    sv.started = (long)time(NULL);
    const int v6 = strchr(host, ':') != NULL;
    fprintf(stderr, "krafla %s, isa %s: model %s, context %d of %d, chat %s, threads %d, kv %s%s\n", ML_VERSION,
            ML_ISA_NAME, sv.model_id, sv.s.ctx, c->max_seq, chat ? chat : "none", ml_pool_size(sv.s.pool),
            kv_type == ML_Q8_0 ? "q8" : kv_type == ML_F16 ? "f16" : "f32", sv.api_key ? ", API key required" : "");
    fprintf(stderr, "listening on http://%s%s%s:%d/v1\n", v6 ? "[" : "", host, v6 ? "]" : "", bound);
    fflush(stderr);
    for (;;) {
        char peer[64];
        const ml_sock cs = ml_net_accept(ls, peer, sizeof peer);
        if (cs == ML_SOCK_BAD) continue;
        ml_net_timeouts(cs, 60);
        handle(&sv, cs, peer);
        ml_net_close(cs);
    }
out_engine:
    srv_engine_free(&sv.eng);
out_state:
    ml_state_free(&sv.s);
out_tok:
    ml_tokenizer_free(&sv.tok);
    ml_model_free(&sv.m);
    return rc;
}
