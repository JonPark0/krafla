#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "server/json.h"

typedef struct {
    js_doc *d;
    const char *s;
    size_t n, i;   /* text, its length, the read position */
    size_t sp;     /* next free byte of d->strs */
    int depth;
} parser;

static int fail(parser *p, const char *msg) {
    if (!p->d->err) { p->d->err = msg; p->d->err_at = p->i; }
    return -1;
}

static void skip_ws(parser *p) {
    while (p->i < p->n && (p->s[p->i] == ' ' || p->s[p->i] == '\t' || p->s[p->i] == '\n' || p->s[p->i] == '\r')) p->i++;
}

static int new_node(parser *p, int type) {
    js_doc *d = p->d;
    if (d->count >= JS_MAX_NODES) return fail(p, "too many values");
    if (d->count == d->cap) {
        const int cap = d->cap ? 2 * d->cap : 64;
        js_node *n = cap > d->cap ? realloc(d->n, sizeof *n * (size_t)cap) : NULL;
        if (!n) { d->err = NULL; return fail(p, "out of memory"); }
        d->n = n;
        d->cap = cap;
    }
    js_node *x = &d->n[d->count];
    memset(x, 0, sizeof *x);
    x->type = type;
    x->child = x->next = -1;
    return d->count++;
}

static int hex4(parser *p, unsigned *v) {
    if (p->n - p->i < 4) return fail(p, "bad \\u escape");
    *v = 0;
    for (int k = 0; k < 4; k++) {
        const char c = p->s[p->i++];
        const unsigned h = c >= '0' && c <= '9' ? (unsigned)(c - '0')
                         : c >= 'a' && c <= 'f' ? (unsigned)(c - 'a' + 10)
                         : c >= 'A' && c <= 'F' ? (unsigned)(c - 'A' + 10) : 16u;
        if (h > 15) return fail(p, "bad \\u escape");
        *v = *v << 4 | h;
    }
    return 0;
}

static void put_utf8(parser *p, unsigned c) {
    char *o = p->d->strs + p->sp;
    if (c < 0x80) { o[0] = (char)c; p->sp += 1; }
    else if (c < 0x800) { o[0] = (char)(0xC0 | c >> 6); o[1] = (char)(0x80 | (c & 63)); p->sp += 2; }
    else if (c < 0x10000) {
        o[0] = (char)(0xE0 | c >> 12); o[1] = (char)(0x80 | (c >> 6 & 63)); o[2] = (char)(0x80 | (c & 63)); p->sp += 3;
    } else {
        o[0] = (char)(0xF0 | c >> 18); o[1] = (char)(0x80 | (c >> 12 & 63));
        o[2] = (char)(0x80 | (c >> 6 & 63)); o[3] = (char)(0x80 | (c & 63)); p->sp += 4;
    }
}

/* At the opening quote. Decodes into strs at sp (never longer than the
 * source, quotes included, so strs of the text's length always has room). */
static int parse_string(parser *p, size_t *off, int *len) {
    p->i++;
    *off = p->sp;
    for (;;) {
        if (p->i >= p->n) return fail(p, "unterminated string");
        const unsigned char c = (unsigned char)p->s[p->i++];
        if (c == '"') break;
        if (c < 0x20) return fail(p, "control character in string");
        if (c != '\\') { p->d->strs[p->sp++] = (char)c; continue; }
        if (p->i >= p->n) return fail(p, "unterminated string");
        const char e = p->s[p->i++];
        unsigned u;
        switch (e) {
        case '"': case '\\': case '/': p->d->strs[p->sp++] = e; break;
        case 'b': p->d->strs[p->sp++] = '\b'; break;
        case 'f': p->d->strs[p->sp++] = '\f'; break;
        case 'n': p->d->strs[p->sp++] = '\n'; break;
        case 'r': p->d->strs[p->sp++] = '\r'; break;
        case 't': p->d->strs[p->sp++] = '\t'; break;
        case 'u':
            if (hex4(p, &u)) return -1;
            if (u >= 0xD800 && u < 0xDC00 && p->n - p->i >= 6 && p->s[p->i] == '\\' && p->s[p->i + 1] == 'u') {
                const size_t at = p->i;
                unsigned lo;
                p->i += 2;
                if (hex4(p, &lo)) return -1;
                if (lo >= 0xDC00 && lo < 0xE000) u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                else p->i = at;   /* not a pair: the high half alone becomes U+FFFD */
            }
            if (u >= 0xD800 && u < 0xE000) u = 0xFFFD;   /* lone surrogate */
            put_utf8(p, u);
            break;
        default:
            p->i--;
            return fail(p, "bad escape in string");
        }
    }
    *len = (int)(p->sp - *off);
    p->d->strs[p->sp++] = '\0';
    return 0;
}

static int parse_number(parser *p, double *v) {
    const size_t start = p->i;
    if (p->i < p->n && p->s[p->i] == '-') p->i++;
    if (p->i < p->n && p->s[p->i] == '0') p->i++;
    else if (p->i < p->n && p->s[p->i] >= '1' && p->s[p->i] <= '9')
        while (p->i < p->n && p->s[p->i] >= '0' && p->s[p->i] <= '9') p->i++;
    else return fail(p, "bad number");
    if (p->i < p->n && p->s[p->i] == '.') {
        p->i++;
        if (!(p->i < p->n && p->s[p->i] >= '0' && p->s[p->i] <= '9')) return fail(p, "bad number");
        while (p->i < p->n && p->s[p->i] >= '0' && p->s[p->i] <= '9') p->i++;
    }
    if (p->i < p->n && (p->s[p->i] == 'e' || p->s[p->i] == 'E')) {
        p->i++;
        if (p->i < p->n && (p->s[p->i] == '+' || p->s[p->i] == '-')) p->i++;
        if (!(p->i < p->n && p->s[p->i] >= '0' && p->s[p->i] <= '9')) return fail(p, "bad number");
        while (p->i < p->n && p->s[p->i] >= '0' && p->s[p->i] <= '9') p->i++;
    }
    char buf[64];
    if (p->i - start >= sizeof buf) return fail(p, "number too long");
    memcpy(buf, p->s + start, p->i - start);
    buf[p->i - start] = '\0';
    *v = strtod(buf, NULL);
    return 0;
}

static int parse_value(parser *p);

/* Parses the members of an array or object (at its opening bracket) into node x. */
static int parse_members(parser *p, int x, int obj) {
    const char close = obj ? '}' : ']';
    if (++p->depth > JS_MAX_DEPTH) return fail(p, "nested too deeply");
    p->i++;
    skip_ws(p);
    int last = -1, count = 0;
    if (p->i < p->n && p->s[p->i] == close) { p->i++; p->depth--; return 0; }
    for (;;) {
        size_t key = 0;
        if (obj) {
            int klen;
            if (p->i >= p->n || p->s[p->i] != '"') return fail(p, "expected a member name");
            if (parse_string(p, &key, &klen)) return -1;
            skip_ws(p);
            if (p->i >= p->n || p->s[p->i] != ':') return fail(p, "expected ':'");
            p->i++;
            skip_ws(p);
        }
        const int v = parse_value(p);
        if (v < 0) return -1;
        p->d->n[v].key = key;
        if (last < 0) p->d->n[x].child = v;
        else p->d->n[last].next = v;
        last = v;
        count++;
        skip_ws(p);
        if (p->i < p->n && p->s[p->i] == ',') { p->i++; skip_ws(p); continue; }
        if (p->i < p->n && p->s[p->i] == close) { p->i++; break; }
        return fail(p, obj ? "expected ',' or '}'" : "expected ',' or ']'");
    }
    p->d->n[x].len = count;
    p->depth--;
    return 0;
}

static int literal(parser *p, const char *word, int type) {
    const size_t k = strlen(word);
    if (p->n - p->i < k || memcmp(p->s + p->i, word, k)) return fail(p, "unexpected character");
    p->i += k;
    return new_node(p, type);
}

/* Returns the new node's index, or -1. */
static int parse_value(parser *p) {
    if (p->i >= p->n) return fail(p, "unexpected end of input");
    const char c = p->s[p->i];
    int x;
    switch (c) {
    case '{':
    case '[':
        x = new_node(p, c == '{' ? JS_OBJ : JS_ARR);
        if (x < 0 || parse_members(p, x, c == '{')) return -1;
        return x;
    case '"': {
        x = new_node(p, JS_STR);
        if (x < 0) return -1;
        size_t off;
        int len;
        if (parse_string(p, &off, &len)) return -1;
        p->d->n[x].str = off;
        p->d->n[x].len = len;
        return x;
    }
    case 't': return literal(p, "true", JS_TRUE);
    case 'f': return literal(p, "false", JS_FALSE);
    case 'n': return literal(p, "null", JS_NULL);
    default:
        if (c == '-' || (c >= '0' && c <= '9')) {
            double v;
            if (parse_number(p, &v)) return -1;
            x = new_node(p, JS_NUM);
            if (x >= 0) p->d->n[x].num = v;
            return x;
        }
        return fail(p, "unexpected character");
    }
}

int js_parse(js_doc *d, const char *text, size_t len) {
    memset(d, 0, sizeof *d);
    d->strs = malloc(len + 1);
    if (!d->strs) { d->err = "out of memory"; return -1; }
    parser p = { d, text, len, 0, 0, 0 };
    skip_ws(&p);
    if (parse_value(&p) < 0) return -1;
    skip_ws(&p);
    if (p.i != len) return fail(&p, "unexpected data after the value");
    return 0;
}

void js_free(js_doc *d) {
    free(d->n);
    free(d->strs);
    d->n = NULL;
    d->strs = NULL;
    d->count = d->cap = 0;
}

int js_get(const js_doc *d, int obj, const char *key) {
    if (obj < 0 || d->n[obj].type != JS_OBJ) return -1;
    for (int i = d->n[obj].child; i >= 0; i = d->n[i].next)
        if (!strcmp(d->strs + d->n[i].key, key)) return i;
    return -1;
}

int js_type(const js_doc *d, int i) { return i < 0 ? JS_NULL : d->n[i].type; }

const char *js_str(const js_doc *d, int i) { return i >= 0 && d->n[i].type == JS_STR ? d->strs + d->n[i].str : NULL; }

const char *js_key(const js_doc *d, int i) { return d->strs + d->n[i].key; }

static int sb_reserve(sbuf *b, size_t extra) {
    if (b->oom) return 0;
    if (b->n + extra + 1 <= b->cap) return 1;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->n + extra + 1) {
        if (cap > (size_t)-1 / 2) { b->oom = 1; return 0; }
        cap *= 2;
    }
    char *p = realloc(b->p, cap);
    if (!p) { b->oom = 1; return 0; }
    b->p = p;
    b->cap = cap;
    return 1;
}

void sb_put(sbuf *b, const char *s, size_t n) {
    if (!sb_reserve(b, n)) return;
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = '\0';
}

void sb_puts(sbuf *b, const char *s) { sb_put(b, s, strlen(s)); }

void sb_printf(sbuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int k = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (k < 0 || !sb_reserve(b, (size_t)k)) return;
    va_start(ap, fmt);
    vsnprintf(b->p + b->n, (size_t)k + 1, fmt, ap);
    va_end(ap);
    b->n += (size_t)k;
}

/* Length of the well-formed UTF-8 sequence at s[0, n), or 0 if it is
 * ill-formed; then *bad is the length of its maximal subpart (the longest
 * start of a well-formed sequence, at least 1 byte), which becomes one
 * U+FFFD, as Unicode recommends and Python and browsers decode. */
static size_t utf8_seq(const unsigned char *s, size_t n, size_t *bad) {
    const unsigned c = s[0];
    unsigned lo = 0x80, hi = 0xBF;   /* allowed range of the second byte */
    size_t k;
    if (c < 0x80) return 1;
    if (c >= 0xC2 && c <= 0xDF) k = 2;
    else if (c >= 0xE0 && c <= 0xEF) { k = 3; if (c == 0xE0) lo = 0xA0; if (c == 0xED) hi = 0x9F; }
    else if (c >= 0xF0 && c <= 0xF4) { k = 4; if (c == 0xF0) lo = 0x90; if (c == 0xF4) hi = 0x8F; }
    else { *bad = 1; return 0; }
    size_t j = 1;
    for (; j < k && j < n; j++) {
        const unsigned b = s[j];
        if (b < (j == 1 ? lo : 0x80u) || b > (j == 1 ? hi : 0xBFu)) break;
    }
    if (j == k) return k;
    *bad = j;
    return 0;
}

void sb_json_str(sbuf *b, const char *s, size_t n) {
    const unsigned char *u = (const unsigned char *)s;
    sb_put(b, "\"", 1);
    size_t run = 0;   /* bytes copied as they are, pending */
    for (size_t i = 0; i < n;) {
        const unsigned c = u[i];
        const char *esc = c == '"' ? "\\\"" : c == '\\' ? "\\\\" : c == '\n' ? "\\n" : c == '\r' ? "\\r"
                        : c == '\t' ? "\\t" : c == '\b' ? "\\b" : c == '\f' ? "\\f" : NULL;
        size_t k = 1, bad = 1;
        if (!esc && c >= 0x20) {
            k = utf8_seq(u + i, n - i, &bad);
            if (k) { run += k; i += k; continue; }
        }
        sb_put(b, s + i - run, run);
        run = 0;
        if (esc) sb_puts(b, esc);
        else if (c < 0x20) sb_printf(b, "\\u%04x", c);
        else { sb_puts(b, "\\ufffd"); k = bad; }
        i += k;
    }
    sb_put(b, s + n - run, run);
    sb_put(b, "\"", 1);
}

void sb_free(sbuf *b) {
    free(b->p);
    b->p = NULL;
    b->n = b->cap = 0;
    b->oom = 0;
}

size_t utf8_complete(const char *s, size_t n) {
    const unsigned char *u = (const unsigned char *)s;
    for (size_t k = 1; k <= 3 && k <= n; k++) {
        const unsigned c = u[n - k];
        if ((c & 0xC0) == 0x80) continue;   /* continuation byte: look further back */
        const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return k < need ? n - k : n;
    }
    return n;
}
