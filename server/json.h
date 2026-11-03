/* JSON for the HTTP server: a parser into an array of nodes, and a growable
 * string buffer with a JSON string writer. */
#ifndef ML_JSON_H
#define ML_JSON_H
#include <stddef.h>

enum { JS_NULL, JS_FALSE, JS_TRUE, JS_NUM, JS_STR, JS_ARR, JS_OBJ };

typedef struct {
    int type;            /* JS_* */
    int child, next;     /* first member (arrays, objects) and next sibling, by index; -1: none */
    int len;             /* JS_STR: bytes; JS_ARR, JS_OBJ: members */
    size_t key;          /* object members: offset of the key in strs */
    size_t str;          /* JS_STR: offset of the value in strs (NUL-terminated) */
    double num;          /* JS_NUM */
} js_node;

/* A parsed document. Node 0 is the root. Strings are stored decoded and
 * NUL-terminated in strs (a string holding \u0000 ends there for C use). */
typedef struct {
    js_node *n;
    int count, cap;
    char *strs;
    const char *err;     /* the first error, for messages */
    size_t err_at;       /* its byte offset */
} js_doc;

#define JS_MAX_DEPTH 64
#define JS_MAX_NODES (1 << 20)   /* values per document */

/* Parses text[0, len). 0 on success; -1 on a syntax error (d->err, d->err_at)
 * or when memory runs out (d->err "out of memory"). Free with js_free either way. */
int js_parse(js_doc *d, const char *text, size_t len);
void js_free(js_doc *d);
/* Index of member `key` of object node obj, or -1 (also when obj is not an object). */
int js_get(const js_doc *d, int obj, const char *key);
/* The type of node i, JS_NULL for -1 (a missing member). */
int js_type(const js_doc *d, int i);
/* The value of a JS_STR node, or NULL. */
const char *js_str(const js_doc *d, int i);
const char *js_key(const js_doc *d, int i);

/* Growable byte buffer, kept NUL-terminated. On allocation failure oom is set
 * and further appends are dropped. */
typedef struct {
    char *p;
    size_t n, cap;
    int oom;
} sbuf;

void sb_put(sbuf *b, const char *s, size_t n);
void sb_puts(sbuf *b, const char *s);
void sb_printf(sbuf *b, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;
/* s[0, n) as a quoted JSON string: escapes quotes, backslashes and control
 * characters, and replaces bytes that are not valid UTF-8 with U+FFFD. */
void sb_json_str(sbuf *b, const char *s, size_t n);
void sb_free(sbuf *b);

/* End of the longest prefix of s[0, n) that does not stop inside a UTF-8
 * sequence: n, or the start of a lead byte at the end whose continuation
 * bytes have not all arrived yet. */
size_t utf8_complete(const char *s, size_t n);

#endif
