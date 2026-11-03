/* The HTTP server's pure parts: JSON parsing and writing, UTF-8 boundaries,
 * and stop strings over text arriving in pieces. */
#include <string.h>
#include "server/engine.h"
#include "server/json.h"
#include "tests/tt.h"

static int parse(js_doc *d, const char *s) { return js_parse(d, s, strlen(s)); }

static void test_json_values(void) {
    js_doc d;
    TT_EQ_INT(parse(&d, " {\"a\": [1, -2.5e2, true, false, null], \"b\": {\"c\": \"x\\ty\"}, \"e\": {}} "), 0);
    TT_EQ_INT(js_type(&d, 0), JS_OBJ);
    TT_EQ_INT(d.n[0].len, 3);
    const int a = js_get(&d, 0, "a");
    TT_EQ_INT(js_type(&d, a), JS_ARR);
    TT_EQ_INT(d.n[a].len, 5);
    int i = d.n[a].child;
    TT_ASSERT(d.n[i].num == 1.0);
    i = d.n[i].next;
    TT_ASSERT(d.n[i].num == -250.0);
    i = d.n[i].next;
    TT_EQ_INT(js_type(&d, i), JS_TRUE);
    i = d.n[i].next;
    TT_EQ_INT(js_type(&d, i), JS_FALSE);
    i = d.n[i].next;
    TT_EQ_INT(js_type(&d, i), JS_NULL);
    TT_EQ_INT(d.n[i].next, -1);
    const char *c = js_str(&d, js_get(&d, js_get(&d, 0, "b"), "c"));
    TT_ASSERT(c && !strcmp(c, "x\ty"));
    TT_EQ_INT(js_type(&d, js_get(&d, 0, "e")), JS_OBJ);
    TT_EQ_INT(js_get(&d, 0, "missing"), -1);
    TT_EQ_INT(js_type(&d, js_get(&d, 0, "missing")), JS_NULL);
    TT_EQ_INT(js_get(&d, a, "x"), -1);   /* not an object */
    js_free(&d);
}

static void test_json_unicode(void) {
    js_doc d;
    /* BMP escape, a surrogate pair, a lone surrogate, raw UTF-8 */
    TT_EQ_INT(parse(&d, "[\"\\u00e9\\ud83d\\ude00\\ud800x\", \"\xea\xb0\x80\"]"), 0);
    const char *s = js_str(&d, d.n[0].child);
    TT_ASSERT(s && !strcmp(s, "\xc3\xa9\xf0\x9f\x98\x80\xef\xbf\xbdx"));
    TT_ASSERT(!strcmp(js_str(&d, d.n[d.n[0].child].next), "\xea\xb0\x80"));
    js_free(&d);
}

static void test_json_errors(void) {
    static const char *const bad[] = {
        "", "{", "[1,]", "{\"a\" 1}", "{\"a\":1,}", "[01]", "[1.]", "[-]", "[1e]", "\"abc", "\"a\nb\"",
        "\"\\x\"", "\"\\u12\"", "tru", "[1] 2", "{1:2}", "nul", "[\"\\uZZZZ\"]",
    };
    for (size_t k = 0; k < sizeof bad / sizeof bad[0]; k++) {
        js_doc d;
        const int rc = parse(&d, bad[k]);
        if (rc != -1 || !d.err) fprintf(stderr, "accepted: %s\n", bad[k]);
        TT_EQ_INT(rc, -1);
        TT_ASSERT(d.err != NULL);
        js_free(&d);
    }
    char deep[2 * JS_MAX_DEPTH + 8];
    memset(deep, '[', JS_MAX_DEPTH + 1);
    memset(deep + JS_MAX_DEPTH + 1, ']', JS_MAX_DEPTH + 1);
    deep[2 * JS_MAX_DEPTH + 2] = '\0';
    js_doc d;
    TT_EQ_INT(parse(&d, deep), -1);
    js_free(&d);
    deep[JS_MAX_DEPTH] = ' ';   /* exactly JS_MAX_DEPTH levels */
    deep[JS_MAX_DEPTH + 1] = ' ';
    TT_EQ_INT(parse(&d, deep), 0);
    js_free(&d);
}

static void test_json_writer(void) {
    sbuf b = { 0 };
    const char s[] = "a\"b\\c\n\x01\xc3\xa9\xff\xe2\x82";
    sb_json_str(&b, s, sizeof s - 1);
    /* \xff alone, then the truncated \xe2\x82 as one U+FFFD (its maximal subpart) */
    TT_ASSERT(b.p && !strcmp(b.p, "\"a\\\"b\\\\c\\n\\u0001\xc3\xa9\\ufffd\\ufffd\""));
    sb_free(&b);
    const char s2[] = "\xe0\x80x\xed\xa0\x80\xf4\x90";   /* overlong, surrogate, above U+10FFFF */
    sb_json_str(&b, s2, sizeof s2 - 1);
    TT_ASSERT(b.p && !strcmp(b.p, "\"\\ufffd\\ufffdx\\ufffd\\ufffd\\ufffd\\ufffd\\ufffd\""));
    sb_free(&b);
    sb_printf(&b, "%d-%s", 42, "x");
    TT_ASSERT(b.p && !strcmp(b.p, "42-x") && b.n == 4);
    sb_free(&b);
    /* round trip: what the writer escapes, the parser restores */
    sb_json_str(&b, "tab\there \xe2\x82\xac", 13);
    js_doc d;
    TT_EQ_INT(js_parse(&d, b.p, b.n), 0);
    TT_ASSERT(!strcmp(js_str(&d, 0), "tab\there \xe2\x82\xac"));
    js_free(&d);
    sb_free(&b);
}

static void test_utf8_complete(void) {
    TT_EQ_INT(utf8_complete("abc", 3), 3);
    TT_EQ_INT(utf8_complete("a\xe2\x82", 3), 1);
    TT_EQ_INT(utf8_complete("a\xe2\x82\xac", 4), 4);
    TT_EQ_INT(utf8_complete("\xf0\x9f\x98", 3), 0);
    TT_EQ_INT(utf8_complete("\xf0\x9f\x98\x80", 4), 4);
    TT_EQ_INT(utf8_complete("a\xc3", 2), 1);
    TT_EQ_INT(utf8_complete("", 0), 0);
}

/* Feeds pieces, collecting what is released; returns whether a stop hit. */
static int feed_all(const char *const *stop, int nstop, const char *const *pieces, int n, char *out) {
    srv_stop t = { { NULL, 0, 0, 0 }, 0, stop, nstop };
    out[0] = '\0';
    int hit = 0;
    for (int i = 0; i < n && !hit; i++) {
        size_t from, k;
        hit = srv_stop_feed(&t, pieces[i], strlen(pieces[i]), i == n - 1, &from, &k);
        strncat(out, t.text.p + from, k);
    }
    sb_free(&t.text);
    return hit;
}

static void test_stop_strings(void) {
    char out[256];
    static const char *const stop1[] = { "END" }, *const stop2[] = { "\n\n", "User:" };
    static const char *const p1[] = { "hello E", "N", "D more" };
    TT_EQ_INT(feed_all(stop1, 1, p1, 3, out), 1);
    TT_ASSERT(!strcmp(out, "hello "));
    static const char *const p2[] = { "a E", "Nx", " b" };   /* a stop prefix that does not complete */
    TT_EQ_INT(feed_all(stop1, 1, p2, 3, out), 0);
    TT_ASSERT(!strcmp(out, "a ENx b"));
    static const char *const p3[] = { "one\n", "two Us", "er: x" };
    TT_EQ_INT(feed_all(stop2, 2, p3, 3, out), 1);
    TT_ASSERT(!strcmp(out, "one\ntwo "));
    static const char *const p4[] = { "x\xe2", "\x82", "\xac y" };   /* a UTF-8 sequence split three ways */
    srv_stop t = { { NULL, 0, 0, 0 }, 0, NULL, 0 };
    size_t from, k;
    srv_stop_feed(&t, p4[0], 2, 0, &from, &k);
    TT_EQ_INT(k, 1);
    srv_stop_feed(&t, p4[1], 1, 0, &from, &k);
    TT_EQ_INT(k, 0);
    srv_stop_feed(&t, p4[2], 3, 0, &from, &k);
    TT_EQ_INT(k, 5);
    TT_EQ_INT(from, 1);
    sb_free(&t.text);
    static const char *const p5[] = { "ab", "c" };   /* flush releases a held-back prefix */
    static const char *const stop3[] = { "cd" };
    TT_EQ_INT(feed_all(stop3, 1, p5, 2, out), 0);
    TT_ASSERT(!strcmp(out, "abc"));
}

int TT_ENTRY(int argc, char **argv) {
    (void)argc;
    (void)argv;
    TT_RUN(test_json_values);
    TT_RUN(test_json_unicode);
    TT_RUN(test_json_errors);
    TT_RUN(test_json_writer);
    TT_RUN(test_utf8_complete);
    TT_RUN(test_stop_strings);
    TT_MAIN_END("test_server");
}
