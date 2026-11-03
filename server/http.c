#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "server/http.h"

static int ieq(const char *a, const char *b) {
    for (; *a && *b; a++, b++)
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
    return *a == *b;
}

static void copy_trim(char *dst, size_t cap, const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    size_t n = strlen(s);
    while (n && (s[n - 1] == ' ' || s[n - 1] == '\t')) n--;
    if (n >= cap) n = cap - 1;
    memcpy(dst, s, n);
    dst[n] = '\0';
}

/* Parses the request line and headers in h (NUL-terminated, lines ending in
 * \r\n or \n). 0, or 400. */
static int parse_head(char *h, http_req *r) {
    char *line = h, *next;
    int first = 1;
    for (; *line; line = next) {
        char *eol = strchr(line, '\n');
        next = eol ? eol + 1 : line + strlen(line);
        if (eol) {
            if (eol > line && eol[-1] == '\r') eol[-1] = '\0';
            *eol = '\0';
        } else if (*line && line[strlen(line) - 1] == '\r') {
            line[strlen(line) - 1] = '\0';
        }
        if (first) {
            char *sp1 = strchr(line, ' '), *sp2 = sp1 ? strchr(sp1 + 1, ' ') : NULL;
            if (!sp1 || !sp2 || (size_t)(sp1 - line) >= sizeof r->method || strncmp(sp2 + 1, "HTTP/1.", 7)) return 400;
            memcpy(r->method, line, (size_t)(sp1 - line));
            r->method[sp1 - line] = '\0';
            *sp2 = '\0';
            char *q = strchr(sp1 + 1, '?');
            if (q) *q = '\0';
            if (strlen(sp1 + 1) >= sizeof r->path || sp1[1] != '/') return 400;
            strcpy(r->path, sp1 + 1);
            first = 0;
            continue;
        }
        if (!*line) break;
        char *colon = strchr(line, ':');
        if (!colon) return 400;
        *colon = '\0';
        const char *v = colon + 1;
        if (ieq(line, "Content-Length")) {
            char *end;
            while (*v == ' ' || *v == '\t') v++;
            errno = 0;
            const long n = strtol(v, &end, 10);
            while (*end == ' ' || *end == '\t') end++;
            if (end == v || *end || n < 0 || errno == ERANGE) return 400;
            r->content_length = n;
        } else if (ieq(line, "Authorization")) copy_trim(r->auth, sizeof r->auth, v);
        else if (ieq(line, "Expect")) {
            char e[32];
            copy_trim(e, sizeof e, v);
            r->expect_continue = ieq(e, "100-continue");
        } else if (ieq(line, "Transfer-Encoding")) {
            char e[64];
            copy_trim(e, sizeof e, v);
            r->chunked = !ieq(e, "identity");
        }
    }
    return first ? 400 : 0;
}

int http_read(ml_sock c, http_req *r, size_t max_body) {
    memset(r, 0, sizeof *r);
    r->content_length = -1;
    char *buf = malloc(HTTP_MAX_HEADER + 1);
    if (!buf) return -1;
    size_t n = 0;
    char *end = NULL;
    while (!end) {
        if (n == HTTP_MAX_HEADER) { free(buf); return 431; }
        const long k = ml_net_recv(c, buf + n, HTTP_MAX_HEADER - n);
        if (k <= 0) { free(buf); return n ? 408 : -1; }
        n += (size_t)k;
        buf[n] = '\0';
        end = strstr(buf, "\r\n\r\n");
        if (end) end += 4;
        else if ((end = strstr(buf, "\n\n")) != NULL) end += 2;
    }
    const size_t head = (size_t)(end - buf), extra = n - head;
    end[-1] = '\0';
    int st = parse_head(buf, r);
    if (!st && r->chunked) st = 411;
    if (!st && r->content_length > 0 && (size_t)r->content_length > max_body) st = 413;
    if (st || r->content_length <= 0) { free(buf); return st; }
    const size_t len = (size_t)r->content_length;
    r->body = malloc(len + 1);
    if (!r->body) { free(buf); return 413; }
    const size_t have = extra < len ? extra : len;
    memcpy(r->body, buf + head, have);
    free(buf);
    if (have < len && r->expect_continue) {
        static const char cont[] = "HTTP/1.1 100 Continue\r\n\r\n";
        if (ml_net_send(c, cont, sizeof cont - 1)) return -1;
    }
    for (size_t got = have; got < len;) {
        const long k = ml_net_recv(c, r->body + got, len - got);
        if (k <= 0) return 408;
        got += (size_t)k;
    }
    r->body[len] = '\0';
    return 0;
}

void http_req_free(http_req *r) {
    free(r->body);
    r->body = NULL;
}

const char *http_reason(int status) {
    switch (status) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    default: return "Error";
    }
}

int http_respond(ml_sock c, int status, const char *ctype, const char *extra, const char *body, size_t n) {
    char head[2048];
    const int k = snprintf(head, sizeof head,
                           "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %lu\r\nConnection: close\r\n%s\r\n",
                           status, http_reason(status), ctype, (unsigned long)n, extra ? extra : "");
    if (k < 0 || (size_t)k >= sizeof head) return -1;
    if (ml_net_send(c, head, (size_t)k)) return -1;
    return n ? ml_net_send(c, body, n) : 0;
}

int http_stream_start(ml_sock c, const char *extra) {
    char head[2048];
    const int k = snprintf(head, sizeof head,
                           "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                           "Connection: close\r\n%s\r\n", extra ? extra : "");
    if (k < 0 || (size_t)k >= sizeof head) return -1;
    return ml_net_send(c, head, (size_t)k);
}
