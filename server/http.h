/* HTTP/1.1 for the server: one request per connection, answered and closed. */
#ifndef ML_HTTP_H
#define ML_HTTP_H
#include <stddef.h>
#include "server/net.h"

#define HTTP_MAX_HEADER (16 * 1024)

typedef struct {
    char method[16];
    char path[1024];         /* without the query string */
    char auth[512];          /* Authorization header */
    long content_length;     /* -1: none */
    int expect_continue;     /* Expect: 100-continue */
    int chunked;             /* Transfer-Encoding: chunked (not supported) */
    char *body;              /* content_length bytes, NUL-terminated; NULL without a body */
} http_req;

/* Reads one request. Returns 0, -1 if the client closed or timed out before
 * a request arrived, or an HTTP status (400, 408, 411, 413, 431) to answer. */
int http_read(ml_sock c, http_req *r, size_t max_body);
void http_req_free(http_req *r);
const char *http_reason(int status);
/* Sends a complete response with Content-Length and Connection: close.
 * extra: more header lines, each ending in \r\n, or NULL. */
int http_respond(ml_sock c, int status, const char *ctype, const char *extra, const char *body, size_t n);
/* Sends the head of an event stream (text/event-stream, Connection: close). */
int http_stream_start(ml_sock c, const char *extra);

#endif
