#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")
#endif
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif
#include <stdio.h>
#include <string.h>
#include "server/net.h"

#ifdef _WIN32
#define SOCK(s) ((SOCKET)(s))
#else
#define SOCK(s) (s)
#endif

int ml_net_init(void) {
#ifdef _WIN32
    WSADATA w;
    return WSAStartup(MAKEWORD(2, 2), &w) == 0 ? 0 : -1;
#else
    signal(SIGPIPE, SIG_IGN);
    return 0;
#endif
}

/* Addresses are parsed, not resolved: getaddrinfo in a static glibc build
 * would load the system's NSS libraries at run time. */
ml_sock ml_net_listen(const char *host, int port, int *bound, char *err, size_t errlen) {
    struct sockaddr_storage ss;
    struct sockaddr_in *a4 = (struct sockaddr_in *)&ss;
    struct sockaddr_in6 *a6 = (struct sockaddr_in6 *)&ss;
    socklen_t sl;
    memset(&ss, 0, sizeof ss);
    const char *addr = strcmp(host, "localhost") ? host : "127.0.0.1";
    if (inet_pton(AF_INET, addr, &a4->sin_addr) == 1) {
        a4->sin_family = AF_INET;
        a4->sin_port = htons((unsigned short)port);
        sl = sizeof *a4;
    } else if (inet_pton(AF_INET6, addr, &a6->sin6_addr) == 1) {
        a6->sin6_family = AF_INET6;
        a6->sin6_port = htons((unsigned short)port);
        sl = sizeof *a6;
    } else {
        snprintf(err, errlen, "--host takes an IP address (127.0.0.1, 0.0.0.0, ::1, ...) or localhost, not '%s'", host);
        return ML_SOCK_BAD;
    }
    const ml_sock s = (ml_sock)socket(ss.ss_family, SOCK_STREAM, 0);
    if (s == ML_SOCK_BAD) { snprintf(err, errlen, "cannot create a socket"); return ML_SOCK_BAD; }
#ifndef _WIN32
    const int one = 1;   /* Windows: SO_REUSEADDR would let another process take the port */
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
#endif
    if (bind(SOCK(s), (struct sockaddr *)&ss, sl) != 0 || listen(SOCK(s), 16) != 0) {
        ml_net_close(s);
        snprintf(err, errlen, "cannot listen on %s port %d (in use, or not an address of this machine)", host, port);
        return ML_SOCK_BAD;
    }
    sl = sizeof ss;
    *bound = port;
    if (getsockname(SOCK(s), (struct sockaddr *)&ss, &sl) == 0)
        *bound = ss.ss_family == AF_INET6 ? ntohs(a6->sin6_port) : ntohs(a4->sin_port);
    return s;
}

ml_sock ml_net_accept(ml_sock s, char *peer, size_t peerlen) {
    struct sockaddr_storage ss;
    socklen_t sl = sizeof ss;
    const ml_sock c = (ml_sock)accept(SOCK(s), (struct sockaddr *)&ss, &sl);
    if (c == ML_SOCK_BAD) return c;
    char host[64] = "?";
    if (ss.ss_family == AF_INET) inet_ntop(AF_INET, &((struct sockaddr_in *)&ss)->sin_addr, host, sizeof host);
    else if (ss.ss_family == AF_INET6) inet_ntop(AF_INET6, &((struct sockaddr_in6 *)&ss)->sin6_addr, host, sizeof host);
    snprintf(peer, peerlen, "%s", host);
    return c;
}

void ml_net_timeouts(ml_sock s, int seconds) {
#ifdef _WIN32
    const DWORD ms = (DWORD)seconds * 1000;
    setsockopt(SOCK(s), SOL_SOCKET, SO_RCVTIMEO, (const char *)&ms, sizeof ms);
    setsockopt(SOCK(s), SOL_SOCKET, SO_SNDTIMEO, (const char *)&ms, sizeof ms);
#else
    struct timeval tv = { seconds, 0 };
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
}

long ml_net_recv(ml_sock s, char *buf, size_t n) {
    const int k = n > 1 << 30 ? 1 << 30 : (int)n;
    return (long)recv(SOCK(s), buf, k, 0);
}

int ml_net_send(ml_sock s, const char *buf, size_t n) {
    while (n > 0) {
        const int k = n > 1 << 30 ? 1 << 30 : (int)n;
#if defined(MSG_NOSIGNAL)
        const long w = (long)send(SOCK(s), buf, k, MSG_NOSIGNAL);
#else
        const long w = (long)send(SOCK(s), buf, k, 0);
#endif
        if (w <= 0) return -1;
        buf += w;
        n -= (size_t)w;
    }
    return 0;
}

void ml_net_close(ml_sock s) {
#ifdef _WIN32
    closesocket(SOCK(s));
#else
    close(s);
#endif
}
