/* TCP sockets for the HTTP server: POSIX sockets, or Winsock on Windows. */
#ifndef ML_NET_H
#define ML_NET_H
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
typedef uintptr_t ml_sock;   /* SOCKET */
#define ML_SOCK_BAD ((ml_sock)~(uintptr_t)0)
#else
typedef int ml_sock;
#define ML_SOCK_BAD (-1)
#endif

/* Once per process: starts Winsock; elsewhere ignores SIGPIPE, so that
 * writing to a client that went away fails instead of ending the process. */
int ml_net_init(void);
/* Listens on host:port (port 0: any free port). Returns the socket, or
 * ML_SOCK_BAD with a message in err. *bound receives the actual port. */
ml_sock ml_net_listen(const char *host, int port, int *bound, char *err, size_t errlen);
/* Waits for a connection; peer receives the client's address as text. */
ml_sock ml_net_accept(ml_sock s, char *peer, size_t peerlen);
/* Receive and send timeouts in seconds. */
void ml_net_timeouts(ml_sock s, int seconds);
/* Bytes received (0: closed, -1: error or timeout). */
long ml_net_recv(ml_sock s, char *buf, size_t n);
/* Sends all n bytes; 0 on success, -1 if the client is gone. */
int ml_net_send(ml_sock s, const char *buf, size_t n);
void ml_net_close(ml_sock s);

#endif
