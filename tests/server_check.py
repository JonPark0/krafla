"""End-to-end checks of ml_server, with the standard library only.

    python3 tests/server_check.py "RUN-prefix" BUILD_DIR MODELS_DIR

Starts ml_server on a random test model (port 0: any free port) and checks
the endpoints, streaming, KV-cache reuse, stop strings, errors and API keys.
Greedy replies are compared with ml_run's for the same conversation. With
the openai package installed, its client is tried too. Prints one FAIL line
per failed check and exits 1 if there was one.
"""
import http.client
import json
import os
import shlex
import socket
import subprocess
import sys
import time

RUN, B, M = sys.argv[1], sys.argv[2], sys.argv[3]
PRE = shlex.split(RUN)
CHAT = [f"{M}/chat_q8.mlb", "-z", f"{M}/bpe_tiny.mlt"]
passed = failed = 0


def check(cond, what):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: ml_server: {what}")
    return cond


def start(args, env=None):
    p = subprocess.Popen(PRE + [f"{B}/ml_server"] + args + ["--port", "0"], stderr=subprocess.PIPE, text=True,
                         env=env)
    deadline = time.time() + 120
    while time.time() < deadline:
        line = p.stderr.readline()
        if not line:
            break
        if line.startswith("listening on http://127.0.0.1:"):
            return p, int(line.split(":")[2].split("/")[0])
    p.kill()
    raise SystemExit(f"FAIL: ml_server did not start: {args}")


def request(port, method, path, body=None, headers=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    out = r.read()
    c.close()
    return r.status, r.getheader("Content-Type", ""), out


def post(port, path, body, headers=None):
    st, ctype, out = request(port, "POST", path, body, headers)
    try:
        return st, json.loads(out) if ctype.startswith("application/json") else out
    except ValueError:
        return st, out


def stream(port, path, body):
    """The data events of a streamed response, parsed (the final [DONE] as the string)."""
    st, ctype, out = request(port, "POST", path, dict(body, stream=True))
    events = []
    for block in out.decode().split("\n\n"):
        if block.startswith("data: "):
            payload = block[6:]
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return st, ctype, events


def ml_run(args):
    """ml_run's output, decoded as the server's JSON writer does (a random
    model prints bytes that are not UTF-8: each maximal subpart becomes U+FFFD)."""
    out = subprocess.run(PRE + [f"{B}/ml_run"] + CHAT + args, capture_output=True, check=True).stdout
    return out.decode("utf-8", "replace")


def main():
    srv, port = start(CHAT + ["--chat", "llama3"])
    try:
        run_checks(port)
    finally:
        srv.kill()
        srv.wait()
    srv, port = start([f"{M}/hyb_q8.mlb", "-z", f"{M}/tok96.bin"])   # linear attention (Qwen3.5)
    try:
        hybrid_checks(port)
    finally:
        srv.kill()
        srv.wait()
    env = dict(os.environ, KRAFLA_API_KEY="sekret")
    srv, port = start(CHAT, env)   # no --chat: completions only
    try:
        auth_checks(port)
    finally:
        srv.kill()
        srv.wait()
    print(f"server: {passed} passed" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


def run_checks(port):
    st, _, out = request(port, "GET", "/health")
    check(st == 200 and json.loads(out) == {"status": "ok"}, f"/health: {st} {out!r}")
    st, _, out = request(port, "GET", "/v1/models")
    check(st == 200 and json.loads(out)["data"][0]["id"] == "chat_q8", f"/v1/models: {st} {out!r}")
    check(request(port, "GET", "/v1/models/chat_q8")[0] == 200, "/v1/models/chat_q8")
    check(request(port, "GET", "/v1/models/other")[0] == 404, "/v1/models/other should be 404")

    # a greedy reply equals ml_run --chat's for the same turn
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello world"}]
    body = {"model": "chat_q8", "messages": msgs, "max_tokens": 6, "temperature": 0}
    st, r = post(port, "/v1/chat/completions", body)
    ref = ml_run(["--chat", "llama3", "-T", "0", "-n", "6", "--system", "be brief", "-i", "hello world"])
    reply = r["choices"][0]["message"]["content"] if st == 200 else None
    check(st == 200 and reply == ref[:-1], f"chat reply {reply!r} differs from ml_run's {ref[:-1]!r}")
    check(r.get("object") == "chat.completion" and r["choices"][0]["message"]["role"] == "assistant", "chat object")
    u = r.get("usage", {})
    check(u.get("total_tokens") == u.get("prompt_tokens", 0) + u.get("completion_tokens", 0) > 0, f"usage {u}")
    fin = r["choices"][0]["finish_reason"]
    check(fin == "stop" or (fin == "length" and u["completion_tokens"] == 6), f"finish {fin}, usage {u}")

    # the same request again: everything but the last prompt token is cached
    st, r2 = post(port, "/v1/chat/completions", body)
    check(r2["choices"][0]["message"]["content"] == reply, "repeated request gives another reply")
    check(r2["usage"]["prompt_tokens_details"]["cached_tokens"] == u["prompt_tokens"] - 1,
          f"cached tokens {r2['usage']} (prompt {u['prompt_tokens']})")

    # streaming: the deltas add up to the same reply
    st, ctype, ev = stream(port, "/v1/chat/completions", dict(body, stream_options={"include_usage": True}))
    check(st == 200 and ctype.startswith("text/event-stream"), f"stream status {st} {ctype}")
    check(ev and ev[-1] == "[DONE]", "stream ends with [DONE]")
    chunks = [e for e in ev if isinstance(e, dict)]
    check(chunks and chunks[0]["choices"][0]["delta"].get("role") == "assistant", "first chunk carries the role")
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    check(text == reply, f"streamed {text!r}, not {reply!r}")
    fins = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    check(fins == [fin], f"stream finish reasons {fins}")
    check(chunks[-1]["choices"] == [] and chunks[-1]["usage"]["prompt_tokens"] == u["prompt_tokens"],
          f"usage chunk {chunks[-1]}")
    check(all(c["object"] == "chat.completion.chunk" for c in chunks), "chunk objects")

    # cached or not, a multi-turn request gets the same reply
    multi = dict(body, messages=msgs + [{"role": "assistant", "content": reply}, {"role": "user", "content": "what now"}])
    st, a = post(port, "/v1/chat/completions", multi)
    post(port, "/v1/chat/completions", dict(body, messages=[{"role": "user", "content": "something else"}]))
    st, b = post(port, "/v1/chat/completions", multi)
    check(a["choices"][0]["message"]["content"] == b["choices"][0]["message"]["content"],
          "multi-turn reply depends on the cache")
    check(a["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, f"multi-turn reused nothing: {a['usage']}")

    # stop strings cut the reply before their first occurrence
    at = [i for i in range(1, len(reply) - 1) if reply[i:i + 2].isascii() and reply[i:i + 2].isprintable()]
    if at:
        s = reply[at[0]:at[0] + 2]
        st, r = post(port, "/v1/chat/completions", dict(body, stop=[s, "\x01never"]))
        got = r["choices"][0]["message"]["content"]
        check(got == reply[:reply.find(s)] and r["choices"][0]["finish_reason"] == "stop",
              f"stop {s!r}: {got!r} from {reply!r}")
        st, ctype, ev = stream(port, "/v1/chat/completions", dict(body, stop=s))
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in ev if isinstance(c, dict) and c["choices"])
        check(text == reply[:reply.find(s)], f"streamed stop {s!r}: {text!r}")

    # text completion: the continuation ml_run prints after the prompt
    st, r = post(port, "/v1/completions", {"prompt": "hello world", "max_tokens": 8, "temperature": 0})
    text, np = r["choices"][0]["text"], r["usage"]["prompt_tokens"]
    ref = ml_run(["-T", "0", "-n", str(np + 7), "-i", "hello world"])[:-1]
    check(st == 200 and r["object"] == "text_completion" and text and ref.endswith(text),
          f"completion {text!r} is not the end of ml_run's {ref!r}")
    st, r2 = post(port, "/v1/completions", {"prompt": [[1, 2, 3]], "max_tokens": 3})
    check(st == 200 and r2["usage"]["prompt_tokens"] == 3, f"token-id prompt: {st} {r2}")
    st, ctype, ev = stream(port, "/v1/completions", {"prompt": "hello world", "max_tokens": 8, "temperature": 0})
    t2 = "".join(c["choices"][0]["text"] for c in ev if isinstance(c, dict) and c["choices"])
    check(t2 == text and all(c["object"] == "text_completion" for c in ev if isinstance(c, dict)),
          f"streamed completion {t2!r}")

    # errors, in OpenAI's format
    for what, path, raw, code in [
        ("invalid JSON", "/v1/chat/completions", b'{"messages": [', 400),
        ("no messages", "/v1/chat/completions", b'{"messages": []}', 400),
        ("n=2", "/v1/chat/completions", b'{"messages": [{"role":"user","content":"x"}], "n": 2}', 400),
        ("logprobs", "/v1/chat/completions", b'{"messages": [{"role":"user","content":"x"}], "logprobs": true}', 400),
        ("bad role", "/v1/chat/completions", b'{"messages": [{"role":"tool","content":"x"}]}', 400),
        ("image content", "/v1/chat/completions",
         b'{"messages": [{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}', 400),
        ("bad temperature", "/v1/completions", b'{"prompt": "x", "temperature": -1}', 400),
        ("bad token id", "/v1/completions", b'{"prompt": [99999]}', 400),
        ("not an object", "/v1/completions", b'[1]', 400),
        ("unknown path", "/v1/nothing", b'{}', 404),
    ]:
        st, ctype, out = request(port, "POST", path, raw=raw)
        ok = st == code and "message" in json.loads(out).get("error", {})
        check(ok, f"{what}: {st} {out[:200]!r}")
    st, _, out = request(port, "GET", "/v1/chat/completions")
    check(st == 405, f"GET on chat: {st}")
    st, r = post(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "tell me more " * 60}]})
    check(st == 400 and r["error"]["code"] == "context_length_exceeded", f"long prompt: {st} {r}")

    # Expect: 100-continue (curl sends it for larger bodies)
    data = json.dumps({"prompt": "hello", "max_tokens": 2}).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=120)
    s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Expect: 100-continue\r\nContent-Length: %d\r\n\r\n" % len(data))
    first = s.recv(4096)
    s.sendall(data)
    rest = b""
    while True:
        k = s.recv(4096)
        if not k:
            break
        rest += k
    s.close()
    check(first.startswith(b"HTTP/1.1 100") and b" 200 OK" in rest, f"100-continue: {first!r} {rest[:80]!r}")

    # a client that leaves mid-stream does not stop the server
    s = socket.create_connection(("127.0.0.1", port), timeout=120)
    data = json.dumps({"prompt": "hello", "max_tokens": 100, "stream": True}).encode()
    s.sendall(b"POST /v1/completions HTTP/1.1\r\nContent-Length: %d\r\n\r\n" % len(data) + data)
    s.recv(64)
    s.close()
    check(request(port, "GET", "/health")[0] == 200, "server down after a client left mid-stream")

    try:
        import openai
    except ImportError:
        return
    c = openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="none")
    r = c.chat.completions.create(model="chat_q8", messages=msgs, max_tokens=6, temperature=0)
    check(r.choices[0].message.content == reply, f"openai client: {r.choices[0].message.content!r}")
    s = c.chat.completions.create(model="chat_q8", messages=msgs, max_tokens=6, temperature=0, stream=True)
    text = "".join(ch.choices[0].delta.content or "" for ch in s if ch.choices)
    check(text == reply, f"openai client stream: {text!r}")


def hybrid_checks(port):
    """Linear-attention states are continued, never cut back: a request that
    continues the last one reuses it, any other starts over, and the replies
    are the same either way."""
    def complete(prompt):
        st, r = post(port, "/v1/completions", {"prompt": prompt, "max_tokens": 8, "temperature": 0})
        check(st == 200, f"hybrid completion: {st} {r}")
        return r["choices"][0]["text"], r["usage"]
    t1, u1 = complete("abcdefghij")
    ref = subprocess.run(PRE + [f"{B}/ml_run", f"{M}/hyb_q8.mlb", "-z", f"{M}/tok96.bin", "-T", "0",
                                "-n", str(u1["prompt_tokens"] + 7), "-i", "abcdefghij"],
                         capture_output=True, check=True).stdout.decode("utf-8", "replace")[:-1]
    check(ref.endswith(t1) and t1, f"hybrid completion {t1!r} is not the end of ml_run's {ref!r}")
    longer = "abcdefghij" + t1 + "klm"
    t2, u2 = complete(longer)
    check(u2["prompt_tokens_details"]["cached_tokens"] > 0, f"hybrid continuation reused nothing: {u2}")
    complete("zzz")
    t3, u3 = complete(longer)
    check(u3["prompt_tokens_details"]["cached_tokens"] == 0, f"hybrid state cut back: {u3}")
    check(t3 == t2, f"hybrid reply depends on the cache: {t2!r} vs {t3!r}")


def auth_checks(port):
    check(request(port, "GET", "/health")[0] == 200, "/health needs no key")
    st, _, out = request(port, "GET", "/v1/models")
    check(st == 401 and json.loads(out)["error"]["code"] == "invalid_api_key", f"no key: {st} {out!r}")
    check(request(port, "GET", "/v1/models", headers={"Authorization": "Bearer wrong"})[0] == 401, "wrong key")
    key = {"Authorization": "Bearer sekret"}
    check(request(port, "GET", "/v1/models", headers=key)[0] == 200, "right key")
    st, r = post(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]}, key)
    check(st == 400 and "--chat" in r["error"]["message"], f"chat without a template: {st} {r}")
    st, r = post(port, "/v1/completions", {"prompt": "hello", "max_tokens": 2}, key)
    check(st == 200 and r["usage"]["completion_tokens"] >= 1, f"completion with key: {st} {r}")


if __name__ == "__main__":
    sys.exit(main())
