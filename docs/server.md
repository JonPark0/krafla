---
title: "OpenAI-compatible server: run a local model behind the OpenAI API"
description: "ml_server serves a Krafla model over HTTP with the OpenAI API: chat and text completions, streaming, the KV cache kept between turns, API keys. Works with the openai client libraries and tools that take a base URL."
nav_order: 4
nav_title: "Server"
---

# OpenAI-compatible server

`ml_server` serves one model over HTTP with the endpoints of the OpenAI API.
Programs written for that API, such as the `openai` Python and JavaScript
libraries and tools that take a base URL, can then use a model on your own
CPU. It comes in the release archives next to `ml_run`, and `make ISA=avx2`
builds it as `build/avx2/ml_server`.

```sh
./ml_server Qwen2.5-0.5B-Instruct-Q4_0.mlb -z tokenizer.mlt --chat qwen -t 4
# krafla 0.7.0, isa avx2: model Qwen2.5-0.5B-Instruct-Q4_0, context 4096 of 32768, chat qwen, threads 4, kv f32
# listening on http://127.0.0.1:8080/v1
```

`-z` and `--chat` take the same tokenizer and template as `ml_run --chat` (see
[Chat](user-guide.md#chat)). Without `--chat` the server answers only text
completions.

## Sending requests

With curl:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "max_tokens": 64
}'
```

With the `openai` Python library:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
messages = [{"role": "user", "content": "What is the capital of France?"}]

reply = client.chat.completions.create(model="local", messages=messages, max_tokens=64)
print(reply.choices[0].message.content)

for chunk in client.chat.completions.create(model="local", messages=messages, stream=True):
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The server has one model, so the `model` field of a request is not checked;
responses carry the server's model id.

## Endpoints

| Endpoint | What it does |
|----------|--------------|
| `POST /v1/chat/completions` | Chat: renders `messages` with the `--chat` template and generates the assistant's reply. |
| `POST /v1/completions` | Continues `prompt`, a string or a list of token ids. |
| `GET /v1/models`, `GET /v1/models/{id}` | The model. Its id is `--alias`, or the file name without `.mlb`. |
| `GET /health` | `{"status":"ok"}`. Needs no API key. |

The same paths without the `/v1` prefix also work.

## Request fields

| Field | Meaning |
|-------|---------|
| `messages` | Chat. Roles `system` and `developer` (both become the system prompt), `user` and `assistant`. Content is a string or a list of `{"type": "text"}` parts. |
| `prompt` | Completions. A string, encoded with BOS if the tokenizer adds one, or token ids. One prompt per request. |
| `max_tokens`, `max_completion_tokens` | Reply tokens at most. Default: `-n`, which by default lets the reply run until the context is full. |
| `temperature` | `0` is greedy. Default: `-T`, which is `0`. OpenAI's default is 1, so set it when you want sampling. |
| `top_p` | Default: `-p` (0.9). |
| `top_k` | Not in OpenAI's API. Default: `--top-k` (0, off). |
| `repeat_penalty` or `repetition_penalty`, `repeat_last_n` | Not in OpenAI's API; as `ml_run`'s `--repeat-penalty` and `--repeat-last-n`. |
| `seed` | Default: `-s`, or a new seed for each request if `-s` is not given. |
| `stop` | A string or up to 4 strings. The reply ends before the first one. |
| `stream` | Send the reply as server-sent events, as OpenAI does, ending with `data: [DONE]`. |
| `stream_options.include_usage` | With `stream`: a last chunk with `usage`. |
| `n` | Must be 1. |

Other fields, such as `frequency_penalty` and `user`, are ignored. The server
refuses what it cannot do: `logprobs`, a non-empty `tools` list, a
`response_format` other than `text`, and for completions `echo`, `suffix` and
`best_of` above 1.

## Responses

Responses have OpenAI's shape: a `chat.completion` or `text_completion`
object, or `chat.completion.chunk` events when streaming.

- **`finish_reason`:** `stop` when the model produced its end token or the
  reply reached a stop string; `length` when it reached `max_tokens` or
  filled the context.
- **`usage`:** `prompt_tokens`, `completion_tokens` and `total_tokens`;
  `prompt_tokens_details.cached_tokens` counts the prompt tokens whose KV cache
  came from the previous request.
- **Errors:** `{"error": {"message", "type", "param", "code"}}` with status
  400 for an invalid request (`code` is `context_length_exceeded` when the
  prompt does not fit the context), 401 for a missing or wrong API key, 404
  and 405.

## How requests run

- **One at a time.** Requests are answered in the order they arrive; a request
  waits until the one before it has finished.
- **The KV cache stays between requests.** When a request's prompt starts with
  the tokens of the previous one, as the next turn of a chat does, only the new
  tokens run. This saves time, not tokens: a request gets the same reply from
  an empty cache. Models with linear-attention layers (Qwen3.5) keep a state
  that cannot be cut back, so for them only a request that continues all of
  the previous one reuses it.
- **Conversations are rendered as `ml_run --chat` renders them,** turn by turn.
  An earlier assistant message is encoded with its turn and closed with the
  template's stop token. If the last message is from the assistant, the reply
  continues it.
- **The context holds the prompt and the reply.** It is `--ctx` tokens long
  (by default the model's `max_seq`, at most 4096). A prompt that does not
  fit is refused, and a reply that fills the context ends with `length`.
- **Replies are deterministic** for a given seed. With `temperature` 0, the
  default, a chat request with one user message gets the same reply as
  `ml_run --chat -T 0` for that message.
- **Text is UTF-8.** Streamed pieces never end inside a character, and bytes
  the model produces that are not valid UTF-8 arrive as U+FFFD.

## Options

| Option | Meaning |
|--------|---------|
| `-z FILE` | Tokenizer. Required. |
| `--chat TEMPLATE` | Chat template, as for `ml_run`: `llama3`, `zephyr`, `llama2`, `chatml`, `qwen`, `smollm`, `gemma`, `qwen3.5` or `qwen3.5-nothink`. |
| `--system TEXT` | System prompt for chat requests that have none. |
| `--host ADDR` | IP address to listen on. Default `127.0.0.1`; `0.0.0.0` listens on every IPv4 interface. Takes an address or `localhost`, not other host names. |
| `--port N` | Default 8080; `0` picks a free port, printed at start. |
| `--api-key KEY` | Require `Authorization: Bearer KEY`. Without it, the `KRAFLA_API_KEY` environment variable is used if set. |
| `--alias NAME` | Model id in responses and `/v1/models`. |
| `--cors ORIGIN` | Let web pages from `ORIGIN` (`*` for any) call the server from a browser. |
| `--ctx N` | Context length, as for `ml_run`. |
| `-n N` | Reply tokens at most for requests without `max_tokens`. Default 0: until the context is full. |
| `-T`, `-p`, `--top-k`, `--repeat-penalty`, `--repeat-last-n`, `-s` | Sampling for requests that do not set it. |
| `--eos ID` | Token that ends a reply, instead of the tokenizer's. |
| `-t N`, `--kv TYPE`, `--batch N`, `--no-pack`, `--pack-cache` | As for `ml_run`. |

Each request prints one line to stderr: the client, the path, the status and,
for completions, the prompt and reply token counts with their speed.

## Security

- **Network:** by default the server listens on 127.0.0.1, so only programs on
  the same machine reach it. If you listen on the network with `--host`, set an
  API key. `KRAFLA_API_KEY` keeps the key out of the process list.
- **No TLS:** across a network, put the server behind a reverse proxy that
  terminates HTTPS.
- **Browsers:** a web page on another site can send requests to a local server
  but cannot read the replies unless `--cors` allows its origin. Allow `*` only
  if every page you open may use the model.
- **Limits:** request bodies up to 8 MiB, headers up to 16 KiB; a connection
  that stays silent for 60 seconds is closed.

## Limits

- One model, one request at a time, one reply per request.
- No embeddings, images, audio, tool calls or log probabilities.
- HTTP/1.1 without keep-alive or chunked request bodies: every response closes
  its connection, which OpenAI's client libraries handle.
