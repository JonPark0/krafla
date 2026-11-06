# Changelog

## Unreleased

### Added

- [Q4_0 block scales](https://github.com/JonPark0/krafla/blob/main/docs/q4-scales.md),
  a page of the docs site, compares Krafla's fitted Q4_0 scales with
  llama.cpp's Q4_0 on twelve models from 135M to 3B. All variants have the
  same tensors and are scored by llama-perplexity on 32 chunks: KL divergence
  from the unquantized model, perplexity and same top token. Two scripts
  reproduce the runs:
  - `scripts/q4_patch_gguf.py` rewrites a GGUF's Q4_0 tensors with
    `tools/mlb.py` (`--rule fit|flat|round`, `--tensors REGEX` for some of
    them).
  - `scripts/compare_q4.sh` builds and scores every variant.
- Charts in the docs, drawn by `scripts/sync_docs.py` from the pages' own
  tables and checked in CI like the decode chart:
  - the KL divergence of the fit and flat Q4 rules against llama.cpp's Q4_0
    on twelve models (Q4_0 block scales);
  - prefill speed against llama.cpp at each ISA level (Comparison with
    llama.cpp);
  - Llama 3.2 1B's file size against perplexity with `--q8` classes (user
    guide).

  The decode chart takes the same colors, checked for color-vision
  deficiency and contrast in light and dark mode.
- `tools/convert.py --q4-rule flat`: the Q4 fit with the error weighted by
  `sqrt(s2 + x²)`, as llama.cpp weights its importance search when every
  channel is equally important. The embedding and classifier keep the `x²`
  fit.

### Changed

- Q4 conversion of Llama-architecture checkpoints (`model_type` `llama`:
  Llama, TinyLlama, SmolLM2) now uses `--q4-rule flat` by default, without an
  importance matrix. On seven such models, from 135M to 3B, it is closer to
  the unquantized model than the `x²` fit on all seven, by 0.6–9%, and than
  llama.cpp's Q4_0 on six, by 1.5–5.6% in KL divergence. The `x²` fit was 10%
  worse than llama.cpp's Q4_0 to 3% better on them. Other architectures keep
  their rules. Q4 files converted earlier still load; convert again for the
  new scales.
- `--q4-rule round` writes `d` = -0 for a block of zeros, as llama.cpp does,
  so that its bytes equal `llama-quantize`'s in every block (TinyLlama's
  embedding has such blocks). The values are unchanged.
- The README, the docs home page, the FAQ and the Korean page no longer claim
  a lower Q4 perplexity from a 4-chunk run (Qwen2.5 0.5B, 15.1 against
  15.8–15.9). On 32 chunks the two rules score 15.88 and 16.11. Without an
  importance matrix, the default rules lower the KL divergence by 6–9% on four
  Qwen models and by 1.5–5.6% on six of seven Llama-architecture models. With
  an importance matrix, llama.cpp's own search matches the fit or does better
  on five of the seven models tried.

## 0.9.1 — 2026-11-05

### Added

- `scripts/hf_collection.py` and `.github/workflows/hf-collection.yml`, run
  after every publishing run of `hf-models.yml` (or by hand): they keep a
  "Krafla" collection on Hugging Face with every published model of
  `scripts/hf_models.json`, smallest first, each noted with the Krafla version
  it needs, its `--chat` template and its Q4_0 size. The catalog is now in
  that order and holds the collection's title and description.
- The README, the docs site (Models link, footer, Korean page) and `llms.txt`
  link to the collection, and the user guide lists all 15 published models
  with the Krafla version each needs.

### Changed

- [The benchmarks](https://github.com/JonPark0/krafla/blob/main/docs/benchmarks.md)
  add the AMX kernel measured again on 0.9.0 with Qwen3 0.6B and 8B: on four
  threads it prefills the 8B model 5-19% faster than the AVX-512 VNNI build
  and the small one about as fast, so AMX stays out of the build. The kernel
  is on the `exp/amx` and `exp/amx-v2` branches.

### Fixed

- `tools/convert.py` ran out of memory on a 16 GB machine converting Qwen3.5
  9B: a matrix was quantized whole, and Q4's temporaries took four times its
  size on top of it (its 248K × 4K embedding table: over 16 GB). Matrices are
  now quantized a block of rows at a time, which gives the same bytes (every
  block of 32 is quantized on its own); that table now needs 1.1 GB beyond
  itself. BF16 tensors are widened to float32 in place, without a second
  copy.

## 0.9.0 — 2026-11-03

### Added

- Qwen3.5 and Qwen3.6 (dense): linear-attention layers with the gated delta
  rule (`ops/delta.c`; a causal convolution and a fixed state per head in
  place of a KV cache, so their memory does not grow with the context),
  attention layers with a gated output and RoPE on part of each head, and
  their tokenizer, which keeps combining marks with the letters
  (`ML_TOKF_MARKS`, with a Unicode mark table). The KV cache now covers
  attention layers only. New header fields at offsets 132–159 and layer flag
  bit 3; `tools/convert.py` reads `model_type` `qwen3_5` (the language model
  of the image-text checkpoints, without the multi-token prediction head), and
  `tools/ref_forward.py` and `tools/gen_random.py` follow. Against llama.cpp
  on the same machine (4 threads, AVX-512 VNNI): Qwen3.5 0.8B Q8_0 perplexity
  15.80 against 15.79, decoding 1.28× and prefill 1.55× faster; its Q4_0 18.41
  against 18.95 in a 24% smaller file, decoding 1.69× and prefill 1.97×
  faster; Qwen3.5 4B Q8_0 8.18 against 8.15, prefill 1.7× faster.
- Chat templates `qwen3.5` (the reply opens with `<think>`) and
  `qwen3.5-nothink` (an empty think block), for `ml_run` and `ml_server`.
- `scripts/hf_models.json`: Qwen3.5 0.8B, 2B, 4B and 9B, in the default set.

### Changed

- `ml_server` reuses the previous request on a model with linear attention
  only when the prompt continues all of it: the state cannot be cut back.

### Fixed

- Packing released the original weight pages only after every matrix was
  packed, so loading briefly held the model twice: 13.7 GB resident for
  Qwen3 8B in Q8_0. On a 16 GB machine that pushed part of the packed weights
  to swap, and the published card measured 4 tok/s prefill where 20 is right.
  Each matrix's pages are now released as soon as it is packed (peak 9.6 GB;
  5.6 GB instead of 9.4 GB for the Q4_0 file).

## 0.8.0 — 2026-11-03

### Added

- SmolLM3: the converter reads `model_type` `smollm3` and marks the layers
  that leave out RoPE (`no_rope_layers`, every fourth layer) with a new
  per-layer flag, bit 2 (`ML_LAYER_NO_ROPE`), which the runtime and
  `tools/ref_forward.py` follow. SmolLM3 3B in Q8_0: perplexity 11.61 on 4
  chunks of wikitext-2, against 15.43 with RoPE in every layer; its tokenizer
  matches the `tokenizers` library on all 304 test texts. Older versions refuse
  files with the new flag.
- `scripts/hf_models.json`: Qwen3 1.7B, Qwen3 4B Instruct 2507, Qwen3 8B,
  Qwen2.5 7B Instruct and SmolLM3 3B in the default set; Llama 3.2 3B, Llama
  3.1 8B, Mistral 7B Instruct v0.3 and Gemma 3 4B and 12B for accounts that
  have accepted their licenses.

### Changed

- `scripts/hf_publish.py` converts at the model's full context (a `max_seq`
  entry can still cap it) and says on the card how `--ctx` sizes the KV cache;
  cards show `ml_server` next to `ml_run`. It converts all files before
  measuring them, so the safetensors are gone by then, and the workflow frees
  about 30 GB of runner disk and allows 5 hours per model, for 7B and 8B models.

## 0.7.0 — 2026-11-03

### Added

- `ml_server`, an OpenAI-compatible HTTP server for one model, in C11 with no
  dependencies (POSIX sockets, Winsock on Windows): `POST /v1/chat/completions`
  (with the `--chat` templates), `POST /v1/completions`, `GET /v1/models` and
  `GET /health`; streaming as server-sent events with an optional usage chunk;
  `temperature`, `top_p`, `top_k`, `repeat_penalty`, `seed`, `stop` and
  `max_tokens`. Requests run one at a time, and the KV cache is kept between
  them, so the next turn of a conversation runs only its new tokens
  (`usage.prompt_tokens_details.cached_tokens`). It listens on 127.0.0.1 by
  default; `--api-key` (or `KRAFLA_API_KEY`) and `--cors` are optional. The
  release archives include it, and `tests/server_check.py` checks it over
  HTTP in every e2e run. See `docs/server.md`.
- `--ctx N` for `ml_run`: the context length, which sizes the KV cache, the
  attention scratch and the RoPE tables at run time, from 1 to the file's
  `max_seq`. The default is `max_seq` capped at 4096, so a file converted at
  the model's full context no longer reserves its whole KV cache. Logits do
  not depend on it. `ml_state_init` takes the context length (0 for the
  default); `ml_bench` sizes its state to the positions it runs, and
  `tools/perplexity.py` to each chunk. When a chat fills the context,
  `ml_run` says how far `--ctx` can raise it.
- Zenodo DOI `10.5281/zenodo.23880653` (all versions): a badge in the README, `doi` in
  `CITATION.cff` and the BibTeX entries, and the site's structured data.
- `.github/workflows/zenodo-doi.yml`, run by `release.yml` after publishing:
  waits for Zenodo to archive the release and adds its DOI to the release
  notes, through Zenodo's public API (`.github/scripts/zenodo_doi.py`).

### Changed

- `tools/convert.py --max-seq` is no longer needed to keep the KV cache small:
  without it the file records the model's full context, and `--ctx` sizes the
  cache at run time.
- Workflows use the Node 24 versions of their actions (`actions/checkout@v5`,
  `actions/cache@v5`, `actions/upload-artifact@v6`,
  `softprops/action-gh-release@v3`); GitHub has deprecated Node 20 on its
  runners.

## 0.6.1 — 2026-11-02

### Changed

- Documentation: corrected statements that no longer matched the code or the
  measurements (kernel count, prefill batch size, the importance-matrix file
  header, the Q8_0/Q4_0 format table, mixed files for SmolLM2), model file
  sizes labelled in MiB, decode speed against llama.cpp described as "at least
  as fast" at every ISA level measured, plainer wording in English and Korean,
  and ASCII characters in code spans.
- Model cards written by `scripts/hf_publish.py` use the same wording.
- First release archived on Zenodo, which gives it a DOI.

## 0.6.0 — 2026-11-01

### Changed

- Renamed from isa-lm to **Krafla** (Kernels Retargeted Across Families of
  Legacy Architectures): repository `JonPark0/krafla`, site
  <https://jonpark0.github.io/krafla/>, release archives `krafla-VER-*`,
  converted models `jonpark0/<model>-krafla` with the `krafla` tag, and
  `KRAFLA_ISA` for the release wrappers (`ISA_LM_ISA` still works). Programs
  (`ml_run`, `ml_bench`, `ml_tok`), the C API and the `.mlb`/`.mlt` formats are
  unchanged.

### Added

- Documentation site on Just the Docs: a sidebar, search, dark mode (following
  the system, with a switch), heading links, copy buttons on code, "Edit this
  page" links, a 404 page and icons. The home page gains download and model
  buttons, a chart of decode speed against llama.cpp
  (`docs/assets/decode-ratio.svg`, drawn from `docs/benchmarks.md` by
  `scripts/sync_docs.py` and checked in CI) and a table of supported CPUs.
- A Korean home page (`/ko/`, linked both ways with hreflang), a changelog
  page generated from `CHANGELOG.md` (`scripts/sync_docs.py`), and Mermaid
  diagrams of compile-time kernel binding and the CPU guard in the
  architecture page.
- `.github/workflows/hf-models.yml` and `scripts/hf_publish.py`: convert the
  models of `scripts/hf_models.json` (Qwen2.5, Qwen3, SmolLM2, TinyLlama;
  Llama 3.2 and Gemma 3 on request), measure each file (perplexity, speed, a
  chat reply) and publish them on Hugging Face with a model card. Converted
  SmolLM2 135M/360M, Qwen2.5 0.5B/1.5B, Qwen3 0.6B and TinyLlama 1.1B files are
  at `jonpark0/<model>-krafla` (README, user guide and FAQ link them).

## 0.5.0 — 2026-11-01

### Added

- Gemma 3 4B, 12B and 27B: their image-text checkpoints (`model_type`
  `gemma3`) convert to the text model, leaving the vision tower out. Gemma 3
  4B Instruct Q8 logits match an F32 numpy forward (KL 0.002); Q4 takes 2083
  MiB and decodes at 14.8 tok/s on 4 vCPUs.
- Gemma 2: soft-capping of attention scores and logits (`.mlb` header
  offsets 124 and 128), and the sliding window on even-numbered layers.
- F16 KV cache (`--kv f16`): K and V are stored in half precision, which
  halves the cache memory compared with F32. Prefill speed is unchanged
  (Qwen2.5 0.5B Q4, 1500 tokens: 722 tok/s with F16 against 718 with F32),
  and perplexity barely moves (21.71 with F16 against 21.74 with F32; the Q8
  cache gives 22.82).

### Changed

- `tools/convert.py` converts checkpoint tensors to float32 one at a time, so
  a 4B model converts in about the memory of its largest tensor instead of
  16 GB.
- AVX-512 VNNI prompt prefill feeds each weight vector to 6 tokens instead
  of 4: matmuls 1.05–1.08× and Qwen2.5 0.5B prefill 1.05× faster on one
  thread (GitHub runner; four threads within noise).
- AVX2 Q8 prompt prefill takes |x| from a buffer made once per tile, saving a
  `psignb` per token and step: Q8 matmuls 1.10–1.16× and Qwen2.5 0.5B Q8
  prefill 1.14× faster on one thread (four threads within noise).

## 0.4.0 — 2026-10-31

### Added

- Gemma 3 text models (Gemma 3 1B): GELU, block output norms, embedding and
  attention scales (`.mlb` header offsets 104-116), local and global layers;
  the Gemma tokenizer (no U+2581 before the text, MLT1 flag 128) and
  `--chat gemma`. Gemma 3 1B Q8 perplexity 42.20 against 42.32 for an F32
  reference.
- Per-layer sliding windows and RoPE bases: a flag byte per layer (header
  offset 96) puts a layer behind the window and on the local RoPE base
  (offset 100). The converter writes them when only some layers slide
  (`layer_types`, Qwen2's `max_window_layers`) instead of capping the context.
- f16 block scales (`.mlb` header offset 120): Q4_0 takes 4.5 bits per
  weight instead of 5, Q8_0 8.5 instead of 9. Both kinds of file load in every
  build.
- `tools/convert.py --q4-rule round`: llama.cpp's Q4_0 rounding, now the
  default for Gemma 3 (`--q4-rule fit` for the fitted rule). Gemma 3 1B Q4
  perplexity 49.83 against 53.25 fitted; its quantization-aware checkpoint
  (`gemma-3-1b-it-qat-q4_0-unquantized`) reaches 40.50, below its Q8 (42.53).
  The fit stays the default for other models (Qwen2.5 0.5B 21.73 against
  22.45).

### Changed

- `tools/convert.py` writes f16 scales by default (`--scales f32` keeps the
  older files): perplexity changes by at most 0.15% (Qwen2.5 0.5B, Llama 3.2
  1B), and Qwen2.5 0.5B Q4 shrinks from 309 to 278 MB. Versions before 0.4.0
  cannot read files with f16 scales.
- AVX2, AVX-VNNI and AVX-512 VNNI builds keep f16 scales as f16 in the packed
  weights, so decoding reads fewer bytes: on GitHub's x86-64 runner
  (Qwen2.5 0.5B) Q4 decode is 1.04-1.12x faster and Q8 1.05-1.11x. These
  builds now require F16C, which every AVX2 CPU has.
- Prompt prefill takes up to 128 tokens per batched step (was 32; `--batch`
  1-128), so each step's pass over the weights serves 4x the tokens: prefill
  up to 1.4x faster on the AVX-512 reference VM (Qwen2.5 0.5B, 4 threads) and
  1.06-1.15x on GitHub's Neoverse N2 runner; level where the model fits the
  cache (GitHub's EPYC runner). Results stay identical to `--batch 1`.
- AArch64 dotprod builds (neon_dotprod, neon_i8mm, sve): Q4 nibbles are packed
  in signed form, so no per-block activation sums are subtracted; the dotprod
  batched kernels take 8 tokens per weight block, and the i8mm ones
  accumulate in smmla's layout. On GitHub's Neoverse N2 runner (Qwen2.5 0.5B)
  Q4 prefill is 1.16-1.24x faster, Q4 decode 1.07-1.12x, and Q8 prefill on
  neon_dotprod and sve 1.04x. Pack cache files from earlier builds are
  rewritten.
- [The benchmarks](https://github.com/JonPark0/isa-lm/blob/main/docs/benchmarks.md)
  explain why AMX stays out of the build: on the reference CPU the tile
  products and the AVX-512 rescaling of every 32-weight block do not overlap,
  and a second AMX kernel, though 1.2-1.4x the AVX-512 VNNI kernel for Q8,
  prefills no faster on a whole model.

## 0.3.0 — 2026-10-31

### Added

- Qwen3: a `head_dim` other than `dim / n_heads` and per-head q/k RMSNorm
  (`.mlb` header offsets 80 and 84). Qwen3 0.6B Q8 perplexity 27.91 against
  28.10 for an F32 reference.
- SmolLM2 and GPT-2 tokenizers: the GPT-2 pre-tokenizer regex (MLT1 flag 32),
  after a digit split if the tokenizer has one (flag 64); identical to the
  tokenizers library on every checked text. `--chat smollm`.
- RoPE scaling of the linear and yarn kinds (header `rope_type` 2 and 3, yarn's
  attention factor at offset 88), with transformers' formulas.
- Sliding-window attention (offset 92): a window on every layer, as Mistral
  v0.1 has, is kept instead of capping the context.
- Windows: CI builds the scalar, SSE2 and AVX2 configurations with MSVC and
  checks them against the numpy reference.
- `scripts/compare_llamacpp.sh` runs on AArch64 too (native, dotprod and Armv8.0
  levels), and the manual `llama.cpp comparison` workflow runs it on GitHub's
  x86-64 and AArch64 runners. [The comparison](https://github.com/JonPark0/isa-lm/blob/main/docs/benchmarks.md)
  now covers both: on x86-64 without AVX-512 isa-lm is ahead or level
  everywhere; on AArch64 decode is level to 1.2× faster with dotprod and i8mm
  and 1.3–1.7× faster on Armv8.0, while llama.cpp prefills Q4_0 faster with
  dotprod. On the reference machine llama.cpp's AVX-512 prefill lead comes
  from AMX; without it llama.cpp prefills at about a third of isa-lm's speed.

### Changed

- `tools/convert.py hf` refuses partial rotary embeddings, and q/k projections
  that do not match the configured heads.
- Importance matrix files record the attention width when it differs from
  `dim` (Qwen3); files for other models are unchanged.

## 0.2.0 — 2026-10-27

### Added

- Qwen2 / Qwen2.5 models: q/k/v projection biases (`.mlb` header flag at
  offset 76), the Qwen2 tokenizer (NFC normalization, single-digit numbers;
  MLT1 flags 8 and 16), and the `qwen` and `chatml` chat templates.
  Qwen2.5 0.5B Instruct Q8: perplexity 19.05 against 19.00 for an F32
  reference.
- `scripts/compare_llamacpp.sh` and [the comparison with llama.cpp](https://github.com/JonPark0/isa-lm/blob/main/docs/benchmarks.md):
  decode is 1.1–1.5× faster than llama.cpp with AVX-512 VNNI, about 2× with
  AVX2 and 2.2–7× on SSE builds; llama.cpp still prefills Q8_0 faster with
  AVX-512.
- `tools/perplexity.py --llama-cpp` chunks and scores as `llama-perplexity`.
- `tools/convert.py hf` reads Mistral checkpoints and caps the context of
  sliding-window models at the window.
- Importance matrix: `ml_run --imatrix FILE` sums the squares of every matrix
  input per channel (over runs), and `tools/convert.py --imatrix FILE`
  weights the Q4 fits by it as llama.cpp does. Qwen2.5 0.5B Q4 with one from
  32K tokens of wikitext-2 train: perplexity 21.72 → 20.92 (llama.cpp
  scoring: 15.13 → 14.52, against 13.33 for Q8). A tied embedding takes the
  classifier's importance. The file is replaced only once fully written.
- Releases from main: merging a version bump (`ML_VERSION` and its
  CHANGELOG.md section) builds, tests and publishes the release
  (`.github/workflows/auto-release.yml`); CI fails a bump without notes.
- `scripts/bench.py` and the Benchmark workflow: A/B speed comparison of two
  versions, run alternately on one machine (GitHub's x86-64 and AArch64
  runners for the workflow).

### Changed

- Faster prompt prefill: the packed matmul works through L2-sized blocks of
  rows, the AVX-512 VNNI tiles share each activation broadcast between two
  row groups, and a batched step's attention and per-token work run across
  threads in one pass each. 128-token prefill of a Qwen2.5-0.5B-shaped model
  on a 4-vCPU AVX-512 VNNI Xeon: Q8 180 → 260 tok/s on 1 thread and 600 →
  925 on 4; Q4 250 → 290 and about 660 → 890. Logits are unchanged.
- Batched kernels sum activation blocks with `psadbw` (Qwen2.5 0.5B, one
  AVX-512 VNNI thread: Q4 prefill 240 → 287 tok/s, Q8 231 → 246), and the
  AVX-512 VNNI and AVX-VNNI tiles are unrolled so that GCC no longer copies
  every accumulator (kernel throughput: AVX-512 VNNI Q8 +6–18%, AVX-VNNI Q8
  +12–51%).
- Q4 blocks of values below 1e-6 are fitted rescaled: their float32 fit
  underflowed (zero scale below about 1e-12). Other blocks convert to the
  same bytes.
- `tools/convert.py hf` refuses architectures it cannot run correctly
  (`model_type` other than llama, mistral or qwen2; activations other than
  SiLU; explicit `head_dim`; biases other than q/k/v) instead of writing a
  wrong model.
- `tools/perplexity.py` works with tokenizers that add no BOS.
- Q4 scales are fitted per block (weighted least squares over candidate
  mappings, all 16 levels) instead of `amax / 7`: Llama 3.2 1B Q4 perplexity
  20.46 → 19.91, Qwen2.5 0.5B 23.58 → 21.73. The format is unchanged;
  convert again to benefit.

## 0.1.0 — 2026-10-23

First release.

### Runtime

- C11 inference for Llama-style decoders (RMSNorm, SwiGLU, GQA, RoPE with
  Llama 3 scaling) with Q8_0 and Q4_0 weights, and mixed Q4 models that keep
  chosen matrix classes in Q8 (`--q8 v,o,down`).
- Compile-time kernel binding for scalar, SSE2, SSE4.1, AVX2, AVX-VNNI,
  AVX-512 VNNI, ARMv7/AArch64 NEON, NEON dotprod, i8mm and SVE; a CPU guard
  exits with code 3 instead of crashing on processors without the build's ISA.
- Row-interleaved weight packing for each ISA's integer dot products,
  batched prompt prefill (on `smmla` with i8mm), software prefetch on x86,
  blocked GQA attention, an optional Q8 KV cache (`--kv q8`), a thread pool
  with bit-identical results for any thread count, and a pack cache
  (`--pack-cache`) for fast start-up.

### Models and tokenizers

- Converters for llama2.c checkpoints and Hugging Face safetensors (Llama 2,
  Llama 3/3.1/3.2, TinyLlama, Mistral).
- Hugging Face `tokenizer.json` support for SentencePiece-style BPE and Llama
  3 byte-level BPE (`tools/convert_tokenizer.py`), identical to the
  `tokenizers` library on every text tested; `ml_tok` encodes and decodes.

### Tools

- `ml_run`: generation with top-k, top-p and repetition penalty, and a chat
  mode (`--chat llama3|zephyr|llama2`).
- `ml_bench`, `tools/perplexity.py`, `--profile`.

### Verification

- Kernel tests against scalar and double-precision references, end-to-end
  logits against a numpy reference, qemu for ARM and older x86 CPUs, native
  Arm CI, and real-model checks (stories15M, TinyLlama 1.1B, Llama 3.2 1B).
- Release binaries for Linux x86-64, AArch64 and ARMv7 (static) and Windows
  x86-64 (MinGW, checked under Wine), each run before publishing.
