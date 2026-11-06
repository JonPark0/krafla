#!/bin/sh
# Scores Krafla's Q4 rule against llama.cpp's Q4_0 on the same tensors: every
# variant is a GGUF of one model scored by llama-perplexity (perplexity, KL
# divergence from the unquantized model, same top token). docs/q4-scales.md
# describes the runs.
#
#   scripts/compare_q4.sh LLAMA_CPP_DIR REF_GGUF TRAIN_TEXT TEST_TEXT [OUT]
#
# LLAMA_CPP_DIR is a llama.cpp checkout built in build-native (llama-quantize,
# llama-imatrix, llama-perplexity); its gguf-py is used to read and patch the
# files. REF_GGUF is the unquantized model (F16 or BF16). The importance
# matrix is collected on TRAIN_TEXT (64 chunks of 512 tokens); TEST_TEXT is
# scored (CHUNKS chunks of 512, default 32). OUT (default build/q4) gets a log
# per variant; VARIANTS picks some of them:
#   round      llama-quantize --pure Q4_0 (no importance matrix)
#   fit        the same tensors, quantized again by tools/mlb.py (--q4-rule fit)
#   flat       --q4-rule flat: the fit weighted by sqrt(s2 + x^2)
#   fit_layers fit for the layers' matrices, the embedding and classifier rounded
#   fit_emb    fit for the embedding and classifier only
#   lsearch    llama-quantize with an importance matrix of equal weights
#   kunif      Krafla's rule with that matrix (the same bytes as flat)
#   limat      llama-quantize --pure --imatrix
#   kimat      Krafla's rule with the same importance matrix
#   q4km       Q4_K_M; q4km_imat with the importance matrix; q8 Q8_0 --pure
set -eu
[ $# -ge 4 ] || { sed -n '2,24p' "$0"; exit 2; }
LCPP=$1 REF=$2 TRAIN=$3 TEST=$4 OUT=${5:-build/q4}
CHUNKS=${CHUNKS:-32}
VARIANTS=${VARIANTS:-"round fit flat lsearch limat kimat q4km q4km_imat q8"}
B=$LCPP/build-native/bin
PATCH="python3 $(dirname "$0")/q4_patch_gguf.py"
export GGUF_PY=$LCPP/gguf-py
mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

if [ ! -f "$OUT/imat.gguf" ]; then
    log "importance matrix"
    "$B/llama-imatrix" -m "$REF" -f "$TRAIN" -c 512 --chunks 64 -o "$OUT/imat.gguf" > "$OUT/imatrix.log" 2>&1
fi
# equal weights: llama-quantize reads a count of 0 as weight 1 for every channel
if [ ! -f "$OUT/uniform.gguf" ]; then
    cp "$OUT/imat.gguf" "$OUT/uniform.gguf"
    python3 - "$OUT/uniform.gguf" <<'EOF'
import os, sys
import numpy as np
sys.path.insert(0, os.environ["GGUF_PY"])
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
mm = np.memmap(sys.argv[1], mode="r+")
for t in r.tensors:
    if t.name.endswith(".counts"):
        mm[int(t.data_offset): int(t.data_offset) + int(t.n_bytes)] = 0
mm.flush()
EOF
fi

log "reference logits"
[ -f "$OUT/base.kld" ] || "$B/llama-perplexity" -m "$REF" -f "$TEST" -c 512 --chunks $CHUNKS \
    --kl-divergence-base "$OUT/base.kld" > "$OUT/ref.log" 2>&1

# llama.cpp's own Q4_0; with --rule round, Krafla's quantizer must give the same bytes
[ -f "$OUT/round.gguf" ] || "$B/llama-quantize" --pure "$REF" "$OUT/round.gguf" Q4_0 > "$OUT/q_round.log" 2>&1
$PATCH "$REF" "$OUT/round.gguf" --rule round --check

V=$OUT/variant.gguf
for v in $VARIANTS; do
    log "$v"
    case $v in
        round)     cp "$OUT/round.gguf" "$V" ;;
        fit)       cp "$OUT/round.gguf" "$V"; $PATCH "$REF" "$V" ;;
        flat)      cp "$OUT/round.gguf" "$V"; $PATCH "$REF" "$V" --rule flat ;;
        fit_layers) cp "$OUT/round.gguf" "$V"; $PATCH "$REF" "$V" --tensors '^blk\.' ;;
        fit_emb)   cp "$OUT/round.gguf" "$V"; $PATCH "$REF" "$V" --tensors '^(token_embd|output)\.' ;;
        kunif)     cp "$OUT/round.gguf" "$V"; $PATCH "$REF" "$V" --imatrix "$OUT/uniform.gguf" ;;
        kimat)     cp "$OUT/round.gguf" "$V"; $PATCH "$REF" "$V" --imatrix "$OUT/imat.gguf" ;;
        lsearch)   "$B/llama-quantize" --pure --imatrix "$OUT/uniform.gguf" "$REF" "$V" Q4_0 ;;
        limat)     "$B/llama-quantize" --pure --imatrix "$OUT/imat.gguf" "$REF" "$V" Q4_0 ;;
        q4km)      "$B/llama-quantize" "$REF" "$V" Q4_K_M ;;
        q4km_imat) "$B/llama-quantize" --imatrix "$OUT/imat.gguf" "$REF" "$V" Q4_K_M ;;
        q8)        "$B/llama-quantize" --pure "$REF" "$V" Q8_0 ;;
        *)         log "unknown variant $v"; exit 2 ;;
    esac > "$OUT/q_$v.log" 2>&1
    "$B/llama-perplexity" -m "$V" -f "$TEST" -c 512 --chunks $CHUNKS \
        --kl-divergence-base "$OUT/base.kld" --kl-divergence > "$OUT/$v.log" 2>&1
    printf '%-10s %s MiB  PPL %s  KLD %s  same top %s\n' "$v" $(( $(wc -c < "$V") / 1048576 )) \
        "$(sed -n 's/^Mean PPL(Q) *: *\([0-9.]*\).*/\1/p' "$OUT/$v.log")" \
        "$(sed -n 's/^Mean *KLD: *\([0-9.]*\).*/\1/p' "$OUT/$v.log")" \
        "$(sed -n 's/^Same top p: *\([0-9.]*\).*/\1/p' "$OUT/$v.log")%" | tee -a "$OUT/results.txt"
    rm -f "$V"
done
