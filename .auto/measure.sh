#!/bin/bash
set -euo pipefail

MODEL_PATH=$(./smollm2 --inspect 2>/dev/null | head -1 | grep -q 'GGUF' && echo 'auto' || echo 'fail')
# model resolves automatically via Ollama manifest

# Correctness gate: argmax for 4-token prompt must be 504 ("The")
# This is the verified golden value from handoff 0002/0003
ARGMAX=$(./smollm2 --logits '<|im_start|>assistant
' 2>&1 | grep '^argmax:' | awk '{print $2}')
if [ "$ARGMAX" != "504" ]; then
    echo "CORRECTNESS FAIL: argmax=$ARGMAX expected=57" >&2
    exit 1
fi

# Run decode trials: 2 warmup (discarded) then 5 measured, take median.
# This avoids CPU boost spike on first run and measures steady-state performance.
for i in 1 2; do
    ./smollm2 -p "Hello, how are you?" -n 50 --temp 0.0 > /dev/null 2>&1
done
TOKS=""
for i in 1 2 3 4 5; do
    OUT=$(./smollm2 -p "Hello, how are you?" -n 50 --temp 0.0 2>&1)
    T=$(echo "$OUT" | grep -o '[0-9.]* tok/s' | grep -o '[0-9.]*')
    TOKS="$TOKS $T"
done

# Sort and take median (3rd of 5)
MED=$(echo $TOKS | tr ' ' '\n' | grep -v '^$' | sort -n | sed -n '3p')

# Prefill speed
PREFILL=$(./smollm2 --logits 'Hello' 2>&1 | grep 'prefill:' | grep -o '[0-9.]* tok/s' | grep -o '[0-9.]*')

echo "METRIC tok_s=$MED"
echo "METRIC prefill_toks=${PREFILL:-0}"
