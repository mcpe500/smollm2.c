#!/bin/bash
set -euo pipefail

MODEL_PATH=$(./smollm2 --inspect 2>/dev/null | head -1 | grep -q 'GGUF' && echo 'auto' || echo 'fail')
# model resolves automatically via Ollama manifest

# Correctness gate: argmax for 4-token prompt must be 504 ("The")
# This is the verified golden value from handoff 0002/0003
ARGMAX=$(./smollm2 --logits '<|im_start|>assistant
' 2>&1 | grep '^argmax:' | awk '{print $2}')
if [ "$ARGMAX" != "504" ]; then
    echo "CORRECTNESS FAIL: argmax=$ARGMAX expected=504" >&2
    exit 1
fi

# Run 3 decode trials, take median tok/s
TOKS=""
for i in 1 2 3; do
    OUT=$(./smollm2 -p "Hello, how are you?" -n 50 --temp 0.0 2>&1)
    T=$(echo "$OUT" | grep -o '[0-9.]* tok/s' | grep -o '[0-9.]*')
    TOKS="$TOKS $T"
done

# Median of 3
MED=$(echo $TOKS | tr ' ' '\n' | grep -v '^$' | sort -n | sed -n '2p')

# Prefill speed
PREFILL=$(./smollm2 --logits 'Hello' 2>&1 | grep 'prefill:' | grep -o '[0-9.]* tok/s' | grep -o '[0-9.]*')

echo "METRIC tok_s=$MED"
echo "METRIC prefill_toks=${PREFILL:-0}"
