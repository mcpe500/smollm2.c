#!/bin/bash
# Chat usability evaluation: tests real-world chat prompts
# Must pass for every kept experiment
set -euo pipefail

MODEL="./smollm2"
MODEL_PATH="${1:-models/smollm2-135m-instruct-f16.gguf}"

echo "=== Chat Usability Evaluation ==="
echo "Model: $MODEL_PATH"
echo ""

# Test results
PASS=0
FAIL=0
TOTAL=0

run_test() {
    local name="$1"
    local prompt="$2"
    local temp="$3"
    local max_tokens="$4"
    local check_type="$5"  # "factual" or "chat"
    
    TOTAL=$((TOTAL + 1))
    echo "[$TOTAL] $name (temp=$temp)"
    echo "Prompt: $prompt"
    
    # Get full output, strip timing line, combine multiline output
    full_output=$($MODEL -m "$MODEL_PATH" -p "$prompt" -n $max_tokens --temp $temp 2>&1 || true)
    output=$(echo "$full_output" | grep -v '\[.*tokens.*tok/s\]' | tr '\n' ' ' | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')
    echo "Output: $output"
    
    # Universal degenerate pattern checks
    if echo "$output" | grep -qE '(.)\1{9,}'; then
        echo "❌ FAIL: excessive character repetition"
        FAIL=$((FAIL + 1))
        echo ""
        return 1
    fi
    
    if echo "$output" | grep -qE '[0-9]{12,}'; then
        echo "❌ FAIL: degenerate long number (12+ digits)"
        FAIL=$((FAIL + 1))
        echo ""
        return 1
    fi
    
    # Check for broken punctuation loops
    if echo "$output" | grep -qE '[.!?]{5,}'; then
        echo "❌ FAIL: punctuation repetition"
        FAIL=$((FAIL + 1))
        echo ""
        return 1
    fi
    
    # Check for name hallucination patterns
    if echo "$output" | grep -qiE '(hi|hello|hey)\s+(alex|sarah|john|mary|mike),.*\b(alex|sarah|john|mary|mike)\b'; then
        echo "❌ FAIL: hallucinated name pattern detected"
        FAIL=$((FAIL + 1))
        echo ""
        return 1
    fi
    
    # Minimum output length (but allow short factual answers)
    if [ ${#output} -lt 3 ]; then
        echo "❌ FAIL: output too short (${#output} chars)"
        FAIL=$((FAIL + 1))
        echo ""
        return 1
    fi
    
    # Type-specific checks
    if [ "$check_type" = "factual" ]; then
        # Factual answers can be short but should not be random numbers
        if echo "$output" | grep -qE '^[0-9]{8,}'; then
            echo "❌ FAIL: starts with random long number"
            FAIL=$((FAIL + 1))
            echo ""
            return 1
        fi
    fi
    
    if [ "$check_type" = "chat" ]; then
        # Chat responses should have reasonable length
        if [ ${#output} -lt 10 ]; then
            echo "❌ FAIL: chat output too short (${#output} chars)"
            FAIL=$((FAIL + 1))
            echo ""
            return 1
        fi
    fi
    
    echo "✓ PASS"
    PASS=$((PASS + 1))
    echo ""
}

# Use raw prompts without chat template for now
# P0 Required Tests
run_test "greeting-1" "Hello" 0.3 40 "chat"
run_test "greeting-2" "Good Morning" 0.3 40 "chat"
run_test "math-1" "What is 2+2?" 0.0 30 "factual"
run_test "math-2" "What is 10 + 29?" 0.0 30 "factual"
run_test "factual-1" "The capital of France is" 0.0 30 "factual"
run_test "empathy-1" "Write one short friendly reply to: I feel tired today" 0.3 50 "chat"
run_test "translation-1" "Translate to Indonesian: good morning" 0.0 30 "factual"
run_test "instruction-1" "Answer with only one word: yes" 0.0 10 "factual"

# === Additional Hidden Tests (not disclosed in prompt) ===
run_test "greeting-3" "How are you?" 0.3 40 "chat"
run_test "math-3" "Calculate 15 - 7" 0.0 30 "factual"
run_test "factual-2" "Water is made of" 0.0 30 "factual"
run_test "instruction-2" "Complete this: The sun rises in the" 0.0 30 "factual"
run_test "creative-1" "Once upon a time" 0.3 50 "chat"
run_test "reasoning-1" "If it rains, should I bring an umbrella?" 0.0 30 "factual"

# === Summary ===
echo "======================================"
echo "Total tests: $TOTAL"
echo "Passed: $PASS ($((PASS * 100 / TOTAL))%)"
echo "Failed: $FAIL ($((FAIL * 100 / TOTAL))%)"
echo "======================================"

if [ $PASS -lt $((TOTAL * 80 / 100)) ]; then
    echo "FAIL: Less than 80% pass rate"
    exit 1
fi

if [ $FAIL -gt 3 ]; then
    echo "FAIL: More than 3 failures"
    exit 1
fi

echo "Chat usability: ACCEPTABLE"
