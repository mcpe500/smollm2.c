#!/usr/bin/env python3
"""Python reference implementation to verify SmolLM2-135M expected outputs."""

import subprocess
import sys

# Install dependencies
print("Installing transformers and torch...")
subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "torch", "-q"],
               capture_output=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test 1: Tokenization
print("\n=== Tokenization ===")
print(f"'hello' -> {tokenizer.encode('hello')}")
print(f"'hello' decoded: '{tokenizer.decode([28120])}'")

# Test 2: Plain "hello"
print("\n=== Test: Plain 'hello' ===")
input_ids = tokenizer.encode("hello", return_tensors="pt")
print(f"Input IDs: {input_ids.tolist()}")
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, 10)
    print("Top 10 tokens (with probabilities):")
    for i in range(10):
        tok = tokenizer.decode([top_ids[i].item()])
        print(f"  {top_ids[i].item():5d}: {probs[top_ids[i]].item():.4f} '{tok}'")

# Test 3: With chat template
print("\n=== Test: Chat template ===")
chat = tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True)
print(f"Chat text: '{chat}'")
input_ids = tokenizer.encode(chat, return_tensors="pt")
print(f"Input IDs: {input_ids.tolist()}")
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, 10)
    print("Top 10 tokens for chat:")
    for i in range(10):
        tok = tokenizer.decode([top_ids[i].item()])
        print(f"  {top_ids[i].item():5d}: {probs[top_ids[i]].item():.4f} '{tok}'")

# Test 4: Generate with chat template
print("\n=== Generation from chat template ===")
input_ids = tokenizer.encode(chat, return_tensors="pt")
with torch.no_grad():
    generated = model.generate(input_ids, max_new_tokens=20, do_sample=True, temperature=0.7)
    output_text = tokenizer.decode(generated[0])
    print(f"Output: {output_text}")

# Additional: Check logits for first token position vs last
print("\n=== Checking logits at different positions ===")
input_ids = tokenizer.encode("hello", return_tensors="pt")
print(f"Input length: {input_ids.shape[1]}")
with torch.no_grad():
    outputs = model(input_ids)
    # Logits shape: [batch, seq_len, vocab_size]
    print(f"Logits shape: {outputs.logits.shape}")
    # Check first position logits
    first_logits = outputs.logits[0, 0]
    first_probs = torch.softmax(first_logits, dim=-1)
    first_top_probs, first_top_ids = torch.topk(first_probs, 5)
    print("First position top 5:")
    for i in range(5):
        tok = tokenizer.decode([first_top_ids[i].item()])
        print(f"  {first_top_ids[i].item():5d}: {first_probs[first_top_ids[i]].item():.4f} '{tok}'")

    # Check last position logits
    last_logits = outputs.logits[0, -1]
    last_probs = torch.softmax(last_logits, dim=-1)
    last_top_probs, last_top_ids = torch.topk(last_probs, 5)
    print("Last position top 5:")
    for i in range(5):
        tok = tokenizer.decode([last_top_ids[i].item()])
        print(f"  {last_top_ids[i].item():5d}: {last_probs[last_top_ids[i]].item():.4f} '{tok}'")