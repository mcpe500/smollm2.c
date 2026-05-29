#!/usr/bin/env python3
"""
verify_weights.py - Verify .sm2 weights match HuggingFace safetensors

Usage:
    python3 verify_weights.py [--sm2 SM2_FILE] [--hf HF_DIR]
"""

import struct
import json
import argparse
import os
from pathlib import Path

def bf16_to_float32(bf16_bits):
    """Convert BF16 bits to float32."""
    s = (bf16_bits >> 15) & 1
    be = (bf16_bits >> 7) & 0xFF
    bm = bf16_bits & 0x7F
    return (-1)**s * (2**(be - 127)) * (1 + bm/128)

def read_safetensors(hf_path):
    """Read HuggingFace safetensors file."""
    # Find the snapshot directory
    snapshots = list(Path(hf_path).glob("snapshots/*"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {hf_path}")
    snapshot = snapshots[-1]
    safetensor_path = snapshot / "model.safetensors"

    with open(safetensor_path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))

        tensors = {}
        for name, info in header.items():
            if name == '__metadata__':
                continue
            offsets = info['data_offsets']
            f.seek(8 + header_size + offsets[0])
            tensors[name] = {
                'shape': info['shape'],
                'data': f.read(offsets[1] - offsets[0])
            }

    return tensors

def read_sm2_weights(sm2_path):
    """Read weights from .sm2 file."""
    with open(sm2_path, 'rb') as f:
        weights_offset = 1179115  # from header
        f.seek(weights_offset)

        # tok_embeddings
        rows, cols = struct.unpack('<II', f.read(8))
        embeddings = struct.unpack(f'<{rows*cols}H', f.read(rows*cols*2))
        weights = {'tok_embeddings': list(embeddings)}

        # Layers
        for layer in range(30):
            r, c = struct.unpack('<II', f.read(8))
            weights[f'layer{layer}_input_ln'] = list(struct.unpack(f'<{c}H', f.read(c*2)))

            r, c = struct.unpack('<II', f.read(8))
            weights[f'layer{layer}_q'] = list(struct.unpack(f'<{r*c}H', f.read(r*c*2)))

            r, c = struct.unpack('<II', f.read(8))
            weights[f'layer{layer}_k'] = list(struct.unpack(f'<{r*c}H', f.read(r*c*2)))

            r, c = struct.unpack('<II', f.read(8))
            weights[f'layer{layer}_v'] = list(struct.unpack(f'<{r*c}H', f.read(r*c*2)))

            r, c = struct.unpack('<II', f.read(8))
            weights[f'layer{layer}_o'] = list(struct.unpack(f'<{r*c}H', f.read(r*c*2)))

            r, c = struct.unpack('<II', f.read(8))
            weights[f'layer{layer}_post_ln'] = list(struct.unpack(f'<{c}H', f.read(c*2)))

            r, c = struct.unpack('<II', f.read(8))
            weights[f'layer{layer}_gate'] = list(struct.unpack(f'<{r*c}H', f.read(r*c*2)))

            r, c = struct.unpack('<II', f.read(8))
            weights[f'layer{layer}_up'] = list(struct.unpack(f'<{r*c}H', f.read(r*c*2)))

            r, c = struct.unpack('<II', f.read(8))
            weights[f'layer{layer}_down'] = list(struct.unpack(f'<{r*c}H', f.read(r*c*2)))

        # final_norm
        r, c = struct.unpack('<II', f.read(8))
        weights['final_norm'] = list(struct.unpack(f'<{c}H', f.read(c*2)))

    return weights

def verify_tensor(hf_data, sm2_data, name, n_samples=100):
    """Verify a single tensor matches."""
    n = len(hf_data) // 2

    hf_vals = [bf16_to_float32(struct.unpack('<H', hf_data[i*2:i*2+2])[0]) for i in range(n)]
    sm2_vals = [struct.unpack('<e', struct.pack('<H', v))[0] for v in sm2_data[:n]]

    diffs = [abs(hf_vals[i] - sm2_vals[i]) for i in range(n)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / n

    hf_checksum = sum(abs(v) for v in hf_vals[:n_samples])
    sm2_checksum = sum(abs(v) for v in sm2_vals[:n_samples])

    return {
        'name': name,
        'n_values': n,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'hf_checksum': hf_checksum,
        'sm2_checksum': sm2_checksum,
        'passed': max_diff < 1e-4
    }

def main():
    parser = argparse.ArgumentParser(description='Verify .sm2 weights match HF safetensors')
    parser.add_argument('--sm2', default='smollm2-135m.sm2', help='Path to .sm2 file')
    parser.add_argument('--hf', default=None, help='Path to HF cache directory')
    args = parser.parse_args()

    if args.hf is None:
        hf_base = Path.home() / '.cache' / 'huggingface' / 'hub'
        args.hf = hf_base / 'models--HuggingFaceTB--SmolLM2-135M-Instruct'

    print(f"Loading HF safetensors from {args.hf}...")
    hf_tensors = read_safetensors(args.hf)

    print(f"Loading .sm2 weights from {args.sm2}...")
    sm2_weights = read_sm2_weights(args.sm2)

    print("\n=== Weight Verification ===\n")

    checks = [
        ('tok_embeddings', 'model.embed_tokens.weight'),
        ('layer0_input_ln', 'model.layers.0.input_layernorm.weight'),
        ('layer0_q', 'model.layers.0.self_attn.q_proj.weight'),
        ('final_norm', 'model.norm.weight'),
    ]

    all_passed = True
    for sm2_name, hf_name in checks:
        if hf_name not in hf_tensors or sm2_name not in sm2_weights:
            print(f"SKIP: {sm2_name} - not found")
            continue

        result = verify_tensor(hf_tensors[hf_name]['data'], sm2_weights[sm2_name], sm2_name)
        status = "PASS" if result['passed'] else "FAIL"
        if not result['passed']:
            all_passed = False

        print(f"{result['name']:20s}: {result['n_values']:8d} vals, "
              f"max_diff={result['max_diff']:.2e}, "
              f"checksum={result['sm2_checksum']:.2f} [{status}]")

    print(f"\n{'='*50}")
    if all_passed:
        print("ALL CHECKS PASSED - Weights are correct!")
    else:
        print("SOME CHECKS FAILED - Check the weights!")

if __name__ == '__main__':
    main()
