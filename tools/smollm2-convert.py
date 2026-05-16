#!/usr/bin/env python3
"""
smollm2-convert: Convert HuggingFace SmolLM2 .safetensors to .sm2 binary format.
Usage:
    python3 smollm2-convert.py [--model MODEL] [--output OUTPUT] [--tokenizer TOKENIZER]
    python3 smollm2-convert.py --download 135M    # Download and convert SmolLM2-135M-Instruct
    python3 smollm2-convert.py --download 360M    # Download SmolLM2-360M-Instruct
    python3 smollm2-convert.py --download 1.7B   # Download SmolLM2-1.7B-Instruct
"""

import struct, json, sys, os, argparse

# Config per model variant
MODEL_CONFIGS = {
"135M": {
        "variant": "smollm2-135m",
        "variant_id": 135,
        "n_layers": 30,
        "dim": 576,
        "hidden_dim": 1536,
        "n_heads": 9,
        "n_kv_heads": 3,
        "head_dim": 64,
        "vocab_size": 49152,
    },
    "360M": {
        "variant": "smollm2-360m",
        "variant_id": 360,
        "n_layers": 32,
        "dim": 960,
        "hidden_dim": 2560,
        "n_heads": 15,
        "n_kv_heads": 5,
        "head_dim": 64,
        "vocab_size": 49152,
    },
    "1.7B": {
        "variant": "smollm2-1.7b",
        "variant_id": 1700,
        "n_layers":24,
        "dim": 2048,
        "hidden_dim": 5504,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 64,
        "vocab_size": 49152,
    },
}

# HuggingFace repo per variant
HF_REPOS = {
    "135M": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "360M": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "1.7B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

def download_file(hf_repo, filename, dest_path, token=None):
    """Download a file from HuggingFace Hub."""
    import urllib.request
    
    url = f"https://huggingface.co/{hf_repo}/resolve/main/{filename}"
    print(f"  Downloading {url}")
    print(f"  -> {dest_path}")
    
    req = urllib.request.Request(url, headers={
        "User-Agent": "smollm2-convert/1.0"
    })
    
    with urllib.request.urlopen(req, timeout=300) as response:
        total = int(response.headers.get('Content-Length', 0))
        with open(dest_path, 'wb') as f:
            downloaded = 0
            chunk_size = 8192
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = f"{downloaded*100//total}%"
                else:
                    pct = f"{downloaded//1024//1024}MB"
                print(f"\r  Downloaded: {pct}", end="", flush=True)
    print()
    return dest_path


def read_safetensors(path):
    """Read a safetensors file, return dict of tensor name -> (shape, dtype, bytes)."""
    import struct as st
    
    tensors = {}
    with open(path, 'rb') as f:
        header_size = st.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
        
        for name, info in header.items():
            if name == '__metadata__':
                continue
            offsets = info['data_offsets']
            shape = info['shape']
            dtype = info['dtype']
            tensors[name] = {
                'shape': shape,
                'dtype': dtype,
                'offset_start': offsets[0],
                'offset_end': offsets[1],
                'size': offsets[1] - offsets[0],
            }
        
        # Read raw bytes for each tensor
        for name, info in tensors.items():
            f.seek(8 + header_size + info['offset_start'])
            tensors[name]['data'] = f.read(info['size'])
    
    return tensors


def bf16_to_f16(bf16_bytes):
    """Convert BF16 bytes to F16 bytes (list of uint16)."""
    n = len(bf16_bytes) // 2
    out = []
    for i in range(n):
        bf16 = struct.unpack('<H', bf16_bytes[i*2:i*2+2])[0]
        # BF16: sign(1) + exp(8) + mantissa(7)
        # F16:  sign(1) + exp(5) + mantissa(10)
        s  = (bf16 >> 15) & 0x1
        be = (bf16 >> 7) & 0xFF   # BF16 exponent (8 bits)
        bm = bf16 & 0x7F          # BF16 mantissa (7 bits)
        
        if be == 0xFF:
            # Inf or NaN
            if bm == 0:
                f16 = (s << 15) | (0x1F << 10)  # Inf
            else:
                f16 = (s << 15) | (0x1F << 10) | (1 << 9)  # NaN (quiet)
        else:
            # Normalize BF16 exponent: bias 127 -> F16 bias 15
            exp = be - 127 + 15
            if exp >= 0x1F:
                f16 = (s << 15) | (0x1F << 10)  # Inf
            elif exp <= 0:
                # Denormal to F16
                if exp >= -10:
                    m = (1 << 10) | (bm << 3)
                    shift = 1 - exp
                    f16 = (s << 15) | (m >> shift)
                else:
                    f16 = 0
            else:
                # Normal
                f16 = (s << 15) | ((exp & 0x1F) << 10) | (bm << 3)
                # Truncate to 10-bit mantissa
        
        out.append(struct.pack('<H', f16))
    
    result = b''.join(out)
    return result


def f16_to_float(f16_bytes):
    """Convert F16 bytes to float32 bytes."""
    import struct as st
    n = len(f16_bytes) // 2
    out = []
    for i in range(n):
        f16 = st.unpack('<H', f16_bytes[i*2:i*2+2])[0]
        s  = (f16 >> 15) & 0x1
        e  = (f16 >> 10) & 0x1F
        m  = f16 & 0x3FF
        
        if e == 0:
            if m == 0:
                val = 0.0
            else:
                val = 2**(-14) * (m / 1024)
        elif e == 0x1F:
            if m == 0:
                val = float('inf')
            else:
                val = float('nan')
        else:
            val = 2**(e - 15) * (1 + m / 1024)
        
        if s:
            val = -val
        out.append(st.pack('<f', val))
    
    return b''.join(out)


def read_tokenizer_json(path):
    """Read tokenizer.json and return vocab + merges."""
    with open(path) as f:
        data = json.load(f)
    
    vocab = data['model']['vocab']
    merges = data['model']['merges']
    
    # Build id->token map
    # vocab is token->id, convert to id->token
    id_to_token = {}
    for token, idx in vocab.items():
        id_to_token[idx] = token
    
    return id_to_token, merges


def write_sm2(path, config, tensors, id_to_token, merges):
    """Write .sm2 binary file."""
    
    MAGIC = b'SM2C001'
    VERSION = 1
    
    # Map safetensor names -> .sm2 layer field names
    # Per layer: input_layernorm, q_proj, k_proj, v_proj, o_proj,
    #           post_attention_layernorm, gate_proj, up_proj, down_proj
    
    def get_tensor(name_prefix, layer_idx=None):
        if layer_idx is not None:
            name = name_prefix.replace('.', f'.{layer_idx}.', 1)
        else:
            name = name_prefix
        
        for key in tensors:
            if key == name or key.endswith(name):
                return tensors[key]
        return None
    
    with open(path, 'wb') as f:
        # ---- Header (256 bytes total) ----
        f.write(MAGIC)                                    # 8 bytes  (0-7)
        f.write(struct.pack('<I', VERSION))              # 4 bytes  (8-11)
        f.write(struct.pack('<I', config['variant_id'])) # 4 bytes  (12-15) variant
        f.write(struct.pack('<I', 0))                    # 4 bytes  (16-19) quant_type = F16
        f.write(struct.pack('<I', 0))                    # 4 bytes  (20-23) flags
        
        # Config block (8 x u32 = 32 bytes) (24-55)
        cfg = struct.pack('<IIIIIIII',
            config['n_layers'],
            config['dim'],
            config['hidden_dim'],
            config['n_heads'],
            config['n_kv_heads'],
            config['head_dim'],
            config['vocab_size'],
            0,  # quant_type repeated
        )
        f.write(cfg)
        
        # max_seq_len + rms_eps + rope_theta (4 + 4 + 4 = 12 bytes) (56-67)
        f.write(struct.pack('<I', config.get('max_seq_len', 2048)))
        f.write(struct.pack('<f', 1e-5))
        f.write(struct.pack('<f', 100000.0))
        
        # bos/eos/pad token ids (12 bytes) (68-79)
        f.write(struct.pack('<I', config.get('bos_token_id', 1)))
        f.write(struct.pack('<I', config.get('eos_token_id', 2)))
        f.write(struct.pack('<I', config.get('pad_token_id', 0)))
        
        # offsets (6 x u64 = 48 bytes) (80-127)
        # tokenizer_offset, tokenizer_size, tensor_index_offset, tensor_index_size,
        # weights_offset, weights_size
        # We'll fill these after we know where things are
        header_before_offsets = f.tell()  # = 128
        f.write(struct.pack('<QQQQQQ', 0, 0, 0, 0, 0, 0))  # 48 bytes
        
        # checksum (u64 = 8 bytes) (128-135)
        f.write(struct.pack('<Q', 0))  # placeholder
        
        # Padding to 256 bytes
        padding_needed = 256 - f.tell()
        if padding_needed < 0:
            raise AssertionError(f"Header overflow: {f.tell()} bytes written")
        f.write(b'\x00' * padding_needed)
        
        assert f.tell() == 256, f"Header should be 256 bytes, got {f.tell()}"
        
        # ---- Tokenizer section ----
        tokenizer_start = f.tell()
        
        # Write id -> token mapping
        for i in range(config['vocab_size']):
            if i in id_to_token:
                token_bytes = id_to_token[i].encode('utf-8')
            else:
                token_bytes = b''
            f.write(struct.pack('<I', len(token_bytes)))
            f.write(token_bytes)
        
        # Write merges
        f.write(struct.pack('<I', len(merges)))
        for merge in merges:
            merge_bytes = merge.encode('utf-8')
            f.write(struct.pack('<I', len(merge_bytes)))
            f.write(merge_bytes)
        
        tokenizer_end = f.tell()
        tokenizer_size = tokenizer_end - tokenizer_start
        
        # ---- Embeddings ----
        emb = get_tensor('model.embed_tokens.weight')
        if emb is None:
            raise ValueError("embed_tokens not found")
        
        f16_data = bf16_to_f16(emb['data'])
        f.write(struct.pack('<II', *emb['shape']))       # rows, cols
        f.write(f16_data)                               # raw F16 bytes
        
        # ---- Layers ----
        for layer in range(config['n_layers']):
            layer_prefix = f'model.layers.{layer}'
            
            def write_tensor(prefix):
                name = f'{layer_prefix}.{prefix}'
                t = None
                for key, val in tensors.items():
                    if key == name or key == f'{layer_prefix}.self_attn.{prefix}' or key == f'{layer_prefix}.mlp.{prefix}':
                        t = (key, val)
                        break
                    # Try simplified matching
                    if prefix == 'input_layernorm.weight' and 'input_layernorm.weight' in key:
                        t = (key, val)
                        break
                    if prefix in key:
                        t = (key, val)
                        break
                
                if t is None:
                    raise ValueError(f"Tensor not found: {name} (tried prefix={prefix})")
                
                key, val = t
                f16_data = bf16_to_f16(val['data'])
                f.write(struct.pack('<II', *val['shape']))
                f.write(f16_data)
            
            # input_layernorm
            t = None
            for key, val in tensors.items():
                if f'.layers.{layer}.' in key and 'input_layernorm.weight' in key:
                    t = val; break
            if t:
                f.write(struct.pack('<II', *t['shape']))
                f.write(bf16_to_f16(t['data']))
            
            # q_proj (dim x dim)
            for key, val in tensors.items():
                if f'.layers.{layer}.' in key and 'q_proj.weight' in key:
                    f.write(struct.pack('<II', *val['shape']))
                    f.write(bf16_to_f16(val['data']))
                    break
            
            # k_proj (n_kv_heads*head_dim x dim)
            for key, val in tensors.items():
                if f'.layers.{layer}.' in key and 'k_proj.weight' in key:
                    f.write(struct.pack('<II', *val['shape']))
                    f.write(bf16_to_f16(val['data']))
                    break
            
            # v_proj (n_kv_heads*head_dim x dim)
            for key, val in tensors.items():
                if f'.layers.{layer}.' in key and 'v_proj.weight' in key:
                    f.write(struct.pack('<II', *val['shape']))
                    f.write(bf16_to_f16(val['data']))
                    break
            
            # o_proj (dim x n_kv_heads*head_dim)
            for key, val in tensors.items():
                if f'.layers.{layer}.' in key and 'o_proj.weight' in key:
                    f.write(struct.pack('<II', *val['shape']))
                    f.write(bf16_to_f16(val['data']))
                    break
            
            # post_attention_layernorm
            for key, val in tensors.items():
                if f'.layers.{layer}.' in key and 'post_attention_layernorm.weight' in key:
                    f.write(struct.pack('<II', *val['shape']))
                    f.write(bf16_to_f16(val['data']))
                    break
            
            # gate_proj (hidden_dim x dim) = SwiGLU gate
            for key, val in tensors.items():
                if f'.layers.{layer}.' in key and 'gate_proj.weight' in key:
                    f.write(struct.pack('<II', *val['shape']))
                    f.write(bf16_to_f16(val['data']))
                    break
            
            # up_proj (hidden_dim x dim)
            for key, val in tensors.items():
                if f'.layers.{layer}.' in key and 'up_proj.weight' in key:
                    f.write(struct.pack('<II', *val['shape']))
                    f.write(bf16_to_f16(val['data']))
                    break
            
            # down_proj (dim x hidden_dim)
            for key, val in tensors.items():
                if f'.layers.{layer}.' in key and 'down_proj.weight' in key:
                    f.write(struct.pack('<II', *val['shape']))
                    f.write(bf16_to_f16(val['data']))
                    break
        
        # ---- final_norm ----
        for key, val in tensors.items():
            if key == 'model.norm.weight':
                f.write(struct.pack('<II', *val['shape']))
                f.write(bf16_to_f16(val['data']))
                break

        # ---- Update header offsets ----
        file_size = f.tell()
        weights_offset = 256  # weights start right after header
        weights_size = file_size - weights_offset

        f.seek(80)  # back to offsets position in header
        f.write(struct.pack('<QQQQQQ',
            256,             # tokenizer_offset
            tokenizer_size,   # tokenizer_size
            0,                # tensor_index_offset (no separate index)
            0,                # tensor_index_size
            weights_offset,   # weights_offset
            weights_size,     # weights_size
        ))
        # checksum (placeholder)
        f.write(struct.pack('<Q', 0))

    print(f"  Written: {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")


def convert(safetensors_path, tokenizer_path, model_size, output_path):
    """Main conversion."""
    print(f"Loading safetensors: {safetensors_path}")
    tensors = read_safetensors(safetensors_path)
    print(f"  Loaded {len(tensors)} tensors")
    
    print(f"Loading tokenizer: {tokenizer_path}")
    id_to_token, merges = read_tokenizer_json(tokenizer_path)
    print(f"  Vocab: {len(id_to_token)} tokens, {len(merges)} merges")
    
    cfg = MODEL_CONFIGS[model_size]
    print(f"Model config: {cfg}")
    
    print(f"Writing .sm2 file...")
    write_sm2(output_path, cfg, tensors, id_to_token, merges)
    print(f"Done! Output: {output_path}")


def download_and_convert(model_size):
    """Download model from HuggingFace and convert."""
    repo = HF_REPOS[model_size]
    cfg = MODEL_CONFIGS[model_size]
    
    model_file = f"smollm2-{model_size.lower()}.safetensors"
    tokenizer_file = "tokenizer.json"
    output_file = f"smollm2-{model_size.lower()}.sm2"
    
    print(f"\n=== Downloading SmolLM2-{model_size} from HuggingFace ===")
    print(f"Repo: {repo}")
    
    # Download model
    if not os.path.exists(model_file):
        download_file(repo, "model.safetensors", model_file)
    else:
        print(f"  Using existing: {model_file}")
    
    # Download tokenizer
    if not os.path.exists(tokenizer_file):
        download_file(repo, "tokenizer.json", tokenizer_file)
    else:
        print(f"  Using existing: {tokenizer_file}")
    
    print(f"\n=== Converting to .sm2 format ===")
    convert(model_file, tokenizer_file, model_size, output_file)


def main():
    parser = argparse.ArgumentParser(description="Convert SmolLM2 safetensors to .sm2 format")
    parser.add_argument('--model', default='smollm2-135m.safetensors',
                        help='Path to safetensors file')
    parser.add_argument('--tokenizer', default='tokenizer.json',
                        help='Path to tokenizer.json')
    parser.add_argument('--output', default=None,
                        help='Output .sm2 file path')
    parser.add_argument('--size', default='135M',
                        choices=['135M', '360M', '1.7B'],
                        help='Model size (for config)')
    parser.add_argument('--download', metavar='SIZE',
                        choices=['135M', '360M', '1.7B'],
                        help='Download model from HuggingFace and convert')
    
    args = parser.parse_args()
    
    if args.download:
        download_and_convert(args.download)
        return
    
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.model))[0]
        args.output = base + '.sm2'
    
    convert(args.model, args.tokenizer, args.size, args.output)


if __name__ == '__main__':
    main()