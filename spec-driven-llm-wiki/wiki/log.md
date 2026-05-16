# Wiki Log

Append-only operation log.

## [2026-04-24] init | Spec-driven wiki skeleton

Created initial wiki skeleton, prompt files, templates, and tooling plan.

## [2026-04-24] graph | Knowledge graph rebuilt

5 nodes, 0 edges.

## [2026-04-25] graph | Knowledge graph rebuilt

5 nodes, 0 edges.

## [2026-05-16] spec | SPEC.001.smollm2c-master-blueprint.md

Created master blueprint for smollm2.c inference engine. Defines Phase 1-8 implementation roadmap, GQA configs for 135M/360M/1.7B, smollm2dl decode layer architecture, KV Turbo Quant strategy, and .sm2 file format specification.

## [2026-05-16] wiki | Comprehensive wiki update

Updated spec-driven-llm-wiki to maximum completeness:

**SPEC Updates:**
- SPEC 001: Expanded Phase 8 (DFlash) with full z-lab/dflash architecture
- Added sm2_file_header struct definitions
- Added DFlash APIs (sm2_dflash_config, sm2_dflash_draft_block, sm2_dflash_verify)
- Added sm2_kv_dtype quantization modes
- Expanded graph plan with sm2-dflash-module node

**New Components:**
- sm2-kv-cache.md - Paged KV cache management
- sm2-file-format.md - .sm2 binary format specification
- sm2-backend-ref.md - Portable matmul backend

**New Decisions (ADRs):**
- ADR-005: DFlash Integration
- ADR-006: Weight Quantization Strategy

**New Patterns:**
- speculative-decoding.md - Draft-verifier pattern
- Updated gqa-attention.md
- Updated low-memory-mode.md

**Wiki Updates:**
- wiki/overview.md - Comprehensive project overview
- wiki/index.md - Complete component/decision/pattern index
- wiki/components/smollm2dl-decode-layer.md - Updated with DFlash section
- wiki/components/smollm2d-server-daemon.md - Updated with full API

## [2026-05-16] impl | Phase 1-3 implementation

**Status: Code implemented, testing in progress**

**Implementation completed:**
- Full C codebase: 38 source files across decode/, attention/, kv/, quant/, backend/, server/, dflash/
- `smollm2-cli` binary compiles and runs
- `smollm2-135m.sm2` converted from HuggingFace checkpoint (1.17MB)
- `tools/smollm2-convert.py` - Download + convert from HF

**Current issue: Magic byte mismatch**
- .sm2 file has magic `SM2C001\x01` (9 bytes due to Python string null terminator)
- C code expects `SM2C001` (8 bytes, no null terminator)
- Fix needed in converter: `MAGIC = b'SM2C001'` should be 8 bytes exactly

**Files created:**
- `include/smollm2.h`, `include/sm2_utils.h`
- `src/smollm2.c`, `src/sm2_model.c`, `src/sm2_tokenizer.c`, `src/sm2_rmsnorm.c`, `src/sm2_rope.c`, `src/sm2_mlp.c`, `src/sm2_sampling.c`, `src/sm2_context.c`, `src/sm2_matmul_ref.c`
- `src/decode/sm2dl_decode.c`, `sm2dl_flash_decode.c`, `sm2dl_batch_decode.c`, `sm2dl_speculative.c`, `sm2dl_kv_quant.c`, `sm2dl_paged_attention.c`
- `src/attention/sm2_attn_prefill.c`, `sm2_attn_flash_prefill.c`, `sm2_attn_paged.c`
- `src/kv/sm2_kv_pool.c`, `sm2_kv_page.c`, `sm2_kv_quant_q4.c`, `sm2_kv_quant_q8.c`, `sm2_kv_turbo2.c`
- `src/quant/sm2_q4.c`, `sm2_q4k.c`, `sm2_q5k.c`, `sm2_q8.c`
- `src/server/smollm2d.c`, `sm2_http.c`, `sm2_scheduler.c`, `sm2_sse.c`, `sm2_metrics.c`
- `src/dflash/sm2_dflash.c`, `sm2_dflash_model.c`, `sm2_dflash_verify.c`
- `Makefile`
- `tools/smollm2-convert.py`
- `tokenizer.json`, `smollm2-135m.safetensors`, `smollm2-135m.sm2`

## [2026-05-17] fix | Critical bugs fixed - Model now generates tokens

**Date:** May 17, 2026 02:55 AM

**Status:** ✅ Model generating valid tokens - SEGFAULT FIXED

### Critical Bugs Fixed

1. **down_proj matmul indexing** ✅
   - Was: `down_proj[down_off + j * hidden_dim + i]` (wrong - row j, col i)
   - Fixed: `down_proj[down_off + i * hidden_dim + j]` (correct - row i, col j)
   - This was causing memory corruption that manifested as segfault

2. **Post-attention RMSNorm** ✅
   - Was: simple `xb[i] = xb[i] * weight[i]` (not RMSNorm)
   - Fixed: proper RMSNorm formula: `xb[i] = (x[i] / rms) * weight[i]`
   - Must compute `rms = sqrt(sum(x^2)/dim + eps)` first

3. **sm2_decode_next sequence** ✅
   - Was: sample first, then embed, then forward (WRONG)
   - Fixed: embed → forward layers → compute logits → sample
   - Tokens must be embedded through the model before sampling

4. **Final RMSNorm** ✅
   - Was: LayerNorm with mean subtraction
   - Fixed: Pure RMSNorm (no mean subtraction)

### Test Results

```bash
$ ./smollm2-cli -m smollm2-135m-v5.sm2 -p "Hello" -n 10
Model loaded in 262.4 ms
Prefill done in 618.7 ms
Output: [gen_tok=23]'[gen_tok=10]<iss[gen_tok=28],[gen_tok=20]$[gen_tok=6]<fil[EOS=0]
Generated 5 tokens in 3320.5 ms (664.1 ms/token)
Total time: 3939.8 ms
```

### Files Modified

- `src/sm2_context.c` - RMSNorm (final_norm, post_attention), FFN matmul indexing
- `src/smollm2.c` - Fixed sm2_decode_next sequence order
- `smollm2-135m-v5.sm2` - Regenerated model file

### Known Remaining Issues

1. **Tokenizer** - Uses byte fallback, full tokenizer.json not loaded
2. **EOS detection** - Stops at ~5 tokens, token 0 selected too early
3. **Output quality** - Not yet verified against HF reference

### Next Steps

1. Fix tokenizer integration (load tokenizer.json properly)
2. Fix EOS detection
3. Verify output against HuggingFace reference
