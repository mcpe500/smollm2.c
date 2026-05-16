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
