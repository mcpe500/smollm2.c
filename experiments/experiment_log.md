# experiment_log.md - Autoresearch Experiment Log

## Baseline
Date: 2026-05-30
Baseline: ~0.9 tok/s (31 seconds for 92 chars)

## Experiments

| # | Date | Change | Before | After | Status |
|---|------|--------|--------|-------|--------|
| 1 | 2026-05-30 | Removed DEBUG prints from sm2dl_decode.c | 0.9 tok/s | 0.9 tok/s | No change |
| 2 | 2026-05-30 | Optimized F16->F32 (removed powf) | 0.9 tok/s | 2.8 tok/s | ✅ 3x speedup |
| 3 | 2026-05-30 | Precompute RoPE frequencies | 2.8 tok/s | 5.7 tok/s | ✅ 2x speedup |
| 4 | 2026-05-30 | F16 lookup table + 4x unrolling | 5.7 tok/s | 6.2 tok/s | ✅ 10% speedup |
| 5 | 2026-05-30 | Attention 4x unroll (no improvement) | 6.2 tok/s | 6.0 tok/s | ❌ No change |
| 6 | 2026-05-30 | NEON backend (not used, slower) | 6.0 tok/s | 2.3 tok/s | ❌ Backend not in decode path |