# attn_bench 2026-07-14

| rope | kv | attn | parity | tok/s | hello_argmax |
|---|---|---|---|---|---|
| f32 | f32 | naive | PASS | 65.0 | 19556 |
| f32 | f32 | flash | PASS | 67.0 | 19556 |
| f32 | f16 | naive | soft | 43.7 | 57 |
| f32 | f16 | flash | soft | 45.3 | 57 |
| f32 | q8 | naive | soft | 68.8 | 57 |
| f32 | q8 | flash | soft | 69.6 | 57 |
| f16 | f32 | naive | soft | 66.5 | 57 |
| f16 | f32 | flash | soft | 65.8 | 57 |
| f16 | f16 | naive | soft | 47.4 | 57 |
| f16 | f16 | flash | soft | 42.2 | 57 |
| f16 | q8 | naive | soft | 68.2 | 57 |
| f16 | q8 | flash | soft | 57.3 | 57 |
| q8 | f32 | naive | soft | 60.4 | 57 |
| q8 | f32 | flash | soft | 70.1 | 57 |
| q8 | f16 | naive | PASS | 48.0 | 19556 |
| q8 | f16 | flash | PASS | 48.9 | 19556 |
| q8 | q8 | naive | soft | 65.5 | 57 |
| q8 | q8 | flash | soft | 65.2 | 57 |

**Winner (PASS + max tok/s):** f32/f32/flash = 67.0 tok/s
