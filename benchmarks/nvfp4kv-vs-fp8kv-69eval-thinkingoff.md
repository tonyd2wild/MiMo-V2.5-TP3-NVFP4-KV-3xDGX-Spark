# Benchmark — NVFP4 KV vs FP8 KV (MiMo-V2.5 Omni, TP=3, 3× DGX Spark)

Same model (`lukealonso/MiMo-V2.5-NVFP4`), same 3× DGX Spark cluster, same TP=3 / 1M-context / MTP2 / Omni config. The **only** material change is the KV-cache dtype (`fp8_e4m3` → `nvfp4`) and its required DiffKV mod + WMMA decode path.

Measured with the same 69-scenario tool-eval harness (thinking-off), comparing `tok_s_decode` — not a single short-prompt wall-clock.

## NVFP4 KV (this recipe)

```
result file:      mimo-nvfp4kv-wmma-nccl64-69eval-thinkingoff__20260622T041320.json
passed:           66
partial:          3
failed:           0
points:           135 / 138
quality:          97.8
tok_s_decode:     36.4
tok_s_effective:  32.4
GPU KV cache size: 10,592,793 tokens
1M concurrency:   10.59x
```

## FP8 KV (frozen public reference)

```
tok_s_decode:     ~38.2
tok_s_effective:  ~33.9
GPU KV cache size: ~3.06M tokens
1M concurrency:   ~3.06x
```

## Difference

| Metric | FP8 KV | NVFP4 KV | Δ |
|---|---|---|---|
| GPU KV pool | ~3.06M tok | ~10.59M tok | **~3.5× larger** |
| 1M-request concurrency | ~3.06× | ~10.59× | **~3.5×** |
| `tok_s_decode` | ~38.2 | 36.4 | −~5% |
| `tok_s_effective` | ~33.9 | 32.4 | −~4% |

**Takeaway:** moving the KV cache from 8-bit `fp8_e4m3` to 4-bit `nvfp4` (with the DiffKV attention backend) buys **~3.5× the concurrent 1M-context KV headroom** on the same hardware, at the cost of a few percent decode throughput. Pick NVFP4 KV when you need many concurrent long-context sessions / multi-agent headroom; pick FP8 KV for fastest raw single-stream decode.

> "10.59×" is full-1M-request KV capacity (how many 1M-token requests fit in the pool), not a single 10M-token request — single-request max stays `--max-model-len 1000000`.
