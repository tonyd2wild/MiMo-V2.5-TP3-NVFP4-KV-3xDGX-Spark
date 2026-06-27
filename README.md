# MiMo-V2.5 Omni · TP=3 · **NVFP4 KV cache** on 3× DGX Spark

> 🔀 This is the **3-Spark (TP=3)** build. Running **2 Sparks**? → [MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark](https://github.com/tonyd2wild/MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark)

Running [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4) (Omni: text + image + video + audio) tensor-parallel across **three NVIDIA DGX Spark (GB10)** boxes — with the KV cache stored in **4-bit `nvfp4`** instead of 8-bit `fp8`.

The single-request context stays at **1,000,000 tokens**. What changes is the **total KV pool**: 4-bit NVFP4 KV roughly **3.5× the concurrent 1M-context headroom** of the fp8 path, on the *same* hardware.

---

## TL;DR — FP8 KV vs NVFP4 KV (same model, same 3× Spark cluster)

| | **FP8 KV** (public/frozen) | **NVFP4 KV** (this repo) |
|---|---|---|
| `--kv-cache-dtype` | `fp8_e4m3` | **`nvfp4`** |
| GPU KV cache pool | **~3.06M tokens** | **~10.59M tokens** |
| Concurrency @ 1M-token request | ~3.06× | **~10.59×** |
| `tok/s` decode (69-eval, thinking-off) | ~38.2 | ~36.4 |
| Effective `tok/s` | ~33.9 | ~32.4 |
| Quality (69-eval) | baseline | **97.8** (66 pass / 3 partial / 0 fail) |

**Net:** ~**3.5× more 1M-context KV headroom** for a few percent of decode throughput. This is the *high-context / many-concurrent-agent* profile, not the fastest raw-decode profile.

> ⚠️ Experimental. `nvfp4` KV cache only works with the matching patched attention backend (DiffKV). Without the mod stack below, vLLM will reject NVFP4 KV or fail in the attention path.

---

## Hardware & shape

- **Model:** `lukealonso/MiMo-V2.5-NVFP4` (Omni)
- **Cluster:** 3× DGX Spark / GB10 (referred to below as `node-a`, `node-b`, `node-c` — substitute your own hostnames)
- **Parallelism:** tensor-parallel 3, pipeline-parallel 1
- **Context:** `max_model_len = 1000000`
- **Concurrency target:** `max_num_seqs = 6`
- **KV cache dtype:** `nvfp4`
- **Speculative decoding:** MTP, 2 tokens
- **Transport:** RoCE / multirail NCCL (four HCA address ranges)
- **Observed:** `GPU KV cache size: 10,592,793 tokens` → ~10.59× full-1M request capacity

## The one line that does it

```bash
--kv-cache-dtype nvfp4
```

…backed by `mods/nvfp4-kv-diffkv` + `VLLM_NVFP4_INLINE=1` + `VLLM_WMMA_DECODE=1` + `--attention-backend triton_attn_diffkv`. See [`recipes/`](recipes/) for the full launch.

The **frozen public path** uses `--kv-cache-dtype fp8_e4m3` (stable, faster decode, ~3.06M KV pool). This repo's NVFP4 path trades a few % throughput for ~3.5× the KV pool.

## Files

```
.
├── README.md
├── recipes/
│   └── mimo-v25-tp3-omni-1mctx-6seq-mtp2-nvfp4kv.yaml   # the exact run config
└── benchmarks/
    └── nvfp4kv-vs-fp8kv-69eval-thinkingoff.md           # measured comparison
```

The patched **mods** (`nvfp4-kv-diffkv`, `mimo-v2-tp3-virtual-heads`, etc.) are NOT vendored here — they carry their own upstream licenses. See **Credits** for where they come from; pull them from upstream and keep their original headers.

## Launch order (DGX Spark cluster)

1. Apply the mod bundle inside the container on all 3 nodes.
2. Start Ray **workers first**.
3. Start the Ray **head last**.
4. Launch vLLM from the head.
5. Watch logs until serving; confirm `kv_cache_dtype=nvfp4` and the `GPU KV cache size` line is large and symmetric across ranks.
6. Text smoke → image smoke → full eval.

## Runtime tells (it's actually using NVFP4 KV)

```
kv_cache_dtype=nvfp4
[nvfp4-kv-diffkv]                 # mod installed
MiMoV2OmniForCausalLM             # Omni preserved
VLLM_MIMO_V2_TP3_VIRTUAL_HEADS=1  # TP=3 head fix active
GPU KV cache size: <large>        # the payoff line
```

If the `GPU KV cache size` line is low or asymmetric by rank, stop and debug before long-context testing.

## Credits

This is a composition of community work plus our TP=3 / NVFP4-KV integration and validation. Please keep these credits with any copy.

- [`lukealonso`](https://huggingface.co/lukealonso) — the `MiMo-V2.5-NVFP4` checkpoint, the chthonic vLLM work this builds around, and the MiniMax-M3 TP=3 virtual-head sharding idea we adapted for MiMo TP=3.
- [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) (esp. PR #251 / `a3refaat`) — DGX Spark MiMo V2.5 NVFP4 enablement + base mods: Triton DiffKV attention, MXFP8 dense dispatch, DiffKV quantized-KV guards, Prometheus route fix, per-node NCCL HCA preservation.
- [`HeNryous/mimo-spark-optimized`](https://github.com/HeNryous/mimo-spark-optimized) — the MiMo Spark optimization work that inspired the NVFP4 DiffKV cache + WMMA decode experiment ported into the TP=3 path.
- [`vLLM`](https://github.com/vllm-project/vllm), [`FlashInfer`](https://github.com/flashinfer-ai/flashinfer), Ray, and NCCL — the runtime stack.
- Eval methodology adapted from `tool-eval-bench v2.0.1` by `wolttam`.

## Notes / disclaimers

- **Experimental.** NVFP4 KV depends on the matching patched attention backend; treat it as such.
- This README uses generic node names and example RoCE ranges — substitute your own environment values.
- Check the license/redistribution terms of every upstream mod before redistributing its code.

*Validated on 3× DGX Spark, 2026-06. "10.59×" = full-1M-request KV capacity, not a single 10M-token request.*

### NVFP4-KV patch (included)

The `recipe/mods/nvfp4-kv-diffkv/` mod (`triton_attn_diffkv.py`, `triton_unified_attention_diffkv.py`, `wmma_decode.py`, `run.sh`) is the exact cache patch that yields the ~10.59M-token KV pool; same code as the TP2 repo. It is applied by `recipe/apply-mods.sh`.
