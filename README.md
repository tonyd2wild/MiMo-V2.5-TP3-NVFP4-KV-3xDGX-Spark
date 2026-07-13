# MiMo-V2.5 Omni on 3× DGX Spark — NVFP4 KV cache, 1M ctx, ~10.59× KV headroom

> `lukealonso/MiMo-V2.5-NVFP4` (Omni: text + image + video + audio) served tensor-parallel across three DGX Spark (GB10) boxes, with the KV cache in 4-bit `nvfp4` instead of 8-bit `fp8` — a `GPU KV cache size: 10,592,793 tokens` pool at a 1M single-request context.

> 🔀 This is the **3-Spark (TP=3)** build. Running **2 Sparks**? → [MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark](https://github.com/tonyd2wild/MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark)

The single-request context stays at **1,000,000 tokens**. What changes is the **total KV pool**: 4-bit NVFP4 KV gives roughly **3.5× the concurrent 1M-context headroom** of the fp8 path, on the *same* hardware.

> ⚠️ Experimental. `nvfp4` KV cache only works with the matching patched attention backend (DiffKV). Without the mod stack below, vLLM will reject NVFP4 KV or fail in the attention path.

---

## TL;DR

- **What you get:** the same MiMo-V2.5 Omni model and 3× Spark cluster, but the KV cache moved from `fp8_e4m3` to `nvfp4` (4-bit) — trading a few percent of decode throughput for ~**3.5× more 1M-context KV headroom**.
- **The numbers:** `~10.59M`-token KV pool (`10,592,793` tokens) vs `~3.06M` on fp8; `36.4` tok/s decode vs `~38.2`; quality `97.8` on the 69-scenario eval (66 pass / 3 partial / 0 fail).
- **Who it's for:** the high-context / many-concurrent-agent profile — many concurrent long-context sessions and multi-agent headroom, not the fastest raw single-stream decode.

### FP8 KV vs NVFP4 KV (same model, same 3× Spark cluster)

| | **FP8 KV** (public/frozen) | **NVFP4 KV** (this repo) |
|---|---|---|
| `--kv-cache-dtype` | `fp8_e4m3` | **`nvfp4`** |
| GPU KV cache pool | **~3.06M tokens** | **~10.59M tokens** |
| Concurrency @ 1M-token request | ~3.06× | **~10.59×** |
| `tok/s` decode (69-eval, thinking-off) | ~38.2 | ~36.4 |
| Effective `tok/s` | ~33.9 | ~32.4 |
| Quality (69-eval) | baseline | **97.8** (66 pass / 3 partial / 0 fail) |

---

## Hardware

- **Cluster:** 3× NVIDIA DGX Spark / GB10 (referred to below as `node-a`, `node-b`, `node-c` — substitute your own hostnames).
- **Parallelism:** tensor-parallel 3, pipeline-parallel 1.
- **Transport:** RoCE / multirail NCCL (four HCA address ranges).
- **KV cache dtype:** `nvfp4`.
- **Context:** `max_model_len = 1000000`.
- **Concurrency target:** `max_num_seqs = 6`.
- **Speculative decoding:** MTP, 2 tokens.
- **Observed:** `GPU KV cache size: 10,592,793 tokens` → ~10.59× full-1M request capacity.

---

## Quick start

The one line that unlocks the large KV pool:

```bash
--kv-cache-dtype nvfp4
```

…backed by `mods/nvfp4-kv-diffkv` + `VLLM_NVFP4_INLINE=1` + `VLLM_WMMA_DECODE=1` + `--attention-backend triton_attn_diffkv`.

Minimal bring-up on the DGX Spark cluster:

1. Apply the mod bundle inside the container on all 3 nodes: `./recipe/apply-mods.sh <container>`
2. Start Ray **workers first**, then start the Ray **head last**.
3. Launch vLLM from the head using [`recipes/mimo-v25-tp3-omni-1mctx-6seq-mtp2-nvfp4kv.yaml`](recipes/mimo-v25-tp3-omni-1mctx-6seq-mtp2-nvfp4kv.yaml).
4. Watch logs until serving; confirm `kv_cache_dtype=nvfp4` and that the `GPU KV cache size` line is large and symmetric across ranks.
5. Text smoke → image smoke → full eval.

The **frozen public path** instead uses `--kv-cache-dtype fp8_e4m3` (stable, faster decode, ~3.06M KV pool). This repo's NVFP4 path trades a few % throughput for ~3.5× the KV pool.

---

## Setup (detailed)

### Repo layout

```
.
├── README.md
├── LICENSE
├── recipe/
│   ├── apply-mods.sh                                       # applies the mods into a running container
│   └── mods/
│       └── nvfp4-kv-diffkv/                                # the NVFP4 KV cache patch (included)
│           ├── run.sh
│           ├── triton_attn_diffkv.py
│           ├── triton_unified_attention_diffkv.py
│           └── wmma_decode.py
├── recipes/
│   └── mimo-v25-tp3-omni-1mctx-6seq-mtp2-nvfp4kv.yaml      # the exact run config
└── benchmarks/
    └── nvfp4kv-vs-fp8kv-69eval-thinkingoff.md              # measured comparison
```

### Weights

- **HF model id:** `lukealonso/MiMo-V2.5-NVFP4` (Omni: text + image + video + audio).
- **Model path (HF hub cache):** `/root/.cache/huggingface/hub/models--lukealonso--MiMo-V2.5-NVFP4/snapshots/<SNAPSHOT_HASH>`

### Image & mods

- **Container:** `<your-mimo-omni-vllm-image>:<tag>` — built from the upstream mod stack (see Credits).
- The `nvfp4-kv-diffkv` mod (`triton_attn_diffkv.py`, `triton_unified_attention_diffkv.py`, `wmma_decode.py`, `run.sh`) **is included in this repo** at `recipe/mods/nvfp4-kv-diffkv/` — it is the exact cache patch that yields the ~10.59M-token KV pool (same code as the TP2 repo), applied by `recipe/apply-mods.sh`.
- The other mods below are **NOT vendored here** — they carry their own upstream licenses. Pull them from upstream (see Credits) and keep their original headers:

```
mods:
  - mods/drop-caches
  - mods/ray-keep-node-nccl-hca
  - mods/fix-prometheus-instrumentator-router
  - mods/fix-mimo-v2-vllm
  - mods/fix-modelopt-mixed-mxfp8
  - mods/nvfp4-kv-diffkv          # <-- the NVFP4 KV cache enablement
  - mods/mimo-v2-tp3-virtual-heads
```

### Launch

Environment (from the recipe):

```
TORCH_CUDA_ARCH_LIST: "12.1a"
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
VLLM_MIMO_V2_TP3_VIRTUAL_HEADS: "1"
VLLM_NVFP4_GEMM_BACKEND: flashinfer-cutlass
VLLM_USE_FLASHINFER_MOE_FP4: "1"
VLLM_FLASHINFER_MOE_BACKEND: throughput
VLLM_USE_RAY_V2_EXECUTOR_BACKEND: "0"
VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM: "0"
VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
VLLM_NVFP4_INLINE: "1"
VLLM_WMMA_DECODE: "1"
VLLM_WMMA_INSPECT: "0"
VLLM_WMMA_COMPARE: "0"
VLLM_WMMA_AUTHCOMPARE: "0"
NCCL_CUMEM_ENABLE: "0"
NCCL_NVLS_ENABLE: "0"
NCCL_NTHREADS: "64"
NCCL_NSOCKS_PERTHREAD: "2"
NCCL_BUFFSIZE: "8388608"
NCCL_CROSS_NIC: "1"
NCCL_IB_GID_INDEX: "3"
# Example internal RoCE ranges — replace with your own multirail HCA subnets.
NCCL_IB_ADDR_RANGE: 192.168.100.0/24,192.168.101.0/24,192.168.102.0/24,192.168.110.0/24,192.168.111.0/24,192.168.112.0/24
```

Serve command (from the recipe; `{...}` placeholders are the `defaults:` block — `port: 8000`, `host: 0.0.0.0`, `served_model_name: MiMo-V2.5-NVFP4`, `model_path` = the HF hub snapshot path above, `tensor_parallel: 3`, `pipeline_parallel: 1`, `gpu_memory_utilization: 0.89`, `max_model_len: 1000000`, `max_num_batched_tokens: 8192`, `max_num_seqs: 6`):

```bash
vllm serve {model_path} \
    --served-model-name {served_model_name} \
    --trust-remote-code \
    --tensor-parallel-size {tensor_parallel} \
    --pipeline-parallel-size {pipeline_parallel} \
    --distributed-executor-backend ray \
    --load-format instanttensor \
    --hf-overrides '{{"architectures":["MiMoV2OmniForCausalLM"]}}' \
    --limit-mm-per-prompt '{{"image":4,"video":1,"audio":1}}' \
    --mm-encoder-tp-mode data \
    --attention-backend triton_attn_diffkv \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --max-model-len {max_model_len} \
    --max-num-batched-tokens {max_num_batched_tokens} \
    --max-num-seqs {max_num_seqs} \
    --block-size 32 \
    --cudagraph-capture-sizes 1 2 4 8 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --speculative-config '{{"method":"mtp","num_speculative_tokens":2}}' \
    --no-async-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser mimo \
    --reasoning-parser mimo \
    --host {host} \
    --port {port}
```

### Verify — runtime tells (it's actually using NVFP4 KV)

```
kv_cache_dtype=nvfp4
[nvfp4-kv-diffkv]                 # mod installed
MiMoV2OmniForCausalLM             # Omni preserved
VLLM_MIMO_V2_TP3_VIRTUAL_HEADS=1  # TP=3 head fix active
GPU KV cache size: <large>        # the payoff line
```

Then run text smoke → image smoke → full eval.

---

## Benchmarks

Same model (`lukealonso/MiMo-V2.5-NVFP4`), same 3× DGX Spark cluster, same TP=3 / 1M-context / MTP2 / Omni config. The **only** material change is the KV-cache dtype (`fp8_e4m3` → `nvfp4`) and its required DiffKV mod + WMMA decode path. Measured with the same 69-scenario tool-eval harness (thinking-off), comparing `tok_s_decode` — not a single short-prompt wall-clock.

**NVFP4 KV (this recipe):**

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

**FP8 KV (frozen public reference):**

```
tok_s_decode:     ~38.2
tok_s_effective:  ~33.9
GPU KV cache size: ~3.06M tokens
1M concurrency:   ~3.06x
```

| Metric | FP8 KV | NVFP4 KV | Δ |
|---|---|---|---|
| GPU KV pool | ~3.06M tok | ~10.59M tok | **~3.5× larger** |
| 1M-request concurrency | ~3.06× | ~10.59× | **~3.5×** |
| `tok_s_decode` | ~38.2 | 36.4 | −~5% |
| `tok_s_effective` | ~33.9 | 32.4 | −~4% |

**Takeaway:** moving the KV cache from 8-bit `fp8_e4m3` to 4-bit `nvfp4` (with the DiffKV attention backend) buys **~3.5× the concurrent 1M-context KV headroom** on the same hardware, at the cost of a few percent decode throughput. Pick NVFP4 KV when you need many concurrent long-context sessions / multi-agent headroom; pick FP8 KV for fastest raw single-stream decode.

> "10.59×" is full-1M-request KV capacity (how many 1M-token requests fit in the pool), not a single 10M-token request — single-request max stays `--max-model-len 1000000`. Full comparison: [`benchmarks/nvfp4kv-vs-fp8kv-69eval-thinkingoff.md`](benchmarks/nvfp4kv-vs-fp8kv-69eval-thinkingoff.md).

---

## Configuration

Keep fixed for the known-good NVFP4-KV path:

- `gpu_memory_utilization: 0.89`
- `max_model_len: 1000000`
- `max_num_batched_tokens: 8192`
- `max_num_seqs: 6`
- `NCCL_NTHREADS: 64`
- `NCCL_BUFFSIZE: 8388608`

The frozen public path differs **only** by `--kv-cache-dtype fp8_e4m3` (no `nvfp4-kv-diffkv` mod, no `VLLM_NVFP4_INLINE` / `VLLM_WMMA_DECODE`). That path is faster but ~3.06M KV pool vs ~10.59M here.

---

## Troubleshooting

- **vLLM rejects NVFP4 KV or fails in the attention path:** `nvfp4` KV depends on the matching patched attention backend (DiffKV). Without the mod stack (`mods/nvfp4-kv-diffkv` + `--attention-backend triton_attn_diffkv` + `VLLM_NVFP4_INLINE=1` + `VLLM_WMMA_DECODE=1`), it will not work.
- **`GPU KV cache size` line is low or asymmetric by rank:** stop and debug before long-context testing — it should be large and symmetric across ranks.
- **Rejected `NCCL_NTHREADS=32` on first boot:** use `NCCL_NTHREADS: 64` (the NCCL64 variant fixes this).

---

## Credits & links

- **drowzeys ("Keys")** — origin of the NVFP4 KV-cache wiring this build's nvfp4 KV path descends from ([Keys---Full-GLM-5.2-Quantrio…](https://github.com/drowzeys/Keys---Full-GLM-5.2-Quantrio-INT4-INT8-mixed-8bit-Attention-on-4-x-DGX-Spark-GB10-Cluster)).
- [`lukealonso`](https://huggingface.co/lukealonso) — the `MiMo-V2.5-NVFP4` checkpoint, the chthonic vLLM work this builds around, and the MiniMax-M3 TP=3 virtual-head sharding idea we adapted for MiMo TP=3.
- [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) (esp. PR #251 / `a3refaat`) — DGX Spark MiMo V2.5 NVFP4 enablement + base mods: Triton DiffKV attention, MXFP8 dense dispatch, DiffKV quantized-KV guards, Prometheus route fix, per-node NCCL HCA preservation.
- [`HeNryous/mimo-spark-optimized`](https://github.com/HeNryous/mimo-spark-optimized) — the MiMo Spark optimization work that inspired the NVFP4 DiffKV cache + WMMA decode experiment ported into the TP=3 path.
- [`vLLM`](https://github.com/vllm-project/vllm), [`FlashInfer`](https://github.com/flashinfer-ai/flashinfer), Ray, and NCCL — the runtime stack.
- Eval methodology adapted from `tool-eval-bench v2.0.1` by `wolttam`.

This is a composition of community work plus our TP=3 / NVFP4-KV integration and validation. Please keep these credits with any copy, and check the license/redistribution terms of every upstream mod before redistributing its code.

**Related:** [MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark](https://github.com/tonyd2wild/MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark) (the 2-Spark build).

**License:** MIT (see [`LICENSE`](LICENSE)) — covers only the recipe documentation and configuration in this repository, not the upstream mods.

---

*Validated on 3× DGX Spark, 2026-06. This README uses generic node names and example RoCE ranges — substitute your own environment values.*
