# Capacity worksheet: Qwen3-8B FP8 on a 24 GB GPU

Analytical memory and bandwidth budget for the default recipe in
[engine.compose.yaml](../engine.compose.yaml). These are first-principles
numbers computed with the open-source
[ai-infra-book calculator](https://github.com/bojieli/ai-infra-book) — a
pure-Python, dependency-free tool that derives bytes and FLOPs from the
model's official configuration. They are logical budgets, not measurements:
use them to choose settings, then measure your own workload (see
[Validation](#reproduce)).

Provenance: calculator at `bojieli/ai-infra-book@8e0c478a`, model config
`Qwen/Qwen3-8B@b968826d`, device rows from the official NVIDIA Ada
whitepaper. Computation date: 2026-09-13.

## What the model costs

| Item | FP8 | BF16 |
|---|---:|---:|
| Weights (8.19 B parameters) | 8.19 GB | 16.38 GB |
| KV cache per token | 72 KiB | 144 KiB |
| KV per sequence at 24,576 ctx | 1.81 GB | 3.62 GB |
| KV, 8 sequences at 24,576 ctx | 14.50 GB | 28.99 GB |

FP8 weights **and** FP8 KV cache (`QUANTIZATION=fp8`, `KV_CACHE_DTYPE=fp8`)
are what make the default recipe fit a 24 GB card at all: in BF16, even
weights alone exceed half the card.

## Why the defaults are what they are

`GPU_MEMORY_UTILIZATION` is a fraction of the whole device. The KV pool
gets roughly `utilization × 24 GB − weights`:

| Utilization | Budget | Full-length sequences the pool can hold |
|---|---:|---:|
| 0.50 (default) | 12.0 GB | ~2 |
| 0.70 | 16.8 GB | ~4 |
| 0.90 | 21.6 GB | ~7 |
| 0.95 | 22.8 GB | ~8 |

Reading: `MAX_NUM_SEQS=8` is a scheduler cap, not a capacity promise. At the
full 24,576-token context, eight worst-case sequences need ~22.7 GB of
weights+KV — the utilization knob and the context length are the real
capacity levers. Real traffic rarely keeps every sequence at max context,
which is why the recipe works at 0.50; the point is to know which knob buys
back concurrency when it does not.

`--enforce-eager` trades some speed for a predictable memory footprint (no
CUDA-graph workspace), which keeps this budget honest.

## Decode is bandwidth-bound

One decode step, batch 8, every sequence at full context (weights read once,
KV of 8 × 24,576 tokens read, one token appended per sequence):

| Quantity | Value |
|---|---:|
| HBM traffic per step | 22.69 GB |
| Matrix FLOPs per step | 237 GFLOPs |
| Arithmetic intensity | 10.4 FLOPs/byte |
| Ridge point of a 24 GB Ada GPU (1.008 TB/s, 660.6 TFLOPS FP8) | 655 FLOPs/byte |
| Resource time lower bound | **22.5 ms/step** |
| Throughput ceiling | **~355 tok/s aggregate (~44 tok/s per sequence)** |

Arithmetic intensity is ~63× below the ridge point: decode is a bandwidth
problem, not a compute problem. A single stream at short context reads
mostly weights (8.19 GB / 1.008 TB/s ≈ 8.1 ms) — a ~123 tok/s ceiling per
stream. If your measured numbers are far below these ceilings, the gap is
workspace traffic, non-matrix work and scheduling overhead — that gap, not
the peak spec, is what tuning reduces.

Prefill is the opposite: an 8k-token prompt is ~134 TFLOPs — a few hundred
milliseconds of tensor-core work against ~15 ms of weight reads, so prefill
is compute-bound and batch-8 decode pays for it only by pausing generation.

## Reproduce

```sh
git clone https://github.com/bojieli/ai-infra-book.git
cd ai-infra-book && git checkout 8e0c478a2bd5edd37fc0552a524b29813ae775f1

# KV cache footprint (FP8: --element-bytes 1)
python3 calculations/calc.py state --model qwen3-8b --length 24576 \
  --batch 8 --element-bytes 1

# Per-operator decode accounting (batch 8, full history)
python3 calculations/calc.py forward --model qwen3-8b --batch 8 \
  --history 24576 --tokens 1

# Roofline: FLOPs/traffic from the forward output, official device rows
python3 calculations/calc.py roofline --device rtx4090 --precision FP8 \
  --accumulator FP16 --flops 237058392064 --traffic-bytes 22686839808
```

## Limitations

- Logical bytes, not HBM measurements: no activation workspace, allocator
  overhead, framework buffers or kernel-level re-reads are included.
- `1.0` efficiency in the roofline means official peak, not measured
  utilization.
- The calculator's own caveat applies: per-operator accounting shows what
  fusion could eliminate; it is not an engine measurement.
- Throughput claims for this stack require a measured workload; treat this
  worksheet as the ceiling those measurements chase.
