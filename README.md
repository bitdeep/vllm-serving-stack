# vLLM serving stack: self-hosted LLM inference with FP8

Run a local OpenAI-compatible language-model endpoint with a pinned vLLM image, an explicit GPU memory budget and persistent model storage. The default Qwen3-8B recipe uses FP8 weights and KV cache, a 24,576-token context and at most eight active sequences.

[![Release](https://img.shields.io/github/v/release/bitdeep/vllm-serving-stack)](https://github.com/bitdeep/vllm-serving-stack/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Start here:** [Run an LLM](#run) · [Tune the configuration](#configuration-and-embedding) · [Work with me](#work-with-me)

## What this adds

- A Docker Compose starting point for **Qwen3 model serving**, with curl-based chat inference.
- Explicit FP8 quantization, KV cache, context length and sequence limits.
- An image pinned by digest and a reusable service that private deployments can extend.
- A lifecycle boundary that lets an external worker coordinate vLLM with ASR and TTS.

The recipe makes its defaults visible so you can reason about GPU memory and change them deliberately.

## Run

Requires Docker Compose, NVIDIA Container Toolkit, an FP8-capable GPU and sufficient free GPU memory. The reference configuration targets a 24 GB Ada GPU; it is not a promise that any combination of co-resident models will fit.

```sh
docker compose up -d
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-llm","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

The first start downloads weights into the `models` volume. Startup and model loading may take minutes. The host port binds to loopback; authentication and exposure belong to your private gateway.

## Configuration and embedding

Extend service `vllm` from `engine.compose.yaml` in your own deployment. Override its container environment:

| Variable | Default |
|---|---|
| `MODEL_ID` | `Qwen/Qwen3-8B` |
| `SERVED_MODEL_NAME` | `local-llm` |
| `MAX_MODEL_LEN` | `24576` |
| `GPU_MEMORY_UTILIZATION` | `0.50` |
| `QUANTIZATION` / `KV_CACHE_DTYPE` | `fp8` / `fp8` |
| `MAX_NUM_SEQS` | `8` |
| `SWAP_SPACE_GIB` | `1` |

The first-principles walkthrough of this budget — and the bandwidth ceiling your benchmarks should chase — is chapter 01 of the [garage-inference book](https://github.com/bitdeep/garage-inference/blob/main/01-llm-serving/README.md).

For the analytical memory budget and decode bandwidth ceiling behind these defaults, see [docs/capacity-qwen3-fp8.md](docs/capacity-qwen3-fp8.md).

Private deployments supply existing cache mounts, network, model alias and hardware-specific driver paths. No host names, private registries, client prompts or credentials are required by this recipe. `restart: "no"` allows an external lifecycle manager to own demand loading; without one, explicitly start and stop the service.

GPU memory utilization is a fraction of the full device, not of memory currently free. Measure startup peak, prefill, concurrent decoding and other workloads before increasing it. Shared IPC is enabled for this vLLM recipe; deploy within one trusted inference boundary.

## Validation and provenance

`docker compose config -q` checks the recipe. A functional smoke must call `/health` and generate a response; memory/throughput claims require a separate measured workload.

vLLM v0.16.0 is pinned by image digest in `engine.compose.yaml`. This project contributes the deployment recipe, not the inference engine. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Combine with speech inference

The [inference SDK](https://github.com/bitdeep/gpu-worker-orchestrator) provides managed loading, chat requests and a shared GPU lock. Companion [Whisper ASR](https://github.com/bitdeep/whisper-asr-stack) and [Chatterbox PT-BR](https://github.com/bitdeep/chatterbox-ptbr-server) stacks cover speech recognition and voice cloning.

## Work with me

I build self-hosted model serving and inference integrations: memory budgeting, API boundaries, model lifecycle and deployment validation. For consulting or engineering opportunities, [contact bitdeep on X](https://x.com/_wrbr).

For problems with this recipe, [open an issue](https://github.com/bitdeep/vllm-serving-stack/issues) with the GPU family, image version and a minimal synthetic request. Keep private deployment details and prompts out of public issues.
