# Upstream components

- Deployment recipe: MIT, see [LICENSE](LICENSE).
- [vLLM](https://github.com/vllm-project/vllm): inference engine, Apache-2.0.
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B): default model; its model card and weight license remain applicable.

The image and model weights are not redistributed in the source repository. The recipe pins the engine image digest. Deployments needing byte-identical model reproduction should prepopulate a cache from an approved model revision and run it offline.
