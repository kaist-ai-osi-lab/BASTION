# BASTION: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting

Official code release for reproducing the experiments in **BASTION: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting**.

BASTION accelerates block-diffusion speculative decoding by building a query-dependent verification tree from the drafter's position-wise logits. The public release focuses on the paper's **Transformers backend** reproduction path: DFlash block-diffusion drafters, BASTION adaptive tree construction, hardware-calibrated verification latency modeling, and benchmark scripts.

## Scope

This repository is intended for experiment reproduction, not as a general serving framework. The current public release supports:

- Transformers backend only
- batch size 1 experiments
- Qwen3-4B, Qwen3-8B, and Llama-3.1-8B-Instruct target models
- DFlash draft models released on Hugging Face
- automatic per-(GPU, target model) calibration and profiling cache creation

SGLang, vLLM, and MLX backends are not part of this BASTION release.

Attention backend policy:

- If FlashAttention is installed, AR baseline and DFlash use `flash_attention_2`.
- BASTION verification always uses `sdpa`, even when FlashAttention is installed, because FlashAttention does not support the custom tree attention mask used by BASTION verification.
- If FlashAttention is not installed, all three modes use `sdpa`.

## Installation

Create a fresh `uv` environment and install the Transformers reproduction dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[transformers]"
```

For faster AR and DFlash baselines, install FlashAttention separately if it is compatible with your system:

```bash
uv pip install flash-attn --no-build-isolation
```

BASTION's target verification path still uses SDPA after FlashAttention is installed. This is intentional: FlashAttention can trigger CUDA errors with BASTION's custom tree mask.

## Supported Models

| Target model | DFlash draft model | Notes |
|---|---|---|
| `Qwen/Qwen3-4B` | `z-lab/Qwen3-4B-DFlash-b16` | use `enable_thinking=False` |
| `Qwen/Qwen3-8B` | `z-lab/Qwen3-8B-DFlash-b16` | use `enable_thinking=False` |
| `meta-llama/Llama-3.1-8B-Instruct` | `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` | may require Hugging Face access |

Qwen3 DFlash draft models were trained for non-thinking generation. Do not pass `--enable-thinking` for Qwen3-4B or Qwen3-8B.

## Quick Start

Run a small reproduction slice on GSM8K:

```bash
python -m bastion.benchmark \
  --model Qwen/Qwen3-8B \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 128 \
  --max-new-tokens 2048 \
  --temperature 0.0
```

The script compares three modes in one run:

- AR baseline (`block_size=1`)
- DFlash single-path speculative decoding
- BASTION adaptive tree-drafting

The first run for each `(GPU, target model)` pair creates calibration/profile files under `cache/`:

- `calib_probe_<gpu>_<model>.sigma_sequence_v1.jsonl`
- `calibration_<gpu>_<model>.json`
- `calibration_<gpu>_<model>.profile.json`

These caches are reused by later runs. Delete the corresponding files if you want to recalibrate after changing hardware, CUDA kernels, model dtype, or attention backend.

## Reproducing Paper Benchmarks

The benchmark loader supports the short-context datasets used in the paper:

- `gsm8k`
- `math500`
- `aime25` or `aime2025`
- `humaneval` or `human_eval`
- `mbpp`
- `lcb` or `livecodebench`
- `mt-bench`
- `alpaca`

It also supports the LongBench English subsets:

- `longbench-qasper` or `qasper`
- `longbench-multifieldqa_en` or `multifieldqa_en`
- `longbench-gov_report` or `gov_report`
- `longbench-multi_news` or `multi_news`
- `longbench-triviaqa` or `triviaqa`
- `longbench-samsum` or `samsum`
- `longbench-passage_retrieval_en` or `passage_retrieval_en`

For convenience, `paper-short` concatenates the eight short-context benchmarks, and `longbench` concatenates the seven LongBench subsets. For per-dataset reporting, run each dataset separately:

```bash
for dataset in gsm8k math500 aime25 humaneval mbpp lcb mt-bench alpaca; do
  python -m bastion.benchmark \
    --model Qwen/Qwen3-8B \
    --draft-model z-lab/Qwen3-8B-DFlash-b16 \
    --dataset "$dataset" \
    --max-new-tokens 2048 \
    --temperature 0.0
done
```

LongBench reproduction:

```bash
for dataset in qasper multifieldqa_en gov_report multi_news triviaqa samsum passage_retrieval_en; do
  python -m bastion.benchmark \
    --model Qwen/Qwen3-8B \
    --draft-model z-lab/Qwen3-8B-DFlash-b16 \
    --dataset "$dataset" \
    --max-new-tokens 2048 \
    --temperature 0.0
done
```

For stochastic decoding experiments, set `--temperature 1.0`.

## Multi-GPU Runs

The paper reports single-GPU, batch-size-1 measurements. You can use `torchrun` to shard prompts across multiple GPUs for faster benchmark collection, but each rank still runs batch size 1 on its assigned GPU:

```bash
torchrun --nproc_per_node=8 -m bastion.benchmark \
  --model Qwen/Qwen3-8B \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --dataset gsm8k \
  --max-new-tokens 2048 \
  --temperature 0.0
```

## Implementation Map

- `bastion/tree_draft.py`: adaptive best-first tree construction and BASTION generation
- `bastion/cost_model.py`: calibrated roofline verification-latency model
- `bastion/benchmark.py`: AR vs DFlash vs BASTION reproduction harness
- `dflash/model.py`: DFlash Transformers draft model implementation used by BASTION

## Notes

- Datasets are downloaded through Hugging Face `datasets` and cached as JSONL files in `cache/`.
- Calibration is part of the official reproduction flow. BASTION does not use an uncalibrated fallback in this release.
- Llama-3.1 may require accepting the model license and logging in with `huggingface-cli login`.
- When FlashAttention is installed, the benchmark loads a separate SDPA target model for BASTION verification, so peak memory use is higher than an SDPA-only run.
- Throughput numbers can vary with GPU SKU, CUDA version, attention implementation, and driver/runtime state.

## Citation

```bibtex
@article{oh2026bastion,
  title   = {{BASTION: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting}},
  author  = {Oh, Soowon and Cao, Nam and Kim, Yujin and Jung, Hojung and Ahmad, Huzama and Bae, Sangmin and Yun, Se-Young},
  year    = {2026}
}
```
