"""Benchmark entrypoint for reproducing the BASTION experiments.

The script runs three decoding modes on the same prompts: autoregressive
baseline, DFlash, and BASTION tree drafting. The public release currently
supports the Transformers backend.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from loguru import logger
from rich import print
from tqdm import tqdm

logger.remove()
logger.add(sys.stderr, level=os.environ.get("BASTION_LOG_LEVEL", "WARNING"))

try:
    from . import cost_model, tree_draft
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bastion import cost_model, tree_draft

random.seed(42)

CACHE_DIR = Path(__file__).parent.parent / "cache"

DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "aime25": {
        "load_args": ("math-ai/aime25",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: (
            f"{x.get('problem') or x.get('question')}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(**x),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "lcb": {
        "load_args": ("livecodebench/code_generation_lite", "release_v2"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: (
            "Write a complete solution to the following programming problem.\n"
            "Return only the solution code.\n\n"
            f"{x.get('question_content') or x.get('question') or x.get('prompt')}"
        ),
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
    "alpaca": {
        "load_args": ("tatsu-lab/alpaca",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: (
            x["instruction"]
            if not x.get("input")
            else f"{x['instruction']}\n\n{x['input']}"
        ),
    },
}

LONGBENCH_SUBSETS = {
    "qasper",
    "multifieldqa_en",
    "gov_report",
    "multi_news",
    "triviaqa",
    "samsum",
    "passage_retrieval_en",
}

for _longbench_subset in LONGBENCH_SUBSETS:
    _longbench_config = {
        "load_args": ("THUDM/LongBench", _longbench_subset),
        "load_kwargs": {"split": "test", "trust_remote_code": True},
        "format": lambda x: f"{x['context']}\n\n{x['input']}",
    }
    DATASETS[f"longbench-{_longbench_subset}"] = _longbench_config
    DATASETS[_longbench_subset] = _longbench_config

DATASETS["aime2025"] = DATASETS["aime25"]
DATASETS["human_eval"] = DATASETS["humaneval"]
DATASETS["livecodebench"] = DATASETS["lcb"]

DATASET_GROUPS = {
    "paper-short": [
        "gsm8k",
        "math500",
        "aime25",
        "humaneval",
        "mbpp",
        "lcb",
        "mt-bench",
        "alpaca",
    ],
    "longbench": [f"longbench-{name}" for name in sorted(LONGBENCH_SUBSETS)],
}


def _prepare_dataset(name: str) -> Path:
    from datasets import load_dataset

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS.keys())}")

    cfg = DATASETS[name]
    CACHE_DIR.mkdir(exist_ok=True)
    out_path = CACHE_DIR / f"{name}.jsonl"
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")

    print(f"[download] {name} ...")
    dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])

    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in dataset:
            if cfg.get("multi_turn"):
                turns = cfg["format"](row)
            else:
                turns = [cfg["format"](row)]
            f.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)

    with open(out_path, encoding="utf-8") as f:
        num_samples = sum(1 for _ in f)
    print(f"[cached] {out_path} ({num_samples} samples)")
    return out_path


def load_and_process_dataset(data_name: str) -> list[dict]:
    if data_name in DATASET_GROUPS:
        return list(chain.from_iterable(load_and_process_dataset(name) for name in DATASET_GROUPS[data_name]))

    if data_name not in DATASETS:
        available = sorted(list(DATASETS.keys()) + list(DATASET_GROUPS.keys()))
        raise ValueError(f"Unknown dataset '{data_name}'. Available: {available}")

    path = CACHE_DIR / f"{data_name}.jsonl"
    if not path.exists():
        _prepare_dataset(data_name)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _limit_dataset(dataset: list[dict], max_samples: int | None) -> list[dict]:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    random.shuffle(dataset)
    return dataset[:max_samples]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _dist_init(torch_dist) -> None:
    if _dist_size() <= 1:
        return
    missing = [name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ]
    if missing:
        raise RuntimeError(f"Distributed run is missing environment variables: {missing}")
    torch_dist.init_process_group(backend="nccl", init_method="env://")


def _dist_size() -> int:
    return _env_int("WORLD_SIZE", 1)


def _dist_rank() -> int:
    return _env_int("RANK", 0)


def _dist_local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _dist_gather(torch_dist, obj: Any, dst: int = 0):
    if not torch_dist.is_initialized():
        return [obj]
    if _dist_is_main():
        objs = [None for _ in range(_dist_size())]
        torch_dist.gather_object(obj, objs, dst=dst)
        return objs
    torch_dist.gather_object(obj, dst=dst)
    return None


def _dist_destroy(torch_dist) -> None:
    if torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.destroy_process_group()


_TRANSFORMERS_SUPPORTED_PATTERN = re.compile(r"qwen3(?!\.5)[\w-]*|llama.*3\.1.*8b.*instruct", re.IGNORECASE)


def _check_transformers_model(model_name: str) -> None:
    if not _TRANSFORMERS_SUPPORTED_PATTERN.search(model_name):
        raise ValueError(
            f"Transformers backend does not support '{model_name}'. "
            f"Only Qwen3 series and LLaMA-3.1-8B-Instruct are supported."
        )


def _get_baseline_attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401  # pyright: ignore[reportMissingImports]
        return "flash_attention_2"
    except ImportError:
        logger.warning(
            "flash_attn not installed. AR, DFlash, and BASTION will all use torch.sdpa. "
            "For faster AR/DFlash baselines, install: uv pip install flash-attn --no-build-isolation"
        )
        return "sdpa"


def _mean_acceptance_length(rows: list[dict[int, SimpleNamespace]], key: int) -> float | None:
    lengths = list(chain.from_iterable(getattr(row[key], "acceptance_lengths", []) for row in rows))
    if not lengths:
        return None
    return float(np.mean(lengths))


def _print_decode_summary(responses: list[dict[int, SimpleNamespace]], block_size: int) -> None:
    baseline_tpot = np.mean([r[1].time_per_output_token for r in responses])
    spec_tpot = np.mean([r[block_size].time_per_output_token for r in responses])
    tree_tpot = np.mean([r[block_size * 10].time_per_output_token for r in responses])
    spec_accept = _mean_acceptance_length(responses, block_size)
    tree_accept = _mean_acceptance_length(responses, block_size * 10)

    method_w = 23
    throughput_w = 16
    speedup_w = 9
    accept_w = 12
    table_w = method_w + throughput_w + speedup_w + accept_w + 3

    def tps_text(tpot: float) -> str:
        return f"{1 / tpot:.2f} tok/s"

    def accept_text(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    print()
    print(f"{'=' * table_w}")
    print(f"{'Method':<{method_w}} {'Throughput':>{throughput_w}} {'Speedup':>{speedup_w}} {'Avg accept':>{accept_w}}")
    print(f"{'-' * table_w}")
    print(f"{'AR baseline (bs=1)':<{method_w}} {tps_text(baseline_tpot):>{throughput_w}} {'1.00x':>{speedup_w}} {'-':>{accept_w}}")
    print(f"{f'DFlash (bs={block_size})':<{method_w}} {tps_text(spec_tpot):>{throughput_w}} {f'{baseline_tpot / spec_tpot:.2f}x':>{speedup_w}} {accept_text(spec_accept):>{accept_w}}")
    print(f"{'BASTION tree-draft':<{method_w}} {tps_text(tree_tpot):>{throughput_w}} {f'{baseline_tpot / tree_tpot:.2f}x':>{speedup_w}} {accept_text(tree_accept):>{accept_w}}")
    print(f"{'=' * table_w}")
    print()

def _apply_chat_template(tokenizer, messages: list[dict], enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def infer_cost_model_name_from_target(model_name_or_path: str) -> str:
    model_key, _ = cost_model.resolve_model_spec(model_name_or_path)
    return model_key


def infer_cost_gpu_type_from_device(device: Any) -> str:
    import torch as _torch

    if device.type != "cuda":
        raise ValueError("adaptive_best cost modeling requires a CUDA device.")
    if device.index is None:
        raise ValueError("CUDA device index is required to resolve GPU type.")

    gpu_name = _torch.cuda.get_device_name(device.index)
    gpu_key, _ = cost_model.resolve_gpu_spec(gpu_name)
    return gpu_key


def profile_tree_adaptive_constants(
    *,
    draft_model,
    target,
    input_ids: Any,
    block_size: int,
    temperature: float,
    model_name: str,
    gpu_type: str,
    num_trials: int = 3,
):
    """Profile draft latency and non-verify overhead (returns tuple).

    Uses `bastion_generate` in probe mode to collect latency samples and
    returns (mean_draft_latency_s, mean_non_verify_overhead_s).
    """
    if block_size <= 1:
        raise ValueError("adaptive_best profiling requires block_size > 1.")

    draft_samples = []
    overhead_samples = []
    for _ in range(max(1, num_trials)):
        resp = tree_draft.bastion_generate(
            draft_model=draft_model,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max(block_size * 2, 8),
            stop_token_ids=None,
            temperature=temperature,
            block_size=block_size,
            return_stats=True,
            cost_model_name=model_name,
            cost_gpu_type=gpu_type,
            draft_latency_s=0.001,
            non_verify_latency_s=1e-9,
            min_tree_size=1,
            max_tree_size=block_size,
        )
        if getattr(resp, "draft_latencies", None):
            draft_samples.extend(float(x) for x in resp.draft_latencies)

        # Non-verification overhead excludes draft and target verification forward passes.
        step_count = min(
            len(getattr(resp, "tree_build_latencies", [])),
            len(getattr(resp, "tree_select_latencies", [])),
            len(getattr(resp, "tree_kv_reorder_latencies", [])),
            len(getattr(resp, "tree_extract_latencies", [])),
        )
        for i in range(step_count):
            overhead_samples.append(
                float(resp.tree_build_latencies[i])
                + float(resp.tree_select_latencies[i])
                + float(resp.tree_kv_reorder_latencies[i])
                + float(resp.tree_extract_latencies[i])
            )

    if not draft_samples:
        raise RuntimeError("Failed to profile draft latency; no samples collected.")
    if not overhead_samples:
        raise RuntimeError("Failed to profile non-verify overhead; no samples collected.")

    return float(np.mean(draft_samples)), float(np.mean(overhead_samples))


def _run_transformers(args: argparse.Namespace) -> None:
    import torch
    from torch import distributed as torch_dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dflash.model import DFlashDraftModel, dflash_generate

    _check_transformers_model(args.model)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _dist_init(torch_dist)
    torch.cuda.set_device(_dist_local_rank())
    device = torch.device(f"cuda:{_dist_local_rank()}")
    baseline_attn_impl = _get_baseline_attn_impl()
    bastion_attn_impl = "sdpa"

    logger.info("Loading AR/DFlash target model: {} (attn_implementation={})", args.model, baseline_attn_impl)
    target = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation=baseline_attn_impl, dtype=torch.bfloat16
    ).to(device).eval()

    if baseline_attn_impl == bastion_attn_impl:
        bastion_target = target
    else:
        logger.info(
            "Loading BASTION verification target model: {} (attn_implementation={})",
            args.model,
            bastion_attn_impl,
        )
        bastion_target = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation=bastion_attn_impl, dtype=torch.bfloat16
        ).to(device).eval()

    logger.info("Loading draft model: {} (attn_implementation={})", args.draft_model, baseline_attn_impl)
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model, attn_implementation=baseline_attn_impl, dtype=torch.bfloat16
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_and_process_dataset(args.dataset)

    dataset = _limit_dataset(dataset, args.max_samples)

    model_key = infer_cost_model_name_from_target(args.model)
    gpu_key = infer_cost_gpu_type_from_device(device)
    CACHE_DIR.mkdir(exist_ok=True)
    calib_cache = CACHE_DIR / f"calibration_{gpu_key}_{model_key}.json"
    profile_cache = CACHE_DIR / f"calibration_{gpu_key}_{model_key}.profile.json"

    logger.info(
        "Calibration/profile setup for model={} gpu={} (calib_cache={}, profile_cache={})",
        model_key,
        gpu_key,
        calib_cache,
        profile_cache,
    )

    sigma_scale = float(os.environ.get("BASTION_SIGMA_SCALE", "30.0"))
    fit_version = "sigma_sequence_v1"
    probe_path = CACHE_DIR / f"calib_probe_{gpu_key}_{model_key}.{fit_version}.jsonl"

    def _probe_cache_attn_impl(path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        return json.loads(line).get("attn_impl")
        except Exception:
            return None
        return None

    def _calibration_cache_current(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        metadata_current = (
            isinstance(cached.get("fit_parameters"), dict)
            and cached.get("fit_version") == fit_version
            and float(cached.get("sigma_scale", -1.0)) == sigma_scale
        )
        if not metadata_current:
            return False

        cached_attn_impl = cached.get("attn_impl")
        if cached_attn_impl is None:
            cached_attn_impl = _probe_cache_attn_impl(probe_path)
            if cached_attn_impl == bastion_attn_impl and _dist_is_main():
                cached["attn_impl"] = cached_attn_impl
                path.write_text(json.dumps(cached), encoding="utf-8")
        return cached_attn_impl == bastion_attn_impl

    # Create/refit calibration cache if missing or produced by an older fitting path.
    if not _calibration_cache_current(calib_cache) and _dist_is_main():
        if probe_path.exists():
            logger.info(
                "Calibration cache missing/outdated. Fitting from existing probe data: {}",
                probe_path,
            )
        else:
            logger.info(
                "Calibration cache missing. Collecting calibration data for model={} gpu={} ...",
                model_key,
                gpu_key,
            )
            cost_model.collect_calibration_data(
                target=bastion_target,
                tokenizer=tokenizer,
                device=device,
                model_name_or_path=args.model,
                gpu_type=gpu_key,
                output_path=probe_path,
                attn_implementation=bastion_attn_impl,
            )
            logger.info("Calibration data collected at {}. Fitting roofline calibration ...", probe_path)

        params = cost_model.fit_linear_calibration(probe_path, sigma_scale=sigma_scale)
        calib_cache.write_text(
            json.dumps(
                {
                    "fit_parameters": params,
                    "fit_version": fit_version,
                    "sigma_scale": sigma_scale,
                }
            ),
            encoding="utf-8",
        )
        logger.info("Calibration fit complete. Saved parameters to {}", calib_cache)
    elif _dist_is_main():
        logger.info("Using existing calibration cache: {}", calib_cache)

    # Wait for calibration cache to be available across ranks. BASTION uses a
    # calibrated latency model; missing or stale calibration is a hard error.
    if _dist_size() > 1 and not _dist_is_main():
        logger.info("Waiting for calibration cache to become available: {}", calib_cache)
        while not calib_cache.exists():
            time.sleep(0.25)

    if not _calibration_cache_current(calib_cache):
        raise RuntimeError(f"Calibration cache was not created or is stale: {calib_cache}")

    os.environ["CALIBRATION_JSON_PATH"] = str(calib_cache)
    logger.info("Loaded calibration parameters from {}", calib_cache)

    # Profile draft + overhead constants (only once; main creates, others wait).
    if not profile_cache.exists() and _dist_is_main():
        logger.info(
            "Profiling tree-adaptive constants for model={} gpu={} (num_trials={})",
            model_key,
            gpu_key,
            3,
        )
        # Make a single probe input from the first dataset sample.
        probe_input = dataset[0]
        probe_messages = [{"role": "user", "content": probe_input["turns"][0]}]
        probe_text = tokenizer.apply_chat_template(
            probe_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        probe_input_ids = tokenizer.encode(probe_text, return_tensors="pt").to(device)
        dl_s, nv_s = profile_tree_adaptive_constants(
            draft_model=draft_model,
            target=bastion_target,
            input_ids=probe_input_ids,
            block_size=block_size,
            temperature=args.temperature,
            model_name=model_key,
            gpu_type=gpu_key,
            num_trials=3,
        )
        profile_cache.write_text(
            json.dumps(
                {
                    "draft_latency_s": dl_s,
                    "non_verify_latency_s": nv_s,
                    "attn_impl": bastion_attn_impl,
                }
            ),
            encoding="utf-8",
        )
        logger.info(
            "Profiling complete. draft_latency_s={:.6f} non_verify_latency_s={:.6f} saved to {}",
            dl_s,
            nv_s,
            profile_cache,
        )
    elif _dist_is_main():
        logger.info("Using existing profiling cache: {}", profile_cache)

    if _dist_size() > 1 and not _dist_is_main():
        logger.info("Waiting for profiling cache to become available: {}", profile_cache)
        while not profile_cache.exists():
            time.sleep(0.25)

    if not profile_cache.exists():
        raise RuntimeError(f"Profiling cache was not created: {profile_cache}")

    prof = json.loads(profile_cache.read_text(encoding="utf-8"))
    if "draft_latency_s" not in prof or "non_verify_latency_s" not in prof:
        raise ValueError(f"Invalid profile cache: {profile_cache}")
    if prof.get("attn_impl") is None:
        prof_attn_impl = _probe_cache_attn_impl(probe_path)
        if prof_attn_impl == bastion_attn_impl:
            prof["attn_impl"] = prof_attn_impl
            if _dist_is_main():
                profile_cache.write_text(json.dumps(prof), encoding="utf-8")
    if prof.get("attn_impl") != bastion_attn_impl:
        raise ValueError(
            f"Invalid profile cache attention backend in {profile_cache}: "
            f"expected {bastion_attn_impl}, got {prof.get('attn_impl')}"
        )
    draft_latency_s = float(prof["draft_latency_s"])
    non_verify_latency_s = float(prof["non_verify_latency_s"])
    if draft_latency_s <= 0 or non_verify_latency_s <= 0:
        raise ValueError(
            f"Invalid profiling constants in {profile_cache}: "
            f"draft_latency_s={draft_latency_s}, non_verify_latency_s={non_verify_latency_s}"
        )
    logger.info(
        "Loaded profiling constants from {} (draft_latency_s={:.6f}, non_verify_latency_s={:.6f})",
        profile_cache,
        draft_latency_s,
        non_verify_latency_s,
    )

    responses = []
    indices = range(_dist_rank(), len(dataset), _dist_size())
    for idx in tqdm(indices, disable=not _dist_is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = _apply_chat_template(tokenizer, messages, args.enable_thinking)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = dflash_generate(
                    draft_model,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    block_size=bs,
                    return_stats=True,
                )

            # Tree-drafting path using bastion_generate
            response[block_size * 10] = tree_draft.bastion_generate(
                draft_model,
                target=bastion_target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                block_size=block_size,
                return_stats=True,
                cost_model_name=model_key,
                cost_gpu_type=gpu_key,
                draft_latency_s=draft_latency_s,
                non_verify_latency_s=non_verify_latency_s,
            )

            # Extract output from AR baseline for messages
            ar_response = response[1]
            generated_ids = ar_response.output_ids[0, ar_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if _dist_size() > 1:
        responses = _dist_gather(torch_dist, responses, dst=0)
        if not _dist_is_main():
            _dist_destroy(torch_dist)
            return
        responses = list(chain(*responses))

    _print_decode_summary(responses, block_size)
    _dist_destroy(torch_dist)


def main() -> None:
    parser = argparse.ArgumentParser(description="BASTION benchmark: AR vs DFlash vs Tree-drafting")
    parser.add_argument("--backend", choices=["transformers"], default="transformers")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--enable-thinking", action="store_true")

    args = parser.parse_args()

    if args.enable_thinking and any(x in args.model.lower() for x in ["qwen3-4b", "qwen3-8b"]):
        raise ValueError(
            "DFlash draft models for Qwen3-4B and Qwen3-8B were not trained with thinking traces. "
            "Using --enable-thinking will lead to suboptimal performance."
        )

    if args.backend != "transformers":
        raise ValueError(f"Unsupported backend: {args.backend}")
    _run_transformers(args)


if __name__ == "__main__":
    main()
