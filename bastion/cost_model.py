"""Latency cost model and calibration helpers for BASTION.

The benchmark collects target-model verification timings, fits the
roofline calibration parameters, and exposes the resulting cache through
`CALIBRATION_JSON_PATH` before adaptive tree construction.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


MODEL_SPECS = {
    "qwen3-4b": {
        "display_name": "Qwen3-4B",
        "hf_model_name": "Qwen/Qwen3-4B",
        "args": {
            "L": 36,
            "h": 2560,
            "n_q": 32,
            "n_kv": 8,
            "d": 128,
            "h_ffn": 9728,
            "V": 151936,
        },
    },
    "qwen3-8b": {
        "display_name": "Qwen3-8B",
        "hf_model_name": "Qwen/Qwen3-8B",
        "args": {
            "L": 36,
            "h": 4096,
            "n_q": 32,
            "n_kv": 8,
            "d": 128,
            "h_ffn": 12288,
            "V": 151936,
        },
    },
    "llama3.1-8b-instruct": {
        "display_name": "Llama3.1-8B-Instruct",
        "hf_model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "args": {
            "L": 32,
            "h": 4096,
            "n_q": 32,
            "n_kv": 8,
            "d": 128,
            "h_ffn": 14336,
            "V": 128256,
        },
    },
}


MODEL_ALIASES = {
    "qwen34b": "qwen3-4b",
    "qwenqwen34b": "qwen3-4b",
    "qwen38b": "qwen3-8b",
    "qwenqwen38b": "qwen3-8b",
    "llama318binstruct": "llama3.1-8b-instruct",
    "llama31instruct8b": "llama3.1-8b-instruct",
    "metallamallama318binstruct": "llama3.1-8b-instruct",
}


GPU_ALIASES = {
    "a5000": "a5000",
    "rtxa5000": "a5000",
    "nvidiartxa5000": "a5000",
    "a6000": "a6000",
    "rtxa6000": "a6000",
    "nvidiartxa6000": "a6000",
    "a100": "a100",
    "nvidiaa100": "a100",
    "a10080gb": "a100",
    "nvidiaa10080gb": "a100",
    "a10080gbpcie": "a100",
    "nvidiaa10080gbpcie": "a100",
    "h100": "h100",
    "h10080gb": "h100",
    "h10080gbsxm": "h100",
    "nvidiah100": "h100",
    "nvidiah10080gb": "h100",
    "nvidiah10080gbsxm": "h100",
    "nvidiah10080gbhbm3": "h100",
    "h10080gbhbm3": "h100",
    "h100hbm3": "h100",
    "h200": "h200",
    "h200141gb": "h200",
    "h200141gbsxm": "h200",
    "nvidiah200": "h200",
    "nvidiah200141gb": "h200",
    "nvidiah200141gbsxm": "h200",
    "nvidiah200141gbhbm3e": "h200",
    "h200141gbhbm3e": "h200",
    "h200hbm3e": "h200",
    "b6000": "b6000",
    "rtxpro6000": "b6000",
    "nvidiartxpro6000": "b6000",
    "nvidiartxpro6000blackwell": "b6000",
    "nvidiartxpro6000blackwellserveredition": "b6000",
}


GPU_SPECS = {
    "a5000": {
        "display_name": "NVIDIA RTX A5000",
        "peak_throughput_bf16": 111 * 1e12,
        "memory_bandwidth": 768 * 1e9,
    },
    "a6000": {
        "display_name": "NVIDIA RTX A6000",
        "peak_throughput_bf16": 155 * 1e12,
        "memory_bandwidth": 768 * 1e9,
    },
    "a100": {
        "display_name": "NVIDIA A100 80GB PCIe",
        "peak_throughput_bf16": 312 * 1e12,
        "memory_bandwidth": 1935 * 1e9,
    },
    "h100": {
        "display_name": "NVIDIA H100 80GB SXM",
        "peak_throughput_bf16": 989 * 1e12,
        "memory_bandwidth": 3350 * 1e9,
    },
    "h200": {
        "display_name": "NVIDIA H200 141GB SXM",
        "peak_throughput_bf16": 989 * 1e12,
        "memory_bandwidth": 4800 * 1e9,
    },
    "b6000": {
        "display_name": "NVIDIA RTX PRO 6000 Blackwell",
        "peak_throughput_bf16": 504 * 1e12,
        "memory_bandwidth": 1792 * 1e9,
    },
}


def resolve_model_spec(model_name: str) -> tuple[str, dict]:
    normalized = _normalize_name(model_name)
    model_key = MODEL_ALIASES.get(normalized)
    if model_key is None and model_name in MODEL_SPECS:
        model_key = model_name
    if model_key is None:
        supported = sorted(MODEL_SPECS.keys())
        raise ValueError(f"Unsupported model_name: '{model_name}'. Supported values: {supported}")
    return model_key, MODEL_SPECS[model_key]


def resolve_gpu_spec(gpu_type: str) -> tuple[str, dict]:
    normalized = _normalize_name(gpu_type)
    gpu_key = GPU_ALIASES.get(normalized)
    if gpu_key is None and gpu_type in GPU_SPECS:
        gpu_key = gpu_type
    if gpu_key is None:
        supported = sorted(GPU_SPECS.keys())
        raise ValueError(f"Unsupported gpu_type: '{gpu_type}'. Supported values: {supported}")
    return gpu_key, GPU_SPECS[gpu_key]


def resolve_calibration_params(gpu_key: str, model_key: str) -> dict:
    """Resolve fitted calibration params for (gpu, model).

    CALIBRATION_JSON_PATH must point to the cache produced by the benchmark
    calibration stage. BASTION's public reproduction path does not use an
    uncalibrated fallback.
    """
    json_path = os.environ.get("CALIBRATION_JSON_PATH", "")
    if not json_path:
        raise RuntimeError(
            f"No calibration source for ({gpu_key}, {model_key}). Run the "
            "benchmark calibration stage first or set CALIBRATION_JSON_PATH "
            "to its fit JSON."
        )
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"CALIBRATION_JSON_PATH={json_path} does not exist.")
    with open(json_path) as f:
        data = json.load(f)
    params = data.get("fit_parameters", {})
    if not params:
        raise ValueError(f"CALIBRATION_JSON_PATH={json_path} has no 'fit_parameters' key")
    return dict(params)


@dataclass(frozen=True)
class VerifiedLatencyEstimator:
    model_name: str
    gpu_type: str
    context_tokens: int
    compute_quadratic: float
    compute_linear: float
    compute_constant: float
    memory_quadratic: float
    memory_linear: float
    memory_constant: float
    alpha_m: float
    beta_m: float
    alpha_c: float
    beta_c: float

    def _raw_compute_latency_seconds(self, sequence_tokens: int) -> float:
        s = float(sequence_tokens)
        return self.compute_quadratic * s * s + self.compute_linear * s + self.compute_constant

    def _raw_memory_latency_seconds(self, sequence_tokens: int) -> float:
        s = float(sequence_tokens)
        return self.memory_quadratic * s * s + self.memory_linear * s + self.memory_constant

    def raw_branches(self, sequence_tokens: int) -> tuple[float, float]:
        return (
            self._raw_compute_latency_seconds(sequence_tokens),
            self._raw_memory_latency_seconds(sequence_tokens),
        )

    def calibrated_branches(self, sequence_tokens: int) -> tuple[float, float]:
        raw_compute, raw_memory = self.raw_branches(sequence_tokens)
        return (
            self.alpha_c * raw_compute + self.beta_c,
            self.alpha_m * raw_memory + self.beta_m,
        )

    def estimate(self, sequence_tokens: int) -> float:
        compute_latency_seconds, memory_latency_seconds = self.calibrated_branches(sequence_tokens)
        return max(compute_latency_seconds, memory_latency_seconds)

    def next_delta(self, sequence_tokens: int) -> float:
        return self.estimate(sequence_tokens + 1) - self.estimate(sequence_tokens)


def _fit_quadratic_from_points(y0: float, y1: float, y2: float) -> tuple[float, float, float]:
    constant = float(y0)
    quadratic = float((y2 - 2.0 * y1 + y0) / 2.0)
    linear = float(y1 - y0 - quadratic)
    return quadratic, linear, constant


def calculate_qwen_flops(s, c, L, h, n_q, n_kv, d, h_ffn, V):
    h_q = n_q * d
    h_kv = n_kv * d

    q_proj = 2 * s * h * h_q
    kv_proj = 4 * s * h * h_kv
    qk_matmul = 2 * s * (c + s) * h_q
    sv_matmul = 2 * s * (c + s) * h_q
    o_proj = 2 * s * h_q * h

    attn_flops_per_layer = q_proj + kv_proj + qk_matmul + sv_matmul + o_proj

    gate_up_proj = 4 * s * h * h_ffn
    down_proj = 2 * s * h_ffn * h

    ffn_flops_per_layer = gate_up_proj + down_proj

    total_layer_flops = L * (attn_flops_per_layer + ffn_flops_per_layer)

    output_header_flops = 2 * s * h * V

    total_flops = total_layer_flops + output_header_flops
    return total_flops


def calculate_qwen_inference_memory_footprint(
    s,
    c,
    L,
    h,
    n_q,
    n_kv,
    d,
    h_ffn,
    V,
    bp=2,
    return_breakdown=False,
):
    h_q = n_q * d
    h_kv = n_kv * d

    embed_param = V * h
    attn_param_per_layer = (2 * h * h_q) + (2 * h * h_kv)
    ffn_param_per_layer = 3 * h * h_ffn
    output_param = h * V

    footprint_param = bp * (embed_param + L * (attn_param_per_layer + ffn_param_per_layer) + output_param)

    kv_read_per_layer = 2 * c * h_kv
    kv_write_per_layer = 2 * s * h_kv
    footprint_kv = bp * L * (kv_read_per_layer + kv_write_per_layer)

    qkv_proj_io = (s * h) + (s * h_q + 2 * s * h_kv)
    attn_score_io = (s * h_q) + (n_q * s * (c + s))
    attn_value_io = (n_q * s * (c + s)) + (s * h_q)
    o_proj_io = (s * h_q) + (s * h)
    attn_act_per_layer = qkv_proj_io + attn_score_io + attn_value_io + o_proj_io

    gate_up_proj_io = (s * h) + (2 * s * h_ffn)
    down_proj_io = (2 * s * h_ffn) + (s * h)
    ffn_act_per_layer = gate_up_proj_io + down_proj_io

    output_header_io = (s * h) + (s * V)

    footprint_act = bp * (L * (attn_act_per_layer + ffn_act_per_layer) + output_header_io)

    total_footprint = footprint_param + footprint_kv + footprint_act
    if return_breakdown:
        return {
            "footprint_param": footprint_param,
            "footprint_kv": footprint_kv,
            "footprint_act": footprint_act,
            "total_footprint": total_footprint,
        }
    return total_footprint


def _calculate_model_flops(model_args, sequence_tokens: int, context_tokens: int) -> float:
    return calculate_qwen_flops(**model_args, s=sequence_tokens, c=context_tokens)


def _calculate_model_inference_memory_footprint(model_args, sequence_tokens: int, context_tokens: int) -> float:
    return calculate_qwen_inference_memory_footprint(**model_args, bp=2, s=sequence_tokens, c=context_tokens)


def build_fixed_interval_grid(start: int, stop: int, interval: int) -> list[int]:
    if start <= 0 or stop < start or interval <= 0:
        raise ValueError("Invalid grid bounds or interval")

    values = list(range(start, stop + 1, interval))
    if values[-1] != stop:
        values.append(stop)
    return values


def build_power_of_two_grid(start: int = 16, stop: int = 2048) -> list[int]:
    if start <= 0 or stop < start:
        raise ValueError("Invalid grid bounds")
    values: list[int] = []
    current = start
    while True:
        values.append(current)
        if current >= stop:
            break
        current *= 2
    return values


def load_jsonl_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def aggregate_latency_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for record in records:
        if "s" in record:
            sequence_size = int(record["s"])
        else:
            sequence_size = int(record["sequence_size"])

        if "c" in record:
            context_size = int(record["c"])
        else:
            context_size = int(record["context_size"])

        latency_value = float(record["verification_latency_s"])
        grouped.setdefault((sequence_size, context_size), []).append(latency_value)

    aggregated: list[dict] = []
    for (sequence_size, context_size), latencies in sorted(grouped.items()):
        latency_array = np.asarray(latencies, dtype=np.float64)
        aggregated.append(
            {
                "s": sequence_size,
                "c": context_size,
                "verification_latency_s": float(latency_array.mean()),
                "verification_latency_std_s": float(latency_array.std()),
                "num_measurements": int(latency_array.size),
            }
        )
    return aggregated


def compute_cost_model_latencies(
    *,
    model_name: str,
    gpu_type: str,
    s: int,
    c: int,
) -> tuple[float, float]:
    _, model_spec = resolve_model_spec(model_name)
    _, gpu_spec = resolve_gpu_spec(gpu_type)
    model_args = model_spec["args"]

    flops = calculate_qwen_flops(s=s, c=c, **model_args)
    memory_breakdown = calculate_qwen_inference_memory_footprint(
        s=s,
        c=c,
        bp=2,
        return_breakdown=True,
        **model_args,
    )

    lat_comp_s = float(flops / gpu_spec["peak_throughput_bf16"])
    lat_mem_s = float(memory_breakdown["total_footprint"] / gpu_spec["memory_bandwidth"])
    return lat_mem_s, lat_comp_s


def prepare_arrays_for_fitting(
    aggregated_records: list[dict],
    *,
    model_name: str,
    gpu_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    t_mem_ana_list: list[float] = []
    t_comp_ana_list: list[float] = []
    y_measured_list: list[float] = []
    sc_pairs_list: list[tuple[int, int]] = []

    for record in aggregated_records:
        sequence_size = int(record["s"])
        context_size = int(record["c"])
        measured_latency = float(record["verification_latency_s"])

        lat_mem_s, lat_comp_s = compute_cost_model_latencies(
            model_name=model_name,
            gpu_type=gpu_type,
            s=sequence_size,
            c=context_size,
        )

        t_mem_ana_list.append(lat_mem_s)
        t_comp_ana_list.append(lat_comp_s)
        y_measured_list.append(measured_latency)
        sc_pairs_list.append((sequence_size, context_size))

    return (
        np.asarray(t_mem_ana_list, dtype=np.float64),
        np.asarray(t_comp_ana_list, dtype=np.float64),
        np.asarray(y_measured_list, dtype=np.float64),
        sc_pairs_list,
    )


def _sigma_weights(s_values: np.ndarray, *, sigma_scale: float) -> np.ndarray:
    """Build curve_fit sigma weights from sequence sizes.

    sigma = 1 + sigma_scale * normalized_s, then normalized to mean 1.0.
    This mirrors spec-dllm/scripts/fit_linear_calibration.py.
    """
    s_arr = np.asarray(s_values, dtype=np.float64)
    if s_arr.size == 0:
        return np.asarray([], dtype=np.float64)
    s_min = float(np.min(s_arr))
    s_max = float(np.max(s_arr))
    if s_max <= s_min:
        sigma = np.ones_like(s_arr, dtype=np.float64)
    else:
        normalized = (s_arr - s_min) / (s_max - s_min)
        sigma = 1.0 + float(sigma_scale) * normalized
    sigma_mean = float(np.mean(sigma)) if sigma.size > 0 else 1.0
    if sigma_mean != 0.0:
        sigma = sigma / sigma_mean
    return sigma


def fit_roofline_calibration(
    t_mem_ana: np.ndarray,
    t_comp_ana: np.ndarray,
    y_measured: np.ndarray,
    s_values: np.ndarray | None = None,
    sigma_scale: float = 0.0,
) -> tuple[float, float, float, float, float]:
    from scipy.optimize import curve_fit  # pyright: ignore[reportMissingImports]

    def calibrated_roofline(indices, alpha_m, beta_m, alpha_c, beta_c):
        idx_int = np.round(indices).astype(int)
        t_m_cal = alpha_m * t_mem_ana[idx_int] + beta_m
        t_c_cal = alpha_c * t_comp_ana[idx_int] + beta_c
        return np.maximum(t_m_cal, t_c_cal)

    indices = np.arange(len(y_measured), dtype=np.float64)
    sigma = None
    if s_values is not None and float(sigma_scale) != 0.0:
        sigma = _sigma_weights(np.asarray(s_values, dtype=np.float64), sigma_scale=float(sigma_scale))

    initial_guess = [1.0, 0.0, 1.0, 0.0]
    param_bounds = ([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf])

    popt, _ = curve_fit(
        calibrated_roofline,
        indices,
        y_measured,
        p0=initial_guess,
        bounds=param_bounds,
        sigma=sigma,
        maxfev=5000,
    )

    alpha_m, beta_m, alpha_c, beta_c = popt
    y_pred = calibrated_roofline(indices, alpha_m, beta_m, alpha_c, beta_c)
    mse = float(np.mean((y_pred - y_measured) ** 2))
    return float(alpha_m), float(beta_m), float(alpha_c), float(beta_c), mse


def collect_calibration_data(
    *,
    target,
    tokenizer,
    device,
    model_name_or_path: str,
    gpu_type: str,
    output_path: str | Path,
    sequence_start: int = 16,
    sequence_stop: int = 512,
    sequence_interval: int = 16,
    context_start: int = 64,
    context_stop: int = 2048,
    context_interval: int | None = None,
    warmup_sequence_size: int = 16,
    warmup_context_size: int = 16,
    measurement_warmup_repeats: int = 3,
    repeats: int = 5,
    attn_implementation: str = "sdpa",
) -> Path:
    import torch
    from transformers import DynamicCache
    from tqdm import tqdm

    logger.info(
        "Collecting calibration data for model={} gpu={} -> {}",
        model_name_or_path,
        gpu_type,
        output_path,
    )

    def _dummy_token_id() -> int:
        for candidate in (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id):
            if candidate is not None:
                return int(candidate)
        return 0

    def _make_dummy_batch(token_id: int, length: int) -> torch.Tensor:
        return torch.full((1, length), token_id, dtype=torch.long, device=device)

    @torch.inference_mode()
    def _warmup_forward(token_id: int, *, context_size: int, sequence_size: int) -> None:
        cache = DynamicCache()
        if context_size > 0:
            context_ids = _make_dummy_batch(token_id, context_size)
            context_position_ids = torch.arange(context_size, device=device).unsqueeze(0)
            _ = target(
                context_ids,
                position_ids=context_position_ids,
                past_key_values=cache,
                use_cache=True,
            )

        verify_ids = _make_dummy_batch(token_id, sequence_size)
        verify_position_ids = torch.arange(context_size, context_size + sequence_size, device=device).unsqueeze(0)
        _ = target(
            verify_ids,
            position_ids=verify_position_ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
        if context_size > 0:
            cache.crop(context_size)

    @torch.inference_mode()
    def _measure_single_point(
        token_id: int,
        *,
        cache: DynamicCache,
        sequence_size: int,
        context_size: int,
        warmup_repeats: int,
        repeats: int,
    ) -> tuple[list[float], DynamicCache]:
        latencies: list[float] = []
        verify_ids = _make_dummy_batch(token_id, sequence_size)
        verify_position_ids = torch.arange(context_size, context_size + sequence_size, device=device).unsqueeze(0)

        for _ in range(warmup_repeats):
            _ = target(
                verify_ids,
                position_ids=verify_position_ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            cache.crop(context_size)

        def _cuda_time() -> float:
            torch.cuda.synchronize(device)
            return time.perf_counter()

        for _ in range(repeats):
            t0 = _cuda_time()
            _ = target(
                verify_ids,
                position_ids=verify_position_ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            latencies.append(_cuda_time() - t0)
            cache.crop(context_size)

        return latencies, cache

    if sequence_start <= 0 or context_start <= 0:
        raise ValueError("sequence_start and context_start must be positive")

    torch.cuda.set_device(device.index or 0)
    dummy_token_id = _dummy_token_id()
    warmup_sequence_size = max(int(warmup_sequence_size), 1)
    warmup_context_size = max(int(warmup_context_size), 0)
    _warmup_forward(
        dummy_token_id,
        context_size=warmup_context_size,
        sequence_size=warmup_sequence_size,
    )
    torch.cuda.synchronize(device)

    sequence_sizes = build_fixed_interval_grid(sequence_start, sequence_stop, sequence_interval)
    if context_interval is None:
        context_sizes = build_power_of_two_grid(context_start, context_stop)
    else:
        context_sizes = build_fixed_interval_grid(context_start, context_stop, context_interval)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for sequence_size in tqdm(sequence_sizes, desc="calibration sequences"):
        cache = DynamicCache()
        current_context_size = 0

        for context_size in tqdm(context_sizes, desc=f"s={sequence_size}", leave=False):
            if context_size < current_context_size:
                raise ValueError("Context sizes must be monotonically increasing")

            context_delta = context_size - current_context_size
            if context_delta > 0:
                context_ids = _make_dummy_batch(dummy_token_id, context_delta)
                context_position_ids = torch.arange(
                    current_context_size,
                    current_context_size + context_delta,
                    device=device,
                ).unsqueeze(0)
                _ = target(
                    context_ids,
                    position_ids=context_position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
                current_context_size = context_size

            latencies, cache = _measure_single_point(
                dummy_token_id,
                cache=cache,
                sequence_size=sequence_size,
                context_size=context_size,
                warmup_repeats=max(int(measurement_warmup_repeats), 0),
                repeats=max(int(repeats), 1),
            )

            latency_array = np.asarray(latencies, dtype=np.float64)
            records.append(
                {
                    "model_name_or_path": model_name_or_path,
                    "gpu_type": gpu_type,
                    "attn_impl": attn_implementation,
                    "s": int(sequence_size),
                    "c": int(context_size),
                    "verification_latency_s": float(latency_array.mean()),
                    "verification_latency_ms": float(latency_array.mean() * 1000.0),
                    "verification_latency_std_s": float(latency_array.std()),
                    "verification_latency_std_ms": float(latency_array.std() * 1000.0),
                    "num_repeats": int(max(int(repeats), 1)),
                    "warmup_sequence_size": int(warmup_sequence_size),
                    "warmup_context_size": int(warmup_context_size),
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Calibration data collection complete: {} records written to {}", len(records), output_path)

    return output_path


def build_verified_latency_estimator(*, context_tokens: int, model_name: str = "qwen3-4b", gpu_type: str = "a5000") -> VerifiedLatencyEstimator:
    model_key, model_spec = resolve_model_spec(model_name)
    gpu_key, gpu_spec = resolve_gpu_spec(gpu_type)

    model_args = model_spec["args"]
    peak_throughput_bf16 = gpu_spec["peak_throughput_bf16"]
    memory_bandwidth = gpu_spec["memory_bandwidth"]

    def _raw_compute_latency_seconds(sequence_tokens: int) -> float:
        flops = _calculate_model_flops(model_args, sequence_tokens, context_tokens)
        return flops / peak_throughput_bf16

    def _raw_memory_latency_seconds(sequence_tokens: int) -> float:
        memory_footprint = _calculate_model_inference_memory_footprint(model_args, sequence_tokens, context_tokens)
        return memory_footprint / memory_bandwidth

    compute_quad, compute_linear, compute_constant = _fit_quadratic_from_points(
        _raw_compute_latency_seconds(0),
        _raw_compute_latency_seconds(1),
        _raw_compute_latency_seconds(2),
    )
    memory_quad, memory_linear, memory_constant = _fit_quadratic_from_points(
        _raw_memory_latency_seconds(0),
        _raw_memory_latency_seconds(1),
        _raw_memory_latency_seconds(2),
    )

    calibration_params = resolve_calibration_params(gpu_key, model_key)
    alpha_m = float(calibration_params.get("alpha_m", 1.0))
    beta_m = float(calibration_params.get("beta_m", 0.0))
    alpha_c = float(calibration_params.get("alpha_c", 1.0))
    beta_c = float(calibration_params.get("beta_c", 0.0))

    return VerifiedLatencyEstimator(
        model_name=model_name,
        gpu_type=gpu_type,
        context_tokens=int(context_tokens),
        compute_quadratic=compute_quad,
        compute_linear=compute_linear,
        compute_constant=compute_constant,
        memory_quadratic=memory_quad,
        memory_linear=memory_linear,
        memory_constant=memory_constant,
        alpha_m=alpha_m,
        beta_m=beta_m,
        alpha_c=alpha_c,
        beta_c=beta_c,
    )


def estimate_verified_latency_seconds(*, sequence_tokens: int, context_tokens: int, model_name: str = "qwen3-4b", gpu_type: str = "a5000") -> float:
    return build_verified_latency_estimator(context_tokens=context_tokens, model_name=model_name, gpu_type=gpu_type).estimate(sequence_tokens)


def collect_calibration_data_from_jsonl(
    jsonl_path: str | Path,
    out_path: str | Path,
    max_samples: int | None = 512,
) -> Path:
    """Aggregate calibration measurement JSONL into averaged (s, c) points."""
    jsonl_path = Path(jsonl_path)
    out_path = Path(out_path)
    records = load_jsonl_records(jsonl_path)
    if max_samples is not None:
        records = records[:max_samples]
    aggregated = aggregate_latency_records(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in aggregated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def fit_linear_calibration(data_path: str | Path, *, sigma_scale: float = 30.0) -> dict[str, float]:
    """Fit linear roofline calibration parameters from collected measurement JSONL."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Calibration data not found: {data_path}")

    logger.info("Fitting calibration parameters from {}", data_path)
    records = load_jsonl_records(data_path)
    aggregated = aggregate_latency_records(records)
    if not aggregated:
        raise ValueError("No calibration records found after aggregation")

    first_record = records[0]
    model_name = first_record.get("model_name_or_path") or first_record.get("model_name") or "qwen3-4b"
    gpu_type = first_record.get("gpu_type", "a5000")

    t_mem_ana, t_comp_ana, y_measured, sc_pairs = prepare_arrays_for_fitting(
        aggregated,
        model_name=model_name,
        gpu_type=gpu_type,
    )
    s_values = np.asarray([int(s) for (s, _c) in sc_pairs], dtype=np.float64)
    alpha_m, beta_m, alpha_c, beta_c, _ = fit_roofline_calibration(
        t_mem_ana,
        t_comp_ana,
        y_measured,
        s_values=s_values,
        sigma_scale=sigma_scale,
    )
    logger.info(
        "Calibration fit complete for {}/{}: alpha_m={:.6f} beta_m={:.6f} alpha_c={:.6f} beta_c={:.6f}",
        model_name,
        gpu_type,
        alpha_m,
        beta_m,
        alpha_c,
        beta_c,
    )
    return {
        "alpha_m": alpha_m,
        "beta_m": beta_m,
        "alpha_c": alpha_c,
        "beta_c": beta_c,
    }


__all__ = [
    "resolve_model_spec",
    "resolve_gpu_spec",
    "resolve_calibration_params",
    "build_verified_latency_estimator",
    "estimate_verified_latency_seconds",
    "calculate_qwen_flops",
    "calculate_qwen_inference_memory_footprint",
    "build_fixed_interval_grid",
    "build_power_of_two_grid",
    "_sigma_weights",
    "load_jsonl_records",
    "aggregate_latency_records",
    "compute_cost_model_latencies",
    "prepare_arrays_for_fitting",
    "fit_roofline_calibration",
    "collect_calibration_data",
    "collect_calibration_data_from_jsonl",
    "fit_linear_calibration",
]
