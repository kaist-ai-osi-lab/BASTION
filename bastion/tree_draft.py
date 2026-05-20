"""Adaptive tree-drafting implementation used by BASTION.

This module builds a cost-model-guided verification tree from DFlash
draft logits, verifies candidate paths with the target model, and keeps
the target KV cache aligned to the accepted path.
"""
from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from transformers import DynamicCache

from .cost_model import build_verified_latency_estimator

if TYPE_CHECKING:
    from dflash.model import DFlashDraftModel


_MIN_ADAPTIVE_BEST_TREE_NODES = 32
_MAX_ADAPTIVE_BEST_TREE_NODES = 8192
_MAX_ADAPTIVE_TOP_K = 128
_ADAPTIVE_BUCKET_MODE_DOUBLING = "doubling"
_ADAPTIVE_BUCKET_MODE_FIXED_INTERVAL = "fixed_interval"
_ADAPTIVE_BUCKET_MODES = {
    _ADAPTIVE_BUCKET_MODE_DOUBLING,
    _ADAPTIVE_BUCKET_MODE_FIXED_INTERVAL,
}


@dataclass
class TreeDraft:
    token_ids: torch.LongTensor
    parent_indices: torch.LongTensor
    depths: torch.LongTensor
    tree_mask: torch.BoolTensor
    retrieve_indices: torch.LongTensor
    path_lens: torch.LongTensor


def _next_adaptive_tree_return_size(
    *,
    current_tree_size: int,
    current_return_size: int,
    min_tree_size: int,
    bucket_mode: str,
) -> int:
    if bucket_mode == _ADAPTIVE_BUCKET_MODE_DOUBLING:
        step = current_return_size
    elif bucket_mode == _ADAPTIVE_BUCKET_MODE_FIXED_INTERVAL:
        step_override = os.environ.get("ADAPTIVE_BUCKET_STEP")
        step = int(step_override) if step_override else min_tree_size
    else:
        raise ValueError(
            "adaptive_return_bucket_mode must be one of: "
            f"{_ADAPTIVE_BUCKET_MODE_DOUBLING}, {_ADAPTIVE_BUCKET_MODE_FIXED_INTERVAL}."
        )

    if current_tree_size >= current_return_size + step:
        return current_return_size + step
    return current_return_size


def _build_tree_mask(parent_indices: torch.LongTensor, max_depth: int) -> torch.BoolTensor:
    num_nodes = int(parent_indices.numel())
    device = parent_indices.device
    tree_mask = torch.eye(num_nodes, dtype=torch.bool, device=device)
    node_indices = torch.arange(num_nodes, device=device)
    current_parents = parent_indices.clone()
    neg_one = torch.full_like(current_parents, -1)
    for _ in range(max_depth):
        valid = current_parents >= 0
        tree_mask[node_indices[valid], current_parents[valid]] = True
        next_parents = parent_indices[current_parents.clamp(min=0)]
        current_parents = torch.where(valid, next_parents, neg_one)
    return tree_mask


def _build_retrieve_indices(
    parent_indices: torch.LongTensor,
    depths: torch.LongTensor,
    max_depth: int,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    N = int(parent_indices.numel())
    device = parent_indices.device

    has_child = torch.zeros(N, dtype=torch.bool, device=device)
    child_parents = parent_indices[1:]
    valid_parents = child_parents[child_parents >= 0]
    if valid_parents.numel() > 0:
        has_child.scatter_(0, valid_parents, True)
    leaf_indices = torch.where(~has_child)[0]
    P = int(leaf_indices.numel())

    max_path_len = max_depth + 1
    path_lens = depths[leaf_indices] + 1

    all_leaves_uniform = bool((depths[leaf_indices] == max_depth).all().item())
    if all_leaves_uniform:
        paths_rev = torch.empty((P, max_path_len), dtype=torch.long, device=device)
        current = leaf_indices.clone()
        for step in range(max_path_len):
            paths_rev[:, step] = current
            parents = parent_indices[current.clamp(min=0)]
            current = torch.where(current >= 0, parents, torch.full_like(current, -1))
        return paths_rev.flip(1), path_lens

    retrieve_indices = torch.full((P, max_path_len), -1, dtype=torch.long, device=device)
    current = leaf_indices.clone()
    current_d = depths[current]
    neg_one = torch.full_like(current, -1)
    zeros = torch.zeros_like(current_d)

    for _ in range(max_path_len):
        valid = current >= 0
        if not valid.any():
            break
        rows = torch.where(valid)[0]
        retrieve_indices[rows, current_d[rows]] = current[rows]
        parents = parent_indices[current.clamp(min=0)]
        current = torch.where(valid, parents, neg_one)
        current_d = torch.where(valid, depths[current.clamp(min=0)], zeros)

    return retrieve_indices, path_lens


def finalize_tree_draft(
    token_ids: list[int],
    parent_indices: list[int],
    depths: list[int],
    *,
    device: torch.device,
) -> TreeDraft:
    token_ids_cpu = torch.tensor(token_ids, dtype=torch.long)
    parent_cpu = torch.tensor(parent_indices, dtype=torch.long)
    depths_cpu = torch.tensor(depths, dtype=torch.long)
    max_depth_built = int(depths_cpu.max().item()) if depths_cpu.numel() > 0 else 0
    tree_mask_cpu = _build_tree_mask(parent_cpu, max_depth_built)
    retrieve_indices_cpu, path_lens_cpu = _build_retrieve_indices(parent_cpu, depths_cpu, max_depth_built)
    return TreeDraft(
        token_ids=token_ids_cpu.to(device=device),
        parent_indices=parent_cpu.to(device=device),
        depths=depths_cpu.to(device=device),
        tree_mask=tree_mask_cpu.to(device=device),
        retrieve_indices=retrieve_indices_cpu.to(device=device),
        path_lens=path_lens_cpu.to(device=device),
    )


def build_adaptive_best_tree_from_draft_logits(
    root_token_id: torch.LongTensor,
    draft_logits: torch.FloatTensor,
    *,
    context_prefix_length: int,
    draft_latency_s: float,
    non_verify_latency_s: float,
    cost_model_name: str,
    cost_gpu_type: str,
    min_tree_size: int = _MIN_ADAPTIVE_BEST_TREE_NODES,
    max_tree_size: int = _MAX_ADAPTIVE_BEST_TREE_NODES,
    adaptive_return_bucket_mode: str = _ADAPTIVE_BUCKET_MODE_FIXED_INTERVAL,
) -> TreeDraft:
    if draft_logits.dim() != 3 or draft_logits.shape[0] != 1:
        raise ValueError("draft_logits must have shape [1, depth, vocab].")
    if root_token_id.numel() != 1:
        raise ValueError("root_token_id must contain exactly one token.")
    if context_prefix_length < 0:
        raise ValueError("context_prefix_length must be >= 0.")
    if draft_latency_s <= 0:
        raise ValueError("draft_latency_s must be > 0.")
    if non_verify_latency_s < 0:
        raise ValueError("non_verify_latency_s must be >= 0.")
    _floor_override = os.environ.get("ADAPTIVE_MIN_TREE_SIZE")
    if _floor_override:
        min_tree_size = int(_floor_override)
    if min_tree_size < 1:
        raise ValueError("min_tree_size must be >= 1.")
    if max_tree_size < 1:
        raise ValueError("max_tree_size must be >= 1.")
    if min_tree_size > max_tree_size:
        raise ValueError("min_tree_size must be <= max_tree_size.")
    if adaptive_return_bucket_mode not in _ADAPTIVE_BUCKET_MODES:
        raise ValueError(
            "adaptive_return_bucket_mode must be one of: "
            f"{_ADAPTIVE_BUCKET_MODE_DOUBLING}, {_ADAPTIVE_BUCKET_MODE_FIXED_INTERVAL}."
        )

    device = draft_logits.device
    max_depth = int(draft_logits.shape[1])
    vocab_size = int(draft_logits.shape[2])

    if max_tree_size == 1:
        return finalize_tree_draft(
            [int(root_token_id.reshape(-1)[0].item())],
            [-1],
            [0],
            device=device,
        )

    latency_estimator = build_verified_latency_estimator(
        context_tokens=context_prefix_length,
        model_name=cost_model_name,
        gpu_type=cost_gpu_type,
    )
    current_tree_cost_s: float | None = None

    k_static = min(_MAX_ADAPTIVE_TOP_K, vocab_size)
    step_log_probs = torch.log_softmax(draft_logits[0], dim=-1)
    sorted_vals, sorted_ids = torch.topk(step_log_probs, k=k_static, dim=-1, sorted=True)
    sorted_vals_rows = sorted_vals.float().cpu().tolist()
    sorted_ids_rows = sorted_ids.long().cpu().tolist()

    cand_parent: list[int] = [-1]
    cand_depth: list[int] = [0]
    cand_token: list[int] = [int(root_token_id.reshape(-1)[0].item())]
    cand_rank: list[int] = [-1]
    cand_node_lp: list[float] = [0.0]
    cand_tree_idx: list[int] = [-1]
    heap: list[tuple[float, int]] = [(-0.0, 0)]

    tree_token_ids: list[int] = []
    tree_parent_indices: list[int] = []
    tree_depths: list[int] = []

    # Root path probability is 1.0 and participates in the running expectation term.
    path_prob_sum = 0.0

    tree_return_size = min_tree_size
    terminate = False
    while heap and not terminate:
        neg_path_lp, node_id = heapq.heappop(heap)
        cur_path_lp = -neg_path_lp

        parent_cand_id = cand_parent[node_id]
        tree_parent = -1 if parent_cand_id < 0 else cand_tree_idx[parent_cand_id]
        if parent_cand_id >= 0 and tree_parent < 0:
            raise RuntimeError("Parent must be visited before child in adaptive-best tree search.")

        new_path_probability = math.exp(cur_path_lp)
        prev_tree_size = len(tree_token_ids)

        if prev_tree_size > 0:
            if current_tree_cost_s is None:
                raise RuntimeError("Tree cost must be initialized after the root node.")
            prev_cost = current_tree_cost_s
            next_cost = prev_cost + latency_estimator.next_delta(prev_tree_size)
            lhs = new_path_probability * (draft_latency_s + non_verify_latency_s + prev_cost)
            rhs = path_prob_sum * (next_cost - prev_cost)
            if lhs <= rhs and prev_tree_size >= min_tree_size:
                break

        tree_token_ids.append(cand_token[node_id])
        tree_parent_indices.append(tree_parent)
        tree_depths.append(cand_depth[node_id])
        cand_tree_idx[node_id] = len(tree_token_ids) - 1
        path_prob_sum += new_path_probability

        current_tree_size = len(tree_token_ids)
        if current_tree_size == max_tree_size:
            terminate = True

        tree_return_size = _next_adaptive_tree_return_size(
            current_tree_size=current_tree_size,
            current_return_size=tree_return_size,
            min_tree_size=min_tree_size,
            bucket_mode=adaptive_return_bucket_mode,
        )

        if terminate:
            break

        if current_tree_cost_s is None:
            current_tree_cost_s = latency_estimator.estimate(1)
        else:
            current_tree_cost_s = next_cost

        node_depth = cand_depth[node_id]
        if node_depth < max_depth:
            child_depth = node_depth + 1
            row = child_depth - 1
            child_lp = sorted_vals_rows[row][0]
            child_tok = sorted_ids_rows[row][0]
            child_id = len(cand_parent)
            cand_parent.append(node_id)
            cand_depth.append(child_depth)
            cand_token.append(child_tok)
            cand_rank.append(0)
            cand_node_lp.append(child_lp)
            cand_tree_idx.append(-1)
            heapq.heappush(heap, (-(cur_path_lp + child_lp), child_id))

        if node_depth > 0:
            row = node_depth - 1
            next_rank = cand_rank[node_id] + 1
            if next_rank < k_static:
                sib_lp = sorted_vals_rows[row][next_rank]
                sib_tok = sorted_ids_rows[row][next_rank]
                sib_id = len(cand_parent)
                cand_parent.append(cand_parent[node_id])
                cand_depth.append(node_depth)
                cand_token.append(sib_tok)
                cand_rank.append(next_rank)
                cand_node_lp.append(sib_lp)
                cand_tree_idx.append(-1)
                heapq.heappush(
                    heap,
                    (-(cur_path_lp - cand_node_lp[node_id] + sib_lp), sib_id),
                )
    if not tree_token_ids:
        raise ValueError("adaptive_best failed to produce any tree node.")

    if tree_return_size > 0 and tree_return_size < len(tree_token_ids):
        tree_token_ids = tree_token_ids[:tree_return_size]
        tree_parent_indices = tree_parent_indices[:tree_return_size]
        tree_depths = tree_depths[:tree_return_size]

    return finalize_tree_draft(tree_token_ids, tree_parent_indices, tree_depths, device=device)


def _sample(logits: torch.FloatTensor, temperature: float = 0.0) -> torch.LongTensor:
    """Sample tokens from logits with optional temperature scaling."""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: list[int],
) -> torch.Tensor:
    """Extract target hidden features from selected layer outputs."""
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def _build_tree_attention_mask(
    tree_mask: torch.BoolTensor,
    prefix_len: int,
    dtype: torch.dtype,
) -> torch.FloatTensor:
    """Build attention mask for tree verification."""
    if tree_mask.dim() != 2 or tree_mask.shape[0] != tree_mask.shape[1]:
        raise ValueError("tree_mask must have shape [tree_len, tree_len].")

    tree_len = int(tree_mask.shape[0])
    total_len = prefix_len + tree_len
    min_value = torch.finfo(dtype).min
    attn = torch.full((tree_len, total_len), min_value, dtype=dtype, device=tree_mask.device)
    if prefix_len > 0:
        attn[:, :prefix_len] = 0
    attn[:, prefix_len:].masked_fill_(tree_mask, 0)
    return attn.view(1, 1, tree_len, total_len)


def _reorder_kv_cache_for_accepted_path(
    past_key_values,
    prefix_len: int,
    accepted_node_indices: torch.LongTensor,
) -> None:
    layers = [layer for layer in past_key_values.layers if layer.is_initialized]
    if not layers:
        return

    src_idx = torch.cat(
        [
            torch.arange(prefix_len, dtype=torch.long, device=accepted_node_indices.device),
            prefix_len + accepted_node_indices,
        ]
    )

    stacked_k = torch.stack([layer.keys for layer in layers])
    stacked_v = torch.stack([layer.values for layer in layers])
    new_k = stacked_k.index_select(-2, src_idx)
    new_v = stacked_v.index_select(-2, src_idx)

    for layer_idx, layer in enumerate(layers):
        layer.keys = new_k[layer_idx]
        layer.values = new_v[layer_idx]


@dataclass
class _TreeVerifyResult:
    """Result of tree path verification."""
    accepted: int
    accepted_node_indices: torch.LongTensor
    accepted_token_ids: torch.LongTensor
    next_token_id: int


def _select_best_tree_path(
    tree: TreeDraft,
    verify_logits: torch.FloatTensor,
    temperature: float,
) -> _TreeVerifyResult:
    """Select best path through verification tree and sample next token."""
    if verify_logits.dim() != 3 or verify_logits.shape[0] != 1:
        raise ValueError("verify_logits must have shape [1, tree_len, vocab].")
    if verify_logits.shape[1] != tree.token_ids.shape[0]:
        raise ValueError("verify_logits length does not match tree size.")

    greedy_next = torch.argmax(verify_logits[0], dim=-1)
    num_paths, max_path_len = tree.retrieve_indices.shape

    if max_path_len > 1:
        idx = tree.retrieve_indices.clamp(min=0)
        valid = tree.retrieve_indices >= 0
        prev_nodes = idx[:, :-1]
        curr_nodes = idx[:, 1:]
        valid_steps = valid[:, :-1] & valid[:, 1:]
        match = (greedy_next[prev_nodes] == tree.token_ids[curr_nodes]) & valid_steps
        accepted_counts = match.long().cumprod(dim=1).sum(dim=1)
    else:
        accepted_counts = torch.zeros(num_paths, dtype=torch.long, device=greedy_next.device)

    best_path_idx = int(accepted_counts.argmax().item())
    best_accepted = int(accepted_counts[best_path_idx].item())
    path_len = int(tree.path_lens[best_path_idx].item())
    best_path = tree.retrieve_indices[best_path_idx, :path_len]
    accepted_node_indices = best_path[: best_accepted + 1]
    accepted_token_ids = tree.token_ids[accepted_node_indices]
    next_parent_idx = int(accepted_node_indices[-1].item())
    next_token = int(
        _sample(verify_logits[:, next_parent_idx : next_parent_idx + 1], temperature)[0, 0].item()
    )

    return _TreeVerifyResult(
        accepted=best_accepted,
        accepted_node_indices=accepted_node_indices,
        accepted_token_ids=accepted_token_ids,
        next_token_id=next_token,
    )


@torch.inference_mode()
def bastion_generate(
    draft_model: "DFlashDraftModel",
    target: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float,
    block_size: int | None = None,
    mask_token_id: int | None = None,
    return_stats: bool = False,
    cost_model_name: str = "qwen3-4b",
    cost_gpu_type: str | None = None,
    draft_latency_s: float | None = None,
    non_verify_latency_s: float | None = None,
    min_tree_size: int = _MIN_ADAPTIVE_BEST_TREE_NODES,
    max_tree_size: int = _MAX_ADAPTIVE_BEST_TREE_NODES,
    adaptive_return_bucket_mode: str = _ADAPTIVE_BUCKET_MODE_FIXED_INTERVAL,
) -> torch.LongTensor | SimpleNamespace:
    """Generate tokens using tree-drafting with adaptive best-first search.

    Args:
        draft_model: DFlashDraftModel instance
        target: Target language model
        input_ids: Input token IDs [1, seq_len]
        max_new_tokens: Maximum tokens to generate
        stop_token_ids: Tokens that stop generation
        temperature: Sampling temperature
        block_size: Number of draft tokens per iteration (default: model.block_size)
        mask_token_id: Mask token for initialization
        return_stats: Whether to return generation statistics
        cost_model_name: Model name for cost estimation
        cost_gpu_type: GPU type for cost estimation
        draft_latency_s: Profiled draft latency from `profile_tree_adaptive_constants`.
        non_verify_latency_s: Profiled non-verification overhead from `profile_tree_adaptive_constants`.
        min_tree_size: Minimum tree nodes
        max_tree_size: Maximum tree nodes
        adaptive_return_bucket_mode: Return-size bucket mode for adaptive tree truncation.

    Returns:
        output_ids or SimpleNamespace with stats
    """
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = draft_model.block_size if block_size is None else block_size
    if block_size <= 1:
        raise ValueError("bastion_generate requires block_size > 1; use AR decoding for block_size=1.")

    mask_token_id = draft_model.mask_token_id if mask_token_id is None else mask_token_id
    if draft_latency_s is None or non_verify_latency_s is None:
        raise ValueError(
            "bastion_generate requires profiled draft_latency_s and "
            "non_verify_latency_s. Run the benchmark profile step first "
            "or pass cached profile values."
        )
    draft_latency_s = float(draft_latency_s)
    non_verify_latency_s = float(non_verify_latency_s)
    if draft_latency_s <= 0 or non_verify_latency_s <= 0:
        raise ValueError(
            "Profiled latency constants must be positive: "
            f"draft_latency_s={draft_latency_s}, "
            f"non_verify_latency_s={non_verify_latency_s}"
        )
    device = target.device

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = _cuda_time() if return_stats else None
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = _sample(output.logits, temperature)
    target_hidden = _extract_context_feature(output.hidden_states, draft_model.target_layer_ids)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    start = num_input_tokens
    draft_prefill = True
    target_dtype = next(target.parameters()).dtype

    # Fine-grained timing fields are used by the benchmark profiling cache.
    draft_latencies = []
    verification_latencies = []
    tree_build_latencies = []
    tree_select_latencies = []
    tree_kv_reorder_latencies = []
    tree_extract_latencies = []

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()

        t_draft_start = _cuda_time() if return_stats else None
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, 1 - block_size :, :])
        past_key_values_draft.crop(start)
        t_draft_end = _cuda_time() if return_stats else None
        if draft_prefill and return_stats:
            draft_prefill = False
            decode_start = _cuda_time()
        if return_stats and t_draft_start is not None and t_draft_end is not None:
            draft_latencies.append(t_draft_end - t_draft_start)

        root_id = int(block_output_ids[0, 0].item())
        t_build_start = _cuda_time() if return_stats else None
        tree = build_adaptive_best_tree_from_draft_logits(
            root_token_id=torch.tensor(root_id, device=device),
            draft_logits=draft_logits,
            context_prefix_length=start,
            draft_latency_s=draft_latency_s,
            non_verify_latency_s=non_verify_latency_s,
            cost_model_name=cost_model_name,
            cost_gpu_type=cost_gpu_type if cost_gpu_type is not None else "a5000",
            min_tree_size=min_tree_size,
            max_tree_size=max_tree_size,
            adaptive_return_bucket_mode=adaptive_return_bucket_mode,
        )
        t_build_end = _cuda_time() if return_stats else None
        if return_stats and t_build_start is not None and t_build_end is not None:
            tree_build_latencies.append(t_build_end - t_build_start)

        verify_input_ids = tree.token_ids.unsqueeze(0)
        verify_position_ids = (start + tree.depths).unsqueeze(0)
        prefix_len = past_key_values_target.get_seq_length()
        verify_attention_mask = _build_tree_attention_mask(
            tree_mask=tree.tree_mask,
            prefix_len=prefix_len,
            dtype=target_dtype,
        )

        t_verify_start = _cuda_time() if return_stats else None
        verify_output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        t_verify_end = _cuda_time() if return_stats else None
        if return_stats and t_verify_start is not None and t_verify_end is not None:
            verification_latencies.append(t_verify_end - t_verify_start)

        t_select_start = _cuda_time() if return_stats else None
        tree_result = _select_best_tree_path(tree, verify_output.logits, temperature)
        t_select_end = _cuda_time() if return_stats else None
        if return_stats and t_select_start is not None and t_select_end is not None:
            tree_select_latencies.append(t_select_end - t_select_start)

        acceptance_length = tree_result.accepted
        output_ids[:, start : start + acceptance_length + 1] = tree_result.accepted_token_ids.unsqueeze(0)
        output_ids[:, start + acceptance_length + 1] = tree_result.next_token_id

        t_kv_start = _cuda_time() if return_stats else None
        _reorder_kv_cache_for_accepted_path(
            past_key_values_target,
            prefix_len,
            tree_result.accepted_node_indices,
        )
        t_kv_end = _cuda_time() if return_stats else None
        if return_stats and t_kv_start is not None and t_kv_end is not None:
            tree_kv_reorder_latencies.append(t_kv_end - t_kv_start)

        start += acceptance_length + 1

        t_extract_start = _cuda_time() if return_stats else None
        target_hidden = _extract_context_feature(verify_output.hidden_states, draft_model.target_layer_ids)[
            :, tree_result.accepted_node_indices, :
        ]
        t_extract_end = _cuda_time() if return_stats else None
        if return_stats and t_extract_start is not None and t_extract_end is not None:
            tree_extract_latencies.append(t_extract_end - t_extract_start)

        past_key_values_target.crop(start)
        acceptance_lengths.append(acceptance_length + 1)

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_token_ids_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_token_ids_tensor
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens if num_output_tokens > 0 else 0,
        acceptance_lengths=acceptance_lengths,
        draft_latencies=draft_latencies,
        verification_latencies=verification_latencies,
        tree_build_latencies=tree_build_latencies,
        tree_select_latencies=tree_select_latencies,
        tree_kv_reorder_latencies=tree_kv_reorder_latencies,
        tree_extract_latencies=tree_extract_latencies,
    )


def _cuda_time() -> float:
    """Get current CUDA time."""
    import time
    torch.cuda.synchronize()
    return time.perf_counter()


__all__ = ["build_adaptive_best_tree_from_draft_logits", "bastion_generate"]
