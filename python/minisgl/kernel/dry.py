from __future__ import annotations

from typing import Literal

import torch
import triton

from .triton.dry import (
    dry_apply_penalties_kernel,
    dry_blocked_parallel_kernel,
    dry_candidate_parallel_kernel,
    dry_naive_kernel,
    dry_window_kernel,
    dry_z_kernel,
)

DryAlgorithm = Literal["z", "naive", "candidate_parallel", "blocked_parallel", "window"]


def apply_dry_penalties_triton(
    logits: torch.Tensor,
    token_pool: torch.Tensor,
    table_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    allowed_lengths: torch.Tensor,
    dry_multipliers: torch.Tensor,
    dry_bases: torch.Tensor,
    *,
    algorithm: DryAlgorithm = "z",
    blocked_parallel_block_size: int = 16,
    window_block_size: int = 128,
) -> torch.Tensor:
    """Apply DRY penalties on GPU using the selected matching algorithm."""
    if algorithm not in ("z", "naive", "candidate_parallel", "blocked_parallel", "window"):
        raise ValueError(f"Unknown DRY algorithm: {algorithm!r}")
    if blocked_parallel_block_size <= 0 or (
        blocked_parallel_block_size & (blocked_parallel_block_size - 1)
    ):
        raise ValueError("blocked_parallel_block_size must be a positive power of two")
    if window_block_size <= 0 or window_block_size & (window_block_size - 1):
        raise ValueError("window_block_size must be a positive power of two")

    assert logits.ndim == 2
    assert token_pool.ndim == 2
    batch_size, vocab_size = logits.shape
    assert table_indices.shape == (batch_size,)
    assert seq_lens.shape == (batch_size,)
    assert allowed_lengths.shape == (batch_size,)
    assert dry_multipliers.shape == (batch_size,)
    assert dry_bases.shape == (batch_size,)
    assert logits.stride(1) == 1
    assert token_pool.stride(1) == 1

    inputs = (
        logits,
        token_pool,
        table_indices,
        seq_lens,
        allowed_lengths,
        dry_multipliers,
        dry_bases,
    )
    assert all(tensor.is_cuda for tensor in inputs)
    assert all(tensor.device == logits.device for tensor in inputs)

    dry_logits = logits.clone()
    if batch_size == 0 or vocab_size == 0:
        return dry_logits

    max_matches = torch.full(
        (batch_size, vocab_size),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    common_args = (
        token_pool,
        table_indices,
        seq_lens,
        allowed_lengths,
        max_matches,
    )

    if algorithm == "z":
        z = torch.zeros(
            (batch_size, token_pool.shape[1]),
            dtype=torch.int32,
            device=logits.device,
        )
        dry_z_kernel[(batch_size,)](
            *common_args,
            z,
            TOKEN_POOL_ROW_STRIDE=token_pool.stride(0),
            MAX_MATCH_ROW_STRIDE=max_matches.stride(0),
            Z_ROW_STRIDE=z.stride(0),
            num_warps=1,
        )
    elif algorithm == "naive":
        dry_naive_kernel[(batch_size,)](
            *common_args,
            TOKEN_POOL_ROW_STRIDE=token_pool.stride(0),
            MAX_MATCH_ROW_STRIDE=max_matches.stride(0),
            num_warps=1,
        )
    elif algorithm == "candidate_parallel":
        naive_grid = (batch_size, token_pool.shape[1])
        dry_candidate_parallel_kernel[naive_grid](
            *common_args,
            TOKEN_POOL_ROW_STRIDE=token_pool.stride(0),
            MAX_MATCH_ROW_STRIDE=max_matches.stride(0),
            num_warps=1,
        )
    elif algorithm == "blocked_parallel":
        blocked_grid = (
            batch_size,
            triton.cdiv(token_pool.shape[1], blocked_parallel_block_size),
        )
        dry_blocked_parallel_kernel[blocked_grid](
            *common_args,
            TOKEN_POOL_ROW_STRIDE=token_pool.stride(0),
            MAX_MATCH_ROW_STRIDE=max_matches.stride(0),
            BLOCK_SIZE=blocked_parallel_block_size,
            num_warps=1,
        )
    else:
        window_grid = (
            batch_size,
            triton.cdiv(token_pool.shape[1], window_block_size),
        )
        dry_window_kernel[window_grid](
            *common_args,
            TOKEN_POOL_ROW_STRIDE=token_pool.stride(0),
            MAX_MATCH_ROW_STRIDE=max_matches.stride(0),
            BLOCK_SIZE=window_block_size,
            num_warps=4,
        )

    penalty_block_size = 256
    penalty_grid = (
        batch_size,
        triton.cdiv(vocab_size, penalty_block_size),
    )
    dry_apply_penalties_kernel[penalty_grid](
        dry_logits,
        max_matches,
        allowed_lengths,
        dry_multipliers,
        dry_bases,
        vocab_size,
        LOGITS_ROW_STRIDE=dry_logits.stride(0),
        MAX_MATCH_ROW_STRIDE=max_matches.stride(0),
        BLOCK_SIZE=penalty_block_size,
        num_warps=4,
    )
    return dry_logits
