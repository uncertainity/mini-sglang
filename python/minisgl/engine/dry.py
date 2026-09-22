from __future__ import annotations

from typing import Literal

import torch


DryAlgorithm = Literal["z", "naive", "window"]


def _apply_dry_penalties_one_req_z(
    logit_i: torch.Tensor,
    history_i: torch.Tensor,
    allowed_len: int,
    dry_multiplier: float,
    dry_base: float,
) -> torch.Tensor:
    if len(history_i) <= allowed_len:
        return logit_i

    penalty_targets = {}

    rev = torch.flip(history_i, dims=(0,))
    left = 0
    right = 0
    z = torch.zeros(len(rev), dtype=torch.int32, device=rev.device)
    for idx in range(1, len(rev)):
        if idx <= right:
            z[idx] = min(right - idx + 1, int(z[idx - left].item()))
        while (
            idx + int(z[idx].item()) < len(rev)
            and rev[idx + int(z[idx].item())] == rev[int(z[idx].item())]
        ):
            z[idx] += 1

        matched = int(z[idx].item())
        if idx + matched - 1 > right:
            left = idx
            right = idx + matched - 1

    for idx in range(1, len(z)):
        matched = int(z[idx].item())
        if matched >= allowed_len:
            token_id = int(rev[idx - 1].item())
            penalty_targets[token_id] = max(penalty_targets.get(token_id, 0), matched)

    for token_id, matched in penalty_targets.items():
        penalty = dry_multiplier * (dry_base ** (matched - allowed_len))
        logit_i[token_id] -= penalty

    return logit_i


def _apply_dry_penalties_one_req_window(
    logit_i: torch.Tensor,
    history_i: torch.Tensor,
    allowed_len: int,
    dry_multiplier: float,
    dry_base: float,
) -> torch.Tensor:
    if len(history_i) <= allowed_len:
        return logit_i

    penalty_targets = {}
    candidate_windows = []
    matched_indices = []
    end_point = len(history_i) - allowed_len
    current_suffix = history_i[end_point:]

    for idx in range(end_point):
        candidate_windows.append(history_i[idx : idx + allowed_len])

    for candidate_idx, window in enumerate(candidate_windows):
        if torch.equal(window, current_suffix):
            matched_indices.append(candidate_idx + allowed_len)

    for target_idx in matched_indices:
        candidate_left = target_idx - allowed_len - 1
        current_suffix_left = end_point - 1
        matched = allowed_len
        while candidate_left >= 0 and current_suffix_left >= 0:
            if history_i[candidate_left] == history_i[current_suffix_left]:
                matched += 1
                candidate_left -= 1
                current_suffix_left -= 1
            else:
                break

        token_id = int(history_i[target_idx].item())
        penalty_targets[token_id] = max(penalty_targets.get(token_id, 0), matched)

    for token_id, matched in penalty_targets.items():
        penalty = dry_multiplier * (dry_base ** (matched - allowed_len))
        logit_i[token_id] -= penalty

    return logit_i


def _apply_dry_penalties_one_req_naive(
    logit_i: torch.Tensor,
    history_i: torch.Tensor,
    allowed_len: int,
    dry_multiplier: float,
    dry_base: float,
) -> torch.Tensor:
    if len(history_i) <= allowed_len:
        return logit_i

    penalty_targets = {}
    suffix_start = len(history_i) - 1

    for candidate_start in range(allowed_len, len(history_i)):
        matched = 0
        offset = 1
        while candidate_start - offset >= 0 and suffix_start - (offset - 1) >= 0:
            if history_i[candidate_start - offset] == history_i[suffix_start - (offset - 1)]:
                matched += 1
                offset += 1
            else:
                break

        if matched >= allowed_len:
            token_id = int(history_i[candidate_start].item())
            penalty_targets[token_id] = max(penalty_targets.get(token_id, 0), matched)

    for token_id, matched in penalty_targets.items():
        penalty = dry_multiplier * (dry_base ** (matched - allowed_len))
        logit_i[token_id] -= penalty

    return logit_i


def apply_dry_penalties_python(
    logits: torch.Tensor,
    token_pool: torch.Tensor,
    table_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    allowed_lengths: torch.Tensor,
    dry_multipliers: torch.Tensor,
    dry_bases: torch.Tensor,
    *,
    algorithm: DryAlgorithm = "z",
) -> torch.Tensor:
    if algorithm == "z":
        apply_one_req = _apply_dry_penalties_one_req_z
    elif algorithm == "naive":
        apply_one_req = _apply_dry_penalties_one_req_naive
    elif algorithm == "window":
        apply_one_req = _apply_dry_penalties_one_req_window
    else:
        raise ValueError(f"Unknown DRY algorithm: {algorithm!r}")

    assert len(logits) == len(table_indices)
    assert len(table_indices) == len(seq_lens)

    dry_logits = logits.clone()

    for idx in range(len(logits)):
        table_idx = int(table_indices[idx].item())
        seq_len = int(seq_lens[idx].item())
        history_i = token_pool[table_idx, :seq_len]
        allowed_len = int(allowed_lengths[idx].item())
        dry_multiplier = float(dry_multipliers[idx].item())
        dry_base = float(dry_bases[idx].item())

        if dry_multiplier == 0.0:
            continue

        apply_one_req(
            dry_logits[idx],
            history_i,
            allowed_len,
            dry_multiplier,
            dry_base,
        )

    return dry_logits
