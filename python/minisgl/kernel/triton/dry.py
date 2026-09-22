import triton
import triton.language as tl


@triton.jit
def dry_z_kernel(
    token_pool_ptr,
    table_indices_ptr,
    seq_lens_ptr,
    allowed_lens_ptr,
    max_matches_ptr,
    z_ptr,
    TOKEN_POOL_ROW_STRIDE: tl.constexpr,
    MAX_MATCH_ROW_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    table_idx = tl.load(table_indices_ptr + req_idx)
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    allowed_len = tl.load(allowed_lens_ptr + req_idx).to(tl.int32)
    history_ptr = token_pool_ptr + table_idx * TOKEN_POOL_ROW_STRIDE
    req_z_ptr = z_ptr + req_idx * Z_ROW_STRIDE

    left = 0
    right = 0

    for idx in tl.range(1, seq_len):
        z_i = 0
        if idx <= right:
            z_i = tl.minimum(right - idx + 1, tl.load(req_z_ptr + idx - left))

        keep_matching = idx + z_i < seq_len
        while keep_matching:
            suffix_token = tl.load(history_ptr + seq_len - 1 - z_i)
            candidate_token = tl.load(history_ptr + seq_len - 1 - (idx + z_i))
            tokens_match = suffix_token == candidate_token
            z_i += tokens_match.to(tl.int32)
            keep_matching = tokens_match & (idx + z_i < seq_len)

        tl.store(req_z_ptr + idx, z_i)
        if idx + z_i - 1 > right:
            left = idx
            right = idx + z_i - 1

    for idx in tl.range(1, seq_len):
        matched = tl.load(req_z_ptr + idx)
        if matched >= allowed_len:
            token_id = tl.load(history_ptr + seq_len - idx)
            output_ptr = max_matches_ptr + req_idx * MAX_MATCH_ROW_STRIDE + token_id
            old_match = tl.load(output_ptr)
            tl.store(output_ptr, tl.maximum(matched, old_match))


@triton.jit
def dry_naive_kernel(
    token_pool_ptr,
    table_indices_ptr,
    seq_lens_ptr,
    allowed_lens_ptr,
    max_matches_ptr,
    TOKEN_POOL_ROW_STRIDE: tl.constexpr,
    MAX_MATCH_ROW_STRIDE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    table_idx = tl.load(table_indices_ptr + req_idx)
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    allowed_len = tl.load(allowed_lens_ptr + req_idx).to(tl.int32)
    history_ptr = token_pool_ptr + table_idx * TOKEN_POOL_ROW_STRIDE
    suffix_start = seq_len - 1

    for candidate_start in tl.range(allowed_len, seq_len):
        matched = 0
        candidate_idx = candidate_start - 1
        suffix_idx = suffix_start

        keep_matching = candidate_idx >= 0
        while keep_matching:
            candidate_token = tl.load(history_ptr + candidate_idx)
            suffix_token = tl.load(history_ptr + suffix_idx)
            tokens_match = candidate_token == suffix_token
            matched += tokens_match.to(tl.int32)
            candidate_idx -= 1
            suffix_idx -= 1
            keep_matching = tokens_match & (candidate_idx >= 0)

        if matched >= allowed_len:
            token_id = tl.load(history_ptr + candidate_start)
            output_ptr = max_matches_ptr + req_idx * MAX_MATCH_ROW_STRIDE + token_id
            old_match = tl.load(output_ptr)
            tl.store(output_ptr, tl.maximum(matched, old_match))


@triton.jit
def dry_candidate_parallel_kernel(
    token_pool_ptr,
    table_indices_ptr,
    seq_lens_ptr,
    allowed_lens_ptr,
    max_matches_ptr,
    TOKEN_POOL_ROW_STRIDE: tl.constexpr,
    MAX_MATCH_ROW_STRIDE: tl.constexpr,
):
    """Assign one program to each request/candidate continuation pair."""
    req_idx = tl.program_id(0)
    candidate_slot = tl.program_id(1)

    table_idx = tl.load(table_indices_ptr + req_idx)
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    allowed_len = tl.load(allowed_lens_ptr + req_idx).to(tl.int32)
    history_ptr = token_pool_ptr + table_idx * TOKEN_POOL_ROW_STRIDE

    suffix_idx = seq_len - 1
    candidate_start = candidate_slot + allowed_len
    valid_candidate = (allowed_len > 0) & (candidate_start <= suffix_idx)

    candidate_idx = candidate_start - 1
    matched = 0
    keep_matching = valid_candidate & (candidate_idx >= 0)

    while keep_matching:
        candidate_token = tl.load(history_ptr + candidate_idx)
        suffix_token = tl.load(history_ptr + suffix_idx)
        tokens_match = candidate_token == suffix_token
        matched += tokens_match.to(tl.int32)
        candidate_idx -= 1
        suffix_idx -= 1
        keep_matching = tokens_match & (candidate_idx >= 0) & (suffix_idx >= 0)

    if valid_candidate & (matched >= allowed_len):
        token_id = tl.load(history_ptr + candidate_start)
        output_ptr = max_matches_ptr + req_idx * MAX_MATCH_ROW_STRIDE + token_id
        tl.atomic_max(output_ptr, matched)


@triton.jit
def dry_blocked_parallel_kernel(
    token_pool_ptr,
    table_indices_ptr,
    seq_lens_ptr,
    allowed_lens_ptr,
    max_matches_ptr,
    TOKEN_POOL_ROW_STRIDE: tl.constexpr,
    MAX_MATCH_ROW_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Assign a vector of candidate continuation positions to each program."""
    req_idx = tl.program_id(0)
    candidate_block_idx = tl.program_id(1)

    table_idx = tl.load(table_indices_ptr + req_idx)
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    allowed_len = tl.load(allowed_lens_ptr + req_idx).to(tl.int32)
    history_ptr = token_pool_ptr + table_idx * TOKEN_POOL_ROW_STRIDE

    suffix_idx = seq_len - 1
    candidate_start = (
        allowed_len
        + candidate_block_idx * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    valid_candidate = (allowed_len > 0) & (candidate_start <= suffix_idx)

    candidate_idx = candidate_start - 1
    matched = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    active = valid_candidate & (candidate_idx >= 0) & (suffix_idx >= 0)

    while tl.sum(active.to(tl.int32), axis=0) > 0:
        candidate_token = tl.load(
            history_ptr + candidate_idx,
            mask=active,
            other=-1,
        )
        suffix_token = tl.load(
            history_ptr + suffix_idx,
            mask=suffix_idx >= 0,
            other=-1,
        )
        tokens_match = active & (candidate_token == suffix_token)
        matched += tokens_match.to(tl.int32)

        candidate_idx -= 1
        suffix_idx -= 1
        active = tokens_match & (candidate_idx >= 0) & (suffix_idx >= 0)

    qualifies = valid_candidate & (matched >= allowed_len)
    token_id = tl.load(
        history_ptr + candidate_start,
        mask=qualifies,
        other=0,
    )
    output_ptr = max_matches_ptr + req_idx * MAX_MATCH_ROW_STRIDE + token_id
    tl.atomic_max(output_ptr, matched, mask=qualifies)


@triton.jit
def dry_window_kernel(
    token_pool_ptr,
    table_indices_ptr,
    seq_lens_ptr,
    allowed_lens_ptr,
    max_matches_ptr,
    TOKEN_POOL_ROW_STRIDE: tl.constexpr,
    MAX_MATCH_ROW_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compare a block of candidate windows for one request in parallel."""
    req_idx = tl.program_id(0)
    candidate_block_idx = tl.program_id(1)

    table_idx = tl.load(table_indices_ptr + req_idx)
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    allowed_len = tl.load(allowed_lens_ptr + req_idx).to(tl.int32)
    history_ptr = token_pool_ptr + table_idx * TOKEN_POOL_ROW_STRIDE

    candidate_start = candidate_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    suffix_start = seq_len - allowed_len
    valid_candidate = (
        (allowed_len > 0)
        & (seq_len > allowed_len)
        & (candidate_start < suffix_start)
    )

    window_matches = valid_candidate
    for offset in tl.range(0, allowed_len):
        candidate_token = tl.load(
            history_ptr + candidate_start + offset,
            mask=valid_candidate,
            other=-1,
        )
        suffix_token = tl.load(
            history_ptr + suffix_start + offset,
            mask=seq_len > allowed_len,
            other=-1,
        )
        window_matches = window_matches & (candidate_token == suffix_token)

    matched = tl.where(window_matches, allowed_len, 0).to(tl.int32)
    candidate_left = candidate_start - 1
    suffix_left = suffix_start - 1
    active = window_matches & (candidate_left >= 0) & (suffix_left >= 0)

    while tl.sum(active.to(tl.int32), axis=0) > 0:
        candidate_token = tl.load(
            history_ptr + candidate_left,
            mask=active,
            other=-1,
        )
        suffix_token = tl.load(
            history_ptr + suffix_left,
            mask=suffix_left >= 0,
            other=-1,
        )
        active = active & (candidate_token == suffix_token)
        matched += active.to(tl.int32)
        candidate_left -= 1
        suffix_left -= 1
        active = active & (candidate_left >= 0) & (suffix_left >= 0)

    target_position = candidate_start + allowed_len
    token_id = tl.load(
        history_ptr + target_position,
        mask=window_matches,
        other=0,
    )
    output_ptr = max_matches_ptr + req_idx * MAX_MATCH_ROW_STRIDE + token_id
    tl.atomic_max(output_ptr, matched, mask=window_matches)


@triton.jit
def dry_apply_penalties_kernel(
    logits_ptr,
    max_matches_ptr,
    allowed_lens_ptr,
    dry_multipliers_ptr,
    dry_bases_ptr,
    vocab_size,
    LOGITS_ROW_STRIDE: tl.constexpr,
    MAX_MATCH_ROW_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply the per-token DRY penalties computed by a matching kernel."""
    req_idx = tl.program_id(0)
    vocab_block_idx = tl.program_id(1)
    token_ids = vocab_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_token = token_ids < vocab_size

    matched = tl.load(
        max_matches_ptr + req_idx * MAX_MATCH_ROW_STRIDE + token_ids,
        mask=valid_token,
        other=-1,
    )
    allowed_len = tl.load(allowed_lens_ptr + req_idx).to(tl.int32)
    dry_multiplier = tl.load(dry_multipliers_ptr + req_idx)
    dry_base = tl.load(dry_bases_ptr + req_idx)
    should_apply = valid_token & (matched >= allowed_len) & (dry_multiplier != 0.0)

    exponent = (matched - allowed_len).to(tl.float32)
    penalty = dry_multiplier * tl.exp2(exponent * tl.log2(dry_base))
    logits = tl.load(
        logits_ptr + req_idx * LOGITS_ROW_STRIDE + token_ids,
        mask=should_apply,
        other=0.0,
    )
    tl.store(
        logits_ptr + req_idx * LOGITS_ROW_STRIDE + token_ids,
        logits - penalty,
        mask=should_apply,
    )
