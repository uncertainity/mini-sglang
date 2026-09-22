from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, cast

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

from .config import DryAlgorithm
from .dry import DryAlgorithm as PythonDryAlgorithm
from .dry import apply_dry_penalties_python

if TYPE_CHECKING:
    from minisgl.core import Batch


DryBackend = Literal["cpu", "cuda"]


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None

    token_pool: torch.Tensor | None = None
    table_indices: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None

    dry_multipliers: torch.Tensor | None = None
    dry_bases: torch.Tensor | None = None
    dry_allowed_lengths: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int
    dry_backend: DryBackend = "cpu"
    dry_algorithm: DryAlgorithm = "z"

    def __post_init__(self) -> None:
        if self.dry_backend not in ("cpu", "cuda"):
            raise ValueError(f"Unknown DRY backend: {self.dry_backend!r}")
        if self.dry_algorithm not in (
            "z",
            "naive",
            "candidate_parallel",
            "blocked_parallel",
            "window",
        ):
            raise ValueError(f"Unknown DRY algorithm: {self.dry_algorithm!r}")
        if self.dry_backend == "cpu" and self.dry_algorithm not in ("z", "naive", "window"):
            raise ValueError(
                f"DRY algorithm {self.dry_algorithm!r} requires dry_backend='cuda'"
            )

    def prepare(self, batch: Batch, token_pool: torch.Tensor) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        use_dry = any(p.dry_multiplier > 0.0 for p in params)
        if use_dry:
            table_indices = make_device_tensor(
                [r.table_idx for r in batch.reqs],
                torch.int64,
                self.device,
            )
            seq_lens = make_device_tensor(
                [r.device_len for r in batch.reqs],
                torch.int64,
                self.device,
            )
            dry_multipliers = make_device_tensor(
                [p.dry_multiplier for p in params],
                torch.float32,
                self.device,
            )

            dry_bases = make_device_tensor(
                [p.dry_base for p in params],
                torch.float32,
                self.device,
            )

            dry_allowed_lengths = make_device_tensor(
                [p.dry_allowed_length for p in params],
                torch.int64,
                self.device,
            )
        else:
            token_pool = None
            table_indices = None
            seq_lens = None
            dry_multipliers = None
            dry_bases = None
            dry_allowed_lengths = None

        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(
                temperatures=None,
                token_pool=token_pool,
                table_indices=table_indices,
                seq_lens=seq_lens,
                dry_multipliers=dry_multipliers,
                dry_bases=dry_bases,
                dry_allowed_lengths=dry_allowed_lengths,
            )

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(
            temperatures,
            top_k=top_k,
            top_p=top_p,
            token_pool=token_pool,
            table_indices=table_indices,
            seq_lens=seq_lens,
            dry_multipliers=dry_multipliers,
            dry_bases=dry_bases,
            dry_allowed_lengths=dry_allowed_lengths,
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.dry_multipliers is not None:
                assert args.token_pool is not None
                assert args.table_indices is not None
                assert args.seq_lens is not None
                assert args.dry_allowed_lengths is not None
                assert args.dry_bases is not None

                if self.dry_backend == "cpu":
                    logits = apply_dry_penalties_python(
                        logits=logits,
                        token_pool=args.token_pool,
                        table_indices=args.table_indices,
                        seq_lens=args.seq_lens,
                        allowed_lengths=args.dry_allowed_lengths,
                        dry_multipliers=args.dry_multipliers,
                        dry_bases=args.dry_bases,
                        algorithm=cast(PythonDryAlgorithm, self.dry_algorithm),
                    )
                else:
                    from minisgl.kernel.dry import apply_dry_penalties_triton

                    logits = apply_dry_penalties_triton(
                        logits=logits,
                        token_pool=args.token_pool,
                        table_indices=args.table_indices,
                        seq_lens=args.seq_lens,
                        allowed_lengths=args.dry_allowed_lengths,
                        dry_multipliers=args.dry_multipliers,
                        dry_bases=args.dry_bases,
                        algorithm=self.dry_algorithm,
                    )
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
