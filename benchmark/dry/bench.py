from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, TextIO

import torch
import triton

from minisgl.kernel.dry import apply_dry_penalties_triton


BenchmarkAlgorithm = Literal[
    "z",
    "naive",
    "candidate_parallel",
    "blocked_parallel",
    "window",
]
RepetitionProfile = Literal["low", "normal", "high"]
Sweep = Literal["context", "batch", "repetition"]


def load_python_reference() -> Callable[..., torch.Tensor]:
    reference_path = Path(__file__).parents[2] / "python/minisgl/engine/dry.py"
    spec = importlib.util.spec_from_file_location("minisgl_dry_reference", reference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DRY reference implementation from {reference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_dry_penalties_python


apply_dry_penalties_python = load_python_reference()


@dataclass(frozen=True)
class BenchmarkCase:
    batch_size: int
    context_length: int
    profile: RepetitionProfile = "normal"


@dataclass(frozen=True)
class RunMetadata:
    timestamp: str
    git_commit: str
    device: str
    torch_version: str
    triton_version: str
    torch_cuda_version: str


@dataclass(frozen=True)
class BenchmarkResult:
    sweep: Sweep
    strategy: BenchmarkAlgorithm
    profile: RepetitionProfile
    batch_size: int
    context_length: int
    vocab_size: int
    python_warmup: int
    python_repeats: int
    triton_warmup: int
    triton_repeats: int
    python_ms: float
    triton_ms: float
    python_speedup: float
    gpu_speedup_vs_naive: float
    max_error: float


CSV_FIELDS = tuple(RunMetadata.__dataclass_fields__) + tuple(BenchmarkResult.__dataclass_fields__)


class CsvResultWriter:
    def __init__(self, path: Path, metadata: RunMetadata) -> None:
        self.path = path
        self.metadata = metadata
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def write(self, result: BenchmarkResult) -> None:
        self._writer.writerow({**asdict(self.metadata), **asdict(result)})
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> CsvResultWriter:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def git_revision(repo_root: Path) -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return f"{revision}-dirty" if status else revision
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def safe_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-_")
    return sanitized.lower() or "unknown"


def resolve_output_path(
    requested: Path | None,
    *,
    repo_root: Path,
    timestamp: datetime,
    device_name: str,
    revision: str,
) -> Path:
    if requested is not None:
        return requested.expanduser()

    stamp = timestamp.strftime("%Y%m%d-%H%M%S-%f%z")
    device = safe_filename_component(device_name)
    return repo_root / "benchmark-results" / "dry" / f"{stamp}-{device}-{revision}.csv"


DEFAULT_CONTEXT_LENGTHS = (128, 256, 512, 1024, 2048)
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16)


def make_inputs(
    case: BenchmarkCase,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    token_pool = torch.randint(
        0,
        vocab_size,
        (case.batch_size, case.context_length),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )

    if case.profile == "normal":
        # Inject one repeated suffix into an otherwise random history.
        repeated_length = max(1, min(8, case.context_length // 4))
        source_start = case.context_length // 4
        token_pool[:, -repeated_length:] = token_pool[
            :,
            source_start : source_start + repeated_length,
        ].clone()
    elif case.profile == "high":
        # Copy the suffix into many non-overlapping locations. A random token
        # between copies prevents every match from extending to the full
        # history and overflowing the exponential DRY penalty.
        repeated_length = max(1, min(8, case.context_length // 4))
        suffix_start = case.context_length - repeated_length
        repeated_suffix = token_pool[:, suffix_start:].clone()
        for source_start in range(
            1,
            suffix_start - repeated_length + 1,
            repeated_length + 1,
        ):
            token_pool[:, source_start : source_start + repeated_length] = repeated_suffix

    logits = torch.randn(
        (case.batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    table_indices = torch.arange(case.batch_size, dtype=torch.int64, device=device)
    seq_lens = torch.full(
        (case.batch_size,),
        case.context_length,
        dtype=torch.int64,
        device=device,
    )
    allowed_lengths = torch.tensor(
        [3 + req_idx % 2 for req_idx in range(case.batch_size)],
        dtype=torch.int64,
        device=device,
    )
    dry_multipliers = torch.tensor(
        [0.8 + 0.1 * (req_idx % 3) for req_idx in range(case.batch_size)],
        dtype=torch.float32,
        device=device,
    )
    dry_bases = torch.full(
        (case.batch_size,),
        1.5,
        dtype=torch.float32,
        device=device,
    )
    return (
        logits,
        token_pool,
        table_indices,
        seq_lens,
        allowed_lengths,
        dry_multipliers,
        dry_bases,
    )


def benchmark_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the Python and Triton DRY penalty implementations."
    )
    parser.add_argument(
        "--algorithm",
        choices=("all", "z", "naive", "candidate_parallel", "blocked_parallel", "window"),
        default="all",
        help="DRY strategy to benchmark; the default compares all five.",
    )
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--cpu-warmup", type=int, default=1)
    parser.add_argument("--cpu-repeats", type=int, default=3)
    parser.add_argument("--cuda-warmup", type=int, default=3)
    parser.add_argument("--cuda-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--suite",
        choices=("both", "context", "batch", "repetition"),
        default="both",
        help="Sweep context length, batch size, both, or repetition profiles.",
    )
    parser.add_argument(
        "--profile",
        choices=("all", "low", "normal", "high"),
        default="normal",
        help="History repetition profile; use 'all' for all three profiles.",
    )
    parser.add_argument("--fixed-batch", type=int, default=4)
    parser.add_argument("--fixed-context", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "CSV result path. By default, write a timestamped file below "
            "benchmark-results/dry/."
        ),
    )
    return parser.parse_args()


def benchmark_cases(
    sweep: Sweep,
    title: str,
    cases: tuple[BenchmarkCase, ...],
    algorithms: tuple[BenchmarkAlgorithm, ...],
    args: argparse.Namespace,
    device: torch.device,
    result_writer: CsvResultWriter,
) -> None:
    print(title)
    print(
        f"{'strategy':>18} {'profile':>8} {'batch':>7} {'context':>9} {'cpu/python (ms)':>17} "
        f"{'cuda/triton (ms)':>18} {'py speedup':>11} {'gpu/naive':>11} {'max error':>12}"
    )

    for case_idx, case in enumerate(cases):
        inputs = make_inputs(case, args.vocab_size, device, args.seed + case_idx)
        case_results: list[tuple[BenchmarkAlgorithm, float, float, float]] = []

        for algorithm in algorithms:
            reference_algorithm = (
                "naive"
                if algorithm in ("candidate_parallel", "blocked_parallel")
                else algorithm
            )

            def run_python() -> torch.Tensor:
                return apply_dry_penalties_python(*inputs, algorithm=reference_algorithm)

            def run_triton() -> torch.Tensor:
                return apply_dry_penalties_triton(*inputs, algorithm=algorithm)

            python_output = run_python()
            triton_output = run_triton()
            torch.cuda.synchronize()
            max_error = float((python_output - triton_output).abs().max().item())
            torch.testing.assert_close(triton_output, python_output, rtol=1e-4, atol=1e-4)

            python_ms = benchmark_ms(
                run_python,
                warmup=args.cpu_warmup,
                repeats=args.cpu_repeats,
            )
            triton_ms = benchmark_ms(
                run_triton,
                warmup=args.cuda_warmup,
                repeats=args.cuda_repeats,
            )
            case_results.append((algorithm, python_ms, triton_ms, max_error))

        naive_triton_ms = next(
            (
                triton_ms
                for algorithm, _python_ms, triton_ms, _max_error in case_results
                if algorithm == "naive"
            ),
            None,
        )
        if naive_triton_ms is None:
            def run_naive_python() -> torch.Tensor:
                return apply_dry_penalties_python(*inputs, algorithm="naive")

            def run_naive_triton() -> torch.Tensor:
                return apply_dry_penalties_triton(*inputs, algorithm="naive")

            naive_python_output = run_naive_python()
            naive_triton_output = run_naive_triton()
            torch.testing.assert_close(
                naive_triton_output,
                naive_python_output,
                rtol=1e-4,
                atol=1e-4,
            )
            naive_triton_ms = benchmark_ms(
                run_naive_triton,
                warmup=args.cuda_warmup,
                repeats=args.cuda_repeats,
            )

        for algorithm, python_ms, triton_ms, max_error in case_results:
            python_speedup = python_ms / triton_ms
            gpu_speedup_vs_naive = naive_triton_ms / triton_ms
            result_writer.write(
                BenchmarkResult(
                    sweep=sweep,
                    strategy=algorithm,
                    profile=case.profile,
                    batch_size=case.batch_size,
                    context_length=case.context_length,
                    vocab_size=args.vocab_size,
                    python_warmup=args.cpu_warmup,
                    python_repeats=args.cpu_repeats,
                    triton_warmup=args.cuda_warmup,
                    triton_repeats=args.cuda_repeats,
                    python_ms=python_ms,
                    triton_ms=triton_ms,
                    python_speedup=python_speedup,
                    gpu_speedup_vs_naive=gpu_speedup_vs_naive,
                    max_error=max_error,
                )
            )
            print(
                f"{algorithm:>18} {case.profile:>8} {case.batch_size:>7} "
                f"{case.context_length:>9} "
                f"{python_ms:>17.3f} {triton_ms:>18.3f} "
                f"{python_speedup:>10.2f}x {gpu_speedup_vs_naive:>10.2f}x "
                f"{max_error:>12.3g}"
            )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because both paths receive GPU-resident tensors")

    device = torch.device("cuda")
    repo_root = Path(__file__).parents[2]
    started_at = datetime.now().astimezone()
    revision = git_revision(repo_root)
    device_name = torch.cuda.get_device_name(device)
    output_path = resolve_output_path(
        args.output,
        repo_root=repo_root,
        timestamp=started_at,
        device_name=device_name,
        revision=revision,
    )
    algorithms: tuple[BenchmarkAlgorithm, ...]
    if args.algorithm == "all":
        algorithms = (
            "naive",
            "candidate_parallel",
            "blocked_parallel",
            "z",
            "window",
        )
    else:
        algorithms = (args.algorithm,)

    profiles: tuple[RepetitionProfile, ...]
    if args.profile == "all":
        profiles = ("low", "normal", "high")
    else:
        profiles = (args.profile,)

    context_cases = tuple(
        BenchmarkCase(
            batch_size=args.fixed_batch,
            context_length=context_length,
            profile=profile,
        )
        for context_length in DEFAULT_CONTEXT_LENGTHS
        for profile in profiles
    )
    batch_cases = tuple(
        BenchmarkCase(
            batch_size=batch_size,
            context_length=args.fixed_context,
            profile=profile,
        )
        for batch_size in DEFAULT_BATCH_SIZES
        for profile in profiles
    )
    repetition_cases = tuple(
        BenchmarkCase(
            batch_size=args.fixed_batch,
            context_length=args.fixed_context,
            profile=profile,
        )
        for profile in profiles
    )

    print(f"Device: {device_name}")
    print(f"DRY strategies: {', '.join(algorithms)}")
    print(f"Vocabulary size: {args.vocab_size:,}")
    print(f"Results: {output_path}")
    print()

    sections: list[tuple[Sweep, str, tuple[BenchmarkCase, ...]]] = []
    if args.suite in ("both", "context"):
        sections.append(
            ("context", f"Context sweep (fixed batch={args.fixed_batch})", context_cases)
        )
    if args.suite in ("both", "batch"):
        sections.append(
            ("batch", f"Batch sweep (fixed context={args.fixed_context})", batch_cases)
        )
    if args.suite == "repetition":
        sections.append(
            (
                "repetition",
                f"Repetition sweep (batch={args.fixed_batch}, context={args.fixed_context})",
                repetition_cases,
            )
        )

    metadata = RunMetadata(
        timestamp=started_at.isoformat(),
        git_commit=revision,
        device=device_name,
        torch_version=torch.__version__,
        triton_version=triton.__version__,
        torch_cuda_version=torch.version.cuda or "unknown",
    )
    with CsvResultWriter(output_path, metadata) as result_writer:
        for section_idx, (sweep, title, cases) in enumerate(sections):
            if section_idx:
                print()
            benchmark_cases(sweep, title, cases, algorithms, args, device, result_writer)

    print()
    print(f"Saved CSV results to {output_path}")
    from plot import plot_csv

    plot_path = plot_csv(output_path)
    print(f"Saved JPEG plot to {plot_path}")


if __name__ == "__main__":
    main()
