# DRY penalty microbenchmark

This benchmark compares the Python reference implementations of the DRY repetition penalty with the five Triton strategies implemented by Mini-SGLang:

- `naive`
- `candidate_parallel`
- `blocked_parallel`
- `z`
- `window`

Every case checks the Triton output against the corresponding Python reference before reporting a timing. The benchmark prints a readable table, incrementally saves the same measurements as CSV, and generates a four-panel JPEG summary.

## Environment

The launcher uses the dedicated Mini-SGLang virtual environment at `/data1/sandesh/.venvs/mini-sglang`, CUDA build tools from the `vllm092clean` Conda environment, and GPU 6 by default. Matplotlib is required for JPEG generation and is already declared in Mini-SGLang's development dependencies. Override the GPU normally when needed:

```bash
CUDA_VISIBLE_DEVICES=0 bash benchmark/dry/run.sh
```

The launcher can be called from any working directory.

## Smoke test

Run one small case to compile the kernel, check numerical agreement, and verify CSV output:

```bash
bash benchmark/dry/run.sh \
  --algorithm z \
  --suite repetition \
  --profile low \
  --fixed-batch 1 \
  --fixed-context 32 \
  --vocab-size 1024 \
  --cpu-warmup 0 \
  --cpu-repeats 1 \
  --cuda-warmup 0 \
  --cuda-repeats 1
```

## Benchmark suites

Run the default context-length and batch-size sweeps for all five strategies:

```bash
bash benchmark/dry/run.sh
```

Run the low, normal, and high repetition profiles at a fixed shape:

```bash
bash benchmark/dry/run.sh --suite repetition --profile all
```

Select a single strategy or provide an explicit output file:

```bash
bash benchmark/dry/run.sh \
  --algorithm blocked_parallel \
  --output benchmark-results/dry/blocked-parallel.csv
```

By default, results are written to a timestamped path of the form:

```text
benchmark-results/dry/<timestamp>-<gpu>-<git-revision>.csv
benchmark-results/dry/<timestamp>-<gpu>-<git-revision>.jpg
```

The JPEG contains Triton latency and GPU speedup versus the Triton-naive kernel for both the context-length and batch-size sweeps. Both y-axes use logarithmic scales so all five strategies remain readable.

Plot any existing benchmark CSV independently:

```bash
/data1/sandesh/.venvs/mini-sglang/bin/python \
  benchmark/dry/plot.py benchmark-results/dry/<result>.csv
```

Use `--output` to choose a different JPEG path.

`benchmark-results/` is ignored by Git because timings are machine-specific. For an upstream pull request, include the exact command and a representative table in the PR description or this README rather than committing all raw runs.

## CSV schema

Each result row contains:

- run metadata: timestamp, Git revision, GPU, Torch version, Triton version, and Torch CUDA version;
- case parameters: sweep, strategy, repetition profile, batch size, context length, and vocabulary size;
- timing parameters: Python and Triton warmup/repetition counts;
- measurements: Python-reference latency, Triton latency, Python-reference speedup, GPU speedup versus Triton naive, and maximum numerical error.

Rows are flushed after every completed case, so results that finished before an interruption remain available.
The Git revision is suffixed with `-dirty` when tracked or untracked working-tree changes are present.

## Interpretation

The Python reference receives GPU-resident tensors and performs Python-level scalar reads. Its timing therefore includes host synchronization overhead. `python_speedup` is diagnostic only and must not be used to rank GPU strategies.

Use `triton_ms` for absolute GPU latency and `gpu_speedup_vs_naive` for a normalized comparison with a common denominator:

```text
gpu_speedup_vs_naive = Triton-naive latency / strategy Triton latency
```

A value above 1 means the strategy is faster than Triton naive; a value below 1 means it is slower. When one non-naive strategy is selected, the benchmark times a hidden Triton-naive baseline for the same case so this ratio remains comparable.
