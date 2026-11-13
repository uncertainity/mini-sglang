from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter


Sweep = Literal["context", "batch", "repetition"]

STRATEGY_ORDER = (
    "naive",
    "candidate_parallel",
    "blocked_parallel",
    "z",
    "window",
)
MARKERS = ("o", "s", "^", "D", "P")


@dataclass(frozen=True)
class ResultRow:
    sweep: str
    strategy: str
    profile: str
    batch_size: int
    context_length: int
    triton_ms: float
    gpu_speedup_vs_naive: float | None


def load_results(path: Path) -> tuple[list[ResultRow], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "strategy",
            "profile",
            "batch_size",
            "context_length",
            "triton_ms",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f"No benchmark rows found in {path}")

    metadata_keys = (
        "timestamp",
        "git_commit",
        "device",
        "torch_version",
        "triton_version",
        "torch_cuda_version",
    )
    metadata = {key: raw_rows[0].get(key, "unknown") for key in metadata_keys}
    rows = [
        ResultRow(
            sweep=row.get("sweep", ""),
            strategy=row["strategy"],
            profile=row["profile"],
            batch_size=int(row["batch_size"]),
            context_length=int(row["context_length"]),
            triton_ms=float(row["triton_ms"]),
            gpu_speedup_vs_naive=(
                float(row["gpu_speedup_vs_naive"])
                if row.get("gpu_speedup_vs_naive")
                else None
            ),
        )
        for row in raw_rows
    ]
    return rows, metadata


def _deduplicate(rows: list[ResultRow], *, keep: Literal["first", "last"]) -> list[ResultRow]:
    keyed: dict[tuple[str, str, int, int], ResultRow] = {}
    iterable = rows if keep == "last" else reversed(rows)
    for row in iterable:
        key = (row.strategy, row.profile, row.batch_size, row.context_length)
        keyed[key] = row
    values = list(keyed.values())
    return values if keep == "last" else list(reversed(values))


def split_sweeps(rows: list[ResultRow]) -> tuple[list[ResultRow], list[ResultRow]]:
    if any(row.sweep for row in rows):
        return (
            [row for row in rows if row.sweep == "context"],
            [row for row in rows if row.sweep == "batch"],
        )

    # Legacy CSV files predate the explicit sweep column. The benchmark writes
    # the context sweep first and the batch sweep second. Their fixed point is
    # duplicated, so keep the first copy for context and the last for batch.
    fixed_batch = Counter(row.batch_size for row in rows).most_common(1)[0][0]
    fixed_context = Counter(row.context_length for row in rows).most_common(1)[0][0]
    context_rows = _deduplicate(
        [row for row in rows if row.batch_size == fixed_batch],
        keep="first",
    )
    batch_rows = _deduplicate(
        [row for row in rows if row.context_length == fixed_context],
        keep="last",
    )
    return context_rows, batch_rows


def _plot_panel(
    ax: Axes,
    rows: list[ResultRow],
    *,
    x_field: Literal["context_length", "batch_size"],
    y_field: Literal["triton_ms", "gpu_speedup_vs_naive"],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    if not rows:
        ax.text(0.5, 0.5, "Not included in this run", ha="center", va="center")
        ax.set_title(title)
        ax.set_axis_off()
        return

    grouped: dict[tuple[str, str], list[ResultRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.strategy, row.profile)].append(row)

    profiles = {row.profile for row in rows}
    naive_latency_by_case = {
        (row.profile, row.batch_size, row.context_length): row.triton_ms
        for row in rows
        if row.strategy == "naive"
    }

    def y_value(row: ResultRow) -> float:
        if y_field == "triton_ms":
            return row.triton_ms
        if row.gpu_speedup_vs_naive is not None:
            return row.gpu_speedup_vs_naive
        case_key = (row.profile, row.batch_size, row.context_length)
        try:
            return naive_latency_by_case[case_key] / row.triton_ms
        except KeyError as exc:
            raise ValueError(
                "Cannot derive legacy GPU speedup because the Triton-naive row is missing "
                f"for profile={row.profile}, batch={row.batch_size}, "
                f"context={row.context_length}."
            ) from exc

    strategy_rank = {name: idx for idx, name in enumerate(STRATEGY_ORDER)}
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (strategy_rank.get(item[0][0], len(strategy_rank)), item[0][1]),
    )
    colors = plt.get_cmap("tab10").colors
    for group_idx, ((strategy, profile), group_rows) in enumerate(ordered_groups):
        ordered_rows = sorted(group_rows, key=lambda row: getattr(row, x_field))
        label = strategy if len(profiles) == 1 else f"{strategy} ({profile})"
        ax.plot(
            [getattr(row, x_field) for row in ordered_rows],
            [y_value(row) for row in ordered_rows],
            label=label,
            color=colors[group_idx % len(colors)],
            marker=MARKERS[group_idx % len(MARKERS)],
            linewidth=1.8,
            markersize=5,
        )

    x_values = sorted({getattr(row, x_field) for row in rows})
    if len(x_values) > 1:
        ax.set_xscale("log", base=2)
    ax.set_xticks(x_values)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)


def plot_csv(input_path: Path, output_path: Path | None = None) -> Path:
    rows, metadata = load_results(input_path)
    context_rows, batch_rows = split_sweeps(rows)
    output_path = output_path or input_path.with_suffix(".jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    _plot_panel(
        axes[0, 0],
        context_rows,
        x_field="context_length",
        y_field="triton_ms",
        title="Triton latency vs. context length",
        x_label="Context length (tokens)",
        y_label="Median latency (ms, log scale)",
    )
    _plot_panel(
        axes[0, 1],
        context_rows,
        x_field="context_length",
        y_field="gpu_speedup_vs_naive",
        title="GPU speedup vs. Triton naive by context length",
        x_label="Context length (tokens)",
        y_label="GPU speedup vs. Triton naive (×, log scale)",
    )
    _plot_panel(
        axes[1, 0],
        batch_rows,
        x_field="batch_size",
        y_field="triton_ms",
        title="Triton latency vs. batch size",
        x_label="Batch size",
        y_label="Median latency (ms, log scale)",
    )
    _plot_panel(
        axes[1, 1],
        batch_rows,
        x_field="batch_size",
        y_field="gpu_speedup_vs_naive",
        title="GPU speedup vs. Triton naive by batch size",
        x_label="Batch size",
        y_label="GPU speedup vs. Triton naive (×, log scale)",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=min(5, len(labels)))

    fig.suptitle(
        "DRY penalty benchmark\n"
        f"{metadata['device']} · Torch {metadata['torch_version']} · "
        f"Triton {metadata['triton_version']} · {metadata['git_commit']}",
        fontsize=14,
    )
    fig.savefig(output_path, format="jpg", dpi=180, pil_kwargs={"quality": 95})
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a DRY benchmark CSV as a JPEG summary.")
    parser.add_argument("input", type=Path, help="Benchmark CSV produced by bench.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JPEG output path. Defaults to the input path with a .jpg suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_csv(args.input, args.output)
    print(f"Saved JPEG plot to {output_path}")


if __name__ == "__main__":
    main()
