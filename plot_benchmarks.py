#!/usr/bin/env python3
"""Plot CPU and GPU clustering times from create_benchmarks CSV output."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_benchmarks(csv_path: Path) -> tuple[list[int], list[float], list[int], list[float]]:
    cpu_points: list[int] = []
    cpu_times: list[float] = []
    gpu_points: list[int] = []
    gpu_times: list[float] = []

    with csv_path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                point_count = int(row["total_cloud_points"])
                cpu_time = float(row["cpu_clustering_ms"])
            except (KeyError, TypeError, ValueError):
                continue

            cpu_points.append(point_count)
            cpu_times.append(cpu_time)

            gpu_time = (row.get("gpu_clustering_ms") or "").strip()
            if gpu_time:
                gpu_points.append(point_count)
                gpu_times.append(float(gpu_time))

    if not cpu_points:
        raise ValueError("No numeric benchmark rows found in the CSV file.")

    return cpu_points, cpu_times, gpu_points, gpu_times


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clustering benchmark results.")
    parser.add_argument("csv_path", type=Path, help="CSV output from create_benchmarks")
    parser.add_argument("--output", type=Path, default=Path("benchmark_times.png"), help="Output PNG path")
    args = parser.parse_args()

    cpu_points, cpu_times, gpu_points, gpu_times = read_benchmarks(args.csv_path)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(cpu_points, cpu_times, marker="o", label="CPU clustering")
    axis.plot(gpu_points, gpu_times, marker="o", label="GPU V3 clustering")
    axis.set_xlabel("Total cloud points")
    axis.set_ylabel("Clustering time (ms)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="both", linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
