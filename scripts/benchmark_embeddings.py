"""Benchmark script for measuring embedding throughput and latency across devices."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from rag.embeddings.hf import HuggingFaceEmbedder


def _generate_synthetic_passages(n: int) -> list[str]:
    words = [
        "retrieval",
        "augmented",
        "generation",
        "document",
        "intelligence",
        "vector",
        "database",
        "similarity",
        "search",
        "embedding",
        "pipeline",
        "indexing",
        "tokens",
        "latency",
        "throughput",
        "benchmark",
        "accuracy",
    ]
    passages = []
    for i in range(n):
        # 30-word synthetic passage
        passage = " ".join(words[(i + j) % len(words)] for j in range(30))
        passages.append(passage)
    return passages


def _benchmark_single_device(
    device: str,
    passages: list[str],
    batch_size: int,
    model_name: str,
    repeats: int,
) -> dict[str, str | int | float]:
    print(
        f"Benchmarking {device} (model={model_name}, N={len(passages)}, "
        f"batch_size={batch_size}, repeats={repeats})..."
    )
    embedder = HuggingFaceEmbedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    # 1. Untimed warm-up batch
    warmup_batch = passages[:batch_size]
    embedder.embed_passages(warmup_batch)

    runs_throughput: list[float] = []
    runs_p95: list[float] = []
    runs_total_time: list[float] = []

    # 2. Time N passages, repeat and calculate median
    for _ in range(repeats):
        batch_latencies_ms: list[float] = []
        t_start = time.perf_counter()

        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            b_start = time.perf_counter()
            embedder.embed_passages(batch)
            b_end = time.perf_counter()
            batch_latencies_ms.append((b_end - b_start) * 1000.0)

        total_time = time.perf_counter() - t_start
        throughput = len(passages) / total_time if total_time > 0 else 0.0

        sorted_latencies = sorted(batch_latencies_ms)
        p95_idx = min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)
        p95_ms = sorted_latencies[p95_idx] if sorted_latencies else 0.0

        runs_throughput.append(throughput)
        runs_p95.append(p95_ms)
        runs_total_time.append(total_time)

    return {
        "device": device,
        "model": model_name,
        "passages": len(passages),
        "batch_size": batch_size,
        "texts_per_sec": statistics.median(runs_throughput),
        "p95_ms": statistics.median(runs_p95),
        "total_time_s": statistics.median(runs_total_time),
    }


def run_benchmark(
    n: int = 1000,
    batch_size: int = 32,
    model_name: str = "BAAI/bge-small-en-v1.5",
    repeats: int = 3,
    devices: list[str] | None = None,
) -> None:
    if devices is None:
        devices = []
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
        elif torch.cuda.is_available():
            devices.append("cuda")
        if "cpu" not in devices:
            devices.append("cpu")

    passages = _generate_synthetic_passages(n)
    results: list[dict[str, str | int | float]] = []

    for device in devices:
        res = _benchmark_single_device(
            device=device,
            passages=passages,
            batch_size=batch_size,
            model_name=model_name,
            repeats=repeats,
        )
        results.append(res)

    print("\n### Embedding Benchmark Results (Median of 3 Runs)\n")
    print(
        "| Device | Model | Passages | Batch Size | Median Texts/sec | "
        "Median p95 Latency (ms) | Median Total Time (s) |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        print(
            f"| {r['device']} | {r['model']} | {r['passages']} | {r['batch_size']} | "
            f"{float(r['texts_per_sec']):.1f} | {float(r['p95_ms']):.2f} | "
            f"{float(r['total_time_s']):.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding models.")
    parser.add_argument("--n", type=int, default=1000, help="Number of synthetic passages")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="Model name")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions for median")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Devices to compare (e.g. mps cpu)",
    )
    args = parser.parse_args()

    run_benchmark(
        n=args.n,
        batch_size=args.batch_size,
        model_name=args.model,
        repeats=args.repeats,
        devices=args.devices,
    )


if __name__ == "__main__":
    main()
