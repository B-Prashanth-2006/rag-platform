"""Benchmark script for measuring embedding throughput and latency."""

from __future__ import annotations

import argparse
import time

from rag.embeddings.hf import HuggingFaceEmbedder, resolve_embedding_device


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


def run_benchmark(
    n: int = 100,
    batch_size: int = 32,
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str | None = None,
) -> None:
    resolved_device = resolve_embedding_device(explicit_device=device)
    print(f"Loading embedder {model_name} on {resolved_device}...")
    embedder = HuggingFaceEmbedder(
        model_name=model_name,
        device=resolved_device,
        batch_size=batch_size,
    )

    passages = _generate_synthetic_passages(n)

    # Warmup
    embedder.embed_passages(passages[: min(4, n)])

    # Timed run per batch
    batch_latencies_ms: list[float] = []
    start_total = time.perf_counter()

    for i in range(0, n, batch_size):
        batch = passages[i : i + batch_size]
        t0 = time.perf_counter()
        embedder.embed_passages(batch)
        t1 = time.perf_counter()
        batch_latencies_ms.append((t1 - t0) * 1000.0)

    total_time_s = time.perf_counter() - start_total
    texts_per_second = n / total_time_s if total_time_s > 0 else 0.0

    sorted_latencies = sorted(batch_latencies_ms)
    p95_index = min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)
    p95_latency_ms = sorted_latencies[p95_index] if sorted_latencies else 0.0

    print("\n### Embedding Benchmark Results\n")
    print("| Metric | Value |")
    print("| --- | --- |")
    print(f"| Model | {model_name} |")
    print(f"| Device | {resolved_device} |")
    print(f"| Total Passages | {n} |")
    print(f"| Batch Size | {batch_size} |")
    print(f"| Total Time (s) | {total_time_s:.2f} |")
    print(f"| Texts / Second | {texts_per_second:.1f} |")
    print(f"| p95 Batch Latency (ms) | {p95_latency_ms:.2f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding models.")
    parser.add_argument("--n", type=int, default=100, help="Number of synthetic passages")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="Model name")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    args = parser.parse_args()

    run_benchmark(
        n=args.n,
        batch_size=args.batch_size,
        model_name=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
