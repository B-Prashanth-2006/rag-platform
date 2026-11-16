"""Benchmark script comparing FAISS (Flat & HNSW) and Chroma vector stores."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import chromadb
import faiss
import numpy as np

from rag.domain.entities import VectorRecord
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.faiss_store import FaissVectorStore


def _generate_normalized_vectors(
    count: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    raw = rng.standard_normal((count, dim), dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    res: np.ndarray = np.asarray(raw / np.maximum(norms, 1e-9), dtype=np.float32)
    return res


def _generate_clustered_dataset(
    count: int,
    num_queries: int,
    dim: int,
    n_clusters: int,
    rng: np.random.Generator,
    cluster_std: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    # 1. Cluster centers normalized to unit hypersphere
    centers = rng.standard_normal((n_clusters, dim), dtype=np.float32)
    center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / np.maximum(center_norms, 1e-9)

    # 2. Clustered data vectors
    cluster_ids = rng.integers(0, n_clusters, size=count)
    noise = rng.normal(0.0, cluster_std, size=(count, dim)).astype(np.float32)
    points = centers[cluster_ids] + noise
    points_norm = np.linalg.norm(points, axis=1, keepdims=True)
    vecs: np.ndarray = np.asarray(points / np.maximum(points_norm, 1e-9), dtype=np.float32)

    # 3. Clustered query vectors centered around cluster centers
    q_clusters = rng.integers(0, n_clusters, size=num_queries)
    q_noise = rng.normal(0.0, cluster_std, size=(num_queries, dim)).astype(np.float32)
    queries = centers[q_clusters] + q_noise
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries_norm: np.ndarray = np.asarray(queries / np.maximum(q_norms, 1e-9), dtype=np.float32)

    return vecs, queries_norm


def _benchmark_dataset(
    dataset_name: str,
    vecs_np: np.ndarray,
    queries_np: np.ndarray,
    dim: int,
    k: int,
    faiss_ef_search_values: list[int],
    chroma_ef_construction: int,
    chroma_ef_search: int,
    temp_base_dir: Path,
) -> list[dict[str, Any]]:
    doc_id = uuid.uuid4()
    records: list[VectorRecord] = [
        VectorRecord(
            chunk_id=uuid.UUID(int=i + 1),
            vector=v.tolist(),
            document_id=doc_id,
            metadata={"idx": i},
        )
        for i, v in enumerate(vecs_np)
    ]
    queries = [q.tolist() for q in queries_np]

    results: list[dict[str, Any]] = []

    # 1. FAISS Flat (Ground Truth)
    print(f"[{dataset_name}] Benchmarking FAISS-Flat (Ground Truth)...")
    flat_dir = temp_base_dir / f"faiss_flat_{dataset_name.lower().replace(' ', '_')}"
    flat_store = FaissVectorStore(dimension=dim, index_dir=flat_dir, index_type="flat")

    t0 = time.perf_counter()
    flat_store.add(records)
    flat_build_time = time.perf_counter() - t0

    ground_truth: list[set[uuid.UUID]] = []
    flat_latencies: list[float] = []

    for q in queries:
        t_start = time.perf_counter()
        hits = flat_store.search(q, k=k)
        flat_latencies.append((time.perf_counter() - t_start) * 1000.0)
        ground_truth.append({h.chunk_id for h in hits})

    flat_sorted = sorted(flat_latencies)
    flat_p95 = flat_sorted[int(len(flat_sorted) * 0.95)] if flat_sorted else 0.0
    results.append(
        {
            "dataset": dataset_name,
            "backend": "FAISS-Flat",
            "build_time": flat_build_time,
            "mean_latency": float(np.mean(flat_latencies)),
            "p95_latency": flat_p95,
            "recall": 1.0,
        }
    )

    # 2. FAISS HNSW across different efSearch values
    print(f"[{dataset_name}] Building FAISS-HNSW (M=32, efConst=64)...")
    hnsw_dir = temp_base_dir / f"faiss_hnsw_{dataset_name.lower().replace(' ', '_')}"
    hnsw_store = FaissVectorStore(
        dimension=dim,
        index_dir=hnsw_dir,
        index_type="hnsw",
        m=32,
        ef_search=64,
        ef_construction=64,
    )

    t0 = time.perf_counter()
    hnsw_store.add(records)
    hnsw_build_time = time.perf_counter() - t0

    sub_index: Any = faiss.downcast_index(hnsw_store._index.index)

    for ef in faiss_ef_search_values:
        print(f"[{dataset_name}] Benchmarking FAISS-HNSW (efSearch={ef})...")
        sub_index.hnsw.efSearch = ef

        hnsw_latencies: list[float] = []
        hnsw_recalls: list[float] = []

        for q, gt_set in zip(queries, ground_truth, strict=True):
            t_start = time.perf_counter()
            hits = hnsw_store.search(q, k=k)
            hnsw_latencies.append((time.perf_counter() - t_start) * 1000.0)

            hit_set = {h.chunk_id for h in hits}
            overlap = len(hit_set & gt_set)
            hnsw_recalls.append(overlap / k if k > 0 else 1.0)

        hnsw_sorted = sorted(hnsw_latencies)
        hnsw_p95 = hnsw_sorted[int(len(hnsw_sorted) * 0.95)] if hnsw_sorted else 0.0
        results.append(
            {
                "dataset": dataset_name,
                "backend": f"FAISS-HNSW (efSearch={ef})",
                "build_time": hnsw_build_time,
                "mean_latency": float(np.mean(hnsw_latencies)),
                "p95_latency": hnsw_p95,
                "recall": float(np.mean(hnsw_recalls)),
            }
        )

    # 3. Chroma with explicit search_ef and construction_ef
    chroma_label = f"Chroma (efSearch={chroma_ef_search}, efConst={chroma_ef_construction})"
    print(f"[{dataset_name}] Benchmarking {chroma_label}...")
    chroma_dir = temp_base_dir / f"chroma_{dataset_name.lower().replace(' ', '_')}"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    client.create_collection(
        name="bench_collection",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": chroma_ef_construction,
            "hnsw:search_ef": chroma_ef_search,
        },
    )
    chroma_store = ChromaVectorStore(
        dimension=dim,
        index_dir=chroma_dir,
        collection_name="bench_collection",
    )

    t0 = time.perf_counter()
    batch_size = 5000
    for b in range(0, len(records), batch_size):
        chroma_store.add(records[b : b + batch_size])
    chroma_build_time = time.perf_counter() - t0

    chroma_latencies: list[float] = []
    chroma_recalls: list[float] = []

    for q, gt_set in zip(queries, ground_truth, strict=True):
        t_start = time.perf_counter()
        hits = chroma_store.search(q, k=k)
        chroma_latencies.append((time.perf_counter() - t_start) * 1000.0)

        hit_set = {h.chunk_id for h in hits}
        overlap = len(hit_set & gt_set)
        chroma_recalls.append(overlap / k if k > 0 else 1.0)

    chroma_sorted = sorted(chroma_latencies)
    chroma_p95 = chroma_sorted[int(len(chroma_sorted) * 0.95)] if chroma_sorted else 0.0
    results.append(
        {
            "dataset": dataset_name,
            "backend": chroma_label,
            "build_time": chroma_build_time,
            "mean_latency": float(np.mean(chroma_latencies)),
            "p95_latency": chroma_p95,
            "recall": float(np.mean(chroma_recalls)),
        }
    )

    return results


def _print_table(title: str, results: list[dict[str, Any]], k: int) -> None:
    print(f"\n### {title}\n")
    print(
        f"| Backend / Index | Build Time (s) | Mean Latency (ms) | p95 Latency (ms) | Recall@{k} |"
    )
    print("| --- | --- | --- | --- | --- |")
    for r in results:
        print(
            f"| {r['backend']} | {float(r['build_time']):.2f} | "
            f"{float(r['mean_latency']):.2f} | {float(r['p95_latency']):.2f} | "
            f"{float(r['recall']) * 100.0:.1f}% |"
        )


def run_benchmark(
    n: int = 20000,
    dim: int = 384,
    num_queries: int = 200,
    k: int = 10,
    seed: int = 42,
    n_clusters: int = 200,
    faiss_ef_search_values: list[int] | None = None,
    chroma_ef_construction: int = 64,
    chroma_ef_search: int = 64,
) -> None:
    if faiss_ef_search_values is None:
        faiss_ef_search_values = [16, 64, 128, 256]

    print(
        f"Generating benchmark datasets (N={n}, dim={dim}, queries={num_queries}, "
        f"clusters={n_clusters}, seed={seed})..."
    )
    rng_random = np.random.default_rng(seed)
    random_vecs = _generate_normalized_vectors(n, dim, rng_random)
    random_queries = _generate_normalized_vectors(num_queries, dim, rng_random)

    rng_clustered = np.random.default_rng(seed + 1)
    clustered_vecs, clustered_queries = _generate_clustered_dataset(
        count=n,
        num_queries=num_queries,
        dim=dim,
        n_clusters=n_clusters,
        rng=rng_clustered,
    )

    bench_dir = Path(tempfile.mkdtemp(prefix="rag_vectorstore_bench_"))

    try:
        # Run Random dataset benchmark
        random_results = _benchmark_dataset(
            dataset_name="Random Vectors",
            vecs_np=random_vecs,
            queries_np=random_queries,
            dim=dim,
            k=k,
            faiss_ef_search_values=faiss_ef_search_values,
            chroma_ef_construction=chroma_ef_construction,
            chroma_ef_search=chroma_ef_search,
            temp_base_dir=bench_dir,
        )

        # Run Clustered dataset benchmark
        clustered_results = _benchmark_dataset(
            dataset_name=f"Clustered Vectors ({n_clusters} Clusters)",
            vecs_np=clustered_vecs,
            queries_np=clustered_queries,
            dim=dim,
            k=k,
            faiss_ef_search_values=faiss_ef_search_values,
            chroma_ef_construction=chroma_ef_construction,
            chroma_ef_search=chroma_ef_search,
            temp_base_dir=bench_dir,
        )

        # Output individual tables
        params_str = f"N={n}, dim={dim}, queries={num_queries}, k={k}"
        _print_table(
            f"Vector Store Benchmark Results — Random Vectors ({params_str})",
            random_results,
            k,
        )
        _print_table(
            f"Vector Store Benchmark Results — Clustered Vectors\n"
            f"({n_clusters} Clusters, {params_str})",
            clustered_results,
            k,
        )

        # Output combined summary table
        all_results = random_results + clustered_results
        print(f"\n### Combined Comparison Table ({params_str})\n")
        header = (
            f"| Dataset | Backend / Index | Build Time (s) | Mean Latency (ms) | "
            f"p95 Latency (ms) | Recall@{k} |"
        )
        print(header)
        print("| --- | --- | --- | --- | --- | --- |")
        for r in all_results:
            print(
                f"| {r['dataset']} | {r['backend']} | {float(r['build_time']):.2f} | "
                f"{float(r['mean_latency']):.2f} | {float(r['p95_latency']):.2f} | "
                f"{float(r['recall']) * 100.0:.1f}% |"
            )

    finally:
        shutil.rmtree(bench_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vector stores.")
    parser.add_argument("--n", type=int, default=20000, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=384, help="Vector dimension")
    parser.add_argument("--queries", type=int, default=200, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Top k neighbors")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clusters", type=int, default=200, help="Number of Gaussian clusters")
    parser.add_argument(
        "--chroma-ef-construction",
        type=int,
        default=64,
        help="Chroma HNSW construction ef",
    )
    parser.add_argument(
        "--chroma-ef-search",
        type=int,
        default=64,
        help="Chroma HNSW search ef",
    )
    args = parser.parse_args()

    run_benchmark(
        n=args.n,
        dim=args.dim,
        num_queries=args.queries,
        k=args.k,
        seed=args.seed,
        n_clusters=args.clusters,
        chroma_ef_construction=args.chroma_ef_construction,
        chroma_ef_search=args.chroma_ef_search,
    )


if __name__ == "__main__":
    main()
