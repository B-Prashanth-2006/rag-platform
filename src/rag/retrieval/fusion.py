"""Reciprocal Rank Fusion (RRF) implementation."""

from __future__ import annotations

import uuid

from rag.domain.entities import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists of chunks using Reciprocal Rank Fusion.

    score(d) = sum_{i} w_i / (k + rank_{i}(d)), where rank starts at 1.
    Ties are broken deterministically by chunk_id ascending.
    Output source is always "fused".
    """
    if not ranked_lists:
        return []

    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError(
            f"Length of weights ({len(weights)}) must match "
            f"length of ranked_lists ({len(ranked_lists)})"
        )

    effective_weights = weights if weights is not None else [1.0] * len(ranked_lists)

    fused_scores: dict[uuid.UUID, float] = {}
    chunk_prototypes: dict[uuid.UUID, RetrievedChunk] = {}

    for lst, weight in zip(ranked_lists, effective_weights, strict=True):
        for rank, chunk in enumerate(lst, start=1):
            score_contrib = weight * (1.0 / (k + rank))
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + score_contrib
            if chunk.chunk_id not in chunk_prototypes:
                chunk_prototypes[chunk.chunk_id] = chunk

    # Sort descending by score, then ascending by chunk_id for deterministic tie-breaking
    sorted_chunk_ids = sorted(
        fused_scores.keys(),
        key=lambda cid: (-fused_scores[cid], cid),
    )

    fused_chunks: list[RetrievedChunk] = []
    for cid in sorted_chunk_ids:
        proto = chunk_prototypes[cid]
        fused_chunks.append(
            RetrievedChunk(
                chunk_id=proto.chunk_id,
                document_id=proto.document_id,
                text=proto.text,
                page=proto.page,
                section=proto.section,
                score=fused_scores[cid],
                source="fused",
            )
        )

    return fused_chunks
