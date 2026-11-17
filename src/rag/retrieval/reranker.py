"""Cross-encoder reranker using HuggingFace AutoModelForSequenceClassification."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

# Disable hf_xet to prevent adaptive throttling issues with HuggingFace Hub downloads
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from rag.domain.entities import RetrievedChunk

if TYPE_CHECKING:
    from rag.config import Settings


def resolve_reranker_device(
    settings: Settings | None = None,
    explicit_device: str | None = None,
) -> str:
    """Determine the optimal compute device (cuda -> mps -> cpu) or use explicit override."""
    if explicit_device is not None:
        return explicit_device

    if settings is not None and settings.reranker_device is not None:
        return settings.reranker_device

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


class CrossEncoderReranker:
    """Cross-encoder reranker scoring (query, chunk) pairs using transformer models."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str | None = None,
        batch_size: int = 32,
        rerank_candidates: int = 20,
        max_length: int = 512,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.device = resolve_reranker_device(explicit_device=device)
        self.batch_size = batch_size
        self.rerank_candidates = rerank_candidates
        self.max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Score retrieved chunks against query using cross-encoder and return top_k."""
        if not chunks or top_k <= 0 or not query.strip():
            return []

        candidates = chunks[: self.rerank_candidates]
        pairs = [[query, chunk.text] for chunk in candidates]

        all_scores: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            inputs = self._tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits.view(-1)
                scores_tensor = torch.sigmoid(logits)
                scores = scores_tensor.cpu().tolist()
                if isinstance(scores, float):
                    all_scores.append(scores)
                else:
                    all_scores.extend(scores)

        scored: list[tuple[RetrievedChunk, float]] = list(zip(candidates, all_scores, strict=True))
        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        selected = scored[:top_k]

        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                page=c.page,
                section=c.section,
                score=score,
                source="reranked",
            )
            for c, score in selected
        ]
