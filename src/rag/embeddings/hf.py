"""HuggingFace Transformer-based embedding model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from rag.domain.entities import EmbeddingModelInfo

if TYPE_CHECKING:
    from rag.config import Settings


def resolve_embedding_device(
    settings: Settings | None = None,
    explicit_device: str | None = None,
) -> str:
    """Determine the optimal compute device (cuda -> mps -> cpu) or use explicit override."""
    if explicit_device is not None:
        return explicit_device

    if settings is not None and settings.embedding_device is not None:
        return settings.embedding_device

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


class HuggingFaceEmbedder:
    """Generates normalized vector embeddings using HuggingFace transformer models."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 32,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        pooling: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = resolve_embedding_device(explicit_device=device)
        self.batch_size = batch_size

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()

        # Determine model family defaults if not explicitly provided
        name_lower = model_name.lower()
        if "e5" in name_lower:
            self.query_prefix = query_prefix if query_prefix is not None else "query: "
            self.passage_prefix = passage_prefix if passage_prefix is not None else "passage: "
            self.pooling = pooling or "mean"
        else:
            # Default to BGE configuration
            self.query_prefix = (
                query_prefix
                if query_prefix is not None
                else "Represent this sentence for searching relevant passages: "
            )
            self.passage_prefix = passage_prefix if passage_prefix is not None else ""
            self.pooling = pooling or "cls"

        dimension = getattr(self._model.config, "hidden_size", 384)
        tokenizer_max = getattr(self._tokenizer, "model_max_length", 512)
        if tokenizer_max > 8192:
            tokenizer_max = 512
        self.max_tokens = max_tokens or tokenizer_max

        version = getattr(self._model.config, "_commit_hash", None) or "unknown"
        self._info = EmbeddingModelInfo(
            name=self.model_name,
            version=version,
            dimension=dimension,
            max_tokens=self.max_tokens,
        )

    @property
    def info(self) -> EmbeddingModelInfo:
        """Return model metadata."""
        return self._info

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        inputs: dict[str, Any] = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            token_embeddings = outputs[0]

            if self.pooling == "cls":
                embeddings = token_embeddings[:, 0]
            else:  # mean pooling
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

            # L2 normalize so cosine similarity equals dot product
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        result: list[list[float]] = embeddings.cpu().tolist()
        return result

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embedding vectors for passages."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            prefixed = [f"{self.passage_prefix}{t}" for t in batch]
            all_vectors.extend(self._embed_batch(prefixed))

        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        """Generate a normalized embedding vector for a query."""
        prefixed = f"{self.query_prefix}{text}"
        vectors = self._embed_batch([prefixed])
        return vectors[0]
