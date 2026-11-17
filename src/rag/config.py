"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "dev"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
    api_key: str = "dev-secret-key"
    # Database connection-pool tuning
    db_pool_size: int = 5
    db_max_overflow: int = 10
    # Ingestion limits
    max_upload_mb: int = 50
    # Chunking settings
    tokenizer_model: str = "BAAI/bge-small-en-v1.5"
    chunker_strategy: str = "fixed"
    chunk_size: int = 256
    chunk_overlap: int = 32
    # Embedding settings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str | None = None
    embedding_batch_size: int = 32
    embedding_cache_enabled: bool = True
    embedding_cache_path: str = "data/embeddings_cache.db"
    # Vector store settings
    vector_store_backend: str = "faiss"
    vector_dimension: int = 384
    vector_index_dir: str = "data/indexes"
    faiss_index_type: str = "flat"
    faiss_hnsw_m: int = 32
    faiss_hnsw_ef_search: int = 64
    faiss_hnsw_ef_construction: int = 64
    # Retrieval settings
    retrieval_mode: str = "hybrid"
    use_reranker: bool = False
    candidate_k: int = 50
    fusion_k: int = 10
    rerank_candidates: int = 20
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str | None = None
    reranker_batch_size: int = 32
    # Generation settings
    llm_provider: str = "ollama"
    llm_timeout: float = 60.0
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    context_token_budget: int = 3000
    min_score_reranked: float = 0.5
    min_score_dense: float = 0.0
    generation_refusal_text: str = "I could not find this in the provided documents."


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()
