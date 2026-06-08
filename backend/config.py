"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the backend service."""

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="multimodal-rag", alias="PINECONE_INDEX_NAME")
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")
    vector_store: Literal["chroma", "pinecone"] = Field(default="chroma", alias="VECTOR_STORE")
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    clip_model: str = Field(default="ViT-L-14", alias="CLIP_MODEL")
    clip_pretrained: str = Field(default="openai", alias="CLIP_PRETRAINED")
    text_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="TEXT_EMBEDDING_MODEL",
    )
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_streaming: bool = Field(default=True, alias="LLM_STREAMING")
    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def cors_origin_list(self) -> list[str]:
        """Return configured CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
