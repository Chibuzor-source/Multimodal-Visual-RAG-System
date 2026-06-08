"""Pydantic models shared across the backend."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Metadata describing an indexed document."""

    doc_id: str
    doc_name: str
    file_path: str
    uploaded_at: datetime
    chunk_count: int
    image_count: int
    table_count: int
    status: Literal["processing", "ready", "error"]


class Chunk(BaseModel):
    """A text, table, or image chunk and its embedding."""

    chunk_id: str
    doc_id: str
    page: int
    chunk_index: int
    type: Literal["text", "image", "table"]
    content: str
    embedding: list[float] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    """A persisted query-response exchange."""

    conversation_id: str
    doc_ids: list[str]
    query: str
    response: str
    sources: list[dict[str, Any]]
    timestamp: datetime


class ParsedTextChunk(BaseModel):
    """A parsed text or table chunk ready for embedding."""

    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    chunk_index: int
    type: Literal["text", "table"]
    content: str


class ParsedImageChunk(BaseModel):
    """A parsed image chunk ready for CLIP embedding."""

    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    chunk_index: int
    type: Literal["image"] = "image"
    content: str
    bbox: list[float]
    width: int
    height: int


class ParsedPdf(BaseModel):
    """All extracted content from a PDF."""

    text_chunks: list[ParsedTextChunk]
    image_chunks: list[ParsedImageChunk]
    table_chunks: list[ParsedTextChunk]
