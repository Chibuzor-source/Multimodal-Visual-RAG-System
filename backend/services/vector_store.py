"""ChromaDB vector-store wrapper for multimodal retrieval."""

from __future__ import annotations

from typing import Any, Literal

import chromadb

from backend.config import Settings, get_settings
from backend.models.schemas import ParsedImageChunk, ParsedTextChunk

TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "image_chunks"


class ChromaVectorStore:
    """Manage text and image Chroma collections."""

    def __init__(self, settings: Settings | None = None, persist_dir: str | None = None) -> None:
        """Initialize persistent Chroma client and required collections."""
        self.settings = settings or get_settings()
        self.client = chromadb.PersistentClient(path=persist_dir or self.settings.chroma_persist_dir)
        self.text_collection = self.client.get_or_create_collection(
            name=TEXT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_collection = self.client.get_or_create_collection(
            name=IMAGE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _text_metadata(chunk: ParsedTextChunk) -> dict[str, str | int]:
        """Build Chroma metadata for a text or table chunk."""
        return {
            "doc_id": chunk.doc_id,
            "doc_name": chunk.doc_name,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
            "type": chunk.type,
            "content": chunk.content,
        }

    @staticmethod
    def _image_metadata(chunk: ParsedImageChunk) -> dict[str, str | int | float]:
        """Build Chroma metadata for an image chunk."""
        x0, y0, x1, y1 = chunk.bbox
        return {
            "doc_id": chunk.doc_id,
            "doc_name": chunk.doc_name,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
            "type": chunk.type,
            "content": chunk.content,
            "bbox_x0": x0,
            "bbox_y0": y0,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "width": chunk.width,
            "height": chunk.height,
        }

    def add_text_chunks(self, chunks: list[ParsedTextChunk], embeddings: list[list[float]]) -> None:
        """Add embedded text and table chunks to the text collection."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self.text_collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._text_metadata(chunk) for chunk in chunks],
        )

    def add_image_chunks(self, chunks: list[ParsedImageChunk], embeddings: list[list[float]]) -> None:
        """Add embedded image chunks to the image collection."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self.image_collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._image_metadata(chunk) for chunk in chunks],
        )

    def query_text(self, query_embedding: list[float], top_k: int = 8, doc_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Query the text collection by embedding."""
        return self._query_collection(self.text_collection, query_embedding, top_k, doc_ids)

    def query_images(self, query_embedding: list[float], top_k: int = 8, doc_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Query the image collection by embedding."""
        return self._query_collection(self.image_collection, query_embedding, top_k, doc_ids)

    def delete_document(self, doc_id: str) -> None:
        """Delete all text and image vectors for a document."""
        where = {"doc_id": doc_id}
        self.text_collection.delete(where=where)
        self.image_collection.delete(where=where)

    @staticmethod
    def _where_doc_ids(doc_ids: list[str] | None) -> dict[str, Any] | None:
        """Build a Chroma where clause for optional document filtering."""
        if not doc_ids:
            return None
        if len(doc_ids) == 1:
            return {"doc_id": doc_ids[0]}
        return {"doc_id": {"$in": doc_ids}}

    def _query_collection(
        self,
        collection: Any,
        query_embedding: list[float],
        top_k: int,
        doc_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Query a Chroma collection and normalize results into dictionaries."""
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=self._where_doc_ids(doc_ids),
            include=["documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        normalized: list[dict[str, Any]] = []
        for item_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            normalized.append(
                {
                    "id": item_id,
                    "document": document,
                    "metadata": metadata or {},
                    "distance": distance,
                    "score": 1.0 / (1.0 + float(distance)),
                }
            )
        return normalized


def get_vector_store(kind: Literal["chroma"] = "chroma") -> ChromaVectorStore:
    """Return the configured vector store implementation."""
    if kind != "chroma":
        raise NotImplementedError("Only ChromaDB is implemented for Phase 1")
    return ChromaVectorStore()
