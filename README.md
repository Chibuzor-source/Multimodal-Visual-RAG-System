# Multimodal Visual RAG System

Phase 1 foundation for a dual-index multimodal RAG system.

## Phase 1

- PDF text extraction and 512-token chunks with 64-token overlap.
- PDF image extraction through PyMuPDF with small-image filtering.
- PDF table extraction through pdfplumber with markdown conversion.
- OpenAI text embeddings and open_clip ViT-L-14 image embeddings.
- ChromaDB wrapper with separate `text_chunks` and `image_chunks` collections.
