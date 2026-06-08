"""PDF parsing utilities for text, image, and table extraction."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Iterable

import fitz
import pdfplumber
from PIL import Image

from backend.models.schemas import ParsedImageChunk, ParsedPdf, ParsedTextChunk

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_AREA = 20_000


def _encode_tokens(text: str) -> list[int]:
    """Encode text into token ids using tiktoken when available."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.encode(text)
    except Exception:
        return text.split()


def _decode_tokens(tokens: list[int] | list[str]) -> str:
    """Decode token ids into text using tiktoken when available."""
    if not tokens:
        return ""
    if isinstance(tokens[0], str):
        return " ".join(tokens)
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.decode(tokens)  # type: ignore[arg-type]
    except Exception:
        return " ".join(str(token) for token in tokens)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into token chunks with overlap."""
    stripped = text.strip()
    if not stripped:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = _encode_tokens(stripped)
    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        token_window = tokens[start : start + chunk_size]
        chunk = _decode_tokens(token_window).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
    return chunks


def table_to_markdown(table: list[list[object | None]]) -> str:
    """Convert a pdfplumber table into a GitHub-flavored markdown table."""
    cleaned = [["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row] for row in table]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    normalized = [row + [""] * (width - len(row)) for row in cleaned]
    header = normalized[0]
    separator = ["---"] * width
    rows = normalized[1:]

    def render_row(row: Iterable[str]) -> str:
        """Render one markdown table row."""
        return "| " + " | ".join(row) + " |"

    return "\n".join([render_row(header), render_row(separator), *[render_row(row) for row in rows]])


def _stable_chunk_id(prefix: str, doc_id: str, page: int, index: int, content: str) -> str:
    """Build a deterministic chunk id from chunk metadata and content."""
    digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{doc_id}_{page}_{index}_{digest}"


def extract_text_chunks(pdf_path: str | Path, doc_id: str, doc_name: str) -> list[ParsedTextChunk]:
    """Extract tokenized text chunks from a PDF using PyMuPDF."""
    chunks: list[ParsedTextChunk] = []
    with fitz.open(str(pdf_path)) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text")
            for page_chunk_index, content in enumerate(chunk_text(page_text)):
                chunk_index = len(chunks)
                chunks.append(
                    ParsedTextChunk(
                        chunk_id=_stable_chunk_id("txt", doc_id, page_number, page_chunk_index, content),
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page=page_number,
                        chunk_index=chunk_index,
                        type="text",
                        content=content,
                    )
                )
    return chunks


def _image_is_informative(width: int, height: int) -> bool:
    """Return whether an image passes the minimum figure-size thresholds."""
    return width >= MIN_WIDTH and height >= MIN_HEIGHT and (width * height) >= MIN_AREA


def extract_images(pdf_path: str | Path, doc_id: str, doc_name: str, output_dir: str | Path) -> list[ParsedImageChunk]:
    """Extract informative PDF images with PyMuPDF and save them to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images: list[ParsedImageChunk] = []

    with fitz.open(str(pdf_path)) as document:
        for page_number, page in enumerate(document, start=1):
            for image_index, image_info in enumerate(page.get_images(full=True)):
                xref = image_info[0]
                base_image = document.extract_image(xref)
                width = int(base_image.get("width", 0))
                height = int(base_image.get("height", 0))
                if not _image_is_informative(width, height):
                    continue

                extension = base_image.get("ext", "png")
                image_bytes = base_image["image"]
                chunk_index = len(images)
                image_id = _stable_chunk_id("img", doc_id, page_number, image_index, str(xref))
                image_file = output_path / f"{image_id}.{extension}"
                image_file.write_bytes(image_bytes)

                rects = page.get_image_rects(xref)
                bbox = list(rects[0]) if rects else [0.0, 0.0, float(width), float(height)]
                images.append(
                    ParsedImageChunk(
                        chunk_id=image_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page=page_number,
                        chunk_index=chunk_index,
                        content=str(image_file),
                        bbox=[float(value) for value in bbox],
                        width=width,
                        height=height,
                    )
                )
    return images


def extract_tables(pdf_path: str | Path, doc_id: str, doc_name: str) -> list[ParsedTextChunk]:
    """Extract tables from a PDF with pdfplumber and convert them to markdown chunks."""
    chunks: list[ParsedTextChunk] = []
    with pdfplumber.open(str(pdf_path)) as document:
        for page_number, page in enumerate(document.pages, start=1):
            for table_index, table in enumerate(page.extract_tables()):
                markdown = table_to_markdown(table)
                if not markdown:
                    continue
                for table_chunk_index, content in enumerate(chunk_text(markdown)):
                    chunk_index = len(chunks)
                    chunks.append(
                        ParsedTextChunk(
                            chunk_id=_stable_chunk_id(
                                "tbl",
                                doc_id,
                                page_number,
                                table_index + table_chunk_index,
                                content,
                            ),
                            doc_id=doc_id,
                            doc_name=doc_name,
                            page=page_number,
                            chunk_index=chunk_index,
                            type="table",
                            content=content,
                        )
                    )
    return chunks


def parse_pdf(pdf_path: str | Path, doc_id: str | None = None, doc_name: str | None = None, output_dir: str | Path = "uploads/images") -> ParsedPdf:
    """Parse text chunks, image chunks, and table chunks from a PDF."""
    path = Path(pdf_path)
    resolved_doc_id = doc_id or str(uuid.uuid4())
    resolved_doc_name = doc_name or path.name
    text_chunks = extract_text_chunks(path, resolved_doc_id, resolved_doc_name)
    table_chunks = extract_tables(path, resolved_doc_id, resolved_doc_name)
    image_chunks = extract_images(path, resolved_doc_id, resolved_doc_name, output_dir)
    return ParsedPdf(text_chunks=text_chunks, image_chunks=image_chunks, table_chunks=table_chunks)
