"""Embedding services for text and image chunks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image

from backend.config import Settings, get_settings


class TextEmbedder:
    """Create OpenAI embeddings for text chunks."""

    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None) -> None:
        """Initialize the text embedder with settings and an optional OpenAI client."""
        self.settings = settings or get_settings()
        self.client = client or OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.text_embedding_model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class ClipEmbedder:
    """Create CLIP embeddings for images and text queries."""

    def __init__(self, settings: Settings | None = None, device: str | None = None) -> None:
        """Initialize CLIP model, preprocessing, and tokenizer."""
        import open_clip
        import torch

        self.settings = settings or get_settings()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.settings.clip_model,
            pretrained=self.settings.clip_pretrained,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.settings.clip_model)

    @staticmethod
    def _normalize(vector: np.ndarray) -> list[float]:
        """Normalize a numpy vector and return it as a list."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.astype(float).tolist()
        return (vector / norm).astype(float).tolist()

    def embed_image(self, image: str | Path | Image.Image) -> list[float]:
        """Embed one image path or PIL image into CLIP vector space."""
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.open(image).convert("RGB")

        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            vector = self.model.encode_image(image_tensor).squeeze().detach().cpu().numpy()
        return self._normalize(vector)

    def embed_images(self, images: list[str | Path | Image.Image]) -> list[list[float]]:
        """Embed a batch of image paths or PIL images."""
        return [self.embed_image(image) for image in images]

    def embed_query_for_images(self, text: str) -> list[float]:
        """Embed a text query into CLIP vector space for image retrieval."""
        tokens = self.tokenizer([text]).to(self.device)
        with self.torch.no_grad():
            vector = self.model.encode_text(tokens).squeeze().detach().cpu().numpy()
        return self._normalize(vector)
