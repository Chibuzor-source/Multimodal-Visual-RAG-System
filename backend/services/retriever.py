"""Retrieval helpers for multimodal RAG."""


def reciprocal_rank_fusion(text_results: list[dict], image_results: list[dict], k: int = 60) -> list[tuple[str, float]]:
    """Fuse text and image ranked results with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for rank, result in enumerate(text_results):
        result_id = result["id"]
        scores[result_id] = scores.get(result_id, 0.0) + 1 / (k + rank + 1)
    for rank, result in enumerate(image_results):
        result_id = result["id"]
        scores[result_id] = scores.get(result_id, 0.0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
