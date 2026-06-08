"""Document API routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/health")
def documents_health() -> dict[str, str]:
    """Return a health check for the documents router."""
    return {"status": "ok"}
