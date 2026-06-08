"""Query API routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/query", tags=["query"])


@router.get("/health")
def query_health() -> dict[str, str]:
    """Return a health check for the query router."""
    return {"status": "ok"}
