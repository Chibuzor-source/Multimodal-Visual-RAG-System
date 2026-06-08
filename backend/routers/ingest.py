"""Ingestion API routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.get("/health")
def ingest_health() -> dict[str, str]:
    """Return a health check for the ingestion router."""
    return {"status": "ok"}
