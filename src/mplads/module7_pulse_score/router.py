"""API route for Module 7 -- Development Pulse Score."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from mplads.module7_pulse_score.exceptions import EntityNotFoundError
from mplads.module7_pulse_score.schemas import PulseScoreResponse
from mplads.module7_pulse_score.service import get_pulse_score

router = APIRouter(prefix="/api/v1", tags=["module7-pulse-score"])


@router.get(
    "/pulse-score/{constituency_or_mp_id}",
    response_model=PulseScoreResponse,
    responses={404: {"description": "Unknown constituency/MP, or a Rajya Sabha MP with no attributable works."}},
)
def get_pulse_score_endpoint(constituency_or_mp_id: str) -> PulseScoreResponse:
    try:
        return get_pulse_score(constituency_or_mp_id)
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "entity_not_found", "reason": str(exc)}) from exc
