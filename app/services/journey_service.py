"""Mock journey-completion service.

Stands in for the real Energy journey endpoint. It validates the payload with the same
Pydantic contract the API exposes, refuses to submit anything incomplete, and issues a
receipt so the console can prove a journey was really completed.

Idempotency is scoped to the **call session**, not the lead. Re-submitting the same call
returns the same receipt (safe against retries), but a second call for the same lead gets
its own receipt — otherwise re-running the demo would orphan the new session's submission.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import JourneySubmission, utcnow
from app.schemas import JourneySubmitRequest, JourneySubmitResponse


def _base_submission_id(lead_id: str) -> str:
    """E-1001 -> SUB-1001, matching the spec's example."""
    suffix = lead_id.split("-")[-1] if "-" in lead_id else lead_id
    return f"SUB-{suffix}"


def next_submission_id(db: Session, lead_id: str) -> str:
    """First submission for a lead uses the spec's exact shape; later ones are suffixed."""
    base = _base_submission_id(lead_id)
    if db.get(JourneySubmission, base) is None:
        return base
    counter = 2
    while db.get(JourneySubmission, f"{base}-{counter}") is not None:
        counter += 1
    return f"{base}-{counter}"


def existing_for_session(db: Session, call_session_id: str) -> JourneySubmission | None:
    return db.execute(
        select(JourneySubmission).where(JourneySubmission.call_session_id == call_session_id)
    ).scalars().first()


def submit_journey(
    db: Session,
    payload: dict[str, Any],
    *,
    call_session_id: str | None = None,
    origin: str = "AI_VOICE_AGENT",
) -> JourneySubmission:
    """Validate + persist. Raises `ValueError` on an invalid payload."""
    try:
        validated = JourneySubmitRequest(**payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid journey payload: {exc.errors()}") from exc

    session_id = call_session_id or validated.call_session_id
    if session_id:
        already = existing_for_session(db, session_id)
        if already is not None:
            return already

    clean_payload = validated.model_dump(exclude={"call_session_id"})
    record = JourneySubmission(
        submission_id=next_submission_id(db, validated.lead_id),
        call_session_id=session_id,
        lead_id=validated.lead_id,
        vertical=validated.vertical,
        payload_json=json.dumps(clean_payload),
        status="COMPLETED",
        created_at=utcnow(),
    )
    db.add(record)
    db.flush()
    return record


def to_response(record: JourneySubmission) -> JourneySubmitResponse:
    return JourneySubmitResponse(
        success=True,
        submission_id=record.submission_id,
        status=record.status,
        payload=json.loads(record.payload_json),
    )


def failure_response(errors: list[str]) -> JourneySubmitResponse:
    return JourneySubmitResponse(success=False, status="REJECTED", errors=errors)
