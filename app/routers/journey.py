"""Mock journey-completion endpoint.

This is the sandbox stand-in for the real Energy journey API. Same payload shape, same
success envelope, and it refuses anything incomplete — so a "completed" journey in the
console always means a genuinely valid payload was accepted.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import JourneySubmission, Lead, LeadStatus, utcnow
from app.schemas import JourneySubmitRequest, JourneySubmitResponse, SubmissionOut
from app.services import journey_service

router = APIRouter(tags=["journey"])


@router.post("/journey/submit", response_model=JourneySubmitResponse)
def submit_journey(
    body: JourneySubmitRequest, db: Session = Depends(get_db)
) -> JourneySubmitResponse:
    lead = db.get(Lead, body.lead_id)
    if lead is None:
        raise HTTPException(status_code=404, detail=f"Unknown lead {body.lead_id}")

    payload = body.model_dump(exclude={"call_session_id"})
    try:
        record = journey_service.submit_journey(
            db, payload, call_session_id=body.call_session_id, origin="API"
        )
    except ValueError as exc:
        return journey_service.failure_response([str(exc)])

    # Reflect the outcome back onto the lead so the queue stays truthful.
    lead.status = LeadStatus.COMPLETED.value
    if body.call_session_id:
        from app.models import AuditEvent, CallSession

        call_session = db.get(CallSession, body.call_session_id)
        if call_session is not None:
            call_session.status = "COMPLETED"
            call_session.state = "COMPLETED"
            call_session.ended_at = call_session.ended_at or utcnow()
            db.add(
                AuditEvent(
                    call_session_id=call_session.id,
                    event_type="JOURNEY_SUBMITTED",
                    event_detail=f"submission_id={record.submission_id} origin=API",
                )
            )
    db.commit()
    return journey_service.to_response(record)


@router.get("/journey/submissions", response_model=list[SubmissionOut])
def list_submissions(db: Session = Depends(get_db)) -> list[SubmissionOut]:
    records = (
        db.execute(select(JourneySubmission).order_by(JourneySubmission.created_at.desc()))
        .scalars()
        .all()
    )
    return [
        SubmissionOut(
            submission_id=row.submission_id,
            lead_id=row.lead_id,
            vertical=row.vertical,
            status=row.status,
            payload=json.loads(row.payload_json),
            created_at=row.created_at,
            call_session_id=row.call_session_id,
        )
        for row in records
    ]
