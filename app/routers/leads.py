"""Lead endpoints — the recovery queue."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import FieldSource, FieldStatus
from app.schemas import LeadDetailOut, LeadOut, LeadQueueItem, SessionSummary
from app.services import dnc_service, lead_service, session_service
from app.services.script_service import script_service

router = APIRouter(tags=["leads"])


@router.get("/leads", response_model=list[LeadQueueItem])
def list_leads(db: Session = Depends(get_db)) -> list[LeadQueueItem]:
    items: list[LeadQueueItem] = []
    for lead in lead_service.list_leads(db):
        session = lead_service.latest_session(db, lead.id)
        items.append(
            LeadQueueItem(
                lead=LeadOut.model_validate(lead),
                resume_step=lead_service.resume_step_for(db, lead),
                scenario_notes=script_service.scenario_notes_for_lead(lead.id),
                expected_outcome=script_service.expected_outcome_for_lead(lead.id),
                latest_session_id=session.id if session else None,
                latest_session_status=session.status if session else None,
                outcome_detail=session.outcome_detail if session else None,
                handoff_reason=session.handoff_reason if session else None,
                started_at=session.started_at if session else None,
                ended_at=session.ended_at if session else None,
                can_start=not lead.dnc_status,
            )
        )
    return items


@router.get("/leads/{lead_id}/sessions", response_model=list[SessionSummary])
def lead_sessions(lead_id: str, db: Session = Depends(get_db)) -> list[SessionSummary]:
    """A lead's whole call history. The queue shows only the latest call; this is how an
    earlier one (a handoff, a phone call) stays reachable after a newer call starts."""
    if lead_service.get_lead(db, lead_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown lead {lead_id}")
    return [
        SessionSummary(
            id=session.id,
            status=session.status,
            mode=session.mode,
            dial_provider=session.dial_provider,
            started_at=session.started_at,
            ended_at=session.ended_at,
            outcome_detail=session.outcome_detail,
            handoff_reason=session.handoff_reason,
            transcript_segments=len(session.transcript),
            fields_captured=sum(
                1
                for row in session.journey_fields
                if row.status == FieldStatus.VALID.value
                and row.source != FieldSource.PREEXISTING.value
                and row.field_name in script_service.data_fields
            ),
        )
        for session in lead_service.sessions_for_lead(db, lead_id)
    ]


@router.get("/leads/{lead_id}", response_model=LeadDetailOut)
def get_lead(lead_id: str, db: Session = Depends(get_db)) -> LeadDetailOut:
    lead = lead_service.get_lead(db, lead_id)
    if lead is None:
        raise HTTPException(status_code=404, detail=f"Unknown lead {lead_id}")

    session = lead_service.latest_session(db, lead.id)
    check = dnc_service.check(lead)

    return LeadDetailOut(
        lead=LeadOut.model_validate(lead),
        resume_step=lead_service.resume_step_for(db, lead),
        scenario_notes=script_service.scenario_notes_for_lead(lead.id),
        expected_outcome=script_service.expected_outcome_for_lead(lead.id),
        scripted_turns=script_service.scripted_turns_for_lead(lead.id),
        preexisting_fields=script_service.preexisting_fields_for_lead(lead.id),
        field_prompts=script_service.field_prompts(),
        latest_session=session_service.serialize_session(db, session) if session else None,
        dnc_check={"allowed": check.allowed, "code": check.code, "detail": check.detail},
    )
