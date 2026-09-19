"""Lead endpoints — the recovery queue."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import LeadDetailOut, LeadQueueItem, LeadOut
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
