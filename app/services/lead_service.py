"""Lead service — reads the recovery queue and its latest call state."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import CallSession, Lead
from app.services.script_service import script_service


def list_leads(db: Session) -> list[Lead]:
    return list(db.execute(select(Lead).order_by(Lead.id)).scalars().all())


def get_lead(db: Session, lead_id: str) -> Lead | None:
    return db.get(Lead, lead_id)


def latest_session(db: Session, lead_id: str) -> CallSession | None:
    return db.execute(
        select(CallSession)
        .where(CallSession.lead_id == lead_id)
        .order_by(CallSession.started_at.desc(), CallSession.id.desc())
        .limit(1)
    ).scalars().first()


def lead_with_context(db: Session, lead: Lead) -> dict[str, Any]:
    """Lead row plus everything the console needs to decide what to show."""
    session = latest_session(db, lead.id)
    return {
        "lead": lead,
        "latest_session": session,
        "resume_step": script_service.resume_step_after(lead.last_completed_step),
        "scenario_notes": script_service.scenario_notes_for_lead(lead.id),
        "scripted_turns": script_service.scripted_turns_for_lead(lead.id),
        "preexisting_fields": script_service.preexisting_fields_for_lead(lead.id),
    }
