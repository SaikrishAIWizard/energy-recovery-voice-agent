"""Lead service — reads the recovery queue and its latest call state."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    CallSession,
    FieldStatus,
    Lead,
    SessionStatus,
)
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


def sessions_for_lead(db: Session, lead_id: str) -> list[CallSession]:
    """Every call ever made for the lead, newest first."""
    return list(
        db.execute(
            select(CallSession)
            .where(CallSession.lead_id == lead_id)
            .order_by(CallSession.started_at.desc(), CallSession.id.desc())
        ).scalars()
    )


def carried_fields(db: Session, lead_id: str) -> dict[str, str]:
    """Journey fields an earlier, incomplete attempt already captured for this lead.

    That is an uploaded recording with items missing, or a phone call the customer hung up
    on part-way. It only applies while that attempt is the lead's most recent session: once
    a newer call has started it is no longer the freshest source of truth, so stale values
    are never silently carried into another call.
    """
    session = latest_session(db, lead_id)
    if session is None or session.status != SessionStatus.INCOMPLETE.value:
        return {}
    return {
        row.field_name: row.value
        for row in session.journey_fields
        if row.status == FieldStatus.VALID.value
        and row.value
        and row.field_name in script_service.data_fields
    }


def resume_step_for(db: Session, lead: Lead) -> str:
    """Where the next recovery call for this lead starts."""
    return script_service.skip_known(
        script_service.resume_step_after(lead.last_completed_step),
        set(carried_fields(db, lead.id)),
    )


def lead_with_context(db: Session, lead: Lead) -> dict[str, Any]:
    """Lead row plus everything the console needs to decide what to show."""
    session = latest_session(db, lead.id)
    return {
        "lead": lead,
        "latest_session": session,
        "resume_step": resume_step_for(db, lead),
        "scenario_notes": script_service.scenario_notes_for_lead(lead.id),
        "scripted_turns": script_service.scripted_turns_for_lead(lead.id),
        "preexisting_fields": script_service.preexisting_fields_for_lead(lead.id),
    }
