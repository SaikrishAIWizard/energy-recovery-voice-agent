"""Dashboard summary — the operator's view of the recovery queue.

Also surfaces the two numbers the hackathon judges care about: how much manual typing the
agent avoided, and how many calls ended in a handoff instead of a wrong answer.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import CallSession, FieldSource, FieldStatus, JourneyField, Lead
from app.schemas import DashboardCounts, DashboardRow, DashboardSummary

router = APIRouter(tags=["dashboard"])

# Rough operator-effort model used for the efficiency headline. A human agent reading the
# script, typing each answer and re-keying it into the journey takes ~45s per field.
SECONDS_PER_FIELD_MANUAL = 45.0


@router.get("/dashboard/summary", response_model=DashboardSummary)
def summary(db: Session = Depends(get_db)) -> DashboardSummary:
    leads = db.execute(select(Lead).order_by(Lead.id)).scalars().all()

    sessions_by_lead: dict[str, CallSession] = {}
    for session in db.execute(
        select(CallSession).order_by(CallSession.started_at, CallSession.id)
    ).scalars().all():
        sessions_by_lead[session.lead_id] = session

    counts = DashboardCounts()
    rows: list[DashboardRow] = []

    for lead in leads:
        session = sessions_by_lead.get(lead.id)
        status = session.status if session else "NOT_CALLED"

        if session is None:
            counts.dropped_off += 1
        elif status == "ACTIVE":
            counts.active_calls += 1
        elif status == "COMPLETED":
            counts.completed += 1
        elif status == "HANDOFF_REQUESTED":
            counts.handoffs += 1
        elif status == "DNC_BLOCKED":
            counts.dnc_blocked += 1
        elif status == "DECLINED":
            counts.declined += 1
        else:
            counts.dropped_off += 1

        rows.append(
            DashboardRow(
                lead_id=lead.id,
                name=f"{lead.first_name} {lead.last_name or ''}".strip(),
                phone=lead.phone,
                last_completed_step=lead.last_completed_step,
                dnc_status=lead.dnc_status,
                call_status=status,
                call_session_id=session.id if session else None,
                outcome_detail=session.outcome_detail if session else None,
                handoff_reason=session.handoff_reason if session else None,
                started_at=session.started_at if session else None,
                ended_at=session.ended_at if session else None,
            )
        )

    calls_placed = db.execute(select(func.count(CallSession.id))).scalar_one()
    resolved = counts.completed + counts.handoffs + counts.declined

    auto_fields = db.execute(
        select(func.count(JourneyField.id)).where(
            JourneyField.status == FieldStatus.VALID.value,
            JourneyField.source == FieldSource.CUSTOMER_SPOKEN.value,
        )
    ).scalar_one()

    return DashboardSummary(
        counts=counts,
        rows=rows,
        completion_rate=round(counts.completed / resolved, 3) if resolved else 0.0,
        handoff_rate=round(counts.handoffs / resolved, 3) if resolved else 0.0,
        calls_placed=calls_placed,
        fields_captured_automatically=auto_fields,
        estimated_manual_minutes_saved=round(auto_fields * SECONDS_PER_FIELD_MANUAL / 60, 1),
    )
