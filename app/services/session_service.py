"""Session serialisation — turns ORM state into the single payload the console renders."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.models import (
    CallSession,
    FieldSource,
    FieldStatus,
    JourneyField,
    Speaker,
    TranscriptSegment,
)
from app.schemas import (
    AuditEventOut,
    CallSessionOut,
    HandoffOut,
    JourneyFieldOut,
    LeadOut,
    StepProgress,
    SubmissionOut,
    TranscriptSegmentOut,
)
from app.services import handoff_service
from app.services.script_service import script_service

STEP_LABELS = {
    "recording_consent": "Recording consent",
    "customer_continue": "Continue journey",
    "property_address": "Service address",
    "move_in_date": "Move-in date",
    "energy_requirement": "Supply needed",
    "concession_status": "Concession",
    "life_support": "Life support",
    "contact_preference": "Contact preference",
    "confirmation": "Confirm & submit",
}

DATA_FIELDS = [
    "property_address",
    "move_in_date",
    "energy_requirement",
    "concession_status",
    "life_support",
    "contact_preference",
]


def _progress(session: CallSession) -> list[StepProgress]:
    order = script_service.resume_order
    rows = {row.field_name: row for row in session.journey_fields}
    terminal = session.state in {
        "COMPLETED",
        "DECLINED",
        "HANDOFF_REQUESTED",
        "DNC_BLOCKED",
        "INCOMPLETE",
    }
    resume_index = order.index(session.resume_step) if session.resume_step in order else 0

    progress: list[StepProgress] = []
    for index, step_id in enumerate(order):
        step = script_service.step(step_id) or {}
        field_name = step.get("field_name", step_id)
        row = rows.get(field_name)

        if step_id == "recording_consent":
            status = "DONE" if session.recording_consent_disclosed else "PENDING"
        elif step_id == "confirmation":
            if session.state == "COMPLETED":
                status = "DONE"
            elif session.current_step == "confirmation" and not terminal:
                status = "ACTIVE"
            else:
                status = "PENDING"
        elif row is not None and row.status == FieldStatus.VALID.value:
            status = "DONE"
        elif row is not None and row.status == FieldStatus.INVALID.value and terminal:
            status = "FAILED"
        elif index < resume_index:
            # Already completed on an earlier visit — not re-asked on this call.
            status = "SKIPPED"
        else:
            status = "PENDING"

        if step_id == session.current_step and not terminal and status not in {"DONE", "SKIPPED"}:
            status = "ACTIVE"

        progress.append(
            StepProgress(step_id=step_id, label=STEP_LABELS.get(step_id, step_id), status=status)
        )
    return progress


def serialize_session(db: Session, session: CallSession) -> CallSessionOut:
    rows: list[JourneyField] = list(session.journey_fields)
    transcript: list[TranscriptSegment] = list(session.transcript)

    collected: dict[str, object] = {}
    known: dict[str, object] = {}
    for row in rows:
        if row.value is None:
            continue
        known[row.field_name] = row.value
        if row.status == FieldStatus.VALID.value and row.field_name in DATA_FIELDS:
            collected[row.field_name] = row.value

    missing = [name for name in DATA_FIELDS if collected.get(name) in (None, "")]

    safety_flags: list[str] = []
    handoff_out: HandoffOut | None = None
    if session.handoff is not None:
        payload = handoff_service.serialize_handoff(session.handoff, session.lead)
        handoff_out = HandoffOut(**payload)
        safety_flags = list(payload["safety_flags"])

    submission_out: SubmissionOut | None = None
    if session.submission is not None:
        submission_out = SubmissionOut(
            submission_id=session.submission.submission_id,
            lead_id=session.submission.lead_id,
            vertical=session.submission.vertical,
            status=session.submission.status,
            payload=json.loads(session.submission.payload_json),
            created_at=session.submission.created_at,
            call_session_id=session.submission.call_session_id,
        )

    last_agent = next(
        (s.text for s in reversed(transcript) if s.speaker == Speaker.AI_AGENT.value), None
    )
    last_customer = next(
        (s.text for s in reversed(transcript) if s.speaker == Speaker.CUSTOMER.value), None
    )

    return CallSessionOut(
        id=session.id,
        lead_id=session.lead_id,
        status=session.status,
        state=session.state,
        current_step=session.current_step,
        resume_step=session.resume_step,
        started_at=session.started_at,
        ended_at=session.ended_at,
        recording_consent_disclosed=session.recording_consent_disclosed,
        handoff_reason=session.handoff_reason,
        outcome_detail=session.outcome_detail,
        mode=session.mode,
        dial_provider=session.dial_provider,
        telephony_reference=session.telephony_reference,
        lead=LeadOut.model_validate(session.lead),
        journey_fields=[JourneyFieldOut.model_validate(row) for row in rows],
        transcript=[TranscriptSegmentOut.model_validate(seg) for seg in transcript],
        audit_events=[AuditEventOut.model_validate(event) for event in session.audit_events],
        handoff=handoff_out,
        submission=submission_out,
        journey_progress=_progress(session),
        collected_fields=collected,
        known_fields=known,
        missing_required_fields=missing,
        safety_flags=safety_flags,
        last_agent_message=last_agent,
        last_customer_message=last_customer,
        demo_replies=script_service.demo_replies(session.current_step),
    )


def preexisting_source(row: JourneyField) -> bool:
    return row.source == FieldSource.PREEXISTING.value
