"""Uploaded call recordings.

A finished call recording goes in; a transcript, a checklist verdict and a queue placement
come out:

    audio -> server-side STT (speaker-labelled where the vendor supports it)
          -> safety screen on the customer's side of the call
          -> the six journey fields, extracted and validated
          -> COMPLETED   every field found: the journey is submitted
             INCOMPLETE  something missing: the lead stays in the recovery queue
             DECLINED / HANDOFF_REQUESTED  the customer said something the live agent
                          would have stopped on, so a human decides, not this code

Nothing here is new policy. Fields go through the same extractor and Python validators as
a live call, the same 0.80 confidence floor applies, the safety engine screens every
customer segment, and the journey is submitted through the same mock endpoint. The
transcript is untrusted input exactly like a typed reply — a vendor cannot talk its way
past the guardrails.

The checklist is the six required journey fields. Values already on file for the lead, or
captured by an earlier incomplete recording, count: a second upload can supply what the
first one missed. Recording disclosure is reported but does not gate the outcome.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import MIN_FIELD_CONFIDENCE
from app.conversation import safety_engine
from app.conversation.field_extractor import build_known_context, extract_field
from app.conversation.state_machine import ConversationEngine
from app.conversation.validators import validate_field
from app.models import (
    RECORDING_UPLOAD_MODE,
    AuditEvent,
    CallSession,
    FieldSource,
    FieldStatus,
    Handoff,
    JourneyField,
    Lead,
    LeadStatus,
    SessionStatus,
    Speaker,
    TranscriptSegment,
    utcnow,
)
from app.services import journey_service, lead_service
from app.services.handoff_service import build_context_summary
from app.services.llm_service import get_llm
from app.services.redaction_service import redact_text
from app.services.script_service import script_service
from app.services.voice_service import voice_providers

logger = logging.getLogger(__name__)

# Cap on transcript text handed to the LLM in one request.
_LLM_TRANSCRIPT_CHARS = 6000


class RecordingError(Exception):
    """A problem the caller should see, carrying the HTTP status it maps to."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class RecordingResult:
    session: CallSession
    outcome: str  # COMPLETED | INCOMPLETE | DECLINED | HANDOFF_REQUESTED
    missing_fields: list[str] = field(default_factory=list)
    stt_provider: str = ""
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Transcript segments and speaker roles
# --------------------------------------------------------------------------- #


@dataclass
class Segment:
    speaker_key: str | None
    start: float
    end: float
    text: str
    confidence: float | None
    role: Speaker = Speaker.CUSTOMER
    redacted: bool = False


def _segments_from_stt(result: dict[str, Any]) -> list[Segment]:
    segments: list[Segment] = []
    for raw in result.get("segments") or []:
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        key = raw.get("speaker")
        segments.append(
            Segment(
                speaker_key=None if key is None else str(key),
                start=float(raw.get("start") or 0.0),
                end=float(raw.get("end") or 0.0),
                text=text,
                confidence=raw.get("confidence"),
            )
        )
    if segments:
        return segments

    # No speaker labels (or no vendor support): one block of text.
    text = (result.get("text") or "").strip()
    if not text:
        return []
    return [Segment(None, 0.0, 0.0, text, result.get("confidence"))]


_SPLIT_YEAR = re.compile(r"\b(19|20)\s(\d{2})\b")

# What the agent says that a customer would not: consent disclosure, script openers and
# the read-back.
_AGENT_CUE = re.compile(
    r"call (is|may be|will be) (being )?recorded|energy recovery|may i help you|"
    r"pick up where you left off|comparing energy plans|read that back|"
    r"have i got (all of that|those details) right",
    re.IGNORECASE,
)
_CONSENT = re.compile(
    r"\b(call|conversation)\b.{0,40}\brecord(ed|ing)\b|\brecord(ed|ing)\b.{0,40}\b(call|conversation)\b",
    re.IGNORECASE,
)
_READBACK = re.compile(
    r"read that back|have i got (all|those)|service address:|move-in date:", re.IGNORECASE
)

# The topic of an agent question, in priority order when a line mentions several.
_TOPICS: list[tuple[str, re.Pattern[str]]] = [
    ("life_support", re.compile(r"life[- ]?support", re.IGNORECASE)),
    ("concession_status", re.compile(r"concession", re.IGNORECASE)),
    ("energy_requirement", re.compile(r"\b(electricity|gas)\b", re.IGNORECASE)),
    (
        "contact_preference",
        re.compile(r"\b(phone or (by )?email|contact you|by phone|prefer)\b", re.IGNORECASE),
    ),
    ("move_in_date", re.compile(r"\b(moving in|moving into|move[- ]in|what date)\b", re.IGNORECASE)),
    ("property_address", re.compile(r"\baddress\b", re.IGNORECASE)),
]


def _question_topic(text: str) -> str | None:
    return next((name for name, pattern in _TOPICS if pattern.search(text)), None)


def _agent_score(segment: Segment) -> int:
    score = 2 if _AGENT_CUE.search(segment.text) else 0
    if "?" in segment.text and _question_topic(segment.text):
        score += 1
    return score


def assign_roles(segments: list[Segment]) -> bool:
    """Label each segment agent or customer. Returns True when an agent side was found.

    With a single voice everyone is treated as the customer. With several, the agent is
    the speaker who sounds like the script; if nobody does, the first to speak. Getting it
    wrong fails safe: a customer read as the agent means fields go unfound and the lead is
    queued for recovery, never a false completion.
    """
    keys = list(dict.fromkeys(s.speaker_key for s in segments))
    if len(keys) < 2:
        return False

    scores = {k: sum(_agent_score(s) for s in segments if s.speaker_key == k) for k in keys}
    agent_key = max(keys, key=lambda k: (scores[k], -keys.index(k)))
    ai = any(
        "energy recovery" in s.text.lower() for s in segments if s.speaker_key == agent_key
    )
    for s in segments:
        if s.speaker_key == agent_key:
            s.role = Speaker.AI_AGENT if ai else Speaker.HUMAN_AGENT
    _fix_diarization_slips(segments)
    return True


def _fix_diarization_slips(segments: list[Segment]) -> None:
    """A one-word reply ("No.") is often tagged as the agent's voice by diarization.

    An agent does not answer its own question, so a short non-question segment straight
    after a question from the same speaker is the customer's reply.
    """
    for previous, current in zip(segments, segments[1:]):
        if (
            current.role is not Speaker.CUSTOMER
            and current.speaker_key == previous.speaker_key
            and previous.role is not Speaker.CUSTOMER
            and previous.text.rstrip().endswith("?")
            and "?" not in current.text
            and len(current.text.split()) <= 3
        ):
            current.role = Speaker.CUSTOMER


def _question_answer_pairs(segments: list[Segment]) -> list[tuple[str, str]]:
    """(field, everything the customer said before the agent spoke again)."""
    pairs: list[tuple[str, str]] = []
    topic: str | None = None
    answer: list[str] = []

    def flush() -> None:
        if topic and answer:
            pairs.append((topic, " ".join(answer)))

    for segment in segments:
        if segment.role is Speaker.CUSTOMER:
            if topic:
                answer.append(segment.text)
            continue
        flush()
        answer = []
        topic = None if _READBACK.search(segment.text) else _question_topic(segment.text)
    flush()
    return pairs


def _dialogue(segments: list[Segment], has_agent: bool) -> str:
    if not has_agent:
        text = " ".join(s.text for s in segments)
    else:
        text = "\n".join(
            f"{'CUSTOMER' if s.role is Speaker.CUSTOMER else 'AGENT'}: {s.text}" for s in segments
        )
    return text[:_LLM_TRANSCRIPT_CHARS]


# --------------------------------------------------------------------------- #
# Lead helpers
# --------------------------------------------------------------------------- #


def next_lead_id(db: Session) -> str:
    highest = 1000
    for lead_id in db.execute(select(Lead.id)).scalars().all():
        match = re.fullmatch(r"E-(\d+)", lead_id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"E-{highest + 1}"


def create_lead(db: Session, first_name: str, phone: str, email: str) -> Lead:
    lead = Lead(
        id=next_lead_id(db),
        first_name=first_name.strip(),
        last_name=None,
        phone=phone.strip(),
        email=email.strip(),
        last_completed_step="customer_continue",
        dnc_status=False,
        status=LeadStatus.DROPPED_OFF.value,
    )
    db.add(lead)
    db.flush()
    return lead


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


def process_recording(
    db: Session,
    *,
    audio: bytes,
    mime_type: str,
    filename: str | None,
    lead: Lead | None = None,
    new_lead: dict[str, str] | None = None,
) -> RecordingResult:
    """Transcribe, check the checklist, and place the lead.

    Pass either an existing `lead` or the details for a `new_lead`. The lead (and its
    session) are only written after transcription succeeds, so a failed upload leaves no
    trace behind.
    """
    assert (lead is None) != (new_lead is None), "pass exactly one of lead / new_lead"

    provider = voice_providers.active_stt
    stt = voice_providers.transcribe(audio, mime_type, recording=True)
    segments = _segments_from_stt(stt)
    if not segments:
        raise RecordingError(
            422,
            f"{provider} returned no transcript: {stt.get('error') or 'no speech found'}",
        )

    if lead is None:
        assert new_lead is not None
        lead = create_lead(db, **new_lead)
    carried = lead_service.carried_fields(db, lead.id)

    now = utcnow()
    session = CallSession(
        id=f"CS-{lead.id}-{int(now.timestamp() * 1000) % 1_000_000:06d}",
        lead_id=lead.id,
        status=SessionStatus.INCOMPLETE.value,
        current_step="property_address",
        state="INCOMPLETE",
        resume_step="property_address",
        started_at=now,
        mode=RECORDING_UPLOAD_MODE,
        dial_provider="recording_upload",
    )
    db.add(session)
    db.flush()

    def audit(event_type: str, detail: str = "") -> None:
        db.add(AuditEvent(call_session_id=session.id, event_type=event_type, event_detail=detail))

    audit(
        "RECORDING_UPLOADED",
        json.dumps(
            {
                "filename": filename or "unnamed",
                "bytes": len(audio),
                "mime_type": mime_type,
                "stt_provider": stt.get("provider") or provider,
                "new_lead": new_lead is not None,
            }
        ),
    )

    # Journey rows: contact details, values on file, anything an earlier recording found,
    # and every remaining field PENDING.
    ConversationEngine(db).seed_journey_fields(session, lead, carried)
    if carried:
        audit(
            "RECORDING_FIELDS_CARRIED",
            "Already captured from an earlier uploaded recording: " + ", ".join(sorted(carried)),
        )

    # --- Roles, redaction, safety ------------------------------------------- #
    has_agent = assign_roles(segments)
    audit(
        "TRANSCRIPT_CREATED",
        json.dumps(
            {
                "segments": len(segments),
                "speakers": len({s.speaker_key for s in segments}),
                "agent_side_identified": has_agent,
                "duration_seconds": round(max(s.end for s in segments)),
            }
        ),
    )

    stop: safety_engine.SafetyVerdict | None = None
    stop_text = ""
    for segment in segments:
        if segment.role is not Speaker.CUSTOMER:
            segment.text, segment.redacted = redact_text(segment.text)
            continue
        verdict = safety_engine.evaluate(segment.text)
        segment.text = verdict.safe_text or segment.text
        segment.redacted = verdict.redacted
        if verdict.redacted:
            audit(
                "CARD_DATA_REDACTED",
                "Card-like sequence detected in customer speech and redacted before storage. "
                f"Fingerprint: {verdict.card_fingerprint or 'unavailable'}",
            )
        for flag in verdict.flags:
            audit("SAFETY_SIGNAL", flag)
        if stop is None and (verdict.is_decline or verdict.is_handoff):
            stop = verdict
            stop_text = segment.text

    for segment in segments:
        db.add(
            TranscriptSegment(
                call_session_id=session.id,
                speaker=segment.role.value,
                start_seconds=round(segment.start),
                end_seconds=max(round(segment.start), round(segment.end)),
                text=segment.text,
                transcription_confidence=segment.confidence,
                redacted=segment.redacted,
            )
        )

    consent_pool = [s for s in segments if s.role is not Speaker.CUSTOMER] or segments
    session.recording_consent_disclosed = any(_CONSENT.search(s.text) for s in consent_pool)
    audit(
        "RECORDING_DISCLOSURE_DETECTED"
        if session.recording_consent_disclosed
        else "RECORDING_DISCLOSURE_NOT_DETECTED",
        "Call-recording disclosure "
        + ("heard" if session.recording_consent_disclosed else "not found")
        + " in the transcript.",
    )

    # --- Extract the checklist ---------------------------------------------- #
    _extract_fields(db, session, lead, segments, has_agent, audit)
    db.flush()

    collected = _collected(db, session.id)
    missing = [name for name in script_service.required_fields if name not in collected]
    session.current_step = missing[0] if missing else "confirmation"
    audit(
        "CHECKLIST_EVALUATED",
        json.dumps(
            {name: ("FOUND" if name in collected else "MISSING") for name in script_service.required_fields}
        ),
    )

    # --- Outcome -------------------------------------------------------------- #
    result = RecordingResult(
        session=session,
        outcome=SessionStatus.INCOMPLETE.value,
        missing_fields=missing,
        stt_provider=str(stt.get("provider") or provider),
    )
    life_support_yes = collected.get("life_support") == "YES"

    if stop is not None and stop.is_decline:
        _decline(session, lead, stop, audit)
        result.outcome = SessionStatus.DECLINED.value
        result.notes.append("The customer declined. No journey was submitted.")
    elif stop is not None or life_support_yes:
        _handoff(
            db, session, lead, collected,
            reason=stop.reason if stop else "SENSITIVE_TOPIC",
            signal=stop.escalation_signal if stop else "SENSITIVE",
            flags=(stop.flags if stop else []) + (["LIFE_SUPPORT"] if life_support_yes else []),
            last_message=stop_text,
            note=stop.note if stop else "Life-support equipment declared at the property.",
            consent=session.recording_consent_disclosed,
            audit=audit,
        )
        result.outcome = SessionStatus.HANDOFF_REQUESTED.value
        result.notes.append("A safety signal in the recording needs a human. Nothing was submitted.")
    elif not missing and _submit(db, session, lead, collected, audit):
        result.outcome = SessionStatus.COMPLETED.value
    else:
        _queue_for_recovery(session, lead, missing, audit)
        result.notes.append(
            "Missing from the recording: " + ", ".join(missing) + ". The lead is in the recovery queue."
            if missing
            else "Every field was found but the journey payload failed validation. "
            "The lead is in the recovery queue."
        )

    if not session.recording_consent_disclosed:
        result.notes.append("No recording disclosure was found in the transcript.")

    audit("CALL_ENDED", f"Uploaded recording analysed: outcome={result.outcome}")
    session.ended_at = utcnow()
    db.commit()
    return result


def _collected(db: Session, session_id: str) -> dict[str, Any]:
    rows = db.execute(
        select(JourneyField).where(JourneyField.call_session_id == session_id)
    ).scalars().all()
    return {
        row.field_name: row.value
        for row in rows
        if row.status == FieldStatus.VALID.value
        and row.value is not None
        and row.field_name in script_service.required_fields
    }


def _extract_fields(
    db: Session,
    session: CallSession,
    lead: Lead,
    segments: list[Segment],
    has_agent: bool,
    audit,
) -> None:
    llm = get_llm()
    rows = {
        row.field_name: row
        for row in db.execute(
            select(JourneyField).where(JourneyField.call_session_id == session.id)
        ).scalars()
    }
    steps = {step["field_name"]: step for step in script_service.steps() if step.get("kind") == "FIELD"}
    context = build_known_context(
        lead={
            "first_name": lead.first_name,
            "last_completed_step": lead.last_completed_step,
        },
        collected=_collected(db, session.id),
    )

    def record(name: str, value: str | None, confidence: float, source: str) -> None:
        row = rows[name]
        row.attempts += 1
        row.confidence = confidence
        if value is not None and confidence >= MIN_FIELD_CONFIDENCE:
            row.value = value
            row.status = FieldStatus.VALID.value
            row.source = FieldSource.CUSTOMER_SPOKEN.value
            audit("FIELD_CAPTURED", f"{name}={value} confidence={confidence:.2f} source={source}")
        elif row.status != FieldStatus.VALID.value:
            # Heard, but not clearly enough to trust: reported as unclear, never used.
            row.value = value
            row.status = FieldStatus.INVALID.value
            audit(
                "FIELD_LOW_CONFIDENCE",
                f"{name} candidate={value!r} confidence={confidence:.2f} < {MIN_FIELD_CONFIDENCE}",
            )

    # Pass 1: answers that follow the agent's question about that field.
    attempted: set[str] = set()
    for name, answer in _question_answer_pairs(segments):
        step = steps.get(name)
        if step is None or name not in rows:
            continue
        attempted.add(name)
        if name == "move_in_date":
            # Spoken "twenty twenty-seven" comes back as "20 27".
            answer = _SPLIT_YEAR.sub(r"\1\2", answer)
        extraction = extract_field(
            name, answer, context, validation_type=step["validation_type"], llm=llm
        )
        if extraction.valid:
            record(name, extraction.value, extraction.confidence, extraction.source)
        elif rows[name].status != FieldStatus.VALID.value:
            rows[name].attempts += 1
            rows[name].status = FieldStatus.INVALID.value
            audit("FIELD_VALIDATION_FAILED", f"{name} reason={extraction.reason}")

    # Pass 2: a field nobody was heard asking about (no agent side, or a paraphrased
    # question). Only the LLM can read that, and its candidate is validated in Python and
    # capped by its own confidence, exactly like a live turn.
    if not llm.enabled:
        return
    dialogue = _dialogue(segments, has_agent)
    for name, row in rows.items():
        step = steps.get(name)
        if step is None or name in attempted or row.status == FieldStatus.VALID.value:
            continue
        proposal = llm.extract_field(name, dialogue, context)
        if not proposal or not proposal.value:
            continue
        checked = validate_field(step["validation_type"], proposal.value)
        if checked.ok:
            record(name, checked.value, min(checked.confidence, proposal.confidence), "LLM+RULES")


def _submit(db: Session, session: CallSession, lead: Lead, collected: dict[str, Any], audit) -> bool:
    payload = {"lead_id": lead.id, "vertical": "ENERGY", **collected}
    try:
        submission = journey_service.submit_journey(
            db, payload, call_session_id=session.id, origin="RECORDING_UPLOAD"
        )
    except ValueError as exc:
        audit("SUBMISSION_BLOCKED", str(exc))
        return False

    session.status = SessionStatus.COMPLETED.value
    session.state = "COMPLETED"
    session.current_step = "confirmation"
    lead.status = LeadStatus.COMPLETED.value
    audit(
        "JOURNEY_SUBMITTED",
        f"submission_id={submission.submission_id} origin=RECORDING_UPLOAD "
        f"payload={json.dumps(payload)}",
    )
    return True


def _queue_for_recovery(session: CallSession, lead: Lead, missing: list[str], audit) -> None:
    session.status = SessionStatus.INCOMPLETE.value
    session.state = "INCOMPLETE"
    session.outcome_detail = f"MISSING_{len(missing)}_FIELDS" if missing else "SUBMISSION_BLOCKED"
    lead.status = LeadStatus.DROPPED_OFF.value
    audit(
        "ADDED_TO_RECOVERY_QUEUE",
        f"missing={missing} — a recovery call will resume at the first missing step and "
        "will not re-ask what the recording already captured.",
    )


def _decline(session: CallSession, lead: Lead, verdict, audit) -> None:
    session.status = SessionStatus.DECLINED.value
    session.state = "DECLINED"
    session.outcome_detail = "CUSTOMER_DECLINED"
    lead.status = LeadStatus.DECLINED.value
    audit(
        "CALL_DECLINED",
        f"matched='{verdict.matched}' — refusal respected; no journey submitted, no retry.",
    )


def _handoff(
    db: Session,
    session: CallSession,
    lead: Lead,
    collected: dict[str, Any],
    *,
    reason: str,
    signal: str | None,
    flags: list[str],
    last_message: str,
    note: str,
    consent: bool,
    audit,
) -> None:
    handoff_fields = {name: collected.get(name) for name in script_service.required_fields}
    summary = build_context_summary(
        lead={
            "id": lead.id,
            "first_name": lead.first_name,
            "last_name": lead.last_name,
            "phone": lead.phone,
            "email": lead.email,
            "last_completed_step": lead.last_completed_step,
        },
        resume_step=session.resume_step,
        last_completed_step=lead.last_completed_step,
        collected=collected,
        reason=reason,
        current_step=session.current_step,
    )
    if not consent:
        summary += " (Uploaded recording: no recording disclosure was found in the transcript.)"

    db.add(
        Handoff(
            call_session_id=session.id,
            reason=reason,
            current_step=session.current_step,
            context_summary=summary,
            last_customer_message=last_message or None,
            safety_flags_json=json.dumps(sorted(set(flags))),
            collected_fields_json=json.dumps(handoff_fields),
        )
    )
    session.status = SessionStatus.HANDOFF_REQUESTED.value
    session.state = "HANDOFF_REQUESTED"
    session.handoff_reason = reason
    session.outcome_detail = signal
    lead.status = LeadStatus.HANDOFF_REQUESTED.value
    audit("HANDOFF_CREATED", f"reason={reason} signal={signal} :: {note}")
