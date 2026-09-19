"""Deterministic conversation state machine.

This is the only thing allowed to decide anything. The LLM is an optional input adapter;
it never chooses to continue, submit, decline, or escalate.

States
------
INIT  DNC_CHECK  CONSENT_DISCLOSURE  AWAITING_CONTINUE_RESPONSE  COLLECTING_FIELD
VALIDATING_FIELD  CONFIRMING_DETAILS  SUBMITTING_JOURNEY  COMPLETED  DECLINED
HANDOFF_REQUESTED  DNC_BLOCKED

Rules enforced here
-------------------
* Only the approved prompt for the active step may be spoken (or its single fallback).
* One customer answer per field, and at most one clarification question per field.
* Two failed captures of the same field -> handoff(REPEATED_FAILURE).
* Confidence below 0.80 after clarification -> handoff(LOW_CONFIDENCE).
* Unknown required values are never submitted, and never invented.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import MAX_FIELD_ATTEMPTS, MIN_FIELD_CONFIDENCE, settings
from app.conversation import safety_engine
from app.conversation.field_extractor import build_known_context, extract_field
from app.conversation.validators import validate_confirmation, validate_field
from app.models import (
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
from app.services import dnc_service, journey_service
from app.services.handoff_service import build_context_summary
from app.services.llm_service import get_llm
from app.services.script_service import ScriptService, script_service
from app.services.voice_service import voice_providers

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    "COMPLETED",
    "DECLINED",
    "HANDOFF_REQUESTED",
    "DNC_BLOCKED",
}

FIELD_LABELS = {
    "property_address": "Service address",
    "move_in_date": "Move-in date",
    "energy_requirement": "Supply needed",
    "concession_status": "Concession",
    "life_support": "Life support",
    "contact_preference": "Contact preference",
}


@dataclass
class EngineResult:
    session: CallSession
    agent_messages: list[str] = field(default_factory=list)
    system_notes: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    handoff_triggered: bool = False
    journey_submitted: bool = False
    submission_id: str | None = None
    terminal: bool = False
    blocked: bool = False
    blocked_reason: str | None = None


class ConversationEngine:
    """One instance per request. Stateless between calls — all state lives in SQLite."""

    def __init__(self, db: Session, scripts: ScriptService | None = None) -> None:
        self.db = db
        self.scripts = scripts or script_service
        self.llm = get_llm()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _audit(self, session_id: str, event_type: str, detail: str = "") -> None:
        self.db.add(
            AuditEvent(call_session_id=session_id, event_type=event_type, event_detail=detail)
        )

    def _say(self, session: CallSession, text: str, confidence: float = 1.0) -> None:
        """Append an agent turn to the transcript. TTS is the browser's job."""
        last_end = self._last_timestamp(session.id)
        self.db.add(
            TranscriptSegment(
                call_session_id=session.id,
                speaker=Speaker.AI_AGENT.value,
                start_seconds=last_end,
                end_seconds=last_end + max(2, len(text) // 14),
                text=text,
                transcription_confidence=confidence,
            )
        )

    def _hear(
        self,
        session: CallSession,
        text: str,
        confidence: float | None,
        redacted: bool = False,
    ) -> None:
        last_end = self._last_timestamp(session.id)
        self.db.add(
            TranscriptSegment(
                call_session_id=session.id,
                speaker=Speaker.CUSTOMER.value,
                start_seconds=last_end,
                end_seconds=last_end + max(2, len(text) // 14),
                text=text,
                transcription_confidence=confidence,
                redacted=redacted,
            )
        )

    def _last_timestamp(self, session_id: str) -> int:
        last = self.db.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.call_session_id == session_id)
            .order_by(TranscriptSegment.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        return last.end_seconds if last else 0

    def _get_session(self, session_id: str) -> CallSession | None:
        return self.db.get(CallSession, session_id)

    def _field_row(self, session_id: str, field_name: str) -> JourneyField | None:
        return self.db.execute(
            select(JourneyField).where(
                JourneyField.call_session_id == session_id,
                JourneyField.field_name == field_name,
            )
        ).scalars().first()

    def _collected(self, session_id: str) -> dict[str, Any]:
        rows = self.db.execute(
            select(JourneyField).where(JourneyField.call_session_id == session_id)
        ).scalars().all()
        return {
            row.field_name: row.value
            for row in rows
            if row.status == FieldStatus.VALID.value and row.value is not None
        }

    def _lead_dict(self, lead: Lead) -> dict[str, Any]:
        return {
            "id": lead.id,
            "first_name": lead.first_name,
            "last_name": lead.last_name,
            "phone": lead.phone,
            "email": lead.email,
            "last_completed_step": lead.last_completed_step,
        }

    # ------------------------------------------------------------------ #
    # 1. Start a call
    # ------------------------------------------------------------------ #
    def start_call(
        self, lead_id: str, mode: str = "AGENT_DRIVEN", force: bool = False
    ) -> EngineResult:
        lead = self.db.get(Lead, lead_id)
        if lead is None:
            raise LookupError(f"Unknown lead {lead_id}")

        session = CallSession(
            id=f"CS-{lead_id}-{int(datetime.now(timezone.utc).timestamp() * 1000) % 1_000_000:06d}",
            lead_id=lead.id,
            status=SessionStatus.ACTIVE.value,
            current_step="recording_consent",
            state="INIT",
            mode=mode,
        )
        self.db.add(session)
        self.db.flush()
        self._audit(session.id, "CALL_START_REQUESTED", f"lead={lead.id} mode={mode}")

        # -- State: DNC_CHECK. Runs BEFORE any call is placed. ------------- #
        session.state = "DNC_CHECK"
        check = dnc_service.check(lead, force=force)
        self._audit(
            session.id,
            "DNC_CHECK_PASSED" if check.allowed else "DNC_BLOCKED",
            check.detail,
        )

        if not check.allowed:
            session.state = "DNC_BLOCKED"
            session.status = SessionStatus.DNC_BLOCKED.value
            session.outcome_detail = check.code
            session.ended_at = utcnow()
            session.handoff_reason = None
            lead.status = LeadStatus.DNC_BLOCKED.value
            self._audit(session.id, "CALL_ABORTED", "No call was placed. DNC gate blocked dialling.")
            self.db.commit()
            return EngineResult(
                session=session,
                blocked=True,
                blocked_reason=check.code,
                terminal=True,
                system_notes=[
                    f"Blocked by Do-Not-Call check ({check.code}). No call placed, no data collected."
                ],
            )

        # -- Seed journey state, then resume from the last completed step. -- #
        resume_step = self.scripts.resume_step_after(lead.last_completed_step)
        session.resume_step = resume_step
        self._seed_journey_fields(session, lead)
        lead.status = LeadStatus.IN_CALL.value

        self._audit(
            session.id,
            "RESUME_FROM_STEP",
            f"last_completed_step={lead.last_completed_step} -> resume_at={resume_step}",
        )

        # -- Dial through the telephony adapter. DNC has already cleared. --- #
        dial = voice_providers.dial(lead.phone)
        session.dial_provider = dial.get("provider")
        session.telephony_reference = dial.get("call_reference")
        self._audit(session.id, "DIAL_ATTEMPT", json.dumps(dial))

        # -- State: CONSENT_DISCLOSURE. Must precede any data collection. --- #
        session.state = "CONSENT_DISCLOSURE"
        session.current_step = "recording_consent"
        prompt = self.scripts.render_prompt("recording_consent", first_name=lead.first_name)
        self._say(session, prompt)
        session.recording_consent_disclosed = True
        self._audit(
            session.id,
            "RECORDING_CONSENT_DISCLOSED",
            "AU two-party consent disclosure delivered before any data collection.",
        )
        self._audit(
            session.id,
            "CALL_STARTED",
            f"AI agent connected to {lead.phone} via {session.dial_provider}",
        )

        session.state = "AWAITING_CONTINUE_RESPONSE"
        self.db.commit()

        dial_note = (
            f"Dialled through '{session.dial_provider}'."
            if dial.get("placed")
            else f"No telephony credentials — '{session.dial_provider}' ran the call locally "
            "(browser microphone)."
        )
        return EngineResult(
            session=session,
            agent_messages=[prompt],
            system_notes=[
                f"DNC check passed. Resuming after '{lead.last_completed_step}' at '{resume_step}'.",
                dial_note,
                "Recording disclosure delivered — data collection is now permitted.",
            ],
        )

    def _seed_journey_fields(self, session: CallSession, lead: Lead) -> None:
        preexisting = self.scripts.preexisting_fields_for_lead(lead.id)

        # Contact details already on the lead record.
        for name, value in (("email", lead.email), ("phone", lead.phone)):
            self.db.add(
                JourneyField(
                    call_session_id=session.id,
                    field_name=name,
                    value=value,
                    status=FieldStatus.VALID.value,
                    source=FieldSource.PREEXISTING.value,
                    confidence=1.0,
                )
            )
        for name, value in preexisting.items():
            self.db.add(
                JourneyField(
                    call_session_id=session.id,
                    field_name=name,
                    value=value,
                    status=FieldStatus.VALID.value,
                    source=FieldSource.PREEXISTING.value,
                    confidence=1.0,
                )
            )

        # Every remaining data field starts PENDING.
        for step in self.scripts.steps():
            if step.get("kind") != "FIELD" and step.get("step_id") != "confirmation":
                continue
            field_name = step["field_name"]
            if self._field_row(session.id, field_name):
                continue
            self.db.add(
                JourneyField(
                    call_session_id=session.id,
                    field_name=field_name,
                    value=None,
                    status=FieldStatus.PENDING.value,
                    source=FieldSource.CUSTOMER_SPOKEN.value,
                )
            )
        self.db.flush()

    # ------------------------------------------------------------------ #
    # 2. Handle one customer utterance
    # ------------------------------------------------------------------ #
    def handle_utterance(
        self,
        session_id: str,
        text: str,
        source: str = "SIMULATED",
        stt_confidence: float | None = None,
    ) -> EngineResult:
        session = self._get_session(session_id)
        if session is None:
            raise LookupError(f"Unknown call session {session_id}")

        if session.state in TERMINAL_STATES:
            return EngineResult(
                session=session,
                terminal=True,
                system_notes=[f"Call already ended in state {session.state}. Utterance ignored."],
            )

        # --- Safety engine runs BEFORE extraction, on every utterance. -------- #
        # It also owns redaction, so it must see the RAW utterance: redacting first
        # would strip the digits and destroy the very signal we escalate on.
        verdict = safety_engine.evaluate(text)
        safe_text = verdict.safe_text or text
        if verdict.redacted:
            self._audit(
                session.id,
                "CARD_DATA_REDACTED",
                "Card-like sequence detected in customer speech and redacted before storage. "
                f"Fingerprint: {verdict.card_fingerprint or 'unavailable'}",
            )
        self._hear(session, safe_text, stt_confidence, redacted=verdict.redacted)
        self.db.flush()

        for flag in verdict.flags:
            self._audit(session.id, "SAFETY_SIGNAL", flag)

        if verdict.is_decline:
            return self._decline(session, verdict, safe_text)

        if verdict.is_handoff:
            return self._handoff(
                session,
                reason=verdict.reason or "SENSITIVE_TOPIC",
                signal=verdict.escalation_signal,
                flags=verdict.flags,
                last_message=safe_text,
                note=verdict.note,
                extra_messages=verdict.agent_messages,
            )

        # --- Deterministic dispatch ------------------------------------------ #
        if session.state == "AWAITING_CONTINUE_RESPONSE":
            return self._handle_gate(session, safe_text, verdict)

        if session.state in {"COLLECTING_FIELD", "VALIDATING_FIELD"}:
            return self._handle_field(session, safe_text, verdict)

        if session.state == "CONFIRMING_DETAILS":
            return self._handle_confirmation(session, safe_text, verdict)

        return EngineResult(
            session=session,
            system_notes=[f"No handler for state {session.state}."],
        )

    # ------------------------------------------------------------------ #
    # 3. Gate: consent + "may I help you continue?"
    # ------------------------------------------------------------------ #
    def _handle_gate(
        self, session: CallSession, text: str, verdict: safety_engine.SafetyVerdict
    ) -> EngineResult:
        intent = safety_engine.classify_gate_intent(text)
        gate_field = self._field_row(session.id, "continue_intent")
        if gate_field is None:
            gate_field = JourneyField(
                call_session_id=session.id,
                field_name="continue_intent",
                status=FieldStatus.PENDING.value,
                source=FieldSource.CUSTOMER_SPOKEN.value,
            )
            self.db.add(gate_field)
            self.db.flush()
        gate_field.attempts += 1

        if intent == "CONTINUE":
            gate_field.value = "CONTINUE"
            gate_field.status = FieldStatus.VALID.value
            gate_field.confidence = 0.95
            self._audit(session.id, "CONSENT_GIVEN", "Customer agreed to continue the journey.")
            return self._advance_to_resume_step(session)

        if intent == "BUSY":
            gate_field.value = "BUSY"
            gate_field.status = FieldStatus.VALID.value
            callback = "tomorrow between 10am and 12pm"
            message = (
                "No problem at all. I'll arrange one callback for you "
                f"{callback}, and I won't try again before then. Thanks for your time."
            )
            self._say(session, message)
            self._audit(session.id, "CALLBACK_SCHEDULED", f"single_callback_slot={callback}")
            return self._end_call(
                session,
                status=SessionStatus.DECLINED.value,
                outcome_detail="BUSY_CALLBACK_SCHEDULED",
                note="Customer was busy. One callback offered and booked. No further attempts.",
                agent_messages=[message],
            )

        # Unclear
        self._audit(session.id, "GATE_UNCLEAR", f"attempt={gate_field.attempts}")
        if gate_field.attempts < MAX_FIELD_ATTEMPTS:
            fallback = self.scripts.fallback_prompt("recording_consent")
            self._say(session, fallback)
            self._audit(session.id, "CLARIFICATION_REQUESTED", "gate clarification 1/1")
            self.db.commit()
            return EngineResult(
                session=session,
                agent_messages=[fallback],
                safety_flags=verdict.flags,
                system_notes=["Consent reply unclear — one clarification asked."],
            )

        return self._handoff(
            session,
            reason="LOW_CONFIDENCE",
            signal="LOW_CONF",
            flags=verdict.flags + ["LOW_CONFIDENCE"],
            last_message=text,
            note="Customer's response to the consent gate could not be understood twice.",
        )

    def _advance_to_resume_step(self, session: CallSession) -> EngineResult:
        step_id = session.resume_step or "property_address"
        step = self.scripts.step(step_id)
        if step is None:
            step_id = "property_address"
            step = self.scripts.step(step_id)

        session.current_step = step_id
        session.state = "COLLECTING_FIELD"
        prompt = self.scripts.render_prompt(step_id, **self._collected(session.id))
        self._say(session, prompt)
        self._audit(session.id, "STEP_STARTED", step_id)
        self.db.commit()
        return EngineResult(
            session=session,
            agent_messages=[prompt],
            system_notes=[f"Resumed journey at '{step_id}'."],
        )

    # ------------------------------------------------------------------ #
    # 4. Field capture
    # ------------------------------------------------------------------ #
    def _handle_field(
        self, session: CallSession, text: str, verdict: safety_engine.SafetyVerdict
    ) -> EngineResult:
        step = self.scripts.step(session.current_step)
        if step is None:
            return self._handoff(
                session, "OFF_SCRIPT", "OFF-SCRIPT", verdict.flags, text,
                "Active step is not in the approved script library.",
            )

        field_name = step["field_name"]
        vtype = step["validation_type"]
        row = self._field_row(session.id, field_name)
        if row is None:
            row = JourneyField(
                call_session_id=session.id,
                field_name=field_name,
                status=FieldStatus.PENDING.value,
                source=FieldSource.CUSTOMER_SPOKEN.value,
            )
            self.db.add(row)
            self.db.flush()

        row.attempts += 1
        session.state = "VALIDATING_FIELD"
        self.db.flush()

        context = build_known_context(
            lead=self._lead_dict(session.lead),
            collected=self._collected(session.id),
            current_step=field_name,
        )
        extraction = extract_field(
            field_name,
            text,
            context,
            validation_type=vtype,
            llm=self.llm,
        )
        self._audit(
            session.id,
            "FIELD_ATTEMPT",
            json.dumps(
                {
                    "step": session.current_step,
                    "field": field_name,
                    "attempt": row.attempts,
                    "extraction": extraction.diagnostic(),
                    "llm": self.llm.name,
                }
            ),
        )

        # --- Valid and confident enough ------------------------------------ #
        if extraction.valid and extraction.confidence >= MIN_FIELD_CONFIDENCE:
            row.value = extraction.value
            row.status = FieldStatus.VALID.value
            row.source = FieldSource.CUSTOMER_SPOKEN.value
            row.confidence = extraction.confidence
            self._audit(
                session.id,
                "FIELD_CAPTURED",
                f"{field_name}={extraction.value} confidence={extraction.confidence:.2f} "
                f"source={extraction.source}",
            )

            # life_support = YES is a hard stop: vulnerable customer.
            if field_name == "life_support" and extraction.value == "YES":
                self.db.flush()
                return self._handoff(
                    session,
                    reason=step.get("handoff_on_yes", "SENSITIVE_TOPIC"),
                    signal="SENSITIVE",
                    flags=verdict.flags + ["LIFE_SUPPORT"],
                    last_message=text,
                    note="Life-support equipment declared at the property. "
                         "Agent must not advise; a human takes over.",
                    extra_messages=[
                        "Thank you for telling me — that's important. Because life-support "
                        "equipment is involved, I'm going to bring in a specialist now so "
                        "nothing gets missed. They'll have everything you've told me."
                    ],
                )

            return self._advance_from(session, step, extraction)

        # --- Valid but below the confidence floor -------------------------- #
        if extraction.valid and extraction.confidence < MIN_FIELD_CONFIDENCE:
            row.value = extraction.value
            row.status = FieldStatus.INVALID.value
            row.confidence = extraction.confidence
            self._audit(
                session.id,
                "FIELD_LOW_CONFIDENCE",
                f"{field_name} candidate='{extraction.value}' confidence="
                f"{extraction.confidence:.2f} < {MIN_FIELD_CONFIDENCE}",
            )
            if row.attempts >= MAX_FIELD_ATTEMPTS:
                return self._handoff(
                    session,
                    reason="LOW_CONFIDENCE",
                    signal="LOW_CONF",
                    flags=verdict.flags + ["LOW_CONFIDENCE"],
                    last_message=text,
                    note=f"Confidence for '{field_name}' stayed below "
                         f"{MIN_FIELD_CONFIDENCE} after clarification.",
                )
            return self._clarify(session, step, row, "low_confidence")

        # --- Not extracted at all ------------------------------------------ #
        row.status = FieldStatus.INVALID.value
        self._audit(
            session.id,
            "FIELD_VALIDATION_FAILED",
            f"{field_name} attempt={row.attempts} reason={extraction.reason}",
        )

        # A question that isn't about this field is off-script, not a failed capture.
        if (
            row.attempts == 1
            and safety_engine.is_question_like(text)
            and not safety_engine.is_meta_question(text)
            and not safety_engine.question_targets_field(text, field_name)
        ):
            return self._handoff(
                session,
                reason="OFF_SCRIPT",
                signal="OFF-SCRIPT",
                flags=verdict.flags + ["OFF_SCRIPT"],
                last_message=text,
                note=f"Customer asked something outside the approved script at '{field_name}'.",
                extra_messages=[
                    "That's a fair question, but it's outside what I'm able to answer on "
                    "this call. Let me hand you to a specialist who can help properly."
                ],
            )

        if row.attempts >= MAX_FIELD_ATTEMPTS:
            return self._handoff(
                session,
                reason="REPEATED_FAILURE",
                signal="CONFUSION",
                flags=verdict.flags + ["REPEATED_FAILURE"],
                last_message=text,
                note=f"'{field_name}' could not be captured after "
                     f"{row.attempts} attempts.",
            )

        return self._clarify(session, step, row, extraction.reason)

    def _clarify(
        self, session: CallSession, step: dict[str, Any], row: JourneyField, reason: str
    ) -> EngineResult:
        """Exactly one clarification question per field, using the approved fallback."""
        fallback = self.scripts.fallback_prompt(step["step_id"])
        self._say(session, fallback)
        session.state = "COLLECTING_FIELD"
        self._audit(
            session.id,
            "CLARIFICATION_REQUESTED",
            f"field={step['field_name']} reason={reason} (1 of 1 allowed)",
        )
        self.db.commit()
        return EngineResult(
            session=session,
            agent_messages=[fallback],
            system_notes=[
                f"'{step['field_name']}' not captured ({reason}). One clarification asked; "
                "a second failure will hand off to a human."
            ],
        )

    def _advance_from(
        self, session: CallSession, step: dict[str, Any], extraction: Any
    ) -> EngineResult:
        next_step_id = step.get("next_step") or "confirmation"
        if next_step_id == "SUBMIT":
            next_step_id = "confirmation"

        if next_step_id == "confirmation":
            session.current_step = "confirmation"
            session.state = "CONFIRMING_DETAILS"
            prompt = self.scripts.render_prompt("confirmation", **self._collected(session.id))
            self._say(session, prompt)
            self._audit(session.id, "CONFIRMATION_READBACK", "Reading back call-collected fields only.")
            self.db.commit()
            return EngineResult(
                session=session,
                agent_messages=[prompt],
                system_notes=["All required fields captured. Reading back for confirmation."],
            )

        session.current_step = next_step_id
        session.state = "COLLECTING_FIELD"
        prompt = self.scripts.render_prompt(next_step_id, **self._collected(session.id))
        self._say(session, prompt)
        self._audit(session.id, "STEP_STARTED", next_step_id)
        self.db.commit()
        return EngineResult(
            session=session,
            agent_messages=[prompt],
            system_notes=[f"Captured '{step['field_name']}'. Advanced to '{next_step_id}'."],
        )

    # ------------------------------------------------------------------ #
    # 5. Confirmation
    # ------------------------------------------------------------------ #
    def _handle_confirmation(
        self, session: CallSession, text: str, verdict: safety_engine.SafetyVerdict
    ) -> EngineResult:
        result = validate_confirmation(text)
        self._audit(
            session.id,
            "CONFIRMATION_RESPONSE",
            f"ok={result.ok} reason={result.reason} confidence={result.confidence:.2f}",
        )

        if result.ok:
            return self._submit(session, verdict)

        # No repeated correction loops — a human takes over instead.
        return self._handoff(
            session,
            reason="REPEATED_FAILURE",
            signal="CONFUSION",
            flags=verdict.flags + ["CONFIRMATION_DECLINED"],
            last_message=text,
            note="Customer did not confirm the read-back. Routing to a human rather than "
                 "looping corrections.",
            extra_messages=[
                "Thanks for checking. Rather than risk getting a detail wrong, let me bring "
                "in a colleague who can go through it with you properly."
            ],
        )

    def _submit(self, session: CallSession, verdict: safety_engine.SafetyVerdict) -> EngineResult:
        session.state = "SUBMITTING_JOURNEY"
        self.db.flush()

        collected = self._collected(session.id)
        payload = {
            "lead_id": session.lead_id,
            "vertical": "ENERGY",
            "property_address": collected.get("property_address"),
            "move_in_date": collected.get("move_in_date"),
            "energy_requirement": collected.get("energy_requirement"),
            "concession_status": collected.get("concession_status"),
            "life_support": collected.get("life_support"),
            "contact_preference": collected.get("contact_preference"),
        }

        missing = [key for key, value in payload.items() if value in (None, "")]
        if missing:
            self._audit(session.id, "SUBMISSION_BLOCKED", f"missing={missing}")
            return self._handoff(
                session,
                reason="LOW_CONFIDENCE",
                signal="LOW_CONF",
                flags=verdict.flags + ["MISSING_REQUIRED_FIELD"],
                last_message="",
                note=f"Refused to submit: required fields missing ({', '.join(missing)}). "
                     "The system never submits unknown values.",
            )

        submission = journey_service.submit_journey(
            self.db, payload, call_session_id=session.id, origin="AI_VOICE_AGENT"
        )

        message = (
            "Thank you — that's all submitted. You'll get a confirmation by "
            f"{'email' if payload['contact_preference'] == 'EMAIL' else 'phone'} shortly. "
            "Have a great day."
        )
        self._say(session, message)

        session.state = "COMPLETED"
        session.status = SessionStatus.COMPLETED.value
        session.ended_at = utcnow()
        session.lead.status = LeadStatus.COMPLETED.value
        self._audit(
            session.id,
            "JOURNEY_SUBMITTED",
            f"submission_id={submission.submission_id} payload={json.dumps(payload)}",
        )
        self._audit(session.id, "CALL_ENDED", "Journey completed by AI voice agent.")
        self.db.commit()

        return EngineResult(
            session=session,
            agent_messages=[message],
            terminal=True,
            journey_submitted=True,
            submission_id=submission.submission_id,
            system_notes=[f"Mock journey submitted as {submission.submission_id}."],
        )

    # ------------------------------------------------------------------ #
    # 6. Decline
    # ------------------------------------------------------------------ #
    def _decline(
        self, session: CallSession, verdict: safety_engine.SafetyVerdict, text: str
    ) -> EngineResult:
        message = "Understood. Thank you for your time. I will not continue this call."
        self._say(session, message)
        self._audit(
            session.id,
            "CALL_DECLINED",
            f"matched='{verdict.matched}' — refusal respected, no retry, no pressure.",
        )
        return self._end_call(
            session,
            status=SessionStatus.DECLINED.value,
            outcome_detail="CUSTOMER_DECLINED",
            note="Customer declined. Call ended immediately with no retry.",
            agent_messages=[message],
        )

    # ------------------------------------------------------------------ #
    # 7. Warm handoff
    # ------------------------------------------------------------------ #
    def _handoff(
        self,
        session: CallSession,
        reason: str,
        signal: str | None,
        flags: list[str],
        last_message: str,
        note: str,
        extra_messages: list[str] | None = None,
    ) -> EngineResult:
        collected = self._collected(session.id)
        handoff_fields = {
            key: collected.get(key)
            for key in [
                "property_address",
                "move_in_date",
                "energy_requirement",
                "concession_status",
                "life_support",
                "contact_preference",
            ]
        }
        summary = build_context_summary(
            lead=self._lead_dict(session.lead),
            resume_step=session.resume_step,
            last_completed_step=session.lead.last_completed_step,
            collected=collected,
            reason=reason,
            current_step=session.current_step,
        )

        existing = self.db.execute(
            select(Handoff).where(Handoff.call_session_id == session.id)
        ).scalars().first()
        if existing is None:
            handoff = Handoff(
                call_session_id=session.id,
                reason=reason,
                current_step=session.current_step,
                context_summary=summary,
                last_customer_message=last_message or None,
                safety_flags_json=json.dumps(sorted(set(flags))),
                collected_fields_json=json.dumps(handoff_fields),
            )
            self.db.add(handoff)
        else:
            handoff = existing

        session.state = "HANDOFF_REQUESTED"
        session.status = SessionStatus.HANDOFF_REQUESTED.value
        session.handoff_reason = reason
        session.outcome_detail = signal
        session.ended_at = utcnow()
        session.lead.status = LeadStatus.HANDOFF_REQUESTED.value

        # Attempt the live warm transfer through the telephony adapter. With no provider
        # credentials (or no reference) this reports CONSOLE_QUEUE and the handoff waits in
        # the console instead — the customer experience is identical in the local demo.
        transfer = voice_providers.warm_transfer(
            session.telephony_reference, settings.handoff_transfer_number
        )
        self._audit(session.id, "WARM_TRANSFER_ATTEMPT", json.dumps(transfer))
        self._audit(session.id, "HANDOFF_CREATED", f"reason={reason} signal={signal} :: {note}")

        messages = list(extra_messages or [])
        if not messages:
            messages = [
                "I can hear this isn't working the way it should. Let me bring in a "
                "colleague right now — they can see everything we've covered, so you "
                "won't need to repeat anything."
            ]
        for message in messages:
            self._say(session, message)
        self._audit(session.id, "CALL_ENDED", f"Autonomous collection stopped: {reason}")
        self.db.commit()

        transfer_note = (
            f"Live call transferred to {settings.handoff_transfer_number}."
            if transfer.get("transferred")
            else "Handoff queued in the console (no telephony provider configured)."
        )
        return EngineResult(
            session=session,
            agent_messages=messages,
            safety_flags=sorted(set(flags)),
            handoff_triggered=True,
            terminal=True,
            system_notes=[f"Warm handoff created ({reason}/{signal}). {note}", transfer_note],
        )

    # ------------------------------------------------------------------ #
    # 8. Manual end / manual handoff (console actions)
    # ------------------------------------------------------------------ #
    def end_call(self, session_id: str, reason: str = "AGENT_ENDED") -> EngineResult:
        session = self._get_session(session_id)
        if session is None:
            raise LookupError(f"Unknown call session {session_id}")
        if session.state in TERMINAL_STATES:
            return EngineResult(session=session, terminal=True)
        message = "Thanks for your time. I'll end the call here."
        self._say(session, message)
        self.db.flush()
        return self._end_call(
            session,
            status=SessionStatus.DECLINED.value,
            outcome_detail=reason,
            note="Call ended from the console.",
            agent_messages=[message],
        )

    def _end_call(
        self,
        session: CallSession,
        status: str,
        outcome_detail: str,
        note: str,
        agent_messages: list[str],
    ) -> EngineResult:
        session.status = status
        session.state = "DECLINED"
        session.outcome_detail = outcome_detail
        session.ended_at = utcnow()
        if status == SessionStatus.DECLINED.value:
            session.lead.status = LeadStatus.DECLINED.value
        self._audit(session.id, "CALL_ENDED", note)
        self.db.commit()
        return EngineResult(
            session=session,
            agent_messages=agent_messages,
            terminal=True,
            system_notes=[note],
        )

    def request_handoff(self, session_id: str, reason: str, note: str | None = None) -> EngineResult:
        session = self._get_session(session_id)
        if session is None:
            raise LookupError(f"Unknown call session {session_id}")
        if session.state in TERMINAL_STATES:
            return EngineResult(session=session, terminal=True)
        last_customer = self.db.execute(
            select(TranscriptSegment)
            .where(
                TranscriptSegment.call_session_id == session.id,
                TranscriptSegment.speaker == Speaker.CUSTOMER.value,
            )
            .order_by(TranscriptSegment.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        return self._handoff(
            session,
            reason=reason,
            signal="ASKS" if reason == "CUSTOMER_REQUEST" else None,
            flags=[reason, "CONSOLE_INITIATED"],
            last_message=last_customer.text if last_customer else "",
            note=note or "Handoff raised from the agent console.",
        )

    # ------------------------------------------------------------------ #
    # 9. Human-in-the-loop capture  (Agent-Assisted mode, handout Mode A)
    # ------------------------------------------------------------------ #
    def capture_field_by_human(
        self,
        session_id: str,
        field_name: str,
        value: str,
        agent_name: str = "Human Agent",
    ) -> EngineResult:
        """A human agent supplies or corrects a journey field while the call is live.

        This is the Mode A seam. The human's value still goes through the **same** Python
        validator as a spoken answer — a human cannot push an invalid value into journey
        state either. Valid values are stored with source HUMAN_AGENT so the audit trail
        always shows who captured what.
        """
        session = self._get_session(session_id)
        if session is None:
            raise LookupError(f"Unknown call session {session_id}")
        if session.state in TERMINAL_STATES:
            raise ValueError(
                f"Call already ended in state {session.state}; the journey can no longer be edited."
            )

        step = next(
            (s for s in self.scripts.steps() if s.get("field_name") == field_name),
            None,
        )
        if step is None:
            raise ValueError(f"'{field_name}' is not a field on the Energy journey.")

        if session.submission is not None:
            raise ValueError(
                "This journey has already been submitted. Corrections must go through the "
                "human agent's own tools — the system will not silently rewrite a submitted payload."
            )

        result = validate_field(step["validation_type"], value)
        if not result.ok:
            self._audit(
                session.id,
                "HUMAN_CAPTURE_REJECTED",
                f"field={field_name} reason={result.reason} by={agent_name}",
            )
            self.db.commit()
            raise ValueError(
                f"'{value}' is not a valid {field_name} ({result.reason}). "
                "The same validator that checks spoken answers applies here."
            )

        row = self._field_row(session.id, field_name)
        if row is None:
            row = JourneyField(
                call_session_id=session.id,
                field_name=field_name,
                status=FieldStatus.PENDING.value,
                source=FieldSource.HUMAN_AGENT.value,
            )
            self.db.add(row)
            self.db.flush()

        is_correction = row.status == FieldStatus.VALID.value
        row.value = result.value
        row.status = FieldStatus.VALID.value
        row.source = FieldSource.HUMAN_AGENT.value
        row.confidence = 1.0

        self._audit(
            session.id,
            "FIELD_CORRECTED_BY_HUMAN" if is_correction else "FIELD_CAPTURED_BY_HUMAN",
            f"{field_name}={result.value} by={agent_name} (validated, source=HUMAN_AGENT)",
        )

        # If this was the field we were stuck on, move the journey along.
        if not is_correction and session.current_step == step["step_id"]:
            self.db.flush()
            next_step_id = step.get("next_step") or "confirmation"
            if next_step_id == "SUBMIT":
                next_step_id = "confirmation"
            session.current_step = next_step_id
            session.state = (
                "CONFIRMING_DETAILS" if next_step_id == "confirmation" else "COLLECTING_FIELD"
            )
            prompt = self.scripts.render_prompt(next_step_id, **self._collected(session.id))
            self._say(session, prompt)
            self._audit(session.id, "STEP_STARTED", f"{next_step_id} (after human capture)")
            self.db.commit()
            return EngineResult(
                session=session,
                agent_messages=[prompt],
                system_notes=[
                    f"{agent_name} captured '{field_name}' directly; the journey advanced "
                    f"to '{next_step_id}'.",
                ],
            )

        self.db.commit()
        return EngineResult(
            session=session,
            system_notes=[
                f"{agent_name} {'corrected' if is_correction else 'captured'} '{field_name}'.",
            ],
        )


__all__ = ["ConversationEngine", "EngineResult"]
