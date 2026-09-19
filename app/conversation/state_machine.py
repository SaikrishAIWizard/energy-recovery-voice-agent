"""Deterministic conversation state machine.

This is the only thing allowed to decide anything. The LLM is an optional input adapter;
it never chooses to continue, submit, decline, or escalate.

States
------
INIT  DNC_CHECK  CONSENT_DISCLOSURE  AWAITING_CONTINUE_RESPONSE  AWAITING_BUSY_PREFERENCE
COLLECTING_FIELD  VALIDATING_FIELD  CONFIRMING_FIELD  AWAITING_ELIGIBILITY_CHOICE
CONFIRMING_DETAILS  SUBMITTING_JOURNEY  CLOSING  COMPLETED  DECLINED  HANDOFF_REQUESTED
DNC_BLOCKED  INCOMPLETE

Rules enforced here
-------------------
* Only the approved prompt for the active step may be spoken (or its single fallback).
* One customer answer per field. An unclear answer gets up to MAX_FOLLOW_UPS follow-up
  questions (each worded more simply), then a person takes over.
* Two failed captures of the same field -> handoff(REPEATED_FAILURE).
* Confidence below 0.80 after clarification -> handoff(LOW_CONFIDENCE).
* Unknown required values are never submitted, and never invented.
* Address, date and energy are read back ("I have X. Is that correct?") and only count once
  the customer confirms; everything is read back again at the end. One correction is
  allowed, naming a single field; if it is still not confirmed a human takes over.
* Everything the agent SAYS comes from the script file (docs/energy-agent-script-and-
  checklist.md is its source of truth). No spoken strings live in this module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import MAX_FIELD_ATTEMPTS, MAX_FOLLOW_UPS, MIN_FIELD_CONFIDENCE, settings
from app.conversation import safety_engine
from app.conversation.field_extractor import build_known_context, extract_field
from app.conversation.validators import (
    detect_correction_fields,
    validate_confirmation,
    validate_field,
    validate_yes_no_unsure,
)
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
from app.services import dnc_service, journey_service, lead_service
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
    # An analysed recording: there is no live call left to continue.
    "INCOMPLETE",
}

FIELD_LABELS = {
    "property_address": "Service address",
    "move_in_date": "Move-in date",
    "energy_requirement": "Supply needed",
    "concession_status": "Concession",
    "life_support": "Life support",
    "contact_preference": "Contact preference",
}


class JourneyConflict(ValueError):
    """The request is fine but the call is in the wrong state for it (maps to HTTP 409)."""


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

    def __init__(
        self,
        db: Session,
        scripts: ScriptService | None = None,
        *,
        defer_transfer: bool = False,
    ) -> None:
        self.db = db
        self.scripts = scripts or script_service
        self.llm = get_llm()
        # A Twilio webhook answers with the instructions for the live call itself, so a
        # handoff raised inside one must not also redirect the call over the REST API.
        self.defer_transfer = defer_transfer

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
        if not text.strip():
            return  # silence on a phone call: nothing was said, so nothing is transcribed
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

    def _count_audit(self, session_id: str, event_type: str, detail: str | None = None) -> int:
        """How many times an event has been logged on this call. The "follow up, then a
        person" rules are counted from the audit trail, so they need no extra state."""
        self.db.flush()  # sessions do not autoflush; count what this turn has already logged
        query = select(func.count(AuditEvent.id)).where(
            AuditEvent.call_session_id == session_id, AuditEvent.event_type == event_type
        )
        if detail is not None:
            query = query.where(AuditEvent.event_detail == detail)
        return int(self.db.execute(query).scalar_one())

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

        # Read before the new session exists: the lead's latest session must still be the
        # previous one, or an earlier recording's captured fields would be invisible.
        carried = lead_service.carried_fields(self.db, lead.id)

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
        resume_step = self.scripts.skip_known(
            self.scripts.resume_step_after(lead.last_completed_step), set(carried)
        )
        session.resume_step = resume_step
        self.seed_journey_fields(session, lead, carried)
        if carried:
            self._audit(
                session.id,
                "RECORDING_FIELDS_CARRIED",
                "Already captured from an uploaded recording: " + ", ".join(sorted(carried)),
            )
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

    def seed_journey_fields(
        self, session: CallSession, lead: Lead, carried: dict[str, Any] | None = None
    ) -> None:
        """Seed a session's journey rows: contact details, values already on file, and
        everything still to collect.

        `carried` holds values captured by an earlier uploaded recording. They are newer
        than the lead's static on-file data, so they win over it.
        """
        preexisting = {**self.scripts.preexisting_fields_for_lead(lead.id), **(carried or {})}

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
            if field_name in preexisting:
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

        # After the journey is submitted the agent asks one closing question, with its own
        # rules (a decline there must not undo a completed journey).
        if session.state == "CLOSING":
            return self._handle_closing(session, safe_text, verdict)

        if verdict.is_decline:
            return self._decline(session, verdict, safe_text)

        if verdict.is_handoff:
            # Saying "ventilator" at the life-support question IS the answer: record it as a
            # bare yes so the person taking over sees it (no medical detail is kept).
            if "LIFE_SUPPORT" in verdict.flags and session.current_step == "life_support":
                self._record_life_support_yes(session, verdict.flags)
            return self._handoff(
                session,
                reason=verdict.reason or "SENSITIVE_TOPIC",
                signal=verdict.escalation_signal,
                flags=verdict.flags,
                last_message=safe_text,
                note=verdict.note,
            )

        # --- Deterministic dispatch ------------------------------------------ #
        handlers = {
            "AWAITING_CONTINUE_RESPONSE": self._handle_gate,
            "AWAITING_BUSY_PREFERENCE": self._handle_busy_preference,
            "COLLECTING_FIELD": self._handle_field,
            "VALIDATING_FIELD": self._handle_field,
            "CONFIRMING_FIELD": self._handle_field_readback,
            "AWAITING_ELIGIBILITY_CHOICE": self._handle_eligibility_choice,
            "CONFIRMING_DETAILS": self._handle_confirmation,
        }
        handler = handlers.get(session.state)
        if handler is not None:
            return handler(session, safe_text, verdict)

        return EngineResult(
            session=session,
            system_notes=[f"No handler for state {session.state}."],
        )

    def _flag_do_not_call(self, session: CallSession) -> None:
        """The customer asked not to be contacted again: log it AND flag the lead, so every
        future dial (console, phone, recovery queue) is refused by the DNC gate. Nothing has
        to remember to check; `dnc_service.check` reads the lead's own flag."""
        session.outcome_detail = "DO_NOT_CALL_REQUESTED"
        session.lead.dnc_status = True
        self._audit(
            session.id,
            "DNC_REQUEST_LOGGED",
            "Customer asked not to be contacted again.",
        )
        self._audit(
            session.id,
            "DNC_REGISTER_UPDATED",
            f"Lead {session.lead_id} flagged Do-Not-Call at the customer's request. "
            "Future dialling is blocked and cannot be overridden.",
        )

    def _record_life_support_yes(self, session: CallSession, flags: list[str]) -> None:
        row = self._field_row(session.id, "life_support")
        if row is None or row.status == FieldStatus.VALID.value:
            return
        row.value = "YES"
        row.status = FieldStatus.VALID.value
        row.source = FieldSource.CUSTOMER_SPOKEN.value
        row.confidence = 0.9
        row.attempts += 1
        self._audit(
            session.id, "FIELD_CAPTURED", "life_support=YES confidence=0.90 source=SAFETY_SIGNAL"
        )

    # ------------------------------------------------------------------ #
    # 3. Opening: "Is now an okay time to continue where you left off?"
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
            # Ask once whether they want a callback or would rather finish later. No
            # persuading, no second ask.
            gate_field.value = "BUSY"
            gate_field.status = FieldStatus.VALID.value
            message = self.scripts.message("busy_callback_question")
            self._say(session, message)
            session.state = "AWAITING_BUSY_PREFERENCE"
            self._audit(session.id, "BUSY_PREFERENCE_ASKED", "Callback, or leave the journey for later?")
            self.db.commit()
            return EngineResult(
                session=session,
                agent_messages=[message],
                safety_flags=verdict.flags,
                system_notes=["Customer is busy. Asked once: callback, or continue later?"],
            )

        # Unclear
        self._audit(session.id, "GATE_UNCLEAR", f"attempt={gate_field.attempts}")
        if gate_field.attempts < MAX_FIELD_ATTEMPTS:
            fallback = self.scripts.fallback_prompt("recording_consent", gate_field.attempts)
            self._say(session, fallback)
            self._audit(
                session.id,
                "CLARIFICATION_REQUESTED",
                f"gate follow-up {gate_field.attempts} of {MAX_FOLLOW_UPS}",
            )
            self.db.commit()
            return EngineResult(
                session=session,
                agent_messages=[fallback],
                safety_flags=verdict.flags,
                system_notes=[
                    f"Reply unclear — follow-up {gate_field.attempts} of {MAX_FOLLOW_UPS} asked."
                ],
            )

        return self._handoff(
            session,
            reason="LOW_CONFIDENCE",
            signal="LOW_CONF",
            flags=verdict.flags + ["LOW_CONFIDENCE"],
            last_message=text,
            note=f"Customer's reply to the opening question could not be understood after "
                 f"{MAX_FOLLOW_UPS} follow-ups.",
        )

    def _handle_busy_preference(
        self, session: CallSession, text: str, verdict: safety_engine.SafetyVerdict
    ) -> EngineResult:
        preference = safety_engine.classify_busy_preference(text)

        if preference == "UNCLEAR" and self._count_audit(session.id, "BUSY_PREFERENCE_UNCLEAR") == 0:
            self._audit(session.id, "BUSY_PREFERENCE_UNCLEAR", text[:80])
            message = self.scripts.message("busy_callback_question_again")
            self._say(session, message)
            self.db.commit()
            return EngineResult(
                session=session,
                agent_messages=[message],
                system_notes=["Callback preference unclear — asked once more."],
            )

        if preference == "CALLBACK":
            detail, line = "CALLBACK_REQUESTED", "busy_callback_closing"
            self._audit(session.id, "CALLBACK_REQUESTED", "Customer asked for a callback.")
            note = "Customer is busy and asked for a callback. Logged; call ended."
        else:
            # Asked for later, or still unclear after one re-ask: leave it with them. No
            # third ask.
            detail, line = "LEFT_FOR_LATER", "busy_later_closing"
            self._audit(session.id, "JOURNEY_LEFT_FOR_LATER", "Customer will continue the journey later.")
            note = "Customer is busy and will continue later. Logged; call ended."
        message = self.scripts.message(line)
        self._say(session, message)
        return self._end_call(
            session,
            status=SessionStatus.DECLINED.value,
            outcome_detail=detail,
            note=note,
            agent_messages=[message],
        )

    def _advance_to_resume_step(self, session: CallSession) -> EngineResult:
        """Permission given: go straight to the first missing question. Fields already
        present and valid are never asked again."""
        step_id = session.resume_step or "property_address"
        step = self.scripts.step(step_id)
        if step is None or "prompt" not in step:
            step_id = "property_address"

        if step_id == "confirmation":  # every field was already on file
            return self._final_readback(session)

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

        # "Do I qualify?" — the script never answers; it offers a person and lets them choose.
        if step.get("eligibility_question_offer") and safety_engine.is_eligibility_question(text):
            message = self.scripts.message(step["eligibility_question_offer"])
            self._say(session, message)
            session.state = "AWAITING_ELIGIBILITY_CHOICE"
            self._audit(session.id, "ELIGIBILITY_QUESTION_DEFERRED", field_name)
            self.db.commit()
            return EngineResult(
                session=session,
                agent_messages=[message],
                safety_flags=verdict.flags,
                system_notes=["Customer asked about eligibility. No advice given; a team member was offered."],
            )

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
            # Address, date and energy are read back and only count once confirmed.
            if step.get("readback_prompt"):
                row.value = extraction.value
                row.status = FieldStatus.PENDING.value
                row.source = FieldSource.CUSTOMER_SPOKEN.value
                row.confidence = extraction.confidence
                return self._read_back_field(session, step, row, extraction.source)

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

            # life_support = YES is a hard stop: vulnerable customer. No medical questions.
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
        """The next follow-up question for this field, worded more simply each time. After
        the last one (`MAX_FOLLOW_UPS`) a failed capture hands over to a person instead."""
        fallback = self.scripts.fallback_prompt(step["step_id"], row.attempts)
        self._say(session, fallback)
        session.state = "COLLECTING_FIELD"
        self._audit(
            session.id,
            "CLARIFICATION_REQUESTED",
            f"field={step['field_name']} reason={reason} "
            f"(follow-up {row.attempts} of {MAX_FOLLOW_UPS})",
        )
        self.db.commit()
        return EngineResult(
            session=session,
            agent_messages=[fallback],
            system_notes=[
                f"'{step['field_name']}' not captured ({reason}). Follow-up {row.attempts} of "
                f"{MAX_FOLLOW_UPS} asked; a failure after the last one hands off to a human."
            ],
        )

    # -- per-field read-back ("I have X. Is that correct?") ----------------- #
    def _read_back_field(
        self, session: CallSession, step: dict[str, Any], row: JourneyField, source: str = ""
    ) -> EngineResult:
        prompt = self.scripts.readback_prompt(step["step_id"], **{step["field_name"]: row.value})
        assert prompt is not None
        self._say(session, prompt)
        session.state = "CONFIRMING_FIELD"
        self._audit(
            session.id,
            "FIELD_READBACK",
            f"{step['field_name']}={row.value} confidence={(row.confidence or 0):.2f} source={source}",
        )
        self.db.commit()
        return EngineResult(
            session=session,
            agent_messages=[prompt],
            system_notes=[f"Read '{step['field_name']}' back to the customer for confirmation."],
        )

    def _handle_field_readback(
        self, session: CallSession, text: str, verdict: safety_engine.SafetyVerdict
    ) -> EngineResult:
        step = self.scripts.step(session.current_step)
        if step is None or not step.get("readback_prompt"):
            return self._handoff(
                session, "OFF_SCRIPT", "OFF-SCRIPT", verdict.flags, text,
                "Active step has no read-back in the approved script library.",
            )
        field_name = step["field_name"]
        row = self._field_row(session.id, field_name)
        assert row is not None

        confirmation = validate_confirmation(text)
        self._audit(
            session.id,
            "FIELD_READBACK_RESPONSE",
            f"{field_name} ok={confirmation.ok} reason={confirmation.reason}",
        )

        if confirmation.ok:
            row.status = FieldStatus.VALID.value
            row.source = FieldSource.CUSTOMER_SPOKEN.value
            self._audit(
                session.id,
                "FIELD_CAPTURED",
                f"{field_name}={row.value} confidence={(row.confidence or 0):.2f} "
                "source=CUSTOMER_CONFIRMED_READBACK",
            )
            return self._advance_from(session, step, None)

        # Not a yes. They may have simply given the right value ("no, gas only").
        if len(text.split()) >= 2:
            context = build_known_context(
                lead=self._lead_dict(session.lead),
                collected=self._collected(session.id),
                current_step=field_name,
            )
            fresh = extract_field(
                field_name, text, context, validation_type=step["validation_type"], llm=self.llm
            )
            if fresh.valid and fresh.confidence >= MIN_FIELD_CONFIDENCE and fresh.value != row.value:
                if row.attempts >= MAX_FIELD_ATTEMPTS:
                    return self._handoff(
                        session, "REPEATED_FAILURE", "CONFUSION", verdict.flags + ["REPEATED_FAILURE"],
                        text, f"'{field_name}' was still not confirmed after one correction.",
                    )
                row.attempts += 1
                row.value = fresh.value
                row.confidence = fresh.confidence
                return self._read_back_field(session, step, row, fresh.source)

        if confirmation.reason == "confirmation_declined":
            row.value = None
            row.status = FieldStatus.INVALID.value
            self._audit(session.id, "FIELD_READBACK_REJECTED", field_name)
            if row.attempts >= MAX_FIELD_ATTEMPTS:
                return self._handoff(
                    session, "REPEATED_FAILURE", "CONFUSION", verdict.flags + ["REPEATED_FAILURE"],
                    text, f"'{field_name}' was not confirmed after the follow-ups.",
                )
            return self._clarify(session, step, row, "customer_rejected_readback")

        # Neither yes nor no: ask again (up to the follow-up limit), then a person.
        if self._count_audit(session.id, "FIELD_READBACK_UNCLEAR", field_name) < MAX_FOLLOW_UPS:
            self._audit(session.id, "FIELD_READBACK_UNCLEAR", field_name)
            prompt = self.scripts.readback_prompt(step["step_id"], **{field_name: row.value})
            assert prompt is not None
            self._say(session, prompt)
            self.db.commit()
            return EngineResult(
                session=session,
                agent_messages=[prompt],
                system_notes=["Read-back answer unclear — asked again."],
            )
        return self._handoff(
            session, "LOW_CONFIDENCE", "LOW_CONF", verdict.flags + ["LOW_CONFIDENCE"], text,
            f"The read-back of '{field_name}' stayed unclear after {MAX_FOLLOW_UPS} repeats.",
        )

    def _handle_eligibility_choice(
        self, session: CallSession, text: str, verdict: safety_engine.SafetyVerdict
    ) -> EngineResult:
        """Reply to "I cannot advise on eligibility... Would you like me to connect you now?"."""
        step = self.scripts.step(session.current_step)
        answer = validate_confirmation(text)

        if answer.ok:
            return self._handoff(
                session,
                reason="OFF_SCRIPT",
                signal="OFF-SCRIPT",
                flags=verdict.flags + ["ELIGIBILITY_QUESTION"],
                last_message=text,
                note="Customer asked whether they qualify for a concession. A team member answers that.",
            )
        if answer.reason == "confirmation_declined" and step is not None:
            session.state = "COLLECTING_FIELD"
            prompt = self.scripts.render_prompt(step["step_id"], **self._collected(session.id))
            self._say(session, prompt)
            self._audit(session.id, "ELIGIBILITY_OFFER_DECLINED", "Customer will answer the question instead.")
            self.db.commit()
            return EngineResult(session=session, agent_messages=[prompt])

        if self._count_audit(session.id, "ELIGIBILITY_CHOICE_UNCLEAR") < MAX_FOLLOW_UPS and step is not None:
            self._audit(session.id, "ELIGIBILITY_CHOICE_UNCLEAR", text[:80])
            message = self.scripts.message(step["eligibility_question_offer"])
            self._say(session, message)
            self.db.commit()
            return EngineResult(session=session, agent_messages=[message])
        return self._handoff(
            session, "LOW_CONFIDENCE", "LOW_CONF", verdict.flags + ["LOW_CONFIDENCE"], text,
            "The reply to the eligibility offer stayed unclear.",
        )

    def _advance_from(
        self, session: CallSession, step: dict[str, Any], extraction: Any
    ) -> EngineResult:
        # A single detail was being corrected at the final read-back: go back and read
        # everything back again, once.
        confirmation_row = self._field_row(session.id, "confirmation")
        if confirmation_row is not None and (confirmation_row.value or "").startswith("CORRECTING"):
            confirmation_row.value = None
            return self._final_readback(session)

        next_step_id = step.get("next_step") or "confirmation"
        if next_step_id == "SUBMIT":
            next_step_id = "confirmation"

        if next_step_id == "confirmation":
            return self._final_readback(session)

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
    # 5. Final read-back, correction, confirmation, submission
    # ------------------------------------------------------------------ #
    def _final_readback(self, session: CallSession) -> EngineResult:
        session.current_step = "confirmation"
        session.state = "CONFIRMING_DETAILS"
        prompt = self.scripts.render_prompt("confirmation", **self._collected(session.id))
        self._say(session, prompt)
        self._audit(session.id, "CONFIRMATION_READBACK", "Reading back every collected field.")
        self.db.commit()
        return EngineResult(
            session=session,
            agent_messages=[prompt],
            system_notes=["All required fields captured. Reading back for confirmation."],
        )

    def _handle_confirmation(
        self, session: CallSession, text: str, verdict: safety_engine.SafetyVerdict
    ) -> EngineResult:
        confirmation_row = self._field_row(session.id, "confirmation")
        if confirmation_row is None:
            confirmation_row = JourneyField(
                call_session_id=session.id,
                field_name="confirmation",
                status=FieldStatus.PENDING.value,
                source=FieldSource.CUSTOMER_SPOKEN.value,
            )
            self.db.add(confirmation_row)
            self.db.flush()

        # They were asked which detail to correct; this reply names it.
        if confirmation_row.value == "AWAITING_WHICH":
            fields = detect_correction_fields(text)
            if len(fields) != 1:
                return self._handoff(
                    session, "REPEATED_FAILURE", "CONFUSION", verdict.flags + ["CONFIRMATION_DECLINED"],
                    text, "Could not tell which single detail needed correcting.",
                    extra_messages=None,
                )
            return self._apply_correction(session, confirmation_row, fields[0], text, verdict)

        result = validate_confirmation(text)
        self._audit(
            session.id,
            "CONFIRMATION_RESPONSE",
            f"ok={result.ok} reason={result.reason} confidence={result.confidence:.2f}",
        )

        if result.ok:
            return self._submit(session, verdict)

        readbacks = self._count_audit(session.id, "CONFIRMATION_READBACK")
        if result.reason == "confirmation_declined":
            if readbacks >= 2:
                # Already corrected once and read back again: no loops.
                return self._handoff(
                    session, "REPEATED_FAILURE", "CONFUSION", verdict.flags + ["CONFIRMATION_DECLINED"],
                    text, "Details still not confirmed after one correction. Routing to a human "
                          "rather than looping.",
                )
            fields = detect_correction_fields(text)
            if len(fields) == 1:
                return self._apply_correction(session, confirmation_row, fields[0], text, verdict)
            confirmation_row.value = "AWAITING_WHICH"
            message = self.scripts.message("correction_prompt")
            self._say(session, message)
            self._audit(session.id, "CORRECTION_REQUESTED", "Asked which detail to correct.")
            self.db.commit()
            return EngineResult(
                session=session,
                agent_messages=[message],
                safety_flags=verdict.flags,
                system_notes=["Customer did not confirm. Asked which single detail to correct."],
            )

        # Neither yes nor no: follow up (up to the limit), each time more simply, then a person.
        unclear = self._count_audit(session.id, "CONFIRMATION_UNCLEAR")
        if unclear < MAX_FOLLOW_UPS:
            self._audit(session.id, "CONFIRMATION_UNCLEAR", text[:80])
            message = self.scripts.fallback_prompt("confirmation", unclear + 1)
            self._say(session, message)
            self.db.commit()
            return EngineResult(session=session, agent_messages=[message])
        return self._handoff(
            session, "LOW_CONFIDENCE", "LOW_CONF", verdict.flags + ["LOW_CONFIDENCE"], text,
            f"The final confirmation stayed unclear after {MAX_FOLLOW_UPS} follow-ups.",
        )

    def _apply_correction(
        self,
        session: CallSession,
        confirmation_row: JourneyField,
        field_name: str,
        text: str,
        verdict: safety_engine.SafetyVerdict,
    ) -> EngineResult:
        """Correct only the named field. A new date or address given in the same breath is
        validated and applied; anything else is asked for again, and either way the full
        read-back is repeated once."""
        step = next((s for s in self.scripts.steps() if s.get("field_name") == field_name), None)
        row = self._field_row(session.id, field_name)
        if step is None or row is None:
            return self._handoff(
                session, "OFF_SCRIPT", "OFF-SCRIPT", verdict.flags, text,
                f"'{field_name}' is not a field on the Energy journey.",
            )

        if field_name in ("move_in_date", "property_address"):
            context = build_known_context(
                lead=self._lead_dict(session.lead),
                collected=self._collected(session.id),
                current_step=field_name,
            )
            fresh = extract_field(
                field_name, text, context, validation_type=step["validation_type"], llm=self.llm
            )
            if fresh.valid and fresh.confidence >= MIN_FIELD_CONFIDENCE:
                row.value = fresh.value
                row.status = FieldStatus.VALID.value
                row.source = FieldSource.CUSTOMER_SPOKEN.value
                row.confidence = fresh.confidence
                confirmation_row.value = None
                self._audit(session.id, "FIELD_CORRECTED", f"{field_name}={fresh.value}")
                return self._final_readback(session)

        # Ask for just this detail again; the answer returns to the final read-back.
        confirmation_row.value = f"CORRECTING:{field_name}"
        row.value = None
        row.status = FieldStatus.PENDING.value
        row.attempts = 0
        session.current_step = step["step_id"]
        session.state = "COLLECTING_FIELD"
        prompt = self.scripts.render_prompt(step["step_id"], **self._collected(session.id))
        self._say(session, prompt)
        self._audit(session.id, "CORRECTION_STARTED", field_name)
        self.db.commit()
        return EngineResult(
            session=session,
            agent_messages=[prompt],
            safety_flags=verdict.flags,
            system_notes=[f"Correcting '{field_name}' only."],
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

        # Life support is a person's decision, never a submission.
        if payload["life_support"] == "YES":
            self._audit(session.id, "SUBMISSION_BLOCKED", "life_support=YES")
            return self._handoff(
                session,
                reason="SENSITIVE_TOPIC",
                signal="SENSITIVE",
                flags=verdict.flags + ["LIFE_SUPPORT"],
                last_message="",
                note="Life-support equipment declared: a human takes over instead of submitting.",
            )

        try:
            submission = journey_service.submit_journey(
                self.db, payload, call_session_id=session.id, origin="AI_VOICE_AGENT"
            )
        except ValueError as exc:
            self._audit(session.id, "SUBMISSION_BLOCKED", f"payload failed validation: {exc}"[:300])
            return self._handoff(
                session,
                reason="LOW_CONFIDENCE",
                signal="LOW_CONF",
                flags=verdict.flags + ["INVALID_PAYLOAD"],
                last_message="",
                note="The collected details did not pass the journey's own validation. "
                     "Nothing was submitted.",
            )

        # Submitted. The script then asks one closing question, so the call stays open in
        # CLOSING until it is answered (a "yes" hands over to a person).
        message = self.scripts.message("submitted_closing")
        self._say(session, message)
        session.state = "CLOSING"
        session.status = SessionStatus.COMPLETED.value
        session.ended_at = utcnow()
        session.lead.status = LeadStatus.COMPLETED.value
        self._audit(
            session.id,
            "JOURNEY_SUBMITTED",
            f"submission_id={submission.submission_id} payload={json.dumps(payload)}",
        )
        self.db.commit()

        return EngineResult(
            session=session,
            agent_messages=[message],
            terminal=False,
            journey_submitted=True,
            submission_id=submission.submission_id,
            system_notes=[f"Mock journey submitted as {submission.submission_id}."],
        )

    def _handle_closing(
        self, session: CallSession, text: str, verdict: safety_engine.SafetyVerdict
    ) -> EngineResult:
        """Reply to "Is there anything else you need from a team member?"."""
        if verdict.is_decline:
            self._audit(session.id, "CALL_DECLINED", f"matched='{verdict.matched}' after submission")
            if safety_engine.is_do_not_call_request(text):
                self._flag_do_not_call(session)
            return self._finish_call(session, self.scripts.message("decline_closing"))

        if verdict.is_handoff:
            return self._handoff(
                session,
                reason=verdict.reason or "SENSITIVE_TOPIC",
                signal=verdict.escalation_signal,
                flags=verdict.flags,
                last_message=text,
                note=verdict.note,
            )

        answer = validate_yes_no_unsure(text)
        if answer.ok and answer.value == "YES":
            return self._handoff(
                session,
                reason="CUSTOMER_REQUEST",
                signal="ASKS",
                flags=verdict.flags + ["CUSTOMER_REQUEST"],
                last_message=text,
                note="The journey was submitted; the customer has something else for a team member.",
            )
        return self._finish_call(session, self.scripts.message("goodbye"))

    def _finish_call(self, session: CallSession, message: str) -> EngineResult:
        """Close a call whose journey is already submitted."""
        self._say(session, message)
        session.state = "COMPLETED"
        session.status = SessionStatus.COMPLETED.value
        session.ended_at = utcnow()
        self._audit(session.id, "CALL_ENDED", "Journey completed by AI voice agent.")
        self.db.commit()
        return EngineResult(
            session=session,
            agent_messages=[message],
            terminal=True,
            system_notes=["Call ended after submission."],
        )

    # ------------------------------------------------------------------ #
    # 6. Decline
    # ------------------------------------------------------------------ #
    def _decline(
        self, session: CallSession, verdict: safety_engine.SafetyVerdict, text: str
    ) -> EngineResult:
        """Acknowledge once, log it, end. No retry, no persuasion."""
        message = self.scripts.message("decline_closing")
        self._say(session, message)
        do_not_call = safety_engine.is_do_not_call_request(text)
        self._audit(
            session.id,
            "CALL_DECLINED",
            f"matched='{verdict.matched}' — refusal respected, no retry, no pressure.",
        )
        if do_not_call:
            self._flag_do_not_call(session)
        return self._end_call(
            session,
            status=SessionStatus.DECLINED.value,
            outcome_detail="DO_NOT_CALL_REQUESTED" if do_not_call else "CUSTOMER_DECLINED",
            note="Customer asked not to be called again. Lead flagged Do-Not-Call; call ended immediately."
            if do_not_call
            else "Customer declined. Call ended immediately with no retry.",
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
        # Heard but not confirmed yet (e.g. mid read-back): the human should know it too.
        unconfirmed = {
            row.field_name: row.value
            for row in self.db.execute(
                select(JourneyField).where(JourneyField.call_session_id == session.id)
            ).scalars()
            if row.field_name in handoff_fields
            and row.value
            and row.status != FieldStatus.VALID.value
        }
        summary = build_context_summary(
            lead=self._lead_dict(session.lead),
            resume_step=session.resume_step,
            last_completed_step=session.lead.last_completed_step,
            collected=collected,
            reason=reason,
            current_step=session.current_step,
            unconfirmed=unconfirmed,
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
        if self.defer_transfer:
            transfer = {
                "transferred": False,
                "mode": "PHONE_WEBHOOK",
                "detail": "The live call is moved by the webhook's own response.",
            }
        else:
            transfer = voice_providers.warm_transfer(
                session.telephony_reference,
                settings.handoff_transfer_number,
                session_id=session.id,
            )
        self._audit(session.id, "WARM_TRANSFER_ATTEMPT", json.dumps(transfer))
        self._audit(session.id, "HANDOFF_CREATED", f"reason={reason} signal={signal} :: {note}")

        messages = list(extra_messages or []) or [
            self.scripts.handoff_message(reason, life_support="LIFE_SUPPORT" in flags)
        ]
        for message in messages:
            self._say(session, message)
        self._audit(session.id, "CALL_ENDED", f"Autonomous collection stopped: {reason}")
        self.db.commit()

        if transfer.get("transferred"):
            transfer_note = (
                "Live call moved into a conference; the human agent is being dialled into it."
            )
        elif transfer.get("mode") == "PHONE_WEBHOOK":
            transfer_note = "Live call is being moved into a conference with the human agent."
        else:
            transfer_note = "Handoff queued in the console (no live phone call)."
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
        message = self.scripts.message("goodbye")
        if session.state == "CLOSING":
            result = self._finish_call(session, message)
        else:
            self._say(session, message)
            self.db.flush()
            result = self._end_call(
                session,
                status=SessionStatus.DECLINED.value,
                outcome_detail=reason,
                note="Call ended from the console.",
                agent_messages=[message],
            )
        # A live phone call is a real call: say the line, then hang up on the line.
        if session.dial_provider == "twilio":
            voice_providers.end_live_call(session.telephony_reference, message)
        return result

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
        # A call handed to a person is still theirs to complete; every other ended call is not.
        if session.state in TERMINAL_STATES and session.state != "HANDOFF_REQUESTED":
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

        if session.state == "HANDOFF_REQUESTED":
            # The AI has stepped aside, so there is no next question to ask: just keep the value.
            self.db.commit()
            return EngineResult(
                session=session,
                system_notes=[
                    f"{agent_name} {'corrected' if is_correction else 'captured'} '{field_name}' "
                    "after the handoff."
                ],
            )

        # If this was the field we were stuck on, move the journey along.
        if not is_correction and session.current_step == step["step_id"]:
            self.db.flush()
            next_step_id = step.get("next_step") or "confirmation"
            if next_step_id == "SUBMIT":
                next_step_id = "confirmation"
            note = (
                f"{agent_name} captured '{field_name}' directly; the journey advanced "
                f"to '{next_step_id}'."
            )
            if next_step_id == "confirmation":
                result = self._final_readback(session)
                result.system_notes.insert(0, note)
                return result
            session.current_step = next_step_id
            session.state = "COLLECTING_FIELD"
            prompt = self.scripts.render_prompt(next_step_id, **self._collected(session.id))
            self._say(session, prompt)
            self._audit(session.id, "STEP_STARTED", f"{next_step_id} (after human capture)")
            self.db.commit()
            return EngineResult(session=session, agent_messages=[prompt], system_notes=[note])

        self.db.commit()
        return EngineResult(
            session=session,
            system_notes=[
                f"{agent_name} {'corrected' if is_correction else 'captured'} '{field_name}'.",
            ],
        )


    # ------------------------------------------------------------------ #
    # 10. Human completes a handed-off journey
    # ------------------------------------------------------------------ #
    def submit_by_human(
        self,
        session_id: str,
        agent_name: str = "Human Agent",
        customer_confirmed: bool = False,
        life_support_validated: bool = False,
    ) -> EngineResult:
        """A human agent submits the journey for a call that was handed to them.

        The same rules as the AI's own submission apply, and one more: the agent must say
        they read the details back to the customer and the customer confirmed them, which is
        the "final read-back confirmed" item of the checklist done by a person. Nothing is
        submitted unless every field is valid.

        The AI never submits a journey with life support declared. A person can, but only
        after validating it: confirming with the customer that someone at the property uses
        life-support equipment, and taking the case as a vulnerable customer. That
        confirmation is audited. If the customer says the answer was wrong, the agent
        corrects Life support to NO and no validation is needed.
        """
        session = self._get_session(session_id)
        if session is None:
            raise LookupError(f"Unknown call session {session_id}")
        if session.submission is not None:
            raise JourneyConflict("This journey has already been submitted.")
        if session.state != "HANDOFF_REQUESTED":
            raise JourneyConflict(
                "Only a call that was handed to a person can be completed here "
                f"(this one is {session.state})."
            )

        def blocked(reason: str) -> ValueError:
            self._audit(session.id, "HUMAN_SUBMISSION_BLOCKED", f"{reason} by={agent_name}")
            self.db.commit()
            return ValueError(reason)

        if not customer_confirmed:
            raise blocked(
                "Read the details back to the customer and confirm they are correct before submitting."
            )

        collected = self._collected(session.id)
        payload = {
            "lead_id": session.lead_id,
            "vertical": "ENERGY",
            **{name: collected.get(name) for name in self.scripts.required_fields},
        }
        missing = [name for name in self.scripts.required_fields if not payload.get(name)]
        if missing:
            raise blocked(
                "Cannot submit yet: still missing "
                + ", ".join(FIELD_LABELS.get(name, name) for name in missing)
                + "."
            )
        if payload["life_support"] == "YES":
            if not life_support_validated:
                raise blocked(
                    "Life-support equipment was declared at this property. Confirm it with the "
                    "customer and tick the life-support validation to submit. (If the customer says "
                    "it is not right, correct Life support to No instead.)"
                )
            self._audit(
                session.id,
                "LIFE_SUPPORT_VALIDATED_BY_HUMAN",
                f"life_support=YES confirmed with the customer by={agent_name}; handled as a "
                "vulnerable-customer case; no medical details recorded",
            )

        try:
            submission = journey_service.submit_journey(
                self.db, payload, call_session_id=session.id, origin="HUMAN_AGENT"
            )
        except ValueError as exc:
            raise blocked(f"The details did not pass the journey's own validation: {exc}"[:300]) from exc

        session.state = "COMPLETED"
        session.status = SessionStatus.COMPLETED.value
        session.ended_at = utcnow()
        session.lead.status = LeadStatus.COMPLETED.value
        if session.handoff is not None and not session.handoff.accepted_by:
            session.handoff.accepted_by = agent_name
        self._audit(
            session.id,
            "JOURNEY_SUBMITTED",
            f"submission_id={submission.submission_id} origin=HUMAN_AGENT by={agent_name} "
            f"customer_confirmed=true life_support_validated="
            f"{str(payload['life_support'] == 'YES' and life_support_validated).lower()} "
            f"payload={json.dumps(payload)}",
        )
        self._audit(session.id, "CALL_ENDED", "Journey completed by a human agent after handoff.")
        self.db.commit()
        return EngineResult(
            session=session,
            terminal=True,
            journey_submitted=True,
            submission_id=submission.submission_id,
            system_notes=[f"{agent_name} completed the journey as {submission.submission_id}."],
        )


__all__ = ["ConversationEngine", "EngineResult", "JourneyConflict"]
