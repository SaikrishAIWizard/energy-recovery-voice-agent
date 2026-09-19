"""Live phone calls over Twilio.

The console's "Start call by agent" rings the customer for real. From there Twilio does the
listening and the speaking; this module supplies the instructions (TwiML) for each step:

    customer answers   -> Twilio asks /twilio/voice/{session}   -> we speak the pending prompt
    customer speaks    -> Twilio posts /twilio/turn/{session}   -> the state machine decides,
                                                                    we speak its reply
    handoff            -> customer joins a conference, the human agent's phone is dialled into
                          that same conference after a spoken briefing

Nothing about *who decides* changes. Each turn goes through `ConversationEngine
.handle_utterance`, exactly like a typed or browser-transcribed one: the safety engine, the
Python validators, the confidence floor, the DNC gate and the approved script are all still
in charge, and the agent only ever says what the engine returns. Twilio's speech
recognition is untrusted input like any other transcript.

Every webhook is signed by Twilio. Requests that fail signature validation are refused,
because these endpoints are public and one of them can dial a phone.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from typing import Any
from urllib.parse import parse_qs
from xml.sax.saxutils import escape, quoteattr

from sqlalchemy.orm import Session

from app.config import settings
from app.conversation.state_machine import TERMINAL_STATES, ConversationEngine
from app.models import (
    AuditEvent,
    CallSession,
    LeadStatus,
    SessionStatus,
    Speaker,
    utcnow,
)
from app.services import dnc_service
from app.services.handoff_service import FIELD_LABELS, REASON_SENTENCES
from app.services.voice_service import voice_providers

logger = logging.getLogger(__name__)

# Terminal call states Twilio reports when a call did not connect.
UNANSWERED = {"busy", "no-answer", "failed", "canceled"}

# What the customer hears while the human agent's phone is being dialled. The handoff
# reason text itself comes from the engine.
HOLD_LINE = "One moment, I'm bringing in a colleague now."
NO_AGENT_LINE = (
    "I'm sorry, I couldn't reach a colleague just now. We'll call you back shortly. Goodbye."
)
ENDED_LINE = "This call has ended. Goodbye."


class TwilioError(Exception):
    """A problem to show the caller, carrying the HTTP status it maps to."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# --------------------------------------------------------------------------- #
# Request verification
# --------------------------------------------------------------------------- #


def parse_form(body: bytes) -> dict[str, str]:
    """Twilio posts `application/x-www-form-urlencoded`. Parsed by hand: multipart support
    (and its extra dependency) is not needed for this."""
    parsed = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def expected_signature(url: str, params: dict[str, str], auth_token: str) -> str:
    """Twilio's request signature: HMAC-SHA1 over the full URL plus every POST parameter
    (sorted by name, name immediately followed by value), base64-encoded."""
    payload = url + "".join(f"{key}{params[key]}" for key in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def signature_valid(url: str, params: dict[str, str], header: str | None) -> bool:
    if not settings.twilio_validate_signature:
        return True
    if not header or not settings.twilio_auth_token:
        return False
    return hmac.compare_digest(
        expected_signature(url, params, settings.twilio_auth_token), header
    )


# --------------------------------------------------------------------------- #
# TwiML
# --------------------------------------------------------------------------- #


def _say(text: str) -> str:
    return (
        f"<Say voice={quoteattr(settings.twilio_say_voice)} "
        f"language={quoteattr(settings.twilio_speech_language)}>{escape(text)}</Say>"
    )


def _gather(prompts: list[str], action: str) -> str:
    """Speak the prompt(s) and listen for the reply. The prompt sits inside the Gather so
    the customer can talk over it. `actionOnEmptyResult` sends silence to the same webhook,
    where the state machine treats it as a failed capture (one clarification, then handoff)."""
    return (
        '<Gather input="speech" method="POST" speechTimeout="auto" timeout="6" '
        'speechModel="phone_call" actionOnEmptyResult="true" '
        f"language={quoteattr(settings.twilio_speech_language)} action={quoteattr(action)}>"
        + "".join(_say(p) for p in prompts)
        + "</Gather>"
    )


def _twiml(*parts: str) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response>' + "".join(parts) + "</Response>"


def _hangup(*lines: str) -> str:
    return _twiml(*(_say(line) for line in lines), "<Hangup/>")


def _conference_name(session_id: str) -> str:
    return f"handoff-{session_id}"


def _conference(session_id: str, *, agent: bool) -> str:
    """The customer waits (hold music) until the agent enters; the agent starts the
    conference. Either side leaving ends it, so nobody is left on a dead line."""
    return (
        f'<Conference startConferenceOnEnter="{"true" if agent else "false"}" '
        'endConferenceOnExit="true" beep="false" '
        'statusCallbackEvent="join leave end" statusCallbackMethod="POST" '
        f"statusCallback={quoteattr(settings.public_url(f'/twilio/conference/{session_id}'))}>"
        f"{escape(_conference_name(session_id))}</Conference>"
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _audit(db: Session, session: CallSession, event_type: str, detail: str = "") -> None:
    db.add(AuditEvent(call_session_id=session.id, event_type=event_type, event_detail=detail))


def _mask(number: str | None) -> str:
    digits = "".join(c for c in (number or "") if c.isdigit())
    return f"…{digits[-3:]}" if digits else "—"


def _load(db: Session, session_id: str, params: dict[str, str] | None = None) -> CallSession:
    """The session a webhook is about. When Twilio names the call (`CallSid`), it has to be
    the one this session is actually on — a stale or forged id gets nothing."""
    session = db.get(CallSession, session_id)
    if session is None:
        raise TwilioError(404, f"Unknown call session {session_id}")
    if params is not None and session.telephony_reference:
        if params.get("CallSid") != session.telephony_reference:
            raise TwilioError(403, "CallSid does not belong to this session.")
    return session


def _events(session: CallSession) -> list[str]:
    return [event.event_type for event in session.audit_events]


def _pending_prompt(session: CallSession) -> str | None:
    """The last thing the agent said: the question the customer has not answered yet."""
    for segment in reversed(session.transcript):
        if segment.speaker == Speaker.AI_AGENT.value:
            return segment.text
    return None


def _turn_twiml(session: CallSession, messages: list[str], *, handoff: bool, terminal: bool) -> str:
    if handoff:
        return _twiml(
            *(_say(m) for m in messages),
            f'<Redirect method="POST">'
            f"{escape(settings.public_url(f'/twilio/handoff/{session.id}') + '?spoken=1')}</Redirect>",
        )
    if terminal:
        return _hangup(*messages)
    return _twiml(_gather(messages, settings.public_url(f"/twilio/turn/{session.id}")))


# --------------------------------------------------------------------------- #
# Placing the call
# --------------------------------------------------------------------------- #


def dial_customer(db: Session, session_id: str) -> tuple[CallSession, list[str]]:
    """Ring the customer for a session that is already open in the console."""
    session = db.get(CallSession, session_id)
    if session is None:
        raise TwilioError(404, f"Unknown call session {session_id}")
    if session.state in TERMINAL_STATES:
        raise TwilioError(409, f"This call already ended ({session.state}).")
    if session.mode != "AGENT_DRIVEN":
        raise TwilioError(409, "Phone calls are for agent-driven sessions.")
    if session.dial_provider == "twilio" and session.telephony_reference:
        raise TwilioError(409, "A phone call is already in progress for this session.")
    if not voice_providers.live_calls_available:
        raise TwilioError(
            503,
            "Phone calls are not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
            "TWILIO_FROM_NUMBER and PUBLIC_BASE_URL in backend/.env and restart the backend.",
        )

    # A real call is a real DNC obligation: re-checked now, and never overridden.
    lead = session.lead
    check = dnc_service.check(lead)
    if not check.allowed:
        _audit(db, session, "DNC_BLOCKED", f"live dial refused: {check.detail}")
        db.commit()
        raise TwilioError(409, f"Do-Not-Call: {check.detail}")

    target = settings.twilio_dial_override_number or lead.phone
    result = voice_providers.dial(target, live=True, session_id=session.id)
    if not result.get("placed"):
        _audit(db, session, "PHONE_CALL_FAILED", str(result.get("error") or "unknown error"))
        db.commit()
        raise TwilioError(502, f"Twilio would not place the call: {result.get('error')}")

    session.dial_provider = "twilio"
    session.telephony_reference = result["call_reference"]
    _audit(
        db,
        session,
        "PHONE_CALL_PLACED",
        json.dumps(
            {
                "call_sid": result["call_reference"],
                "to": _mask(target),
                "override_number": bool(settings.twilio_dial_override_number),
            }
        ),
    )
    db.commit()

    notes = [f"Calling {_mask(target)} through Twilio. The assistant speaks when they answer."]
    if settings.twilio_dial_override_number:
        notes.append("TWILIO_DIAL_OVERRIDE_NUMBER is set, so the lead's own number was not dialled.")
    return session, notes


# --------------------------------------------------------------------------- #
# Customer leg
# --------------------------------------------------------------------------- #


def answered(db: Session, session_id: str, params: dict[str, str]) -> str:
    """The customer picked up. Speak the question that is waiting (the consent disclosure
    on a fresh call)."""
    session = _load(db, session_id, params)
    _audit(db, session, "PHONE_CALL_ANSWERED", f"call_sid={params.get('CallSid')}")
    db.commit()

    if session.state in TERMINAL_STATES:
        return _hangup(ENDED_LINE)
    prompt = _pending_prompt(session)
    if prompt is None:
        return _hangup(ENDED_LINE)
    return _twiml(_gather([prompt], settings.public_url(f"/twilio/turn/{session.id}")))


def turn(db: Session, session_id: str, params: dict[str, str]) -> str:
    """One thing the customer said (or a silence). The state machine answers; we speak it."""
    session = _load(db, session_id, params)
    if session.state == "HANDOFF_REQUESTED":
        return _twiml(
            f'<Redirect method="POST">{escape(settings.public_url(f"/twilio/handoff/{session.id}"))}</Redirect>'
        )
    if session.state in TERMINAL_STATES:
        return _hangup(ENDED_LINE)

    text = (params.get("SpeechResult") or "").strip()
    try:
        confidence = float(params["Confidence"]) if params.get("Confidence") else None
    except ValueError:
        confidence = None
    if not text:
        _audit(db, session, "PHONE_SILENCE", "No speech heard before the timeout.")

    engine = ConversationEngine(db, defer_transfer=True)
    result = engine.handle_utterance(
        session.id, text, source="STT_PROVIDER", stt_confidence=confidence
    )
    db.expire_all()
    session = db.get(CallSession, session_id)
    assert session is not None

    messages = result.agent_messages or [m for m in [_pending_prompt(session)] if m]
    return _turn_twiml(
        session, messages, handoff=result.handoff_triggered, terminal=result.terminal
    )


def call_status(db: Session, session_id: str, params: dict[str, str]) -> None:
    """Twilio's status callbacks for the customer's call."""
    session = db.get(CallSession, session_id)
    if session is None:
        raise TwilioError(404, f"Unknown call session {session_id}")
    status = params.get("CallStatus", "")
    if params.get("CallSid") != session.telephony_reference:
        return  # a call from an earlier attempt — nothing to do with the current one

    if status == "ringing":
        _audit(db, session, "PHONE_CALL_RINGING", params.get("CallSid", ""))
    elif status in UNANSWERED or status == "completed":
        if session.state in TERMINAL_STATES:
            _audit(db, session, "PHONE_CALL_ENDED", f"status={status}")
        elif "PHONE_CALL_ANSWERED" not in _events(session):
            # Never picked up: back to the browser channel so the operator can try again.
            _audit(db, session, "PHONE_CALL_UNANSWERED", f"status={status}")
            session.dial_provider = "browser_microphone"
            session.telephony_reference = None
        else:
            _hung_up(db, session, status)
    db.commit()


def _hung_up(db: Session, session: CallSession, status: str) -> None:
    """The customer dropped off part-way. What was captured is kept, and the lead goes back
    to the recovery queue to be resumed rather than started again."""
    session.status = SessionStatus.INCOMPLETE.value
    session.state = "INCOMPLETE"
    session.outcome_detail = "PHONE_HANG_UP"
    session.ended_at = utcnow()
    if session.lead.status == LeadStatus.IN_CALL.value:
        session.lead.status = LeadStatus.DROPPED_OFF.value
    _audit(db, session, "PHONE_CALL_ENDED", f"status={status} — customer hung up before finishing.")
    _audit(db, session, "CALL_ENDED", "Customer hung up. Captured fields are kept for the next call.")


# --------------------------------------------------------------------------- #
# Handoff: the human agent joins the same call
# --------------------------------------------------------------------------- #


def dial_agent(db: Session, session: CallSession, *, force: bool = False) -> dict[str, Any]:
    """Ring the human agent's phone. Their leg joins the customer's conference once they
    accept. Idempotent unless `force`, so a redirect that fires twice rings once."""
    if not force and "HANDOFF_AGENT_DIALLED" in _events(session):
        return {"placed": True, "already": True}
    number = settings.handoff_transfer_number
    if not number:
        _audit(db, session, "HANDOFF_AGENT_UNCONFIGURED", "HANDOFF_TRANSFER_NUMBER is not set.")
        db.commit()
        return {"placed": False, "error": "HANDOFF_TRANSFER_NUMBER is not set."}

    result = voice_providers.place_agent_call(number, session.id)
    if result.get("placed"):
        _audit(
            db,
            session,
            "HANDOFF_AGENT_DIALLED",
            json.dumps({"call_sid": result.get("call_reference"), "to": _mask(number)}),
        )
    else:
        _audit(db, session, "HANDOFF_AGENT_FAILED", str(result.get("error")))
    db.commit()
    return result


def handoff_customer(db: Session, session_id: str, params: dict[str, str], spoken: bool) -> str:
    """Move the customer into the conference and bring the human agent in. `spoken` means the
    engine's own handoff message was already said on the last turn."""
    session = _load(db, session_id, params)
    if session.state != "HANDOFF_REQUESTED":
        return _hangup(ENDED_LINE)

    result = dial_agent(db, session)
    if not result.get("placed"):
        return _hangup(*([] if spoken else [HOLD_LINE]), NO_AGENT_LINE)

    return _twiml(
        *([] if spoken else [_say(HOLD_LINE)]),
        f'<Dial timeLimit="900" method="POST" '
        f"action={quoteattr(settings.public_url(f'/twilio/handoff-ended/{session.id}'))}>"
        f"{_conference(session.id, agent=False)}</Dial>",
    )


def handoff_ended(db: Session, session_id: str, params: dict[str, str]) -> str:
    """The customer's leg left the conference. If the agent never joined, say so."""
    session = _load(db, session_id, params)
    if "HANDOFF_AGENT_JOINED" in _events(session):
        return _twiml("<Hangup/>")
    _audit(db, session, "HANDOFF_AGENT_NEVER_JOINED", "Customer left the hold before an agent joined.")
    db.commit()
    return _hangup(NO_AGENT_LINE)


def _agent_brief(session: CallSession) -> str:
    """What the human hears before joining. Field *names* only: the values are on the
    handoff console, and reading an address or a date aloud to a phone line helps no one."""
    handoff = session.handoff
    lead = session.lead
    parts = [f"Handoff from the energy recovery assistant. The customer is {lead.first_name}."]
    if handoff is not None:
        parts.append(REASON_SENTENCES.get(handoff.reason, "The assistant stepped aside."))
        try:
            collected = json.loads(handoff.collected_fields_json or "{}")
        except json.JSONDecodeError:
            collected = {}
        got = [FIELD_LABELS[k] for k, v in collected.items() if v and k in FIELD_LABELS]
        missing = [FIELD_LABELS[k] for k, v in collected.items() if not v and k in FIELD_LABELS]
        if got:
            parts.append("Already captured: " + ", ".join(got) + ".")
        if missing:
            parts.append("Still needed: " + ", ".join(missing) + ".")
    parts.append("Everything is on your handoff console.")
    return " ".join(parts)


def agent_leg(db: Session, session_id: str) -> str:
    """The agent picked up. Brief them, and only connect them if they press 1 (which also
    keeps a voicemail greeting out of the customer's call)."""
    session = _load(db, session_id)
    _audit(db, session, "HANDOFF_AGENT_ANSWERED", "Agent line answered; briefing in progress.")
    db.commit()
    action = settings.public_url(f"/twilio/agent-accept/{session.id}")
    return _twiml(
        f'<Gather input="dtmf" numDigits="1" timeout="10" method="POST" action={quoteattr(action)}>'
        + _say(_agent_brief(session) + " Press 1 to join the call now.")
        + "</Gather>",
        _say("No response received. Goodbye."),
        "<Hangup/>",
    )


def agent_accept(db: Session, session_id: str, params: dict[str, str]) -> str:
    session = _load(db, session_id)
    if params.get("Digits") != "1":
        _audit(db, session, "HANDOFF_AGENT_DECLINED", f"digits={params.get('Digits')!r}")
        db.commit()
        return _hangup("Understood. Goodbye.")
    _audit(db, session, "HANDOFF_AGENT_ACCEPTED", "Agent pressed 1 and is joining the conference.")
    db.commit()
    if session.state != "HANDOFF_REQUESTED":
        return _hangup("The customer has already left the call. Goodbye.")
    return _twiml(f"<Dial>{_conference(session.id, agent=True)}</Dial>")


def agent_status(db: Session, session_id: str, params: dict[str, str]) -> None:
    session = db.get(CallSession, session_id)
    if session is None:
        raise TwilioError(404, f"Unknown call session {session_id}")
    status = params.get("CallStatus", "")
    if status in UNANSWERED:
        _audit(db, session, "HANDOFF_AGENT_UNAVAILABLE", f"agent line status={status}")
        db.commit()


def conference_status(db: Session, session_id: str, params: dict[str, str]) -> None:
    session = db.get(CallSession, session_id)
    if session is None:
        raise TwilioError(404, f"Unknown call session {session_id}")
    event = params.get("StatusCallbackEvent", "")

    if event == "participant-join" and params.get("CallSid") != session.telephony_reference:
        if "HANDOFF_AGENT_JOINED" not in _events(session):
            _audit(db, session, "HANDOFF_AGENT_JOINED", "Human agent is on the same call.")
            if session.handoff is not None and not session.handoff.accepted_by:
                session.handoff.accepted_by = "Human agent (phone)"
    elif event == "conference-end":
        _audit(db, session, "HANDOFF_CONFERENCE_ENDED", "The handoff call ended.")
    db.commit()
