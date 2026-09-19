"""Warm-handoff package builder.

The contract with the human agent is simple: **the customer never repeats themselves.**
Everything collected, everything said last, and why the agent stepped aside travels with
the call.
"""

from __future__ import annotations

import json
from typing import Any

from app.models import Handoff, Speaker, TranscriptSegment

FIELD_LABELS = {
    "property_address": "Service address",
    "move_in_date": "Move-in date",
    "energy_requirement": "Supply needed",
    "concession_status": "Concession",
    "life_support": "Life support",
    "contact_preference": "Contact preference",
}

REASON_SENTENCES = {
    "CUSTOMER_REQUEST": "The customer asked to speak with a person.",
    "FRUSTRATION": "The customer became frustrated and the agent stepped aside.",
    "REPEATED_FAILURE": "The agent could not capture the required detail reliably.",
    "SENSITIVE_TOPIC": "A sensitive, disputed, or vulnerable-customer topic was raised.",
    "OFF_SCRIPT": "The customer raised something outside the approved script.",
    "LOW_CONFIDENCE": "The agent was not confident enough in what was said to continue.",
}

ESCALATION_LABELS = {
    "CUSTOMER_REQUEST": "ASKS",
    "FRUSTRATION": "ANGER",
    "REPEATED_FAILURE": "CONFUSION",
    "SENSITIVE_TOPIC": "SENSITIVE",
    "OFF_SCRIPT": "OFF-SCRIPT",
    "LOW_CONFIDENCE": "LOW_CONF",
}


def build_context_summary(
    *,
    lead: dict[str, Any],
    resume_step: str | None,
    last_completed_step: str | None,
    collected: dict[str, Any],
    reason: str,
    current_step: str,
    unconfirmed: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []

    parts.append(
        "Recording consent was disclosed before any data collection (AU two-party consent)."
    )

    if last_completed_step:
        parts.append(
            f"Journey resumed from the lead's last completed step '{last_completed_step}'"
            + (f" at '{resume_step}'." if resume_step else ".")
        )

    preexisting = [
        FIELD_LABELS.get(key, key)
        for key, value in collected.items()
        if value is not None and key in FIELD_LABELS
    ]
    if preexisting:
        parts.append("Already captured on this call: " + ", ".join(preexisting) + ".")

    heard = [
        f"{FIELD_LABELS.get(key, key)} ({value})"
        for key, value in (unconfirmed or {}).items()
        if value and collected.get(key) in (None, "")
    ]
    if heard:
        parts.append("Heard but not yet confirmed by the customer: " + ", ".join(heard) + ".")

    missing = [
        FIELD_LABELS.get(key, key)
        for key in FIELD_LABELS
        if collected.get(key) in (None, "")
    ]
    if missing:
        parts.append("Still outstanding: " + ", ".join(missing) + ".")

    parts.append(
        f"{REASON_SENTENCES.get(reason, 'The agent stepped aside.')} "
        f"Handed over while at step '{current_step}'."
    )
    return " ".join(parts)


def serialize_handoff(handoff: Handoff, lead: Any | None = None) -> dict[str, Any]:
    try:
        safety_flags = json.loads(handoff.safety_flags_json or "[]")
    except json.JSONDecodeError:
        safety_flags = []
    try:
        collected_fields = json.loads(handoff.collected_fields_json or "{}")
    except json.JSONDecodeError:
        collected_fields = {}

    # Everything the human needs so the customer never repeats themselves: what was
    # captured (with how sure we are and where it came from), what is still outstanding,
    # and whether the recording disclosure was made.
    session = handoff.call_session
    rows = {row.field_name: row for row in (session.journey_fields if session else [])}
    field_details: dict[str, dict[str, Any]] = {}
    for name in FIELD_LABELS:
        row = rows.get(name)
        if row is not None and row.value is not None:
            field_details[name] = {
                "value": row.value,
                "status": row.status,
                "confidence": row.confidence,
                "source": row.source,
            }
    # Live, not a snapshot: it shrinks as the human agent fills the gaps in.
    outstanding_fields = [
        name
        for name in FIELD_LABELS
        if not (rows.get(name) is not None and rows[name].status == "VALID" and rows[name].value)
    ]

    known_contact: dict[str, Any] = {}
    if lead is not None:
        known_contact = {
            "first_name": lead.first_name,
            "last_name": lead.last_name,
            "phone": lead.phone,
            "email": lead.email,
            "last_completed_step": lead.last_completed_step,
        }

    return {
        "id": handoff.id,
        "reason": handoff.reason,
        "current_step": handoff.current_step,
        "context_summary": handoff.context_summary,
        "last_customer_message": handoff.last_customer_message,
        "safety_flags": safety_flags,
        "collected_fields": collected_fields,
        "created_at": handoff.created_at,
        "accepted_by": handoff.accepted_by,
        "escalation_signal": ESCALATION_LABELS.get(handoff.reason),
        "lead_id": lead.id if lead is not None else None,
        "known_contact": known_contact,
        "outstanding_fields": outstanding_fields,
        "field_details": field_details,
        "recording_disclosed": bool(session.recording_consent_disclosed) if session else False,
    }


def build_handoff_package(
    *,
    lead: dict[str, Any],
    reason: str,
    current_step: str,
    collected: dict[str, Any],
    last_customer_message: str,
    conversation_summary: str,
    safety_flags: list[str],
) -> dict[str, Any]:
    """The exact shape from the spec — used by the console and the API."""
    return {
        "lead_id": lead["id"],
        "handoff_reason": reason,
        "current_step": current_step,
        "collected_fields": {
            key: collected.get(key)
            for key in [
                "property_address",
                "move_in_date",
                "energy_requirement",
                "concession_status",
                "life_support",
                "contact_preference",
            ]
        },
        "last_customer_message": last_customer_message,
        "conversation_summary": conversation_summary,
        "safety_flags": sorted(set(safety_flags)),
        "escalation_signal": ESCALATION_LABELS.get(reason),
        "known_contact": {
            "first_name": lead.get("first_name"),
            "phone": lead.get("phone"),
            "email": lead.get("email"),
        },
    }


def last_customer_utterance(segments: list[TranscriptSegment]) -> str | None:
    for segment in reversed(segments):
        if segment.speaker == Speaker.CUSTOMER.value:
            return segment.text
    return None
