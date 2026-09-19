"""Pydantic request/response contracts.

The console is deliberately dumb: every turn endpoint returns the *entire* session
snapshot, so the UI never has to reconstruct state or replay events to stay in sync.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Core entities
# --------------------------------------------------------------------------- #


class LeadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    first_name: str
    last_name: str | None = None
    phone: str
    email: str
    last_completed_step: str
    dnc_status: bool
    status: str
    vertical: str
    created_at: datetime


class JourneyFieldOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    field_name: str
    value: str | None
    status: str
    source: str
    confidence: float | None
    attempts: int


class TranscriptSegmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    speaker: str
    start_seconds: int
    end_seconds: int
    text: str
    transcription_confidence: float | None
    redacted: bool


class AuditEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_type: str
    event_detail: str
    created_at: datetime


class HandoffOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    reason: str
    current_step: str
    context_summary: str
    last_customer_message: str | None
    safety_flags: list[str] = Field(default_factory=list)
    collected_fields: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    accepted_by: str | None = None
    escalation_signal: str | None = None
    lead_id: str | None = None
    known_contact: dict[str, Any] = Field(default_factory=dict)
    # The rest of the warm-handoff package: what is still needed, how sure we are of each
    # captured value and where it came from, and that the recording was disclosed.
    outstanding_fields: list[str] = Field(default_factory=list)
    field_details: dict[str, dict[str, Any]] = Field(default_factory=dict)
    recording_disclosed: bool = False


class SubmissionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    submission_id: str
    lead_id: str
    vertical: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    call_session_id: str | None = None


class StepProgress(BaseModel):
    step_id: str
    label: str
    status: Literal["DONE", "ACTIVE", "PENDING", "SKIPPED", "FAILED"]


class CallSessionOut(BaseModel):
    id: str
    lead_id: str
    status: str
    state: str
    current_step: str
    resume_step: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    recording_consent_disclosed: bool
    handoff_reason: str | None = None
    outcome_detail: str | None = None
    mode: str
    dial_provider: str | None = None
    telephony_reference: str | None = None

    lead: LeadOut
    journey_fields: list[JourneyFieldOut] = Field(default_factory=list)
    transcript: list[TranscriptSegmentOut] = Field(default_factory=list)
    audit_events: list[AuditEventOut] = Field(default_factory=list)
    handoff: HandoffOut | None = None
    submission: SubmissionOut | None = None

    journey_progress: list[StepProgress] = Field(default_factory=list)
    collected_fields: dict[str, Any] = Field(default_factory=dict)
    known_fields: dict[str, Any] = Field(default_factory=dict)
    missing_required_fields: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    last_agent_message: str | None = None
    last_customer_message: str | None = None
    demo_replies: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class StartCallRequest(BaseModel):
    mode: Literal["AGENT_DRIVEN", "AGENT_ASSISTED"] = "AGENT_DRIVEN"
    force: bool = Field(
        default=False,
        description="Demo-only override that bypasses the DNC gate. Never enable in production.",
    )


class UtteranceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    source: Literal["BROWSER_SPEECH", "SIMULATED", "STT_PROVIDER", "HUMAN_TYPED"] = "SIMULATED"
    transcription_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class HandoffRequest(BaseModel):
    reason: Literal[
        "CUSTOMER_REQUEST",
        "FRUSTRATION",
        "REPEATED_FAILURE",
        "SENSITIVE_TOPIC",
        "OFF_SCRIPT",
        "LOW_CONFIDENCE",
    ] = "CUSTOMER_REQUEST"
    note: str | None = None


class AcceptHandoffRequest(BaseModel):
    accepted_by: str = Field(min_length=1, max_length=120)


class FieldCaptureRequest(BaseModel):
    """Agent-Assisted (Mode A): a human supplies or corrects a journey field live.

    The value still passes through the same Python validator as a spoken answer.
    """

    field_name: Literal[
        "property_address",
        "move_in_date",
        "energy_requirement",
        "concession_status",
        "life_support",
        "contact_preference",
    ]
    value: str = Field(min_length=1, max_length=400)
    agent_name: str = Field(default="Human Agent", min_length=1, max_length=120)


class HandoffSubmitRequest(BaseModel):
    """A human agent completing a handed-off journey."""

    agent_name: str = Field(default="Human Agent", min_length=1, max_length=120)
    customer_confirmed: bool = Field(
        default=False,
        description="The agent read the details back and the customer confirmed them.",
    )
    life_support_validated: bool = Field(
        default=False,
        description=(
            "Required when life support is YES: the agent confirmed it with the customer and is "
            "handling this as a vulnerable-customer case. Ignored otherwise."
        ),
    )


class EndCallRequest(BaseModel):
    reason: str = "AGENT_ENDED"


class JourneySubmitRequest(BaseModel):
    """Exactly the payload shape the console posts today."""

    lead_id: str = Field(min_length=1)
    vertical: Literal["ENERGY"] = "ENERGY"
    property_address: str = Field(min_length=1)
    move_in_date: str = Field(min_length=1)
    energy_requirement: Literal["ELECTRICITY", "GAS", "BOTH"]
    concession_status: Literal["YES", "NO", "UNSURE"]
    life_support: Literal["YES", "NO"]
    contact_preference: Literal["PHONE", "EMAIL"]
    call_session_id: str | None = None

    @field_validator("move_in_date")
    @classmethod
    def _date_must_be_iso(cls, value: str) -> str:
        from datetime import date

        try:
            date.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - exercised via API
            raise ValueError("move_in_date must be an ISO date (YYYY-MM-DD)") from exc
        return value


class JourneySubmitResponse(BaseModel):
    success: bool
    submission_id: str | None = None
    status: str
    errors: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Turn / start results
# --------------------------------------------------------------------------- #


class TurnResponse(BaseModel):
    session: CallSessionOut
    agent_messages: list[str] = Field(default_factory=list)
    system_notes: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    handoff_triggered: bool = False
    journey_submitted: bool = False
    submission: SubmissionOut | None = None
    terminal: bool = False


class StartCallResponse(BaseModel):
    blocked: bool = False
    blocked_reason: str | None = None
    session: CallSessionOut | None = None
    agent_messages: list[str] = Field(default_factory=list)
    system_notes: list[str] = Field(default_factory=list)


class SessionSummary(BaseModel):
    """One call in a lead's history — enough to recognise it and open it."""

    id: str
    status: str
    mode: str
    dial_provider: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    outcome_detail: str | None = None
    handoff_reason: str | None = None
    transcript_segments: int = 0
    fields_captured: int = 0


class HandoffListItem(BaseModel):
    """A call waiting on a human, whether or not it is still the lead's latest call."""

    session_id: str
    lead_id: str
    name: str
    reason: str
    accepted_by: str | None = None
    created_at: datetime
    context_summary: str


class RecordingUploadResponse(BaseModel):
    """Result of analysing an uploaded call recording."""

    session: CallSessionOut
    outcome: Literal["COMPLETED", "INCOMPLETE", "DECLINED", "HANDOFF_REQUESTED"]
    missing_fields: list[str] = Field(default_factory=list)
    stt_provider: str
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Lead queue
# --------------------------------------------------------------------------- #


class LeadQueueItem(BaseModel):
    lead: LeadOut
    resume_step: str
    scenario_notes: str | None = None
    expected_outcome: str | None = None
    latest_session_id: str | None = None
    latest_session_status: str | None = None
    outcome_detail: str | None = None
    handoff_reason: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    can_start: bool = True


class LeadDetailOut(BaseModel):
    lead: LeadOut
    resume_step: str
    scenario_notes: str | None = None
    expected_outcome: str | None = None
    scripted_turns: dict[str, str] = Field(default_factory=dict)
    preexisting_fields: dict[str, Any] = Field(default_factory=dict)
    field_prompts: dict[str, str] = Field(default_factory=dict)
    latest_session: CallSessionOut | None = None
    dnc_check: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #


class DashboardCounts(BaseModel):
    dropped_off: int = 0
    active_calls: int = 0
    completed: int = 0
    handoffs: int = 0
    declined: int = 0
    dnc_blocked: int = 0


class DashboardRow(BaseModel):
    lead_id: str
    name: str
    phone: str
    last_completed_step: str
    dnc_status: bool
    call_status: str
    call_session_id: str | None = None
    outcome_detail: str | None = None
    handoff_reason: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class DashboardSummary(BaseModel):
    counts: DashboardCounts
    rows: list[DashboardRow]
    completion_rate: float = 0.0
    handoff_rate: float = 0.0
    calls_placed: int = 0
    fields_captured_automatically: int = 0
    estimated_manual_minutes_saved: float = 0.0
