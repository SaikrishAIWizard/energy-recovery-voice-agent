"""Database models.

Tables 1-7 below are exactly the ones in the spec. `journey_submissions` is the one
addition: the mock journey endpoint has to persist its own receipt (submission id,
payload, timestamp) so the Completed Journey screen and the dashboard can show it
without reverse-engineering it out of the audit log. It is called out in the README.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------- #
# Enumerations (stored as plain strings for SQLite friendliness)
# --------------------------------------------------------------------------- #
class LeadStatus(str, enum.Enum):
    DROPPED_OFF = "DROPPED_OFF"
    IN_CALL = "IN_CALL"
    COMPLETED = "COMPLETED"
    DECLINED = "DECLINED"
    HANDOFF_REQUESTED = "HANDOFF_REQUESTED"
    DNC_BLOCKED = "DNC_BLOCKED"


class SessionStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    DECLINED = "DECLINED"
    HANDOFF_REQUESTED = "HANDOFF_REQUESTED"
    DNC_BLOCKED = "DNC_BLOCKED"
    # An uploaded recording was analysed and checklist items were missing. The lead stays
    # in the recovery queue; it is not a live call and cannot be continued.
    INCOMPLETE = "INCOMPLETE"


# `CallSession.mode` for a session created by uploading a finished call recording.
RECORDING_UPLOAD_MODE = "RECORDING_UPLOAD"


class FieldStatus(str, enum.Enum):
    PENDING = "PENDING"
    VALID = "VALID"
    INVALID = "INVALID"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FieldSource(str, enum.Enum):
    PREEXISTING = "PREEXISTING"
    CUSTOMER_SPOKEN = "CUSTOMER_SPOKEN"
    HUMAN_AGENT = "HUMAN_AGENT"


class Speaker(str, enum.Enum):
    AI_AGENT = "AI_AGENT"
    CUSTOMER = "CUSTOMER"
    HUMAN_AGENT = "HUMAN_AGENT"
    UNKNOWN = "UNKNOWN"


class HandoffReason(str, enum.Enum):
    CUSTOMER_REQUEST = "CUSTOMER_REQUEST"
    FRUSTRATION = "FRUSTRATION"
    REPEATED_FAILURE = "REPEATED_FAILURE"
    SENSITIVE_TOPIC = "SENSITIVE_TOPIC"
    OFF_SCRIPT = "OFF_SCRIPT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


# --------------------------------------------------------------------------- #
# 1. leads
# --------------------------------------------------------------------------- #
class Lead(Base):
    __tablename__ = "leads"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    first_name: Mapped[str] = mapped_column(String(120), nullable=False)
    last_name: Mapped[str | None] = mapped_column(String(120))
    phone: Mapped[str] = mapped_column(String(40), nullable=False)
    email: Mapped[str] = mapped_column(String(200), nullable=False)
    last_completed_step: Mapped[str] = mapped_column(String(64), nullable=False)
    dnc_status: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=LeadStatus.DROPPED_OFF.value)
    vertical: Mapped[str] = mapped_column(String(32), default="ENERGY")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    call_sessions: Mapped[list["CallSession"]] = relationship(
        back_populates="lead", cascade="all, delete-orphan", order_by="CallSession.started_at"
    )


# --------------------------------------------------------------------------- #
# 2. call_sessions
# --------------------------------------------------------------------------- #
class CallSession(Base):
    __tablename__ = "call_sessions"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    lead_id: Mapped[str] = mapped_column(ForeignKey("leads.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default=SessionStatus.ACTIVE.value)
    current_step: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(String(40), default="INIT")
    resume_step: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    recording_consent_disclosed: Mapped[bool] = mapped_column(Boolean, default=False)
    handoff_reason: Mapped[str | None] = mapped_column(String(40))
    # Free-text qualifier for terminal states that the status enum cannot express on its
    # own — e.g. BUSY_CALLBACK_SCHEDULED, DNC_REGISTER_HIT. Keeps the enum clean.
    outcome_detail: Mapped[str | None] = mapped_column(String(64))
    mode: Mapped[str] = mapped_column(String(32), default="AGENT_DRIVEN")
    # Which telephony adapter placed the call, and its handle for a warm transfer.
    dial_provider: Mapped[str | None] = mapped_column(String(40))
    telephony_reference: Mapped[str | None] = mapped_column(String(120))

    lead: Mapped[Lead] = relationship(back_populates="call_sessions")
    journey_fields: Mapped[list["JourneyField"]] = relationship(
        back_populates="call_session", cascade="all, delete-orphan", order_by="JourneyField.id"
    )
    transcript: Mapped[list["TranscriptSegment"]] = relationship(
        back_populates="call_session", cascade="all, delete-orphan", order_by="TranscriptSegment.id"
    )
    handoff: Mapped["Handoff | None"] = relationship(
        back_populates="call_session", cascade="all, delete-orphan", uselist=False
    )
    audit_events: Mapped[list["AuditEvent"]] = relationship(
        back_populates="call_session", cascade="all, delete-orphan", order_by="AuditEvent.id"
    )
    submission: Mapped["JourneySubmission | None"] = relationship(
        back_populates="call_session", cascade="all, delete-orphan", uselist=False
    )


# --------------------------------------------------------------------------- #
# 3. journey_fields
# --------------------------------------------------------------------------- #
class JourneyField(Base):
    __tablename__ = "journey_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_session_id: Mapped[str] = mapped_column(
        ForeignKey("call_sessions.id"), nullable=False, index=True
    )
    field_name: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[str | None] = mapped_column(String(400))
    status: Mapped[str] = mapped_column(String(20), default=FieldStatus.PENDING.value)
    source: Mapped[str] = mapped_column(String(20), default=FieldSource.CUSTOMER_SPOKEN.value)
    confidence: Mapped[float | None] = mapped_column(Float)
    attempts: Mapped[int] = mapped_column(Integer, default=0)

    call_session: Mapped[CallSession] = relationship(back_populates="journey_fields")


# --------------------------------------------------------------------------- #
# 4. transcript_segments
# --------------------------------------------------------------------------- #
class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_session_id: Mapped[str] = mapped_column(
        ForeignKey("call_sessions.id"), nullable=False, index=True
    )
    speaker: Mapped[str] = mapped_column(String(20), nullable=False)
    start_seconds: Mapped[int] = mapped_column(Integer, default=0)
    end_seconds: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    transcription_confidence: Mapped[float | None] = mapped_column(Float)
    redacted: Mapped[bool] = mapped_column(Boolean, default=False)

    call_session: Mapped[CallSession] = relationship(back_populates="transcript")


# --------------------------------------------------------------------------- #
# 5. handoffs
# --------------------------------------------------------------------------- #
class Handoff(Base):
    __tablename__ = "handoffs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_session_id: Mapped[str] = mapped_column(
        ForeignKey("call_sessions.id"), nullable=False, index=True
    )
    reason: Mapped[str] = mapped_column(String(40), nullable=False)
    current_step: Mapped[str] = mapped_column(String(64), nullable=False)
    context_summary: Mapped[str] = mapped_column(Text, nullable=False)
    last_customer_message: Mapped[str | None] = mapped_column(Text)
    safety_flags_json: Mapped[str] = mapped_column(Text, default="[]")
    collected_fields_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    accepted_by: Mapped[str | None] = mapped_column(String(120))

    call_session: Mapped[CallSession] = relationship(back_populates="handoff")


# --------------------------------------------------------------------------- #
# 6. audit_events
# --------------------------------------------------------------------------- #
class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_session_id: Mapped[str] = mapped_column(
        ForeignKey("call_sessions.id"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    event_detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    call_session: Mapped[CallSession] = relationship(back_populates="audit_events")


# --------------------------------------------------------------------------- #
# 7. journey_submissions  (addition — the mock endpoint's receipt)
# --------------------------------------------------------------------------- #
class JourneySubmission(Base):
    __tablename__ = "journey_submissions"

    submission_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    call_session_id: Mapped[str | None] = mapped_column(ForeignKey("call_sessions.id"), index=True)
    lead_id: Mapped[str] = mapped_column(String(32), nullable=False)
    vertical: Mapped[str] = mapped_column(String(32), default="ENERGY")
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="COMPLETED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    call_session: Mapped[CallSession | None] = relationship(back_populates="submission")
