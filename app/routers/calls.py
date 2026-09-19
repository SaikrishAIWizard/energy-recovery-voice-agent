"""Call endpoints — start a recovery call, drive it turn by turn, and inspect it."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.conversation.state_machine import ConversationEngine
from app.database import SessionLocal, get_db
from app.models import CallSession, Handoff, Lead, Speaker, TranscriptSegment
from app.schemas import (
    AcceptHandoffRequest,
    CallSessionOut,
    EndCallRequest,
    FieldCaptureRequest,
    HandoffOut,
    HandoffRequest,
    StartCallRequest,
    StartCallResponse,
    TranscriptSegmentOut,
    TurnResponse,
    UtteranceRequest,
)
from app.services import handoff_service, session_service
from app.services.journey_service import to_response
from app.services.voice_service import voice_providers

logger = logging.getLogger(__name__)
router = APIRouter(tags=["calls"])


def _load_session(db: Session, call_session_id: str) -> CallSession:
    session = db.get(CallSession, call_session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown call session {call_session_id}")
    return session


def _turn_response(db: Session, result) -> TurnResponse:
    # Refresh so relationships written during the turn (handoff, submission, audit
    # events) are visible, then serialise exactly once.
    db.expire_all()
    session_out = session_service.serialize_session(db, result.session)
    return TurnResponse(
        session=session_out,
        agent_messages=result.agent_messages,
        system_notes=result.system_notes,
        safety_flags=result.safety_flags,
        handoff_triggered=result.handoff_triggered,
        journey_submitted=result.journey_submitted,
        submission=session_out.submission,
        terminal=result.terminal,
    )


@router.post("/calls/start/{lead_id}", response_model=StartCallResponse)
def start_call(
    lead_id: str,
    body: StartCallRequest | None = None,
    db: Session = Depends(get_db),
) -> StartCallResponse:
    request = body or StartCallRequest()
    if db.get(Lead, lead_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown lead {lead_id}")

    engine = ConversationEngine(db)
    result = engine.start_call(lead_id, mode=request.mode, force=request.force)

    if result.blocked:
        return StartCallResponse(
            blocked=True,
            blocked_reason=result.blocked_reason,
            session=session_service.serialize_session(db, result.session),
            agent_messages=[],
            system_notes=result.system_notes,
        )

    return StartCallResponse(
        blocked=False,
        session=session_service.serialize_session(db, result.session),
        agent_messages=result.agent_messages,
        system_notes=result.system_notes,
    )


@router.post("/calls/{call_session_id}/utterance", response_model=TurnResponse)
def post_utterance(
    call_session_id: str,
    body: UtteranceRequest,
    db: Session = Depends(get_db),
) -> TurnResponse:
    _load_session(db, call_session_id)
    engine = ConversationEngine(db)
    try:
        result = engine.handle_utterance(
            call_session_id,
            body.text,
            source=body.source,
            stt_confidence=body.transcription_confidence,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _turn_response(db, result)


@router.post("/calls/{call_session_id}/audio", response_model=TurnResponse)
async def post_audio_turn(
    call_session_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> TurnResponse:
    """Server-side speech-to-text turn.

    Takes the **raw audio body** (not multipart) so no extra dependency is needed, routes
    it through whichever STT provider is configured, and feeds the resulting text into the
    exact same `handle_utterance` pipeline a typed or browser-transcribed turn uses.

    The transcript is untrusted input. It is redacted, screened by the safety engine, and
    validated by the Python validators exactly like anything the customer types — a vendor
    cannot bypass the guardrails by returning a clever transcript.

    Returns 503 when no server-side provider is configured. That is the default state, and
    it is not an error: the browser is the transcriber and needs no credentials.
    """
    _load_session(db, call_session_id)

    provider_name = voice_providers.active_stt
    if provider_name not in voice_providers.server_stt:
        raise HTTPException(
            status_code=503,
            detail=(
                "No server-side STT provider configured. Set DEEPGRAM_API_KEY or "
                "ASSEMBLYAI_API_KEY in backend/.env and restart, or transcribe in the "
                "browser and POST text to /calls/{id}/utterance."
            ),
        )

    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio body.")

    mime_type = (request.headers.get("content-type") or "audio/webm").split(";")[0].strip()
    result = voice_providers.transcribe(audio, mime_type)

    text = (result.get("text") or "").strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{provider_name} returned no transcript: "
                f"{result.get('error') or 'empty result'}"
            ),
        )

    confidence = result.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence = max(0.0, min(1.0, float(confidence)))
    else:
        confidence = None

    engine = ConversationEngine(db)
    try:
        turn = engine.handle_utterance(
            call_session_id,
            text,
            source="STT_PROVIDER",
            stt_confidence=confidence,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _turn_response(db, turn)


@router.get("/calls/{call_session_id}", response_model=CallSessionOut)
def get_call(call_session_id: str, db: Session = Depends(get_db)) -> CallSessionOut:
    return session_service.serialize_session(db, _load_session(db, call_session_id))


@router.get("/calls/{call_session_id}/transcript", response_model=list[TranscriptSegmentOut])
def get_transcript(
    call_session_id: str, db: Session = Depends(get_db)
) -> list[TranscriptSegmentOut]:
    session = _load_session(db, call_session_id)
    return [TranscriptSegmentOut.model_validate(seg) for seg in session.transcript]


@router.get("/calls/{call_session_id}/handoff", response_model=HandoffOut)
def get_handoff(call_session_id: str, db: Session = Depends(get_db)) -> HandoffOut:
    session = _load_session(db, call_session_id)
    if session.handoff is None:
        raise HTTPException(status_code=404, detail="No handoff recorded for this call")
    payload = handoff_service.serialize_handoff(session.handoff, session.lead)
    return HandoffOut(**payload)


@router.post("/calls/{call_session_id}/handoff", response_model=TurnResponse)
def request_handoff(
    call_session_id: str,
    body: HandoffRequest | None = None,
    db: Session = Depends(get_db),
) -> TurnResponse:
    _load_session(db, call_session_id)
    request = body or HandoffRequest()
    engine = ConversationEngine(db)
    result = engine.request_handoff(call_session_id, request.reason, request.note)
    return _turn_response(db, result)


@router.post("/calls/{call_session_id}/handoff/accept", response_model=HandoffOut)
def accept_handoff(
    call_session_id: str,
    body: AcceptHandoffRequest,
    db: Session = Depends(get_db),
) -> HandoffOut:
    session = _load_session(db, call_session_id)
    if session.handoff is None:
        raise HTTPException(status_code=404, detail="No handoff recorded for this call")

    session.handoff.accepted_by = body.accepted_by
    db.add(
        TranscriptSegment(
            call_session_id=session.id,
            speaker=Speaker.HUMAN_AGENT.value,
            start_seconds=session.transcript[-1].end_seconds if session.transcript else 0,
            end_seconds=(session.transcript[-1].end_seconds if session.transcript else 0) + 4,
            text=(
                f"Hi {session.lead.first_name}, this is {body.accepted_by} — I can see "
                "everything so far, no need to repeat anything. Let me sort this out for you."
            ),
        )
    )
    from app.models import AuditEvent

    db.add(
        AuditEvent(
            call_session_id=session.id,
            event_type="HANDOFF_ACCEPTED",
            event_detail=f"accepted_by={body.accepted_by}",
        )
    )
    db.commit()
    db.refresh(session)
    return HandoffOut(**handoff_service.serialize_handoff(session.handoff, session.lead))


@router.post("/calls/{call_session_id}/field", response_model=TurnResponse)
def capture_field(
    call_session_id: str,
    body: FieldCaptureRequest,
    db: Session = Depends(get_db),
) -> TurnResponse:
    """Agent-Assisted mode: a human agent captures or corrects one journey field live."""
    _load_session(db, call_session_id)
    engine = ConversationEngine(db)
    try:
        result = engine.capture_field_by_human(
            call_session_id, body.field_name, body.value, body.agent_name
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _turn_response(db, result)


@router.post("/calls/{call_session_id}/end", response_model=TurnResponse)
def end_call(
    call_session_id: str,
    body: EndCallRequest | None = None,
    db: Session = Depends(get_db),
) -> TurnResponse:
    _load_session(db, call_session_id)
    engine = ConversationEngine(db)
    result = engine.end_call(call_session_id, (body or EndCallRequest()).reason)
    return _turn_response(db, result)


@router.websocket("/calls/{call_session_id}/stream")
async def stream_call(websocket: WebSocket, call_session_id: str) -> None:
    """Live session feed.

    Deliberately a lightweight DB poll rather than an in-process pub/sub bus: it stays
    correct if the turn came from another tab, another client, or a REST call, and it
    needs no cross-thread asyncio plumbing. Simple beats clever for a 12-hour build.
    """
    await websocket.accept()
    last_signature: str | None = None
    try:
        while True:
            db = SessionLocal()
            try:
                session = db.get(CallSession, call_session_id)
                if session is None:
                    await websocket.send_json({"type": "error", "detail": "unknown_session"})
                    await websocket.close()
                    return
                payload = session_service.serialize_session(db, session).model_dump(mode="json")
            finally:
                db.close()

            signature = json.dumps(
                [
                    payload["state"],
                    payload["status"],
                    payload["current_step"],
                    len(payload["transcript"]),
                    len(payload["audit_events"]),
                ]
            )
            if signature != last_signature:
                last_signature = signature
                await websocket.send_json({"type": "session", "session": payload})
            await asyncio.sleep(0.6)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("stream closed: %s", exc)
        return
