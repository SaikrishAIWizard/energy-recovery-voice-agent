"""Upload a finished call recording and have it checked against the journey checklist."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Lead
from app.schemas import RecordingUploadResponse
from app.services import recording_service, session_service
from app.services.voice_service import voice_providers

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recordings"])

# Generous for a phone call, small enough to hold in memory.
MAX_RECORDING_BYTES = 40 * 1024 * 1024


@router.post("/recordings/upload", response_model=RecordingUploadResponse)
async def upload_recording(
    request: Request,
    lead_id: str | None = None,
    new_first_name: str | None = None,
    new_phone: str | None = None,
    new_email: str | None = None,
    filename: str | None = None,
    db: Session = Depends(get_db),
) -> RecordingUploadResponse:
    """Transcribe an uploaded call recording and place the lead by what it contains.

    Takes the **raw audio body** (not multipart), like `POST /calls/{id}/audio`, so no
    extra dependency is needed. Say whose call it was with `lead_id`, or create a lead on
    the fly with `new_first_name`, `new_phone` and `new_email`.

    Outcomes: every checklist field found -> journey submitted (COMPLETED); anything
    missing -> the lead stays in the recovery queue (INCOMPLETE); a refusal or a safety
    signal in the recording -> DECLINED / HANDOFF_REQUESTED, exactly as on a live call.

    Returns 503 when no server-side STT provider is configured.
    """
    lead: Lead | None = None
    new_lead: dict[str, str] | None = None
    if lead_id:
        lead = db.get(Lead, lead_id)
        if lead is None:
            raise HTTPException(status_code=404, detail=f"Unknown lead {lead_id}")
    elif (new_first_name or "").strip() and (new_phone or "").strip() and (new_email or "").strip():
        if "@" not in new_email:
            raise HTTPException(status_code=422, detail="new_email does not look like an email address.")
        new_lead = {
            "first_name": new_first_name,  # type: ignore[dict-item]
            "phone": new_phone,  # type: ignore[dict-item]
            "email": new_email,
        }
    else:
        raise HTTPException(
            status_code=422,
            detail="Choose an existing lead (lead_id), or give new_first_name, new_phone and "
            "new_email to create one.",
        )

    provider_name = voice_providers.active_stt
    if provider_name not in voice_providers.server_stt:
        raise HTTPException(
            status_code=503,
            detail=(
                "No server-side STT provider configured. Set DEEPGRAM_API_KEY or "
                "ASSEMBLYAI_API_KEY in backend/.env and restart the backend."
            ),
        )

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_RECORDING_BYTES:
        raise HTTPException(status_code=413, detail=_too_large())
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio body.")
    if len(audio) > MAX_RECORDING_BYTES:
        raise HTTPException(status_code=413, detail=_too_large())

    mime_type = (request.headers.get("content-type") or "application/octet-stream").split(";")[0].strip()

    try:
        # Transcription and LLM extraction are blocking network calls; keep them off the
        # event loop so the live call streams stay responsive.
        result = await run_in_threadpool(
            recording_service.process_recording,
            db,
            audio=audio,
            mime_type=mime_type,
            filename=filename,
            lead=lead,
            new_lead=new_lead,
        )
    except recording_service.RecordingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    db.expire_all()
    return RecordingUploadResponse(
        session=session_service.serialize_session(db, result.session),
        outcome=result.outcome,  # type: ignore[arg-type]
        missing_fields=result.missing_fields,
        stt_provider=result.stt_provider,
        notes=result.notes,
    )


def _too_large() -> str:
    return f"Recording is larger than {MAX_RECORDING_BYTES // (1024 * 1024)} MB."
