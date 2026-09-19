"""FastAPI application entrypoint.

Run with:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from app.config import ENV_FILE, MIN_FIELD_CONFIDENCE, settings
from app.database import SessionLocal, init_db
from app.models import CallSession, Lead
from app.routers import calls, dashboard, database_viewer, journey, leads
from app.seed import ensure_seeded
from app.services.llm_service import llm_status
from app.services.script_service import script_service
from app.services.voice_service import voice_providers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("energy_recovery")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        if settings.seed_on_startup:
            result = ensure_seeded(db)
            logger.info("Seed check complete: %s", result)
        lead_count = db.execute(select(func.count(Lead.id))).scalar_one()
        session_count = db.execute(select(func.count(CallSession.id))).scalar_one()
        logger.info("Database ready — %d leads, %d call sessions.", lead_count, session_count)
    finally:
        db.close()

    # Print exactly what is live. If you set a key in backend/.env and it does not show up
    # here, the key was not read — far better to learn that from the banner than from a
    # demo that silently behaves identically with and without credentials.
    if ENV_FILE:
        logger.info("Configuration file: backend/%s", ENV_FILE)
    else:
        logger.info("No backend/.env found — using the process environment only.")

    logger.info("LLM adapter: %s", llm_status())
    logger.info("Voice providers: %s", voice_providers.describe()["active"])
    logger.info(
        "Server-side STT: %s",
        ", ".join(voice_providers.server_stt) or "none (the browser transcribes)",
    )
    logger.info(
        "Core demo runs with NO API keys. Confidence floor for a field capture: %.2f",
        MIN_FIELD_CONFIDENCE,
    )
    yield

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "AI-assisted voice recovery for dropped Energy comparison leads. Deterministic "
        "state machine, warm human handoff, and a mock journey-completion endpoint. "
        "Runs fully offline with no API keys."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(leads.router)
app.include_router(calls.router)
app.include_router(journey.router)
app.include_router(dashboard.router)
app.include_router(database_viewer.router)


@app.get("/health", tags=["system"])
def health() -> dict:
    db = SessionLocal()
    try:
        lead_count = db.execute(select(func.count(Lead.id))).scalar_one()
        session_count = db.execute(select(func.count(CallSession.id))).scalar_one()
    finally:
        db.close()

    return {
        "status": "ok",
        "app": settings.app_name,
        "version": "1.0.0",
        "database": settings.database_url,
        "leads": lead_count,
        "call_sessions": session_count,
        "journey": {
            "vertical": "ENERGY",
            "journey_id": script_service.raw["journey_id"],
            "script_version": script_service.raw["version"],
            "steps": len(script_service.steps()),
            "required_fields": script_service.required_fields,
        },
        "capabilities": settings.capability_report(),
        "voice_providers": voice_providers.describe(),
        "llm": llm_status(),
        "guardrails": {
            "recording_disclosure_required_before_data_collection": True,
            "dnc_check_before_call_creation": True,
            "card_data_never_captured": True,
            "no_product_or_financial_advice": True,
            "refusals_respected_no_retry": True,
            "min_field_confidence": MIN_FIELD_CONFIDENCE,
        },
    }


@app.post("/demo/reset", tags=["system"])
def demo_reset(clear_calls: bool = True) -> dict:
    """Reset the demo. Deletes call history and restores the synthetic leads."""
    db = SessionLocal()
    try:
        if clear_calls:
            for session in db.execute(select(CallSession)).scalars().all():
                db.delete(session)
            db.commit()
        result = ensure_seeded(db)
    finally:
        db.close()
    return {"reset": True, "calls_cleared": clear_calls, "leads": result}
