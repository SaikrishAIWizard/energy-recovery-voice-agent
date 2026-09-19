"""Automatic database seeding.

Idempotent: existing leads are left alone (and any prior call history is preserved), so
restarting the backend never wipes a demo you just ran.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import DATA_DIR
from app.models import Lead, LeadStatus
from app.services.script_service import LEADS_PATH

logger = logging.getLogger(__name__)

LEADS_FILE = LEADS_PATH if LEADS_PATH.exists() else DATA_DIR / "synthetic_leads.json"


def load_synthetic_leads() -> list[dict]:
    with LEADS_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)["leads"]


def seed(db: Session, *, reset: bool = False) -> dict[str, int]:
    if reset:
        for lead in db.execute(select(Lead)).scalars().all():
            db.delete(lead)
        db.flush()

    existing = {row for row in db.execute(select(Lead.id)).scalars().all()}
    created = 0
    now = datetime.now(timezone.utc)

    for offset, entry in enumerate(load_synthetic_leads()):
        if entry["id"] in existing:
            continue
        db.add(
            Lead(
                id=entry["id"],
                first_name=entry["first_name"],
                last_name=entry.get("last_name"),
                phone=entry["phone"],
                email=entry["email"],
                last_completed_step=entry["last_completed_step"],
                dnc_status=bool(entry["dnc_status"]),
                status=LeadStatus.DROPPED_OFF.value,
                vertical=entry.get("vertical", "ENERGY"),
                created_at=now - timedelta(hours=30 - offset * 3),
            )
        )
        created += 1

    db.commit()
    total = db.execute(select(func.count(Lead.id))).scalar_one()
    if created:
        logger.info("Seeded %d synthetic Energy leads (%d total).", created, total)
    return {"created": created, "total": total}


def restore_seed_leads(db: Session) -> int:
    """Put the synthetic leads back exactly as seeded.

    A do-not-call request flags the lead itself, so clearing call history alone would leave
    a demo lead blocked for good. Only the seeded leads are touched; a lead created from an
    uploaded recording keeps its flag.
    """
    restored = 0
    for entry in load_synthetic_leads():
        lead = db.get(Lead, entry["id"])
        if lead is None:
            continue
        lead.dnc_status = bool(entry["dnc_status"])
        lead.status = LeadStatus.DROPPED_OFF.value
        restored += 1
    db.commit()
    return restored


def ensure_seeded(db: Session) -> dict[str, int]:
    return seed(db, reset=False)
