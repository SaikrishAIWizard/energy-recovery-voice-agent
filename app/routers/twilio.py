"""Live phone calls: the console's "Start call by agent" and Twilio's webhooks.

Two kinds of endpoint live here:

  * `POST /calls/{id}/dial` and `POST /calls/{id}/handoff/dial-agent` — called by the
    console (no auth, like the rest of the local demo API).
  * `POST /twilio/...` — called by Twilio during a call. They are public, so each one checks
    Twilio's request signature first (see `twilio_service.signature_valid`).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import CallSession
from app.schemas import TurnResponse
from app.services import session_service, twilio_service
from app.services.twilio_service import TwilioError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["twilio"])


def _twiml(xml: str) -> Response:
    return Response(content=xml, media_type="application/xml")


async def verified_form(request: Request) -> dict[str, str]:
    """The form Twilio posted, once its signature checks out."""
    params = twilio_service.parse_form(await request.body())
    url = settings.public_url(request.url.path)
    if request.url.query:
        url += f"?{request.url.query}"
    if not twilio_service.signature_valid(url, params, request.headers.get("X-Twilio-Signature")):
        logger.warning("Rejected a Twilio webhook with a bad signature: %s", request.url.path)
        raise HTTPException(status_code=403, detail="Invalid Twilio signature.")
    return params


# --------------------------------------------------------------------------- #
# Console -> backend
# --------------------------------------------------------------------------- #


@router.post("/calls/{call_session_id}/dial", response_model=TurnResponse)
def dial_customer(call_session_id: str, db: Session = Depends(get_db)) -> TurnResponse:
    """"Start call by agent": the assistant phones the customer through Twilio.

    The DNC register is checked again first and can not be overridden for a real call.
    503 when Twilio or PUBLIC_BASE_URL is not configured.
    """
    session, notes = twilio_service.dial_customer(db, call_session_id)
    db.expire_all()
    return TurnResponse(
        session=session_service.serialize_session(db, session), system_notes=notes
    )


@router.post("/calls/{call_session_id}/handoff/dial-agent", response_model=TurnResponse)
def redial_agent(call_session_id: str, db: Session = Depends(get_db)) -> TurnResponse:
    """Ring the human agent again (they missed it). The customer is still on hold."""
    session = db.get(CallSession, call_session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown call session {call_session_id}")
    if session.state != "HANDOFF_REQUESTED" or session.dial_provider != "twilio":
        raise HTTPException(status_code=409, detail="There is no live phone handoff to add an agent to.")
    result = twilio_service.dial_agent(db, session, force=True)
    if not result.get("placed"):
        raise HTTPException(status_code=502, detail=f"Could not ring the agent: {result.get('error')}")
    db.expire_all()
    return TurnResponse(
        session=session_service.serialize_session(db, session),
        system_notes=["Ringing the human agent again."],
    )


# --------------------------------------------------------------------------- #
# Twilio -> backend: the customer's leg
# --------------------------------------------------------------------------- #


@router.post("/twilio/voice/{call_session_id}")
def voice(
    call_session_id: str, params: dict[str, str] = Depends(verified_form), db: Session = Depends(get_db)
) -> Response:
    return _twiml(twilio_service.answered(db, call_session_id, params))


@router.post("/twilio/turn/{call_session_id}")
def turn(
    call_session_id: str, params: dict[str, str] = Depends(verified_form), db: Session = Depends(get_db)
) -> Response:
    return _twiml(twilio_service.turn(db, call_session_id, params))


@router.post("/twilio/status/{call_session_id}")
def status(
    call_session_id: str, params: dict[str, str] = Depends(verified_form), db: Session = Depends(get_db)
) -> Response:
    twilio_service.call_status(db, call_session_id, params)
    return Response(status_code=204)


# --------------------------------------------------------------------------- #
# Twilio -> backend: handoff conference and the human agent's leg
# --------------------------------------------------------------------------- #


@router.post("/twilio/handoff/{call_session_id}")
def handoff(
    call_session_id: str,
    spoken: int = 0,
    notice: str = "",
    params: dict[str, str] = Depends(verified_form),
    db: Session = Depends(get_db),
) -> Response:
    return _twiml(
        twilio_service.handoff_customer(db, call_session_id, params, bool(spoken), notice)
    )


@router.post("/twilio/handoff-ended/{call_session_id}")
def handoff_ended(
    call_session_id: str, params: dict[str, str] = Depends(verified_form), db: Session = Depends(get_db)
) -> Response:
    return _twiml(twilio_service.handoff_ended(db, call_session_id, params))


@router.post("/twilio/agent/{call_session_id}")
def agent(
    call_session_id: str, params: dict[str, str] = Depends(verified_form), db: Session = Depends(get_db)
) -> Response:
    return _twiml(twilio_service.agent_leg(db, call_session_id))


@router.post("/twilio/agent-accept/{call_session_id}")
def agent_accept(
    call_session_id: str, params: dict[str, str] = Depends(verified_form), db: Session = Depends(get_db)
) -> Response:
    return _twiml(twilio_service.agent_accept(db, call_session_id, params))


@router.post("/twilio/agent-status/{call_session_id}")
def agent_status(
    call_session_id: str, params: dict[str, str] = Depends(verified_form), db: Session = Depends(get_db)
) -> Response:
    twilio_service.agent_status(db, call_session_id, params)
    return Response(status_code=204)


@router.post("/twilio/conference/{call_session_id}")
def conference(
    call_session_id: str, params: dict[str, str] = Depends(verified_form), db: Session = Depends(get_db)
) -> Response:
    twilio_service.conference_status(db, call_session_id, params)
    return Response(status_code=204)


__all__ = ["router", "TwilioError"]
