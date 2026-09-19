"""Do-Not-Call gate.

Stubbed against the ACMA Do Not Call Register, but the *placement* of the check is the
point: it runs before a call session can exist, and a block is terminal and audited.

Swap `_lookup_register` for a real register client (or a dialler pre-dial hook) without
touching anything else.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models import Lead


@dataclass
class DNCCheckResult:
    allowed: bool
    code: str
    detail: str


def _lookup_register(phone: str) -> bool:
    """Returns True when the number is on the register.

    Local stub: the register state lives on the lead record (`dnc_status`), which is how
    the synthetic dataset expresses it. A production build would call the ACMA register
    or the dialler's DNC API here.
    """
    return False


def _customer_opted_out(lead: Lead) -> bool:
    """The lead is flagged because the customer themselves asked us to stop, on a call we
    logged, as opposed to being on the register."""
    return any(
        event.event_type == "DNC_REQUEST_LOGGED"
        for session in lead.call_sessions
        for event in session.audit_events
    )


def check(lead: Lead, force: bool = False) -> DNCCheckResult:
    if lead.dnc_status:
        if _customer_opted_out(lead):
            # A customer's own request is never overridable, demo override included.
            return DNCCheckResult(
                allowed=False,
                code="DNC_CUSTOMER_REQUEST",
                detail=(
                    f"{lead.phone} asked not to be contacted again on an earlier call. "
                    "Dialling refused; this cannot be overridden."
                ),
            )
        if force:
            return DNCCheckResult(
                allowed=True,
                code="DNC_OVERRIDE_DEMO",
                detail=(
                    "DNC override forced from the console. DEMO ONLY — a production build "
                    "must never place this call."
                ),
            )
        return DNCCheckResult(
            allowed=False,
            code="DNC_REGISTER_HIT",
            detail=(
                f"{lead.phone} is flagged on the Do-Not-Call register. "
                "Dialling refused before any call session was created."
            ),
        )

    if _lookup_register(lead.phone):
        return DNCCheckResult(
            allowed=False,
            code="DNC_REGISTER_HIT",
            detail=f"{lead.phone} returned a register hit at dial time.",
        )

    return DNCCheckResult(
        allowed=True,
        code="DNC_CLEAR",
        detail=f"{lead.phone} is clear of the Do-Not-Call register. Dialling permitted.",
    )
