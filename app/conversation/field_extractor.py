"""Field extraction.

Rules-first, LLM-optional. The public signature matches the spec:

    extract_field(current_field, customer_utterance, known_context)
        -> {value, confidence, needs_clarification, handoff_signal}

Python validation is authoritative: an LLM-proposed candidate is re-validated by
`app/conversation/validators.py` and its confidence is capped by the LLM's own
confidence, so a hesitant model can never inflate a weak capture into a valid one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.conversation.validators import validate_field
from app.services.llm_service import LLMExtraction, LLMAdapter, RulesOnlyLLM

# Utterances that are pure filler / acknowledgements and can never be a field value.
_FILLER = re.compile(
    r"^\s*(um+|uh+|er+|hmm+|mm+|ah+|oh+|yeah|yep|ok|okay|right|sure|fine|alright|"
    r"hello|hi|hey|thanks|thank you|no worries)\s*[.!]?\s*$",
    re.IGNORECASE,
)

_STRIP_LEAD_IN = re.compile(
    r"^\s*(um+|uh+|well|so|yeah|yes|ok|okay|sure|look|listen|"
    r"i think|i believe|it'?s|its|that'?s|thats|my|the)\s+",
    re.IGNORECASE,
)


@dataclass
class ExtractionResult:
    value: str | None = None
    display: str | None = None
    confidence: float = 0.0
    needs_clarification: bool = False
    handoff_signal: str | None = None
    valid: bool = False
    source: str = "NONE"
    reason: str = ""
    provider_detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Exactly the shape the spec asks for."""
        return {
            "value": self.value,
            "confidence": round(self.confidence, 2),
            "needs_clarification": self.needs_clarification,
            "handoff_signal": self.handoff_signal,
        }

    def diagnostic(self) -> dict[str, Any]:
        return {**self.as_dict(), "valid": self.valid, "source": self.source, "reason": self.reason}


def _is_filler(text: str) -> bool:
    return bool(_FILLER.match(text or ""))


def _clean_utterance(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,!;\"'")


def extract_field(
    current_field: str,
    customer_utterance: str,
    known_context: dict[str, Any] | None = None,
    *,
    validation_type: str | None = None,
    llm: LLMAdapter | None = None,
) -> ExtractionResult:
    """Extract one field from one customer utterance."""
    context = dict(known_context or {})
    context.setdefault("today", date.today().isoformat())
    adapter: LLMAdapter = llm or RulesOnlyLLM()
    utterance = _clean_utterance(customer_utterance)

    if not utterance:
        return ExtractionResult(
            confidence=0.0,
            needs_clarification=True,
            reason="empty_utterance",
            source="NONE",
        )

    if _is_filler(utterance):
        return ExtractionResult(
            confidence=0.0,
            needs_clarification=True,
            reason="filler_only",
            source="NONE",
        )

    vtype = validation_type or current_field

    # --- Pass 1: deterministic rules ------------------------------------- #
    rules_result = validate_field(vtype, utterance)
    if rules_result.ok:
        return ExtractionResult(
            value=rules_result.value,
            display=rules_result.display or rules_result.value,
            confidence=rules_result.confidence,
            needs_clarification=False,
            valid=True,
            source="RULES",
            reason="rules_match",
        )

    # --- Pass 2: optional LLM candidate, re-validated in Python ---------- #
    llm_extraction: LLMExtraction | None = None
    if adapter.enabled:
        llm_extraction = adapter.extract_field(current_field, utterance, context)
        if llm_extraction and llm_extraction.value:
            confirmed = validate_field(vtype, llm_extraction.value)
            if confirmed.ok:
                return ExtractionResult(
                    value=confirmed.value,
                    display=confirmed.display or confirmed.value,
                    confidence=min(confirmed.confidence, llm_extraction.confidence),
                    needs_clarification=False,
                    valid=True,
                    source="LLM+RULES",
                    reason="llm_candidate_python_validated",
                    provider_detail=llm_extraction.provider,
                )

    # --- Pass 3: nothing usable ------------------------------------------ #
    stripped = _STRIP_LEAD_IN.sub("", utterance).strip()
    if stripped and stripped != utterance:
        retry = validate_field(vtype, stripped)
        if retry.ok:
            return ExtractionResult(
                value=retry.value,
                display=retry.display or retry.value,
                confidence=retry.confidence * 0.97,
                needs_clarification=False,
                valid=True,
                source="RULES",
                reason="rules_match_after_lead_in_strip",
            )

    return ExtractionResult(
        value=None,
        display=None,
        confidence=rules_result.confidence,
        needs_clarification=rules_result.needs_clarification,
        handoff_signal=llm_extraction.handoff_signal if llm_extraction else None,
        valid=False,
        source="NONE",
        reason=rules_result.reason or "no_match",
    )


def build_known_context(
    *,
    lead: dict[str, Any] | None = None,
    collected: dict[str, Any] | None = None,
    current_step: str | None = None,
) -> dict[str, Any]:
    """Context handed to the extractor. Deliberately excludes anything card-shaped and
    never contains free-form transcript text."""
    context: dict[str, Any] = {
        "today": date.today().isoformat(),
        "current_step": current_step,
    }
    if lead:
        context["lead"] = {
            "first_name": lead.get("first_name"),
            "last_completed_step": lead.get("last_completed_step"),
        }
    if collected:
        context["already_collected"] = {
            key: value for key, value in collected.items() if value is not None
        }
    return context
