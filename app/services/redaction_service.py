"""Card-data detection and redaction.

The payment boundary is absolute: no card number, expiry, or CVV may ever be stored,
displayed, echoed back, or forwarded into the LLM prompt. This module is applied to
**every** customer utterance before it touches the database, the extractor, or the
transcript the human agent sees.
"""

from __future__ import annotations

import re

REDACTION_TOKEN = "[REDACTED_CARD_DATA]"

# 13-19 digits, optionally grouped: 4111 1111 1111 1111 / 4111-1111-1111-1111 / 4111111111111111
_GROUPED_PAN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_CVV = re.compile(r"\b(cvv|cvc|cvn|security code|card verification)\b[\s:]*\d{3,4}\b", re.IGNORECASE)
_EXPIRY = re.compile(r"\b(expir(y|es|ation)|exp)\b[\s:]*(0[1-9]|1[0-2])\s*[/-]\s*(\d{2,4})", re.IGNORECASE)
_EXPIRY_BARE = re.compile(r"(?<!\d)(0[1-9]|1[0-2])\s*/\s*(\d{2})(?!\d)")

_CARD_CONTEXT = re.compile(
    r"\b(card|credit|debit|visa|mastercard|amex|american express|payment|pay|billing|"
    r"cvv|cvc|expiry|expiration|account number|bank)\b",
    re.IGNORECASE,
)

_PHONE_CONTEXT = re.compile(r"\b(phone|mobile|cell|call me on|reach me on|contact number)\b", re.IGNORECASE)


def _luhn_ok(digits: str) -> bool:
    total, alternate = 0, False
    for char in reversed(digits):
        value = int(char)
        if alternate:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        alternate = not alternate
    return total % 10 == 0


def find_card_like_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, digits) for every card-like sequence in `text`."""
    spans: list[tuple[int, int, str]] = []
    for match in _GROUPED_PAN.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if not (13 <= len(digits) <= 19):
            continue
        spans.append((match.start(), match.end(), digits))
    return spans


def contains_card_data(text: str) -> bool:
    """True when the utterance looks like it is trying to hand us payment details."""
    if not text:
        return False
    if find_card_like_spans(text):
        return True
    if _CVV.search(text):
        return True
    if _CARD_CONTEXT.search(text) and (_EXPIRY.search(text) or _EXPIRY_BARE.search(text)):
        return True
    # "my card number is 4111 1111 ..." — context word plus a long digit run.
    if _CARD_CONTEXT.search(text):
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 13:
            return True
    return False


def redact_text(text: str) -> tuple[str, bool]:
    """Redact card-like sequences. Returns (safe_text, was_redacted).

    Phone numbers are deliberately preserved — they are legitimately part of the lead
    record, and a 10-digit AU number cannot match the 13-19 digit PAN pattern.
    """
    if not text:
        return text, False

    redacted = False
    result = text

    def _mask(match: re.Match[str]) -> str:
        nonlocal redacted
        digits = re.sub(r"\D", "", match.group(0))
        if not (13 <= len(digits) <= 19):
            return match.group(0)
        redacted = True
        return REDACTION_TOKEN

    result = _GROUPED_PAN.sub(_mask, result)

    def _mask_pattern(pattern: re.Pattern[str], replacement: str, source: str) -> str:
        nonlocal redacted
        if pattern.search(source):
            redacted = True
        return pattern.sub(replacement, source)

    result = _mask_pattern(_CVV, r"\1 [REDACTED]", result)
    result = _mask_pattern(_EXPIRY, r"\1 [REDACTED]", result)
    result = _mask_pattern(_EXPIRY_BARE, "[REDACTED_EXPIRY]", result)
    return result, redacted


def redaction_summary(digits: str) -> str:
    """Audit-safe fingerprint: last 4 only, and only for the audit log."""
    tail = digits[-4:] if len(digits) >= 4 else "****"
    return f"pan_len={len(digits)} last4={tail} luhn_valid={_luhn_ok(digits)}"
