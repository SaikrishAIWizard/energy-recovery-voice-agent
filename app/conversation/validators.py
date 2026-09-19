"""Field validators.

**Python owns validation.** Whatever the extractor (rules or LLM) proposes lands here and
gets independently checked, normalised, and given a confidence. A candidate that fails
here is never written to journey state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

# --------------------------------------------------------------------------- #
# Shared vocabulary
# --------------------------------------------------------------------------- #

MONTHS: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))

WEEKDAYS = {
    "monday", "mon", "tuesday", "tue", "tues", "wednesday", "wed", "thursday", "thu", "thur",
    "thurs", "friday", "fri", "saturday", "sat", "sunday", "sun",
}

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "a": 1, "an": 1, "couple": 2, "few": 3,
}

RELATIVE_MARKERS = re.compile(
    r"\b(next|this|coming|following|tomorrow|today|tonight|soon|later|asap|"
    r"end of (the )?(month|week)|mid[- ]?(month|week)|early|late|sometime|"
    r"in (a|an|\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)\s+"
    r"(day|days|week|weeks|month|months))\b",
    re.IGNORECASE,
)

_VAGUE_ANSWERS = {
    "yes", "no", "yeah", "nope", "yep", "ok", "okay", "sure", "maybe", "huh", "what", "pardon",
    "sorry", "hello", "hi", "i dont know", "i don't know", "dunno", "not sure", "unsure",
    "you tell me", "dont know", "don't know", "um", "uh", "what do you mean", "say again",
    "repeat that", "come again", "excuse me", "who is this", "speak up", "no idea",
    "nothing", "none", "n a", "na",
}


@dataclass
class ValidationResult:
    ok: bool
    value: str | None = None
    display: str | None = None
    confidence: float = 0.0
    needs_clarification: bool = False
    reason: str = ""

    @staticmethod
    def invalid(reason: str, confidence: float = 0.0, clarify: bool = True) -> "ValidationResult":
        return ValidationResult(
            ok=False, confidence=confidence, needs_clarification=clarify, reason=reason
        )


def _clean(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,!?;:\"'")


def _bare(text: str) -> str:
    """Lower-cased, punctuation-stripped, apostrophe-normalised form for lookups."""
    text = _clean(text).lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9'\s/:-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_ordinals(text: str) -> str:
    return re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)


def _is_vague(text: str) -> bool:
    bare = _bare(text)
    if bare in _VAGUE_ANSWERS:
        return True
    words = bare.split()
    return len(words) <= 2 and bare in _VAGUE_ANSWERS


# --------------------------------------------------------------------------- #
# address
# --------------------------------------------------------------------------- #

_STREET_TYPES = (
    r"street|st|road|rd|avenue|ave|av|drive|dr|close|court|ct|crescent|cres|place|pl|lane|ln|"
    r"way|parade|pde|terrace|tce|highway|hwy|boulevard|blvd|circuit|cct|grove|rise|walk"
)
# House number, then the street name (1-4 words), then the street type: "12 Test Street".
# A leading unit ("Unit 4,") is skipped because the name must be words, not another number.
_STREET = re.compile(
    rf"\b(?P<num>\d{{1,4}}[a-z]?(?:\s*-\s*\d{{1,4}})?)\s*,?\s+"
    rf"(?P<name>[a-z][a-z'.\-]*(?:\s+[a-z][a-z'.\-]*){{0,3}}?)\s+(?P<type>{_STREET_TYPES})\b",
    re.IGNORECASE,
)
_STATE_FULL = re.compile(
    r"\b(new south wales|victoria|queensland|western australia|south australia|tasmania|"
    r"australian capital territory|northern territory)\b",
    re.IGNORECASE,
)
_STATE_ABBR = re.compile(r"\b(nsw|vic|qld|wa|sa|tas|act|nt)\b", re.IGNORECASE)
_AU_POSTCODE = re.compile(r"\b\d{4}\b")


def validate_address(raw: str) -> ValidationResult:
    """A service address needs a street number, a street name and type, and a suburb, plus
    the state and/or postcode (the checklist asks for both; one of the two is enough to
    proceed because the customer hears the whole address read back and confirms it).

    Anything less ("Queens Street", "12 Test Street") is asked for again, once.
    """
    text = _clean(raw)
    if not text:
        return ValidationResult.invalid("empty_address", 0.0)
    if _is_vague(text):
        return ValidationResult.invalid("not_an_address", 0.15)

    street = _STREET.search(text)
    if street is None:
        has_digit = bool(re.search(r"\d", text))
        return ValidationResult.invalid(
            "address_missing_street_number_or_name" if has_digit else "address_lacks_street_or_number",
            0.3,
        )

    rest = text[street.end():]
    has_postcode = bool(_AU_POSTCODE.search(rest))
    has_state = bool(_STATE_FULL.search(rest) or _STATE_ABBR.search(rest))
    suburb_words = _AU_POSTCODE.sub(" ", _STATE_ABBR.sub(" ", _STATE_FULL.sub(" ", rest)))
    has_suburb = bool(re.search(r"[A-Za-z]{3,}", suburb_words))

    missing = [
        label
        for label, present in (
            ("suburb", has_suburb),
            ("state or postcode", has_state or has_postcode),
        )
        if not present
    ]
    if missing:
        return ValidationResult.invalid("address_missing_" + "_and_".join(m.replace(" ", "_") for m in missing), 0.5)

    confidence = 0.95 if (has_state and has_postcode) else 0.86
    return ValidationResult(ok=True, value=text, display=text, confidence=confidence)


# --------------------------------------------------------------------------- #
# date
# --------------------------------------------------------------------------- #

_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DMY_NUMERIC = re.compile(r"\b(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\b")
_DAY_MONTH_YEAR = re.compile(
    rf"\b(\d{{1,2}})\s*(?:of\s+)?({_MONTH_ALT})\.?,?\s*(\d{{4}}|\d{{2}})\b", re.IGNORECASE
)
_MONTH_DAY_YEAR = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b,?\s*(\d{{4}}|\d{{2}})\b", re.IGNORECASE
)
_MONTH_YEAR_ONLY = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{4}})\b", re.IGNORECASE)
_MONTH_ONLY = re.compile(rf"\b({_MONTH_ALT})\b", re.IGNORECASE)
_YEAR_ONLY = re.compile(r"\b(20\d{2})\b")


def _coerce_year(year: int) -> int:
    return year + 2000 if year < 100 else year


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def validate_move_in_date(raw: str, today: date | None = None) -> ValidationResult:
    today = today or date.today()
    text = _clean(_strip_ordinals(raw))
    if not text:
        return ValidationResult.invalid("empty_date", 0.0)
    if _is_vague(text):
        return ValidationResult.invalid("not_a_date", 0.1)

    parsed: date | None = None

    if m := _ISO.search(text):
        parsed = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if parsed is None and (m := _DAY_MONTH_YEAR.search(text)):
        month = MONTHS.get(m.group(2).lower()[:4].rstrip(".")) or MONTHS.get(m.group(2).lower())
        parsed = _safe_date(_coerce_year(int(m.group(3))), month, int(m.group(1))) if month else None
    if parsed is None and (m := _MONTH_DAY_YEAR.search(text)):
        month = MONTHS.get(m.group(1).lower()[:4].rstrip(".")) or MONTHS.get(m.group(1).lower())
        parsed = _safe_date(_coerce_year(int(m.group(3))), month, int(m.group(2))) if month else None
    if parsed is None and (m := _DMY_NUMERIC.search(text)):
        day, month, year = int(m.group(1)), int(m.group(2)), _coerce_year(int(m.group(3)))
        parsed = _safe_date(year, month, day)
        if parsed is None:  # US-style fallback
            parsed = _safe_date(year, day, month)

    if parsed is not None:
        if parsed < today:
            return ValidationResult.invalid("date_in_the_past", 0.9)
        return ValidationResult(
            ok=True,
            value=parsed.isoformat(),
            display=parsed.strftime("%d %B %Y"),
            confidence=0.96,
        )

    # Incomplete but date-shaped: month+year, month only, year only, or a relative phrase.
    if _MONTH_YEAR_ONLY.search(text) or _MONTH_ONLY.search(text) or _YEAR_ONLY.search(text):
        return ValidationResult.invalid("date_missing_day", 0.5)
    if RELATIVE_MARKERS.search(text) or any(w in _bare(text).split() for w in WEEKDAYS):
        return ValidationResult.invalid("relative_date_ambiguous", 0.55)
    if re.search(r"\b\d{1,2}\b", text):
        return ValidationResult.invalid("date_missing_month_or_year", 0.45)

    return ValidationResult.invalid("not_a_date", 0.1)


# --------------------------------------------------------------------------- #
# enums
# --------------------------------------------------------------------------- #

_ENERGY_BOTH = re.compile(r"\b(both|all of them|two of them|each|and)\b", re.IGNORECASE)
_BOTH_ONLY = re.compile(r"\b(both|all of them|the two|two of them)\b", re.IGNORECASE)
_ENERGY_ELEC = re.compile(r"\b(electric|electricity|elec|power|energy)\b", re.IGNORECASE)
_ENERGY_GAS = re.compile(r"\b(gas|natural gas|lpg)\b", re.IGNORECASE)

_CONTACT_PHONE = re.compile(r"\b(phone|call|ring|mobile|telephone|talk|voice)\b", re.IGNORECASE)
_CONTACT_EMAIL = re.compile(r"\b(email|e-mail|mail|inbox|online|written)\b", re.IGNORECASE)

_YES = re.compile(r"\b(yes|yeah|yep|yup|correct|affirmative|i do|i have|we do|we have|sure|"
                  r"that's right|thats right|indeed|absolutely)\b", re.IGNORECASE)
_NO = re.compile(r"\b(no|nope|nah|none|negative|i don't|i dont|we don't|we dont|never|"
                 r"not really|no thanks|no thank you)\b", re.IGNORECASE)
_UNSURE = re.compile(r"\b(unsure|not sure|don't know|dont know|dunno|maybe|possibly|"
                     r"i think so|not certain|no idea|can't remember|cant remember)\b", re.IGNORECASE)


def _enum_result(value: str, display: str, confidence: float = 0.95) -> ValidationResult:
    return ValidationResult(ok=True, value=value, display=display, confidence=confidence)


def validate_energy_requirement(raw: str) -> ValidationResult:
    text = _bare(raw)
    if not text or _is_vague(text):
        return ValidationResult.invalid("not_an_energy_answer", 0.1)
    has_e = bool(_ENERGY_ELEC.search(text))
    has_g = bool(_ENERGY_GAS.search(text))
    if has_e and has_g:
        return _enum_result("BOTH", "Both")
    if _ENERGY_BOTH.search(text) and (has_e or has_g):
        return _enum_result("BOTH", "Both", 0.9)
    if _BOTH_ONLY.search(text):
        return _enum_result("BOTH", "Both", 0.9)  # "both" answers this question by itself
    if has_e:
        return _enum_result("ELECTRICITY", "Electricity")
    if has_g:
        return _enum_result("GAS", "Gas")
    return ValidationResult.invalid("not_an_energy_answer", 0.2)


def validate_yes_no_unsure(raw: str) -> ValidationResult:
    text = _bare(raw)
    if not text:
        return ValidationResult.invalid("empty_answer", 0.0)
    if _UNSURE.search(text):
        return _enum_result("UNSURE", "Unsure", 0.88)
    if _NO.search(text):
        return _enum_result("NO", "No")
    if _YES.search(text):
        return _enum_result("YES", "Yes")
    return ValidationResult.invalid("not_a_yes_no", 0.15)


def validate_yes_no(raw: str) -> ValidationResult:
    result = validate_yes_no_unsure(raw)
    if result.ok and result.value == "UNSURE":
        return ValidationResult.invalid("unsure_not_allowed", 0.5)
    return result


def validate_contact_preference(raw: str) -> ValidationResult:
    text = _bare(raw)
    if not text or _is_vague(text):
        return ValidationResult.invalid("not_a_contact_preference", 0.1)
    has_phone = bool(_CONTACT_PHONE.search(text))
    has_email = bool(_CONTACT_EMAIL.search(text))
    if has_email and not has_phone:
        return _enum_result("EMAIL", "Email")
    if has_phone and not has_email:
        return _enum_result("PHONE", "Phone")
    if has_phone and has_email:
        # "email is better than a call" / "either" — pick the one the customer led with.
        first_phone = _CONTACT_PHONE.search(text).start()  # type: ignore[union-attr]
        first_email = _CONTACT_EMAIL.search(text).start()  # type: ignore[union-attr]
        if "either" in text or "both" in text or "any" in text:
            return ValidationResult.invalid("ambiguous_contact_preference", 0.6)
        return _enum_result("EMAIL" if first_email < first_phone else "PHONE", "Email" if first_email < first_phone else "Phone", 0.85)
    return ValidationResult.invalid("not_a_contact_preference", 0.2)


_CONFIRM_NEGATIVE = re.compile(
    r"\b(not right|incorrect|wrong|not correct|that'?s not|thats not|not quite|hold on|"
    r"wait a|actually|not really|nope|no thanks|no thank you|isn'?t right|isnt right|"
    r"doesn'?t match|doesnt match|change|different|fix)\b",
    re.IGNORECASE,
)
_CONFIRM_NEGATIVE_BARE_NO = re.compile(r"\bno\b(?!\s+worries)", re.IGNORECASE)
_CONFIRM_POSITIVE = re.compile(
    r"\b(yes|yeah|yep|yup|correct|that'?s right|thats right|all (correct|good|right|fine)|"
    r"looks (good|right|correct|fine)|perfect|confirm(ed)?|go ahead|submit|sounds good|"
    r"spot on|exactly|no worries|that'?s all correct|thats all correct|absolutely)\b",
    re.IGNORECASE,
)


def validate_confirmation(raw: str) -> ValidationResult:
    text = _bare(raw)
    if not text:
        return ValidationResult.invalid("empty_confirmation", 0.0)

    # Negative is checked first and wins: a customer correcting details must never be
    # read as an approval just because the sentence also contains "right".
    negative = bool(_CONFIRM_NEGATIVE.search(text)) or bool(_CONFIRM_NEGATIVE_BARE_NO.search(text))
    positive = bool(_CONFIRM_POSITIVE.search(text))

    if negative:
        return ValidationResult.invalid("confirmation_declined", 0.9, clarify=False)
    if positive:
        return ValidationResult(ok=True, value="CONFIRMED", display="Confirmed", confidence=0.94)
    return ValidationResult.invalid("confirmation_unclear", 0.4, clarify=False)


# --------------------------------------------------------------------------- #
# which field is a correction about?
# --------------------------------------------------------------------------- #

_FIELD_MENTIONS: list[tuple[str, re.Pattern[str]]] = [
    ("life_support", re.compile(r"life[- ]?support|ventilator|oxygen", re.IGNORECASE)),
    ("concession_status", re.compile(r"concession|pension|discount", re.IGNORECASE)),
    ("energy_requirement", re.compile(r"electric|\bgas\b|energy|supply", re.IGNORECASE)),
    ("contact_preference", re.compile(r"phone|e-?mail|contact", re.IGNORECASE)),
    ("move_in_date", re.compile(r"\bdate\b|moving|move[- ]?in|connection", re.IGNORECASE)),
    ("property_address", re.compile(r"address|street|road|suburb|postcode|\bunit\b|property", re.IGNORECASE)),
]


def detect_correction_fields(text: str) -> list[str]:
    """The journey fields a correction refers to: those named ("the date is wrong") plus
    any the text supplies a new value for ("no, it's 5 January 2031"). Usually one; if it is
    several or none the agent asks which detail to correct rather than guessing."""
    found = [name for name, pattern in _FIELD_MENTIONS if pattern.search(text or "")]
    for name, check in (("move_in_date", validate_move_in_date), ("property_address", validate_address)):
        if name not in found and check(text or "").ok:
            found.append(name)
    return found


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

VALIDATORS = {
    "address": validate_address,
    "date": validate_move_in_date,
    "energy_requirement": validate_energy_requirement,
    "yes_no_unsure": validate_yes_no_unsure,
    "yes_no": validate_yes_no,
    "contact_preference": validate_contact_preference,
    "confirmation": validate_confirmation,
}


def validate_field(validation_type: str, raw_value: str | None) -> ValidationResult:
    """Single entry point used by the state machine."""
    if validation_type == "consent" or validation_type == "continue_intent":
        # Handled by the safety engine / gate logic, not a data validator.
        return ValidationResult(ok=bool(raw_value), value=raw_value, display=raw_value, confidence=0.9)
    validator = VALIDATORS.get(validation_type)
    if validator is None:
        return ValidationResult.invalid(f"no_validator_for:{validation_type}", 0.0, clarify=False)
    return validator(raw_value or "")


def humanise_iso_date(value: str | None) -> str:
    if not value:
        return "not provided"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return f"{parsed.day} {parsed:%B %Y}"  # "1 October 2030": spoken aloud, so no leading zero
