"""Safety & handoff engine.

Runs on **every** customer utterance, before any field extraction is attempted. It is
pure Python pattern matching — deliberately not an LLM call — because escalation must be
deterministic, auditable, and impossible to prompt-inject.

Escalation taxonomy (handout §03) is carried as `escalation_signal` so the console can
show *why* the agent stepped aside:

    ASKS        -> CUSTOMER_REQUEST
    ANGER       -> FRUSTRATION
    CONFUSION   -> REPEATED_FAILURE
    SENSITIVE   -> SENSITIVE_TOPIC
    OFF-SCRIPT  -> OFF_SCRIPT
    LOW CONF    -> LOW_CONFIDENCE
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from app.services.redaction_service import (
    contains_card_data,
    find_card_like_spans,
    redact_text,
    redaction_summary,
)

Action = Literal["CONTINUE", "DECLINE", "HANDOFF"]

# --------------------------------------------------------------------------- #
# Pattern banks
# --------------------------------------------------------------------------- #

_HUMAN_REQUEST = re.compile(
    r"\b("
    r"(talk|speak|chat)\s+(to|with)\s+(a\s+|the\s+|an\s+)?"
    r"(person|human|human being|someone|somebody|agent|advisor|adviser|consultant|"
    r"representative|rep|operator|supervisor|manager|real person|live person|"
    r"customer service|support team)"
    r"|get me (a|an|someone|a human|a person)"
    r"|put me through"
    r"|transfer me"
    r"|connect me (to|with)"
    r"|call me back with a human"
    r"|is (this|that) a (robot|bot|machine|recording)"
    r"|are you (a )?(robot|bot|ai|machine)"
    r")\b",
    re.IGNORECASE,
)

_DECLINE = re.compile(
    r"\b("
    r"not interested|no interest|"
    r"(stop|quit|don't|dont|do not|never)\s+(calling|call)\s*(me|us)?|"
    r"(remove|take)\s+(me|my (number|details|name))\s+(off|out)|"
    r"delete my (details|number|data)|"
    r"unsubscribe|opt\s*-?\s*out|"
    r"do not contact|don't contact|dont contact|"
    r"no more calls|stop contacting|"
    r"take me off (your|the) list|"
    r"remove me from (your|the) list"
    r")\b",
    re.IGNORECASE,
)

_PROFANITY = re.compile(
    r"\b(f+u+c+k+\w*|s+h+i+t+\w*|bullshit|bastard|arsehole|asshole|dickhead|"
    r"piss off|pissed off|crap|damn it|bloody hell|screw this)\b",
    re.IGNORECASE,
)

_FRUSTRATION = re.compile(
    r"\b("
    r"stop wasting my time|wasting my time|waste of time|"
    r"already told you|i told you|i already told|told you (this |that )?(already|before)|"
    r"how many times|"
    r"(second|third|fourth|another|two|three)\s+call(s)?\s+(today|this week)|"
    r"called me (twice|before|already|again)|"
    r"this is ridiculous|ridiculous|absurd|unacceptable|"
    r"useless|incompetent|hopeless|"
    r"fed up|sick of (this|it)|had enough|"
    r"hurry up|get on with it|just get on with it|"
    r"you'?re not listening|not listening to me|listen to me|"
    r"this is a joke|what a joke|"
    r"so frustrating|frustrating|annoying|"
    r"i'?m angry|i am angry|"
    r"don'?t (you )?understand|you don'?t get it"
    r")\b",
    re.IGNORECASE,
)

_LIFE_SUPPORT = re.compile(
    r"\b(life\s*-?\s*support|ventilator|respirator|oxygen concentrator|"
    r"dialysis|feeding tube|nebuliser|nebulizer|home oxygen)\b",
    re.IGNORECASE,
)

_VULNERABILITY = re.compile(
    r"\b("
    r"hardship|financial hardship|"
    r"can'?t afford|cannot afford|struggling to pay|struggling financially|"
    r"unemployed|out of work|lost my job|"
    r"disability|disabled|carer|full[- ]time carer|"
    r"mental health|depression|anxiety|"
    r"family violence|domestic violence|"
    r"terminal(ly)? ill|palliative|"
    r"centrelink|pensioner|health care card"
    r")\b",
    re.IGNORECASE,
)

_COMPLAINT = re.compile(
    r"\b("
    r"complain|complaint|"
    r"dispute|disputed|"
    r"ombudsman|ombud|ewon|accc|fair trading|"
    r"legal action|lawyer|solicitor|sue you|take you to court|"
    r"overcharged|over[- ]?billed|wrong bill|unfair bill|"
    r"report you|breach(ed)? (my )?privacy|"
    r"this is fraud|that'?s illegal"
    r")\b",
    re.IGNORECASE,
)

_ADVICE_REQUEST = re.compile(
    r"\b("
    r"should i|should we|"
    r"which (one|plan|provider|company|is)|"
    r"what'?s the best|whats the best|what is the best|who'?s the cheapest|"
    r"cheapest|cheaper|"
    r"recommend|recommendation|"
    r"advice|advise me|give me advice|"
    r"how much (will|would|does|do) (it|that|this|i)|"
    r"what (will|would) (it|that|this) cost|"
    r"is it worth|worth switching|"
    r"would you (pick|choose|go with)|what would you do|"
    r"help me (choose|decide|pick)|"
    r"compare (the )?(plans|providers|options)|"
    r"guarantee|will i save|how much can i save|"
    r"is .{1,20} better than"
    r")\b",
    re.IGNORECASE,
)

_OFF_SCRIPT_TOPIC = re.compile(
    r"\b("
    r"solar|battery|rebate|feed[- ]in tariff|"
    r"nbn|internet|broadband|mobile plan|"
    r"credit card|personal loan|insurance|health cover|"
    r"crypto|invest|stock"
    r")\b",
    re.IGNORECASE,
)

_META_QUESTION = re.compile(
    r"\b("
    r"what do you mean|what does that mean|what do you mean by|"
    r"why do you need|why do you want|why are you asking|"
    r"explain (that|this|it)|i don'?t understand|i dont understand|"
    r"pardon|sorry\?|come again|say (that|it) again|repeat (that|it)|"
    r"what was that|didn'?t catch|didnt catch|could you repeat|"
    r"what\?|huh\?|excuse me\?"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_LEAD = re.compile(
    r"^\s*(what|why|how|who|where|when|which|whose|"
    r"can you|could you|would you|will you|do you|did you|are you|is it|is that|"
    r"does it|have you|am i)\b",
    re.IGNORECASE,
)

# Gate ("can I help you continue?") intent
_BUSY = re.compile(
    r"\b("
    r"busy|not a good time|bad time|not a great time|"
    r"driving|at work|in (a|the) meeting|with a client|"
    r"can'?t talk|cannot talk|can'?t speak|"
    r"call (me )?back|call later|ring (me )?later|later on|"
    r"another time|some other time|"
    r"not (right )?now|now'?s not|now is not"
    r")\b",
    re.IGNORECASE,
)

_CONTINUE_YES = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|alright|all right|fine|go ahead|go on|"
    r"continue|proceed|happy to|let'?s|lets|that'?s fine|no worries|of course|"
    r"absolutely|please do|i'?m listening|im listening)\b",
    re.IGNORECASE,
)

_CONFUSED_ABOUT_CALL = re.compile(
    r"\b(who is this|who'?s this|what is this (about|call)|why are you calling|"
    r"what do you want|which company|where are you (calling )?from|"
    r"i never (signed up|asked)|i didn'?t sign up|i didnt sign up)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #


@dataclass
class SafetyVerdict:
    action: Action = "CONTINUE"
    reason: str | None = None
    escalation_signal: str | None = None
    flags: list[str] = field(default_factory=list)
    matched: str | None = None
    note: str = ""
    safe_text: str = ""
    redacted: bool = False
    card_detected: bool = False
    # Audit-safe fingerprint of any card seen (length + last 4 + Luhn validity only —
    # never the number itself). Recorded so a reviewer can confirm the boundary held.
    card_fingerprint: str | None = None
    agent_messages: list[str] = field(default_factory=list)

    @property
    def is_handoff(self) -> bool:
        return self.action == "HANDOFF"

    @property
    def is_decline(self) -> bool:
        return self.action == "DECLINE"


def _match(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    return m.group(0).strip() if m else None


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #


def evaluate(utterance: str) -> SafetyVerdict:
    """Evaluate one customer utterance.

    Priority order is deliberate: a human request outranks everything (the customer has
    told us what they want); a refusal outranks frustration (respecting "no" beats
    de-escalating); payment data outranks everything else on the sensitive side.
    """
    safe_text, redacted = redact_text(utterance or "")
    verdict = SafetyVerdict(safe_text=safe_text, redacted=redacted)
    flags: list[str] = []

    if not (safe_text or "").strip():
        verdict.note = "empty_utterance"
        return verdict

    card_detected = contains_card_data(safe_text) or redacted
    verdict.card_detected = card_detected

    # Fingerprint is computed from the RAW utterance, before the digits are masked away.
    spans = find_card_like_spans(utterance or "")
    if spans:
        verdict.card_fingerprint = redaction_summary(spans[0][2])

    human = _match(_HUMAN_REQUEST, safe_text)
    decline = _match(_DECLINE, safe_text)
    life_support = _match(_LIFE_SUPPORT, safe_text)
    vulnerability = _match(_VULNERABILITY, safe_text)
    complaint = _match(_COMPLAINT, safe_text)
    advice = _match(_ADVICE_REQUEST, safe_text)
    off_topic = _match(_OFF_SCRIPT_TOPIC, safe_text)
    profanity = _match(_PROFANITY, safe_text)
    frustration = _match(_FRUSTRATION, safe_text)

    # --- collect every signal we saw (flags accumulate; reason is the top priority) ---
    if human:
        flags.append("CUSTOMER_REQUEST")
    if decline:
        flags.append("DECLINE")
    if card_detected:
        flags.append("CARD_DATA_DETECTED")
    if life_support:
        flags.append("LIFE_SUPPORT")
    if vulnerability:
        flags.append("VULNERABILITY")
    if complaint:
        flags.append("COMPLAINT_OR_DISPUTE")
    if profanity:
        flags.append("PROFANITY")
    if frustration:
        flags.append("FRUSTRATION")
    if advice:
        flags.append("ADVICE_REQUEST")
    if off_topic:
        flags.append("OFF_SCRIPT_TOPIC")

    verdict.flags = flags

    # 1. Explicit request for a person.
    if human:
        return _handoff(
            verdict, "CUSTOMER_REQUEST", "ASKS", human,
            "Customer explicitly asked to speak with a human.",
        )

    # 2. Refusal — respect it immediately, no pressure loop.
    if decline:
        verdict.action = "DECLINE"
        verdict.reason = "CUSTOMER_DECLINED"
        verdict.escalation_signal = "ASKS"
        verdict.matched = decline
        verdict.note = "Customer refused further contact. Thank, log, end."
        return verdict

    # 3. Payment data — never capture, never repeat.
    if card_detected:
        return _handoff(
            verdict, "SENSITIVE_TOPIC", "SENSITIVE", None,
            "Payment card data detected in the customer's speech and redacted. "
            "AI voice channels must never capture card details.",
        )

    # 4. Vulnerability / life support / dispute.
    if life_support:
        return _handoff(
            verdict, "SENSITIVE_TOPIC", "SENSITIVE", life_support,
            "Life-support equipment requirement raised — vulnerable-customer case.",
        )
    if vulnerability:
        return _handoff(
            verdict, "SENSITIVE_TOPIC", "SENSITIVE", vulnerability,
            "Hardship or vulnerability signal raised.",
        )
    if complaint:
        return _handoff(
            verdict, "SENSITIVE_TOPIC", "SENSITIVE", complaint,
            "Complaint or dispute raised.",
        )

    # 5. Anger / profanity.
    if profanity or frustration:
        return _handoff(
            verdict, "FRUSTRATION", "ANGER", profanity or frustration,
            "Customer appears frustrated or angry.",
        )

    # 6. Advice request — we collect, we do not advise.
    if advice:
        verdict.action = "HANDOFF"
        verdict.reason = "OFF_SCRIPT"
        verdict.escalation_signal = "OFF_SCRIPT"
        verdict.matched = advice
        verdict.note = "Customer asked for product or financial advice. Not in scope."
        verdict.agent_messages = [
            "I can collect the details for you, but I'm not able to give advice about "
            "plans or pricing. Let me put you through to a specialist who can help.",
        ]
        return verdict

    # 7. Clearly off-script topic.
    if off_topic:
        verdict.action = "HANDOFF"
        verdict.reason = "OFF_SCRIPT"
        verdict.escalation_signal = "OFF_SCRIPT"
        verdict.matched = off_topic
        verdict.note = "Customer raised a topic outside the approved Energy script."
        verdict.agent_messages = [
            "That's outside what I can help with on this call. Let me pass you to a "
            "specialist who can look at that with you.",
        ]
        return verdict

    return verdict


def _handoff(
    verdict: SafetyVerdict,
    reason: str,
    signal: str,
    matched: str | None,
    note: str,
) -> SafetyVerdict:
    verdict.action = "HANDOFF"
    verdict.reason = reason
    verdict.escalation_signal = signal
    verdict.matched = matched
    verdict.note = note
    return verdict


# --------------------------------------------------------------------------- #
# Gate helpers
# --------------------------------------------------------------------------- #


def classify_gate_intent(utterance: str) -> Literal["CONTINUE", "BUSY", "UNCLEAR"]:
    """Classify the reply to the consent / continue gate. Decline and handoff are
    already handled by `evaluate()` before this is called."""
    text = utterance or ""
    if _BUSY.search(text):
        return "BUSY"
    if _CONTINUE_YES.search(text):
        return "CONTINUE"
    return "UNCLEAR"


def is_meta_question(utterance: str) -> bool:
    """Customer is asking *about* the question rather than answering it."""
    return bool(_META_QUESTION.search(utterance or ""))


def is_question_like(utterance: str) -> bool:
    text = (utterance or "").strip()
    if not text:
        return False
    return bool(_QUESTION_LEAD.search(text)) or text.endswith("?")


def is_confused_about_call(utterance: str) -> bool:
    return bool(_CONFUSED_ABOUT_CALL.search(utterance or ""))


# Keywords that mean "the customer is asking about the thing we just asked them".
# A question mentioning these is a clarification request, not an off-script detour.
_FIELD_KEYWORDS: dict[str, re.Pattern[str]] = {
    "property_address": re.compile(
        r"\b(address|street|suburb|postcode|property|where|unit|apartment)\b", re.IGNORECASE
    ),
    "move_in_date": re.compile(
        r"\b(date|day|month|year|when|moving|move in|move-in|settle|settlement)\b", re.IGNORECASE
    ),
    "energy_requirement": re.compile(
        r"\b(electricity|electric|gas|power|both|supply|energy)\b", re.IGNORECASE
    ),
    "concession_status": re.compile(
        r"\b(concession|card|discount|rebate|pension|eligible|eligibility)\b", re.IGNORECASE
    ),
    "life_support": re.compile(
        r"\b(life support|life-support|equipment|medical|ventilator|power supply)\b",
        re.IGNORECASE,
    ),
    "contact_preference": re.compile(
        r"\b(phone|email|call|contact|reach|number|address)\b", re.IGNORECASE
    ),
}


def question_targets_field(utterance: str, field_name: str) -> bool:
    """True when the question is about the current field — treat as clarification."""
    pattern = _FIELD_KEYWORDS.get(field_name)
    if pattern is None:
        return False
    return bool(pattern.search(utterance or ""))
