"""Optional LLM adapter.

Design contract, enforced by prompt *and* by the caller:

  * The LLM may ONLY propose a structured extraction for the field currently being
    collected. That is its entire surface.
  * The LLM may NEVER decide to continue, submit, decline, or hand off. Every decision
    is made by the deterministic state machine in `app/conversation/state_machine.py`.
  * The LLM may NEVER invent customer data. No stated value -> null. Any proposal is
    re-validated by `app/conversation/validators.py` and confidence-capped.
  * The LLM may NEVER give advice, and never sees or repeats card data.

Why the interface is this narrow
--------------------------------
An earlier sketch of this adapter also carried `rephrase()` (re-word an approved script
line before speaking) and `label_intent()` (tag sentiment for a QA log). Both were
implemented and neither was ever called, which is worse than not having them: `/health`
and the README advertised capabilities that no code path could reach.

They were removed rather than wired up, deliberately:

  * The script copy is already written to be spoken aloud, and re-wording it would add a
    round trip to every agent turn — real latency in a voice agent, for copy that is
    already compliant.
  * Intent classification already belongs to the deterministic safety engine. A second,
    opaque source of intent truth sitting beside the one that actually drives escalation
    is a liability in a regulated flow, not a feature.

If phrasing is ever wanted, it belongs behind this same seam — but it has to be called
from somewhere real, and `provider_tests.py` now asserts that every method on this
protocol has a caller.

With no key set, `RulesOnlyLLM` is used and the whole application still works end-to-end.
This is the default.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, settings

logger = logging.getLogger(__name__)

_ALLOWED_FIELDS = {
    "property_address",
    "move_in_date",
    "energy_requirement",
    "concession_status",
    "life_support",
    "contact_preference",
}

_EXTRACTION_SYSTEM_PROMPT = """You are a strict structured-extraction function inside a \
regulated Australian energy lead-recovery voice agent. You are NOT a conversational agent.

ABSOLUTE RULES — violating any of these is a critical failure:
1. Extract ONLY what the customer literally said. If a required value is not clearly and
   completely stated, return null. NEVER guess, infer, complete, or invent a value.
2. NEVER give advice, opinions, price guidance, plan recommendations, or eligibility
   statements. You are not permitted to answer the customer.
3. NEVER output payment card numbers, CVVs, or expiry dates. If present, return null.
4. NEVER decide what happens next. Do not say whether to continue, submit, or escalate.
   You only return the extraction object.
5. If the customer asks a question, complains, or goes off-topic, set
   "handoff_signal" to one of CUSTOMER_REQUEST | FRUSTRATION | REPEATED_FAILURE |
   SENSITIVE_TOPIC | OFF_SCRIPT | LOW_CONFIDENCE as appropriate, otherwise null.
6. Relative or partial dates ("next Friday", "sometime in October") are NOT complete.
   Return value=null, needs_clarification=true, and a confidence below 0.8.

Return ONLY a JSON object with exactly these keys:
{"value": string|null, "confidence": number, "needs_clarification": boolean, "handoff_signal": string|null}
"""


def _coerce_json(content: Any) -> dict[str, Any] | None:
    """Parse a model response into a dict, tolerating vendor quirks.

    Some OpenAI-compatible vendors ignore `response_format` and wrap the object in a
    markdown fence. Strip that, then fall back to locating the outermost braces.
    """
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None

    text = content.strip()
    if text.startswith("```") and "```" in text[3:]:
        text = text.split("```")[1].removeprefix("json").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


@dataclass
class LLMExtraction:
    value: str | None = None
    confidence: float = 0.0
    needs_clarification: bool = False
    handoff_signal: str | None = None
    provider: str = "RULES_ONLY"

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "confidence": round(self.confidence, 2),
            "needs_clarification": self.needs_clarification,
            "handoff_signal": self.handoff_signal,
            "provider": self.provider,
        }


class LLMAdapter(Protocol):
    name: str
    enabled: bool

    def extract_field(
        self, current_field: str, customer_utterance: str, known_context: dict[str, Any]
    ) -> LLMExtraction | None: ...


class RulesOnlyLLM:
    """Default adapter. Deliberately does nothing — the rules engine is sufficient."""

    name = "RULES_ONLY"
    enabled = False

    def extract_field(self, current_field, customer_utterance, known_context):  # noqa: ANN001
        return None


class OpenAICompatibleLLM:
    """Thin httpx adapter for any OpenAI-compatible Chat Completions endpoint.

    Only instantiated when an LLM key is present.

    Kept dependency-light on purpose: no SDK, one HTTP call, hard 6s timeout, and every
    failure degrades silently to the rules engine rather than breaking the call.

    Because it speaks the plain OpenAI wire format, pointing it at a free-tier provider
    (Groq, Google Gemini, OpenRouter, DeepSeek, …) is a base-URL and model-name swap via
    `LLM_BASE_URL` / `LLM_MODEL` — no code change.
    """

    enabled = True

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_LLM_BASE_URL,
        provider_label: str = "openai",
        timeout: float = 6.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.name = (provider_label or "openai").upper()

    # -- internals ---------------------------------------------------------- #
    def _post(self, system: str, user: str, *, json_mode: bool) -> Any:
        import httpx

        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        return httpx.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout,
        )

    def _chat(self, system: str, user: str) -> dict[str, Any] | None:
        try:
            response = self._post(system, user, json_mode=True)

            # Not every OpenAI-compatible vendor supports `response_format`. A 400 here is
            # almost always that, so retry once without it rather than losing the provider.
            if response.status_code == 400 and "response_format" in response.text:
                response = self._post(system, user, json_mode=False)

            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return _coerce_json(content)
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("LLM call failed, falling back to rules engine: %s", exc)
            return None

    # -- adapter API -------------------------------------------------------- #
    def extract_field(
        self, current_field: str, customer_utterance: str, known_context: dict[str, Any]
    ) -> LLMExtraction | None:
        if current_field not in _ALLOWED_FIELDS:
            return None
        user = json.dumps(
            {
                "field_to_extract": current_field,
                "already_known": known_context or {},
                "customer_utterance": customer_utterance,
                "today": known_context.get("today"),
            },
            ensure_ascii=False,
        )
        payload = self._chat(_EXTRACTION_SYSTEM_PROMPT, user)
        if not payload:
            return None
        value = payload.get("value")
        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return LLMExtraction(
            value=str(value).strip() if value not in (None, "", "null") else None,
            confidence=max(0.0, min(1.0, confidence)),
            needs_clarification=bool(payload.get("needs_clarification")),
            handoff_signal=payload.get("handoff_signal") or None,
            provider=self.name,
        )


_llm_singleton: LLMAdapter | None = None


def get_llm() -> LLMAdapter:
    global _llm_singleton
    if _llm_singleton is None:
        if settings.llm_api_key:
            _llm_singleton = OpenAICompatibleLLM(
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                provider_label=settings.llm_provider,
            )
        else:
            _llm_singleton = RulesOnlyLLM()
    return _llm_singleton


def reset_llm() -> None:
    """Drop the cached adapter. Used by tests that flip provider settings at runtime."""
    global _llm_singleton
    _llm_singleton = None


def llm_status() -> dict[str, Any]:
    adapter = get_llm()
    status: dict[str, Any] = {
        "active_adapter": adapter.name,
        "enabled": adapter.enabled,
        "note": (
            "Rules engine only. All decisions, validation, and escalation are Python."
            if not adapter.enabled
            else "LLM available for structured extraction only; every decision stays in Python."
        ),
    }
    if adapter.enabled:
        status["model"] = settings.llm_model
        status["base_url"] = settings.llm_base_url
    return status
