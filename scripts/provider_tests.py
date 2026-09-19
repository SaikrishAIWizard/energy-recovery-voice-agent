"""Provider wiring tests.

Covers four things that are easy to get wrong and expensive to get wrong:

  1. `backend/.env` is actually read. A .env file that nothing loads is worse than no
     .env file at all, because you trust it and it silently does nothing.
  2. The capability report tells the truth. A provider is only reported live when there is
     a real call path behind it — never because a key happens to be set.
  3. A transcript from an external STT vendor cannot bypass the guardrails. It goes through
     the same safety engine and validators as a typed turn, and an LLM-proposed value still
     has to satisfy the Python validator.
  4. Every declared adapter method has a caller. An interface that nothing calls is dead
     code wearing a capability's clothes.

Runs fully offline and needs no credentials: the provider paths are exercised with stubs,
so the assertions hold whether or not you have configured real keys.

    python scripts/provider_tests.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Import the app package regardless of the working directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import ENV_FILE, settings  # noqa: E402
from app.conversation.field_extractor import extract_field  # noqa: E402
from app.main import app  # noqa: E402
from app.services.llm_service import (  # noqa: E402
    _EXTRACTION_SYSTEM_PROMPT,
    LLMAdapter,
    LLMExtraction,
    OpenAICompatibleLLM,
    _coerce_json,
)
from app.services.voice_service import (  # noqa: E402
    AssemblyAISTT,
    DeepgramSTT,
    OpenAIWhisperSTT,
    STTProvider,
    TelephonyProvider,
    TTSProvider,
    voice_providers,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []
BACKEND_DIR = Path(__file__).resolve().parent.parent


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))
    print(f"  [{PASS if condition else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}")


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


class StubSTT:
    """Stands in for a real vendor so the endpoint can be tested without a network call."""

    name = "stub"
    available = True

    def __init__(self, text: str, confidence: float | None = 0.95) -> None:
        self.text = text
        self.confidence = confidence
        self.calls = 0
        self.last_size = 0

    def transcribe(self, audio: bytes, mime_type: str = "audio/webm") -> dict[str, Any]:
        self.calls += 1
        self.last_size = len(audio)
        return {"text": self.text, "confidence": self.confidence}


class StubLLM:
    """An adapter that proposes a value the customer never said."""

    name = "STUB"
    enabled = True

    def __init__(self, value: str | None, confidence: float = 0.99) -> None:
        self.value = value
        self.confidence = confidence

    def extract_field(self, current_field, customer_utterance, known_context):  # noqa: ANN001
        return LLMExtraction(value=self.value, confidence=self.confidence, provider=self.name)


def env_file_values(path: Path) -> dict[str, str]:
    """Minimal .env parser — good enough to compare the file against Settings."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


# --------------------------------------------------------------------------- #
# 1. The .env file is read
# --------------------------------------------------------------------------- #


def test_env_file() -> None:
    section("1. backend/.env is actually loaded")
    env_path = BACKEND_DIR / ".env"

    if not env_path.is_file():
        check(
            "no backend/.env present -> app correctly reports none",
            ENV_FILE is None,
            f"ENV_FILE={ENV_FILE!r}",
        )
        print("       (copy .env.example to .env to exercise the rest of this section)")
        return

    check("app reports the .env file it loaded", ENV_FILE == ".env", f"ENV_FILE={ENV_FILE!r}")

    values = env_file_values(env_path)
    for key, actual in (
        ("LLM_BASE_URL", settings.llm_base_url),
        ("LLM_MODEL", settings.llm_model),
        ("LLM_PROVIDER", settings.llm_provider),
    ):
        expected = values.get(key)
        if expected is None:
            continue
        check(f".env value reaches Settings ({key})", actual == expected, f"{actual!r}")


# --------------------------------------------------------------------------- #
# 2. The capability report tells the truth
# --------------------------------------------------------------------------- #


def test_capability_report() -> None:
    section("2. The capability report tells the truth")
    report = settings.capability_report()

    check(
        "stt.active agrees with the registry",
        report["stt"]["active"] == voice_providers.active_stt,
        f"{report['stt']['active']!r}",
    )
    check("deepgram flag tracks its credential", report["stt"]["deepgram"] == bool(settings.deepgram_api_key))
    check("assemblyai flag tracks its credential", report["stt"]["assemblyai"] == bool(settings.assemblyai_api_key))
    check("whisper flag tracks the LLM credential", report["stt"]["openai_whisper"] == bool(settings.llm_api_key))
    check(
        "llm flag tracks the LLM credential",
        report["llm_extraction"]["external_adapter"] == bool(settings.llm_api_key),
    )
    check(
        "no external TTS vendor is claimed",
        report["tts"]["external_provider"] is False,
        "the browser speaks every turn; there is no TTS call path to claim",
    )

    # The assertion that matters: nothing is advertised without somewhere to send the audio.
    key_to_provider = {"openai_whisper": "openai", "deepgram": "deepgram", "assemblyai": "assemblyai"}
    for name, provider_key in key_to_provider.items():
        if report["stt"].get(name):
            check(
                f"'{name}' is reported live AND is reachable",
                provider_key in voice_providers.server_stt,
                f"server_stt={list(voice_providers.server_stt)}",
            )


# --------------------------------------------------------------------------- #
# 3. Adapters are real and cannot overreach
# --------------------------------------------------------------------------- #


def test_adapters() -> None:
    section("3. Adapters are real, and cannot overreach")

    # An env var with no class behind it is vapour. These three must exist.
    for cls in (DeepgramSTT, AssemblyAISTT, OpenAIWhisperSTT):
        check(f"{cls.__name__} exists and accepts audio", callable(getattr(cls, "transcribe", None)))

    check(
        "AssemblyAI is not registered without a key",
        "assemblyai" not in voice_providers.server_stt or bool(settings.assemblyai_api_key),
    )

    adapter = OpenAICompatibleLLM(
        api_key="test", model="test-model", base_url="https://example.test/v1/", provider_label="groq"
    )
    check("base URL trailing slash is normalised", adapter._base_url == "https://example.test/v1", adapter._base_url)
    check("provider label surfaces in logs and /health", adapter.name == "GROQ", adapter.name)
    check(
        "adapter refuses a non-journey field without a network call",
        adapter.extract_field("card_number", "4111 1111 1111 1111", {}) is None,
    )

    for phrase in (
        "NEVER guess",
        "NEVER give advice",
        "NEVER output payment card numbers",
        "NEVER decide what happens next",
    ):
        check(f"extraction prompt still enforces '{phrase}'", phrase in _EXTRACTION_SYSTEM_PROMPT)


# --------------------------------------------------------------------------- #
# 4. Vendor response quirks
# --------------------------------------------------------------------------- #


def test_json_coercion() -> None:
    section("4. Vendor response quirks are tolerated")
    check("plain JSON object parses", _coerce_json('{"value": "x"}') == {"value": "x"})
    check("markdown-fenced JSON parses", _coerce_json('```json\n{"value": "x"}\n```') == {"value": "x"})
    check("JSON embedded in prose parses", _coerce_json('Sure, here you go: {"value": "x"}') == {"value": "x"})
    check("garbage returns None (so the rules engine takes over)", _coerce_json("not json at all") is None)
    check("None input returns None", _coerce_json(None) is None)


# --------------------------------------------------------------------------- #
# 5. The audio endpoint, and the guardrails behind it
# --------------------------------------------------------------------------- #


def test_audio_endpoint() -> None:
    section("5. The audio endpoint routes through the guardrails")

    saved = voice_providers.server_stt
    created_sessions: list[str] = []
    try:
        with TestClient(app) as client:
            # --- no provider configured: a clear refusal, not a crash ---
            voice_providers.server_stt = {}
            started = client.post("/calls/start/E-1001", json={}).json()
            sid = started["session"]["id"]
            created_sessions.append(sid)

            response = client.post(
                f"/calls/{sid}/audio", content=b"x" * 1024, headers={"Content-Type": "audio/webm"}
            )
            check("503 when no server-side provider is configured", response.status_code == 503, str(response.status_code))
            check(
                "...and the error says how to fix it",
                "DEEPGRAM_API_KEY" in (response.json().get("detail") or ""),
            )

            # --- empty body is rejected before the provider is called ---
            stub = StubSTT("hello")
            voice_providers.server_stt = {"stub": stub}
            response = client.post(
                f"/calls/{sid}/audio", content=b"", headers={"Content-Type": "audio/webm"}
            )
            check("400 on an empty audio body", response.status_code == 400, str(response.status_code))
            check("...and the provider was never called", stub.calls == 0)

            # --- an empty transcript is surfaced, not silently ignored ---
            silent = StubSTT("")
            voice_providers.server_stt = {"stub": silent}
            response = client.post(
                f"/calls/{sid}/audio", content=b"x" * 2048, headers={"Content-Type": "audio/webm"}
            )
            check("422 when the vendor returns no transcript", response.status_code == 422, str(response.status_code))

            # --- a usable transcript drives a real turn ---
            usable = StubSTT("yes that's fine, go ahead")
            voice_providers.server_stt = {"stub": usable}
            response = client.post(
                f"/calls/{sid}/audio", content=b"x" * 4096, headers={"Content-Type": "audio/webm"}
            )
            check("200 for a usable transcript", response.status_code == 200, str(response.status_code))
            turn = response.json()
            check("the provider received the whole audio body", usable.last_size == 4096, str(usable.last_size))
            customer_text = " ".join(
                seg["text"] for seg in turn["session"]["transcript"] if seg["speaker"] == "CUSTOMER"
            )
            check(
                "the vendor transcript landed in the call transcript",
                "go ahead" in customer_text,
                customer_text[:60],
            )
            check("the turn produced an agent reply", len(turn["agent_messages"]) > 0)

            # --- THE ONE THAT MATTERS: a vendor cannot bypass the guardrails ---
            card = StubSTT("my card is 4111 1111 1111 1111 and the expiry is 09/27")
            voice_providers.server_stt = {"stub": card}
            started = client.post("/calls/start/E-1001", json={}).json()
            sid2 = started["session"]["id"]
            created_sessions.append(sid2)

            response = client.post(
                f"/calls/{sid2}/audio", content=b"x" * 4096, headers={"Content-Type": "audio/webm"}
            )
            check("card data in a vendor transcript still returns a turn", response.status_code == 200, str(response.status_code))
            turn = response.json()
            blob = json.dumps(turn["session"])

            check(
                "card data still triggers escalation",
                turn["handoff_triggered"] is True,
                f"status={turn['session']['status']}",
            )
            check("...the PAN never reaches the transcript", "4111 1111 1111 1111" not in blob)
            check(
                "...and the redaction is audited",
                "CARD_DATA_REDACTED" in json.dumps(turn["session"]["audit_events"]),
            )
    finally:
        voice_providers.server_stt = saved

    if created_sessions:
        print(
            f"\n  note: this run created {len(created_sessions)} call session(s) in "
            "backend/energy_recovery.db. POST /demo/reset clears them."
        )


# --------------------------------------------------------------------------- #
# 6. An LLM proposal still has to pass the Python validator
# --------------------------------------------------------------------------- #


def test_llm_cannot_override_validation() -> None:
    section("6. An LLM proposal is re-validated in Python")

    # Rules cannot parse this, so pass 2 (the LLM) is the only way through.
    utterance = "honestly it's hard to say, maybe the middle of next month"

    rejected = extract_field(
        "move_in_date", utterance, {}, validation_type="date", llm=StubLLM("sometime next month")
    )
    check("a structurally invalid LLM value is rejected", rejected.value is None, f"value={rejected.value!r}")
    check("...and it is not marked valid", rejected.valid is False, f"source={rejected.source}")

    accepted = extract_field(
        "move_in_date", utterance, {}, validation_type="date", llm=StubLLM("2026-10-15")
    )
    check("a valid LLM value is accepted", accepted.value == "2026-10-15", f"value={accepted.value!r}")
    check("...and labelled LLM+RULES so the path is auditable", accepted.source == "LLM+RULES", accepted.source)

    hesitant = extract_field(
        "move_in_date", utterance, {}, validation_type="date", llm=StubLLM("2026-10-15", confidence=0.4)
    )
    check(
        "a hesitant model cannot inflate a capture",
        hesitant.confidence <= 0.4,
        f"confidence={hesitant.confidence} (model said 0.4)",
    )


# --------------------------------------------------------------------------- #
# 7. No dead interfaces
# --------------------------------------------------------------------------- #

_ALL_PROTOCOLS: dict[str, type] = {
    "LLMAdapter": LLMAdapter,
    "STTProvider": STTProvider,
    "TTSProvider": TTSProvider,
    "TelephonyProvider": TelephonyProvider,
}

# Documented exceptions. An entry here has to justify itself as "honestly reported as not
# built", not as "we did not get around to it".
_KNOWN_UNCALLED = {
    "TTSProvider.synthesize": (
        "Extension point for a server-side TTS vendor. Nothing calls it because the "
        "browser speaks every turn, and /health reports tts.external_provider = false "
        "rather than claiming it is live."
    ),
}


def _protocol_methods(protocol: type) -> list[str]:
    return sorted(
        name for name, value in vars(protocol).items() if not name.startswith("_") and callable(value)
    )


def test_no_dead_interfaces() -> None:
    """Every declared adapter method must have a caller inside app/.

    This is the regression guard for the bug that recurred three times while building this:
    an interface gets declared and implemented, `/health` and the README advertise it, and
    no code path ever reaches it. A capability nobody can reach is worse than no capability,
    because it is trusted — the first version of this file reported `assemblyai` as live
    when no `AssemblyAISTT` class existed at all.

    Scanning source text is crude, but it is exactly the right level of crude here: it fails
    loudly the moment a method loses its last caller, which is the moment the claim becomes
    a lie.
    """
    section("7. Every declared adapter method has a real caller")
    sources = {
        path.name: path.read_text(encoding="utf-8") for path in (BACKEND_DIR / "app").rglob("*.py")
    }

    for label, protocol in _ALL_PROTOCOLS.items():
        for method in _protocol_methods(protocol):
            key = f"{label}.{method}"
            callers = sorted(
                name
                for name, text in sources.items()
                if re.search(rf"\.{re.escape(method)}\s*\(", text)
            )
            if key in _KNOWN_UNCALLED:
                check(
                    f"{key}() is a declared extension point, not a live claim",
                    not callers,
                    _KNOWN_UNCALLED[key],
                )
            else:
                check(
                    f"{key}() has a caller in app/",
                    bool(callers),
                    ", ".join(callers) or "NO CALLER — this interface is dead code",
                )


def main() -> int:
    print("Provider wiring tests")
    print("=" * 72)

    test_env_file()
    test_capability_report()
    test_adapters()
    test_json_coercion()
    test_audio_endpoint()
    test_llm_cannot_override_validation()
    test_no_dead_interfaces()

    failures = [row for row in results if row[1] == FAIL]
    print("\n" + "=" * 72)
    print(f"{len(results) - len(failures)}/{len(results)} checks passed")
    if failures:
        print("\nFailures:")
        for name, _, detail in failures:
            print(f"  - {name}" + (f" ({detail})" if detail else ""))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
