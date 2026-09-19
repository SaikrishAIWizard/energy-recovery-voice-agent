"""Runtime configuration.

Everything here is optional-by-default: the demo must run end-to-end with **no API keys**
and **no telephony credentials**. Presence of an env var is what switches an optional
adapter on. Nothing in the core journey depends on any of them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"

_logger = logging.getLogger(__name__)

# Confidence floor for a required field. Below this after a clarification attempt we
# stop guessing and hand the customer to a human (handoff reason LOW_CONFIDENCE).
MIN_FIELD_CONFIDENCE = 0.80

# How many times we will attempt to capture one field in total (initial ask + 1
# clarification). The spec caps clarification at exactly one question, so this is 2.
MAX_FIELD_ATTEMPTS = 2

# Default endpoint + model for the optional LLM adapter. Both are overridable, because the
# adapter speaks the OpenAI Chat Completions wire format — which means a free-tier provider
# (Groq, Google Gemini, OpenRouter, DeepSeek, …) is a base-URL and model-name swap.
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


# --------------------------------------------------------------------------- #
# Load backend/.env BEFORE any Settings field reads os.environ.
#
# This matters: `Settings` reads os.environ directly, so without this call a .env file is
# *silently ignored*. You would paste a key, restart, and nothing would change — which is
# a genuinely confusing way to lose an hour.
#
# `override=False` keeps the real environment authoritative: an exported var beats the
# file, so CI or a shell profile can still override it.
# --------------------------------------------------------------------------- #
def _load_env_file() -> str | None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:  # pragma: no cover - dotenv ships with uvicorn[standard]
        _logger.debug("python-dotenv not installed; skipping .env load.")
        return None

    for candidate in (BACKEND_DIR / ".env", BACKEND_DIR / ".env.local"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return candidate.name
    return None


ENV_FILE = _load_env_file()


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _first_env(*names: str) -> str | None:
    for name in names:
        value = _env(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class Settings:
    app_name: str = "Energy Recovery Voice Agent"
    api_prefix: str = ""
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"sqlite:///{(BACKEND_DIR / 'energy_recovery.db').as_posix()}"
        )
    )
    seed_on_startup: bool = field(default_factory=lambda: not _env_flag("SKIP_SEED"))

    # --- LLM: extraction + phrasing only, never decisions ---------------------- #
    # `LLM_API_KEY` is the preferred name; `OPENAI_API_KEY` is accepted as an alias so
    # existing setups keep working.
    llm_api_key: str | None = field(
        default_factory=lambda: _first_env("LLM_API_KEY", "OPENAI_API_KEY")
    )
    llm_base_url: str = field(
        default_factory=lambda: _first_env("LLM_BASE_URL", "OPENAI_BASE_URL")
        or DEFAULT_LLM_BASE_URL
    )
    llm_model: str = field(
        default_factory=lambda: _env("LLM_MODEL") or DEFAULT_LLM_MODEL
    )
    # Cosmetic only — used in logs and /health so you can see *which* vendor answered.
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER") or "openai")

    # --- Speech-to-text providers --------------------------------------------- #
    deepgram_api_key: str | None = field(default_factory=lambda: _env("DEEPGRAM_API_KEY"))
    assemblyai_api_key: str | None = field(default_factory=lambda: _env("ASSEMBLYAI_API_KEY"))
    # Force one provider to the front of the queue. Unset -> preference order below.
    stt_provider: str | None = field(default_factory=lambda: _env("STT_PROVIDER"))
    stt_timeout_seconds: float = field(
        default_factory=lambda: float(_env("STT_TIMEOUT_SECONDS") or 25.0)
    )

    # --- Telephony (outbound dialling + warm transfer) ------------------------- #
    twilio_account_sid: str | None = field(default_factory=lambda: _env("TWILIO_ACCOUNT_SID"))
    twilio_auth_token: str | None = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN"))
    twilio_from_number: str | None = field(default_factory=lambda: _env("TWILIO_FROM_NUMBER"))

    # Where a warm handoff transfers the live call. Unset -> the handoff is queued in the
    # console instead, which is how the local demo runs.
    handoff_transfer_number: str | None = field(
        default_factory=lambda: _env("HANDOFF_TRANSFER_NUMBER")
    )

    # -- derived ---------------------------------------------------------------- #
    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def telephony_enabled(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token and self.twilio_from_number)

    @property
    def server_stt_providers(self) -> list[str]:
        """Server-side STT vendors that are genuinely usable, in preference order.

        An explicitly named `STT_PROVIDER` always wins; otherwise Deepgram and AssemblyAI
        (both free-tier) are preferred over Whisper, which shares the LLM key.
        """
        found: list[str] = []
        if self.stt_provider:
            found.append(self.stt_provider)
        if self.deepgram_api_key and "deepgram" not in found:
            found.append("deepgram")
        if self.assemblyai_api_key and "assemblyai" not in found:
            found.append("assemblyai")
        if self.llm_api_key and "openai" not in found:
            found.append("openai")
        return found

    @property
    def active_stt_provider(self) -> str:
        providers = self.server_stt_providers
        return providers[0] if providers else "browser_web_speech"

    def capability_report(self) -> dict[str, object]:
        """Surfaced on /health so judges can see exactly what is live vs. simulated.

        Every flag here is backed by a call path that actually exists. There is no
        aspirational reporting: a capability report that claims a provider is live when
        nothing calls it is worse than no report at all, because it makes the whole
        guardrail story untrustworthy.
        """
        return {
            "core_demo_requires_api_keys": False,
            "env_file_loaded": ENV_FILE,
            "stt": {
                "browser_web_speech": True,
                "simulated_turns": True,
                "server_upload_endpoint": True,
                "deepgram": bool(self.deepgram_api_key),
                "assemblyai": bool(self.assemblyai_api_key),
                "openai_whisper": bool(self.llm_api_key),
                "active": self.active_stt_provider,
            },
            "tts": {
                # The browser speaks every agent turn. There is deliberately no external
                # TTS vendor wired up — it would add a paid dependency for no demo gain.
                "browser_speech_synthesis": True,
                "external_provider": False,
            },
            "llm_extraction": {
                "rules_engine": True,
                "external_adapter": self.llm_enabled,
                "provider": self.llm_provider if self.llm_enabled else "RULES_ONLY",
                "model": self.llm_model if self.llm_enabled else None,
                "base_url": self.llm_base_url if self.llm_enabled else None,
            },
            "telephony": {
                "browser_microphone": True,
                "twilio": self.telephony_enabled,
            },
        }


settings = Settings()
