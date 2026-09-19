"""Voice provider abstraction.

The core demo needs **no credentials at all**: the browser handles STT (Web Speech API)
and TTS (SpeechSynthesis), and the console can drive simulated transcript turns. Provider
adapters exist behind this interface so a real STT/TTS/telephony vendor can be dropped in
without touching the state machine.

Each adapter declares `available`; the registry only activates one when its credentials
are present in the environment.

Two kinds of STT live here, and the distinction matters:

  * **Client-side** (`BrowserWebSpeechSTT`, `SimulatedTurnSTT`) — the browser transcribes
    and posts text. They cannot accept audio bytes, so they are not in `server_stt`.
  * **Server-side** (`DeepgramSTT`, `AssemblyAISTT`, `OpenAIWhisperSTT`) — these accept
    audio bytes and are reached through `POST /calls/{id}/audio`.

Only providers in `server_stt` are advertised as live in `/health`. A provider that is
registered but never called would be a lie, so the registry keeps the two groups apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import DEFAULT_LLM_BASE_URL, settings

logger = logging.getLogger(__name__)


class STTProvider(Protocol):
    """`recording=True` means a whole uploaded call rather than one live turn: use the long
    timeout and, where the vendor supports it, return speaker-labelled `segments`
    (`[{"speaker", "start", "end", "text", "confidence"}]`, times in seconds)."""

    name: str
    available: bool

    def transcribe(
        self, audio: bytes, mime_type: str = "audio/webm", recording: bool = False
    ) -> dict[str, Any]: ...


class TTSProvider(Protocol):
    name: str
    available: bool

    def synthesize(self, text: str) -> bytes | None: ...


class TelephonyProvider(Protocol):
    name: str
    available: bool

    def place_call(self, to_number: str, from_number: str | None = None) -> dict[str, Any]: ...

    def transfer(self, call_reference: str, to_number: str) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# Built-in, always-available providers
# --------------------------------------------------------------------------- #


@dataclass
class BrowserWebSpeechSTT:
    """No server work at all — the browser transcribes and posts text turns."""

    name: str = "browser_web_speech"
    available: bool = True

    def transcribe(
        self, audio: bytes, mime_type: str = "audio/webm", recording: bool = False
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Browser Web Speech API transcribes client-side. Post the text to "
            "POST /calls/{id}/utterance with source=BROWSER_SPEECH instead."
        )


@dataclass
class SimulatedTurnSTT:
    """Typed / scripted customer turns — the demo's reliable fallback."""

    name: str = "simulated_turns"
    available: bool = True

    def transcribe(
        self, audio: bytes, mime_type: str = "audio/webm", recording: bool = False
    ) -> dict[str, Any]:
        raise NotImplementedError("Simulated mode posts text directly; there is no audio.")


@dataclass
class BrowserSpeechSynthesisTTS:
    name: str = "browser_speech_synthesis"
    available: bool = True

    def synthesize(self, text: str) -> bytes | None:
        return None  # the browser speaks it


@dataclass
class NoOpTelephony:
    name: str = "browser_microphone"
    available: bool = True

    def place_call(self, to_number: str, from_number: str | None = None) -> dict[str, Any]:
        return {
            "placed": False,
            "mode": "LOCAL_BROWSER",
            "detail": "No telephony provider configured — the console runs the call locally.",
        }

    def transfer(self, call_reference: str, to_number: str) -> dict[str, Any]:
        return {
            "transferred": False,
            "mode": "LOCAL_BROWSER",
            "detail": "Warm handoff is simulated in the console handoff queue.",
        }


# --------------------------------------------------------------------------- #
# Optional STT adapters (inert without credentials)
# --------------------------------------------------------------------------- #


@dataclass
class DeepgramSTT:
    """Deepgram pre-recorded listen. One HTTP call, audio in the request body."""

    api_key: str
    name: str = "deepgram"
    available: bool = field(init=False, default=True)

    def transcribe(
        self, audio: bytes, mime_type: str = "audio/webm", recording: bool = False
    ) -> dict[str, Any]:
        try:
            import httpx

            query = "model=nova-2&smart_format=true&language=en-AU"
            if recording:
                # Not smart_format: it rewrites a spoken date into US-order numerals
                # ("3rd of April" -> "04/03"), which an Australian reader takes as 4 March.
                # punctuate+numerals keeps the month as a word. Diarize + utterances give
                # speaker labels and boundaries, so the caller can tell agent from customer.
                query = (
                    "model=nova-2&language=en-AU&punctuate=true&numerals=true"
                    "&diarize=true&utterances=true"
                )
            response = httpx.post(
                f"https://api.deepgram.com/v1/listen?{query}",
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": mime_type,
                },
                content=audio,
                timeout=settings.stt_upload_timeout_seconds
                if recording
                else settings.stt_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            alt = body["results"]["channels"][0]["alternatives"][0]
            result: dict[str, Any] = {
                "text": alt.get("transcript", ""),
                "confidence": alt.get("confidence"),
            }
            if recording:
                result["segments"] = [
                    {
                        "speaker": u.get("speaker"),
                        "start": u.get("start"),
                        "end": u.get("end"),
                        "text": u.get("transcript", ""),
                        "confidence": u.get("confidence"),
                    }
                    for u in body["results"].get("utterances") or []
                ]
            return result
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("Deepgram STT failed: %s", exc)
            return {"text": "", "confidence": 0.0, "error": str(exc)}


@dataclass
class AssemblyAISTT:
    """AssemblyAI v2 — upload the audio, then poll for the transcript.

    Two calls rather than one, because AssemblyAI is asynchronous by design. Free tier
    available (signup credit), which is why it is offered alongside Deepgram.
    """

    api_key: str
    name: str = "assemblyai"
    available: bool = field(init=False, default=True)

    def transcribe(
        self, audio: bytes, mime_type: str = "audio/webm", recording: bool = False
    ) -> dict[str, Any]:
        import time

        timeout = settings.stt_upload_timeout_seconds if recording else settings.stt_timeout_seconds
        try:
            import httpx

            headers = {"authorization": self.api_key}
            with httpx.Client(timeout=timeout) as client:
                upload = client.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers={**headers, "Content-Type": "application/octet-stream"},
                    content=audio,
                )
                upload.raise_for_status()
                upload_url = upload.json()["upload_url"]

                job = client.post(
                    "https://api.assemblyai.com/v2/transcript",
                    headers={**headers, "Content-Type": "application/json"},
                    json={
                        "audio_url": upload_url,
                        "language_code": "en_au",
                        **({"speaker_labels": True} if recording else {}),
                    },
                )
                job.raise_for_status()
                transcript_id = job.json()["id"]

                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    poll = client.get(
                        f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                        headers=headers,
                    )
                    poll.raise_for_status()
                    body = poll.json()
                    status = body.get("status")
                    if status == "completed":
                        result: dict[str, Any] = {
                            "text": body.get("text") or "",
                            "confidence": body.get("confidence"),
                        }
                        if recording:
                            # AssemblyAI reports milliseconds.
                            result["segments"] = [
                                {
                                    "speaker": u.get("speaker"),
                                    "start": (u.get("start") or 0) / 1000,
                                    "end": (u.get("end") or 0) / 1000,
                                    "text": u.get("text", ""),
                                    "confidence": u.get("confidence"),
                                }
                                for u in body.get("utterances") or []
                            ]
                        return result
                    if status == "error":
                        raise RuntimeError(body.get("error") or "AssemblyAI reported an error")
                    time.sleep(0.7)

            return {"text": "", "confidence": 0.0, "error": "AssemblyAI transcription timed out"}
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("AssemblyAI STT failed: %s", exc)
            return {"text": "", "confidence": 0.0, "error": str(exc)}


@dataclass
class OpenAIWhisperSTT:
    """OpenAI `/audio/transcriptions`. Shares the LLM key — no separate signup."""

    api_key: str
    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = "whisper-1"
    name: str = "openai"
    available: bool = field(init=False, default=True)

    def transcribe(
        self, audio: bytes, mime_type: str = "audio/webm", recording: bool = False
    ) -> dict[str, Any]:
        try:
            import httpx

            response = httpx.post(
                f"{self.base_url.rstrip('/')}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("turn.webm", audio, mime_type)},
                data={"model": self.model, "language": "en"},
                timeout=settings.stt_upload_timeout_seconds
                if recording
                else settings.stt_timeout_seconds,
            )
            response.raise_for_status()
            return {"text": response.json().get("text", ""), "confidence": None}
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("OpenAI Whisper STT failed: %s", exc)
            return {"text": "", "confidence": 0.0, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Optional telephony adapter (inert without credentials)
# --------------------------------------------------------------------------- #


@dataclass
class TwilioTelephony:
    account_sid: str
    auth_token: str
    from_number: str
    name: str = "twilio"
    available: bool = field(init=False, default=True)

    def place_call(self, to_number: str, from_number: str | None = None) -> dict[str, Any]:
        try:
            import httpx

            response = httpx.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Calls.json",
                auth=(self.account_sid, self.auth_token),
                data={
                    "To": to_number,
                    "From": from_number or self.from_number,
                    "Url": "http://demo.invalid/twiml",
                },
                timeout=20.0,
            )
            response.raise_for_status()
            body = response.json()
            return {"placed": True, "call_reference": body.get("sid"), "provider": self.name}
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("Twilio place_call failed: %s", exc)
            return {"placed": False, "error": str(exc)}

    def transfer(self, call_reference: str, to_number: str) -> dict[str, Any]:
        try:
            import httpx

            response = httpx.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}"
                f"/Calls/{call_reference}.json",
                auth=(self.account_sid, self.auth_token),
                data={"Twiml": f"<Response><Dial>{to_number}</Dial></Response>"},
                timeout=20.0,
            )
            response.raise_for_status()
            return {"transferred": True, "provider": self.name}
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("Twilio transfer failed: %s", exc)
            return {"transferred": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class VoiceProviderRegistry:
    def __init__(self) -> None:
        self.stt: dict[str, STTProvider] = {
            "browser_web_speech": BrowserWebSpeechSTT(),
            "simulated_turns": SimulatedTurnSTT(),
        }
        self.tts: dict[str, TTSProvider] = {
            "browser_speech_synthesis": BrowserSpeechSynthesisTTS(),
        }
        self.telephony: dict[str, TelephonyProvider] = {
            "browser_microphone": NoOpTelephony(),
        }

        # Insertion order *is* the preference order — see `active_stt`.
        self.server_stt: dict[str, STTProvider] = {}
        if settings.deepgram_api_key:
            deepgram = DeepgramSTT(api_key=settings.deepgram_api_key)
            self.stt["deepgram"] = deepgram
            self.server_stt["deepgram"] = deepgram
        if settings.assemblyai_api_key:
            assemblyai = AssemblyAISTT(api_key=settings.assemblyai_api_key)
            self.stt["assemblyai"] = assemblyai
            self.server_stt["assemblyai"] = assemblyai
        if settings.llm_api_key:
            whisper = OpenAIWhisperSTT(
                api_key=settings.llm_api_key, base_url=settings.llm_base_url
            )
            self.stt["openai"] = whisper
            self.server_stt["openai"] = whisper

        if settings.telephony_enabled:
            self.telephony["twilio"] = TwilioTelephony(
                account_sid=settings.twilio_account_sid or "",
                auth_token=settings.twilio_auth_token or "",
                from_number=settings.twilio_from_number or "",
            )

    @property
    def active_stt(self) -> str:
        """The server-side provider audio uploads will be routed to.

        `STT_PROVIDER` wins if it names a provider that actually has credentials;
        otherwise the first configured provider in preference order does. With no
        credentials at all, the browser stays the transcriber.
        """
        preferred = settings.stt_provider
        if preferred and preferred in self.server_stt:
            return preferred
        for name in self.server_stt:
            return name
        return "browser_web_speech"

    @property
    def active_tts(self) -> str:
        return "browser_speech_synthesis"

    @property
    def active_telephony(self) -> str:
        return "twilio" if "twilio" in self.telephony else "browser_microphone"

    def describe(self) -> dict[str, Any]:
        return {
            "stt": {name: provider.available for name, provider in self.stt.items()},
            "tts": {name: provider.available for name, provider in self.tts.items()},
            "telephony": {
                name: provider.available for name, provider in self.telephony.items()
            },
            "server_stt": list(self.server_stt),
            "active": {
                "stt": self.active_stt,
                "tts": self.active_tts,
                "telephony": self.active_telephony,
            },
            "requires_api_key_for_demo": False,
        }

    # -- speech-to-text ----------------------------------------------------- #
    # Reached by POST /calls/{id}/audio and POST /recordings/upload. Without a configured provider the endpoint
    # refuses up front, so this method only ever runs with somewhere to send the bytes.

    def transcribe(
        self, audio: bytes, mime_type: str = "audio/webm", recording: bool = False
    ) -> dict[str, Any]:
        name = self.active_stt
        provider = self.server_stt.get(name)
        if provider is None:
            return {"text": "", "confidence": 0.0, "error": "no_server_stt_provider"}
        try:
            # A live turn calls providers exactly as it always has; only an uploaded
            # recording asks for the long timeout and speaker labels.
            result = (
                provider.transcribe(audio, mime_type, recording=True)
                if recording
                else provider.transcribe(audio, mime_type)
            )
        except Exception as exc:  # pragma: no cover - provider path
            logger.warning("%s transcribe failed: %s", name, exc)
            result = {"text": "", "confidence": 0.0, "error": str(exc)}
        result.setdefault("provider", name)
        return result

    # -- telephony ---------------------------------------------------------- #
    # The state machine dials and transfers through these, so the adapter seam is a real
    # call path rather than a decorative interface. With no credentials the browser
    # provider answers and the console runs the call locally.

    @property
    def telephony_provider(self) -> TelephonyProvider:
        return self.telephony[self.active_telephony]

    def dial(self, to_number: str) -> dict[str, Any]:
        provider = self.telephony_provider
        try:
            result = provider.place_call(to_number)
        except Exception as exc:  # pragma: no cover - provider path
            logger.warning("%s place_call failed: %s", provider.name, exc)
            result = {"placed": False, "provider": provider.name, "error": str(exc)}
        result.setdefault("provider", provider.name)
        return result

    def warm_transfer(self, call_reference: str | None, to_number: str | None) -> dict[str, Any]:
        provider = self.telephony_provider
        if not call_reference or not to_number:
            return {
                "transferred": False,
                "provider": provider.name,
                "mode": "CONSOLE_QUEUE",
                "detail": "No telephony reference — handoff is queued in the console instead.",
            }
        try:
            result = provider.transfer(call_reference, to_number)
        except Exception as exc:  # pragma: no cover - provider path
            logger.warning("%s transfer failed: %s", provider.name, exc)
            result = {"transferred": False, "provider": provider.name, "error": str(exc)}
        result.setdefault("provider", provider.name)
        return result


voice_providers = VoiceProviderRegistry()
