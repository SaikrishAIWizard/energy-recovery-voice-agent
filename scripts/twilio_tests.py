"""Live-phone-call tests (Twilio).

Runs the FastAPI app in-process against a throwaway SQLite file with Twilio's REST API
stubbed and every webhook signed the way Twilio signs it, so it needs no Twilio account, no
tunnel and no backend running:

    python scripts/twilio_tests.py

What it proves: "Start call by agent" places the call only when asked; the webhooks drive
the same state machine as a browser call (a whole journey completes over the phone);
forged or stale webhooks are refused; silence and a customer hanging up are handled; and a
handoff moves the customer into a conference and dials the human agent into that same call.

What it can NOT prove is Twilio itself: speech recognition quality, ring/answer timing, and
that Twilio accepts the TwiML. That needs one real call (see the README).
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlencode

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

TOKEN = "test-auth-token"
PUBLIC = "https://demo.example.test"
AGENT_PHONE = "+61400000999"

_tmp = tempfile.mkdtemp(prefix="twilio_tests_")
os.environ.update(
    {
        "DATABASE_URL": f"sqlite:///{Path(_tmp, 'test.db').as_posix()}",
        "TWILIO_ACCOUNT_SID": "ACtest",
        "TWILIO_AUTH_TOKEN": TOKEN,
        "TWILIO_FROM_NUMBER": "+15005550006",
        "PUBLIC_BASE_URL": PUBLIC,
        "HANDOFF_TRANSFER_NUMBER": AGENT_PHONE,
        "TWILIO_DIAL_OVERRIDE_NUMBER": "",
        "TWILIO_VALIDATE_SIGNATURE": "true",
    }
)

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import llm_service, twilio_service  # noqa: E402
from app.services.llm_service import RulesOnlyLLM  # noqa: E402
from app.services.voice_service import voice_providers  # noqa: E402

llm_service._llm_singleton = RulesOnlyLLM()  # rules engine only: deterministic, no network

PASS, FAIL = "PASS", "FAIL"
failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    failures += 0 if condition else 1
    print(f"  [{PASS if condition else FAIL}] {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# Twilio REST stub
# --------------------------------------------------------------------------- #
rest_calls: list[tuple[str, dict]] = []
_counter = {"customer": 0, "agent": 0}


class FakeResponse:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self) -> dict:
        return self._body


def fake_post(url: str, **kwargs):
    data = kwargs.get("data") or {}
    rest_calls.append((url, data))
    if url.endswith("/Calls.json"):
        kind = "agent" if data.get("To") == AGENT_PHONE else "customer"
        _counter[kind] += 1
        return FakeResponse(201, {"sid": f"CA{kind}{_counter[kind]}"})
    return FakeResponse(200, {"sid": "updated"})


httpx.post = fake_post  # type: ignore[assignment]


def created_calls() -> list[dict]:
    return [d for u, d in rest_calls if u.endswith("/Calls.json")]


def updates() -> list[tuple[str, dict]]:
    return [(u, d) for u, d in rest_calls if "/Calls/" in u]


# --------------------------------------------------------------------------- #
# Signed webhooks
# --------------------------------------------------------------------------- #
def hook(client: TestClient, path: str, params: dict, query: str = "", sign: bool = True, sid: str | None = None):
    params = dict(params)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if sign:
        headers["X-Twilio-Signature"] = twilio_service.expected_signature(PUBLIC + path + query, params, TOKEN)
    return client.post(path + query, content=urlencode(params), headers=headers)


class Phone:
    """One customer on the phone: start, dial, answer, then talk."""

    def __init__(self, client: TestClient, lead_id: str, force: bool = False) -> None:
        self.client = client
        started = client.post(f"/calls/start/{lead_id}", json={"force": force}).json()
        self.id = started["session"]["id"]
        self.sid: str | None = None

    def dial(self):
        r = self.client.post(f"/calls/{self.id}/dial")
        if r.status_code == 200:
            self.sid = r.json()["session"]["telephony_reference"]
        return r

    def hook(self, endpoint: str, extra: dict | None = None, query: str = ""):
        return hook(self.client, f"/twilio/{endpoint}/{self.id}", {"CallSid": self.sid, **(extra or {})}, query)

    def answer(self):
        return self.hook("voice", {"CallStatus": "in-progress"})

    def say(self, text: str | None):
        return self.hook("turn", {"SpeechResult": text, "Confidence": "0.93"} if text else {"SpeechResult": ""})

    def session(self) -> dict:
        return self.client.get(f"/calls/{self.id}").json()

    def audit(self) -> list[str]:
        return [e["event_type"] for e in self.session()["audit_events"]]


def main() -> int:
    with TestClient(app) as client:
        print("\n0. Signature algorithm")
        vector = twilio_service.expected_signature(
            "https://mycompany.com/myapp.php?foo=1&bar=2",
            {"CallSid": "CA1234567890ABCDE", "Caller": "+14158675309", "Digits": "1234",
             "From": "+14158675309", "To": "+18005551212"},
            "12345",
        )
        check("matches Twilio's published test vector", vector == "RSOYDt4T1cUTdK1PDd93/VVr8B8=", vector)
        health = client.get("/health").json()["capabilities"]["telephony"]
        check("health reports phone calls on", health["live_calls"] is True, str(health))

        print("\n1. Starting a recovery does not ring anyone")
        call = Phone(client, "E-1001")
        check("no Twilio call made at start", created_calls() == [], str(len(created_calls())))
        check("session is on the browser channel", call.session()["dial_provider"] == "browser_microphone")

        print("\n2. Start call by agent")
        r = call.dial()
        check("200 OK", r.status_code == 200, r.text[:150])
        placed = created_calls()[-1]
        check("customer's phone dialled (E.164, spaces stripped)", placed["To"] == "+61400000001", placed["To"])
        check("answer webhook points at this session", placed["Url"] == f"{PUBLIC}/twilio/voice/{call.id}", placed["Url"])
        check("status callback registered", placed["StatusCallback"] == f"{PUBLIC}/twilio/status/{call.id}")
        check("call sid stored on the session", call.sid == "CAcustomer1", str(call.sid))
        check("PHONE_CALL_PLACED audited", "PHONE_CALL_PLACED" in call.audit())
        check("dialling twice is refused", client.post(f"/calls/{call.id}/dial").status_code == 409)

        print("\n3. Webhooks must be signed by Twilio")
        check("unsigned -> 403", hook(client, f"/twilio/voice/{call.id}", {"CallSid": call.sid}, sign=False).status_code == 403)
        forged = client.post(f"/twilio/voice/{call.id}", content=urlencode({"CallSid": call.sid}),
                             headers={"Content-Type": "application/x-www-form-urlencoded", "X-Twilio-Signature": "AAAA"})
        check("wrong signature -> 403", forged.status_code == 403)
        r = hook(client, f"/twilio/turn/{call.id}", {"CallSid": "CAsomeoneelse", "SpeechResult": "yes"})
        check("valid signature but another call's sid -> 403", r.status_code == 403, str(r.status_code))
        r = hook(client, "/twilio/voice/CS-NOPE", {"CallSid": "CAx"})
        check("unknown session -> 404", r.status_code == 404, str(r.status_code))

        print("\n4. A whole journey over the phone")
        r = call.answer()
        xml = r.text
        check("answer -> TwiML", r.status_code == 200 and r.headers["content-type"].startswith("application/xml"))
        check("speaks the recording disclosure", "This call is recorded" in xml, xml[:200])
        check("listens for speech, replies to /twilio/turn", "<Gather" in xml and f"/twilio/turn/{call.id}" in xml)
        check("the customer picked up (audited)", "PHONE_CALL_ANSWERED" in call.audit())
        n_prompts = sum(1 for s in call.session()["transcript"] if s["speaker"] == "AI_AGENT")
        check("the disclosure was not duplicated in the transcript", n_prompts == 1, str(n_prompts))

        script = [
            ("Yes, go ahead.", "service address"),
            ("12 Test Street, Sydney NSW 2000", "moving into"),
            ("1 December 2030", "electricity, gas, or both"),
            ("electricity and gas", "concession"),
            ("No", "life-support"),
            ("No", "phone or by email"),
            ("email please", "read that back"),
        ]
        for said, expect in script:
            xml = call.say(said).text
            check(f"after '{said}' the agent asks about: {expect}", expect in xml and "<Gather" in xml, xml[:160] if expect not in xml else "")
        xml = call.say("yes that's right").text
        check("confirmation ends the call with a hangup", "<Hangup/>" in xml and "submitted" in xml, xml[:200])
        session = call.session()
        check("journey COMPLETED and submitted", session["status"] == "COMPLETED" and session["submission"] is not None, session["status"])
        check("spoken answers are the payload", (session["submission"] or {}).get("payload", {}).get("energy_requirement") == "BOTH")
        check("customer's words are in the transcript", any(s["speaker"] == "CUSTOMER" and "12 Test Street" in s["text"] for s in session["transcript"]))
        hook(client, f"/twilio/status/{call.id}", {"CallSid": call.sid, "CallStatus": "completed"})
        check("the hang-up after completion is just logged", "PHONE_CALL_ENDED" in call.audit() and call.session()["status"] == "COMPLETED")

        print("\n5. Silence is a failed capture: one clarification, then a human")
        quiet = Phone(client, "E-1004")
        quiet.dial()
        quiet.answer()
        xml = quiet.say(None).text
        check("first silence -> the approved fallback prompt", "Sorry, I didn" in xml and "<Gather" in xml, xml[:200])
        n = sum(1 for s in quiet.session()["transcript"] if s["speaker"] == "CUSTOMER")
        check("silence adds no empty transcript line", n == 0, str(n))
        xml = quiet.say(None).text
        check("second silence -> handoff, customer redirected to the conference", f"/twilio/handoff/{quiet.id}?spoken=1" in xml and "<Redirect" in xml, xml[:250])
        check("the engine did not also redirect over REST", not any(quiet.sid in u for u, _ in updates()), str(updates()))
        check("session is HANDOFF_REQUESTED", quiet.session()["status"] == "HANDOFF_REQUESTED")

        print("\n6. Handoff: the human agent joins the same call")
        before = len(created_calls())
        r = quiet.hook("handoff", query="?spoken=1")
        xml = r.text
        check("customer put in the conference to wait", "<Conference" in xml and 'startConferenceOnEnter="false"' in xml and f"handoff-{quiet.id}" in xml, xml[:300])
        agent_call = created_calls()[-1]
        check("agent's phone dialled", len(created_calls()) == before + 1 and agent_call["To"] == AGENT_PHONE, str(agent_call.get("To")))
        check("agent leg URL is this session's", agent_call["Url"] == f"{PUBLIC}/twilio/agent/{quiet.id}", agent_call["Url"])
        quiet.hook("handoff", query="?spoken=1")
        check("a repeated redirect does not ring the agent twice", len(created_calls()) == before + 1)
        check("HANDOFF_AGENT_DIALLED audited", "HANDOFF_AGENT_DIALLED" in quiet.audit())

        agent = {"CallSid": "CAagent1"}
        r = hook(client, f"/twilio/agent/{quiet.id}", agent)
        check("agent hears a briefing that names the customer", "Daniel" in r.text and "Press 1" in r.text, r.text[:250])
        check("briefing reads field names, not values", "Test Street" not in r.text)
        r = hook(client, f"/twilio/agent-accept/{quiet.id}", {**agent, "Digits": "2"})
        check("agent who does not press 1 is not connected", "<Conference" not in r.text and "<Hangup/>" in r.text)
        r = hook(client, f"/twilio/agent-accept/{quiet.id}", {**agent, "Digits": "1"})
        check("agent pressing 1 joins the same conference and starts it", "<Conference" in r.text and 'startConferenceOnEnter="true"' in r.text and f"handoff-{quiet.id}" in r.text, r.text[:250])
        hook(client, f"/twilio/conference/{quiet.id}", {"CallSid": "CAagent1", "StatusCallbackEvent": "participant-join"})
        check("HANDOFF_AGENT_JOINED audited", "HANDOFF_AGENT_JOINED" in quiet.audit())
        check("handoff marked accepted by the phone agent", quiet.session()["handoff"]["accepted_by"] == "Human agent (phone)")
        check("when the customer's leg ends after the agent joined -> plain hangup", quiet.hook("handoff-ended").text.endswith("<Hangup/></Response>") and "Sorry" not in quiet.hook("handoff-ended").text)
        r = hook(client, f"/twilio/agent-status/{quiet.id}", {"CallSid": "CAagent1", "CallStatus": "no-answer"})
        check("agent line status callback accepted", r.status_code == 204)

        print("\n7. Agent never answers: the customer is told, and can be re-dialled")
        lonely = Phone(client, "E-1006")
        lonely.dial()
        lonely.answer()
        lonely.say("Can you put me through to a person?")
        lonely.hook("handoff", query="?spoken=1")
        check("never-joined hold ends with an apology", "HANDOFF_AGENT_JOINED" not in lonely.audit() and "couldn't reach a colleague" in lonely.hook("handoff-ended").text)
        n = len(created_calls())
        r = client.post(f"/calls/{lonely.id}/handoff/dial-agent")
        check("console can ring the agent again", r.status_code == 200 and len(created_calls()) == n + 1, r.text[:120])

        print("\n8. Handoff from the console during a live call")
        live = Phone(client, "E-1007")
        live.dial()
        live.answer()
        client.post(f"/calls/{live.id}/handoff", json={"reason": "CUSTOMER_REQUEST"})
        moved = [d for u, d in updates() if live.sid in u]
        check("customer's call is redirected into the handoff webhook", len(moved) == 1 and moved[0]["Url"] == f"{PUBLIC}/twilio/handoff/{live.id}", str(moved))
        check("WARM_TRANSFER_ATTEMPT audited", "WARM_TRANSFER_ATTEMPT" in live.audit())

        print("\n9. Ending from the console hangs up the phone")
        ender = Phone(client, "E-1002")
        ender.dial()
        ender.answer()
        client.post(f"/calls/{ender.id}/end")
        ended = [d for u, d in updates() if ender.sid in u]
        check("live call is told to say goodbye and hang up", len(ended) == 1 and "<Hangup/>" in ended[0].get("Twiml", ""), str(ended))

        print("\n10. Not answered -> back to the browser channel, retry allowed")
        ring = Phone(client, "E-1005")
        ring.dial()
        hook(client, f"/twilio/status/{ring.id}", {"CallSid": ring.sid, "CallStatus": "no-answer"})
        s = ring.session()
        check("PHONE_CALL_UNANSWERED audited", "PHONE_CALL_UNANSWERED" in ring.audit())
        check("session still ACTIVE on the browser channel", s["status"] == "ACTIVE" and s["dial_provider"] == "browser_microphone", f"{s['status']}/{s['dial_provider']}")
        check("can dial again", ring.dial().status_code == 200)

        print("\n11. Customer hangs up mid-call -> what was captured is kept")
        drop = Phone(client, "E-1005")
        drop.dial()
        drop.answer()
        drop.say("Yes, go ahead.")
        drop.say("1 December 2030")
        hook(client, f"/twilio/status/{drop.id}", {"CallSid": drop.sid, "CallStatus": "completed"})
        s = drop.session()
        check("session INCOMPLETE, outcome PHONE_HANG_UP", s["status"] == "INCOMPLETE" and s["outcome_detail"] == "PHONE_HANG_UP", f"{s['status']}/{s['outcome_detail']}")
        item = next(i for i in client.get("/leads").json() if i["lead"]["id"] == "E-1005")
        check("lead is back in the recovery queue", item["lead"]["status"] == "DROPPED_OFF" and item["latest_session_status"] == "INCOMPLETE")
        check("queue resumes after what was captured", item["resume_step"] == "energy_requirement", item["resume_step"])

        print("\n12. Guardrails still apply on the phone")
        dnc = Phone(client, "E-1003", force=True)
        n = len(created_calls())
        r = dnc.dial()
        check("Do-Not-Call lead cannot be phoned, even if the console was forced", r.status_code == 409 and len(created_calls()) == n, r.text[:120])
        card = Phone(client, "E-1006")
        card.dial()
        card.answer()
        xml = card.say("my card number is 4111 1111 1111 1111").text
        check("card number -> handoff, never echoed", "<Redirect" in xml and "4111" not in xml)
        check("card number redacted in the transcript", not any("4111 1111" in s["text"] for s in card.session()["transcript"]))

        print("\n12b. An earlier call is never buried by a newer one")
        # E-1004's phone call (section 5/6) ended in a handoff. Start another call for the lead.
        newer = Phone(client, "E-1004")
        queue_row = next(i for i in client.get("/leads").json() if i["lead"]["id"] == "E-1004")
        check("the queue now points at the newer, empty call", queue_row["latest_session_id"] == newer.id)
        handoffs = client.get("/handoffs").json()
        check("the earlier handoff is still listed as waiting for a human", any(h["session_id"] == quiet.id for h in handoffs), str([h["session_id"] for h in handoffs]))
        history = client.get("/leads/E-1004/sessions").json()
        ids = [h["id"] for h in history]
        check("the lead's history lists both calls, newest first", ids[:2] == [newer.id, quiet.id], str(ids))
        old = next(h for h in history if h["id"] == quiet.id)
        check("the old call keeps its transcript", old["transcript_segments"] >= 3, str(old["transcript_segments"]))
        check("the old call is a phone call in HANDOFF_REQUESTED", old["dial_provider"] == "twilio" and old["status"] == "HANDOFF_REQUESTED")
        check("the old call's full record is still served", len(client.get(f"/calls/{quiet.id}").json()["transcript"]) == old["transcript_segments"])
        check("unknown lead history -> 404", client.get("/leads/E-9999/sessions").status_code == 404)

        print("\n13. Configuration edges")
        original = twilio_service.settings
        twilio_service.settings = dataclasses.replace(original, twilio_dial_override_number="+61411111111")
        over = Phone(client, "E-1006")
        over.dial()
        twilio_service.settings = original
        check("override number is dialled instead of the lead's", created_calls()[-1]["To"] == "+61411111111", created_calls()[-1]["To"])
        check("override is audited", '"override_number": true' in next(e["event_detail"] for e in over.session()["audit_events"] if e["event_type"] == "PHONE_CALL_PLACED"))

        twilio_service.settings = dataclasses.replace(original, handoff_transfer_number=None)
        noagent = Phone(client, "E-1006")
        noagent.dial()
        noagent.answer()
        noagent.say("Can you put me through to a person?")
        n = len(created_calls())
        xml = noagent.hook("handoff", query="?spoken=1").text
        twilio_service.settings = original
        check("no agent number -> caller told, nobody dialled", "<Conference" not in xml and "<Hangup/>" in xml and len(created_calls()) == n)
        check("HANDOFF_AGENT_UNCONFIGURED audited", "HANDOFF_AGENT_UNCONFIGURED" in noagent.audit())

        saved = voice_providers.telephony.pop("twilio")
        cold = Phone(client, "E-1002")
        r = cold.dial()
        voice_providers.telephony["twilio"] = saved
        check("Twilio not configured -> 503 with what to set", r.status_code == 503 and "PUBLIC_BASE_URL" in r.text, r.text[:120])

    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
