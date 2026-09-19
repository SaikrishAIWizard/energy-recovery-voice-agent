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
import json
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


refuse_agent = {"on": False}


def fake_post(url: str, **kwargs):
    data = kwargs.get("data") or {}
    rest_calls.append((url, data))
    if url.endswith("/Calls.json") and data.get("To") == AGENT_PHONE and refuse_agent["on"]:
        return FakeResponse(
            400,
            {"message": "The number +61400000999 is unverified. Trial accounts may only make calls to verified numbers."},
        )
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
            ("12 Test Street, Sydney NSW 2000", "I have 12 Test Street, Sydney NSW 2000. Is that correct?"),
            ("yes", "moving into"),
            ("1 December 2030", "I have 1 December 2030. Is that correct?"),
            ("yes", "electricity, gas, or both"),
            ("electricity and gas", "Is that right?"),
            ("yes", "concession"),
            ("No", "life-support"),
            ("No", "phone or email"),
            ("email please", "check I have this right"),
        ]
        for said, expect in script:
            xml = call.say(said).text
            check(f"after '{said}' the agent asks about: {expect}", expect in xml and "<Gather" in xml, xml[:160] if expect not in xml else "")
        xml = call.say("yes that's right").text
        check("confirmation is followed by the closing question, not a hangup", "<Gather" in xml and "submitted" in xml and "anything else you need from a team member" in xml and "<Hangup/>" not in xml, xml[:200])
        session = call.session()
        check("journey COMPLETED and submitted", session["status"] == "COMPLETED" and session["submission"] is not None, session["status"])
        check("call stays open for the closing answer", session["state"] == "CLOSING", session["state"])
        xml = call.say("No, that's everything, thanks.").text
        check("'nothing else' ends the call with goodbye and a hangup", "<Hangup/>" in xml and "Goodbye" in xml, xml[:160])
        check("call is now fully COMPLETED", call.session()["state"] == "COMPLETED")
        check("spoken answers are the payload", (session["submission"] or {}).get("payload", {}).get("energy_requirement") == "BOTH")
        check("customer's words are in the transcript", any(s["speaker"] == "CUSTOMER" and "12 Test Street" in s["text"] for s in session["transcript"]))
        hook(client, f"/twilio/status/{call.id}", {"CallSid": call.sid, "CallStatus": "completed"})
        check("the hang-up after completion is just logged", "PHONE_CALL_ENDED" in call.audit() and call.session()["status"] == "COMPLETED")

        print("\n5. Silence is a failed capture: three follow-ups, then a human")
        quiet = Phone(client, "E-1004")
        quiet.dial()
        quiet.answer()
        xml = quiet.say(None).text
        check("first silence -> follow-up 1, the approved re-ask", "Sorry, I didn" in xml and "<Gather" in xml, xml[:200])
        n = sum(1 for s in quiet.session()["transcript"] if s["speaker"] == "CUSTOMER")
        check("silence adds no empty transcript line", n == 0, str(n))
        xml = quiet.say(None).text
        check("second silence -> follow-up 2, simpler", "Please answer yes or no" in xml and "<Gather" in xml, xml[:200])
        xml = quiet.say(None).text
        check("third silence -> follow-up 3", "Just yes or no" in xml and "<Gather" in xml, xml[:200])
        check("still on the call, not handed off yet", quiet.session()["status"] == "ACTIVE")
        xml = quiet.say(None).text
        check("fourth silence -> handoff, customer redirected to the conference", f"/twilio/handoff/{quiet.id}?spoken=1" in xml and "<Redirect" in xml, xml[:250])
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

        print("\n7. Agent busy: the customer is told to hold, and the agent is rung again")
        def new_handoff(lead: str) -> Phone:
            c = Phone(client, lead)
            c.dial()
            c.answer()
            c.say("Can you put me through to a person?")
            c.hook("handoff", query="?spoken=1")
            return c

        def agent_sid() -> str:
            return f"CAagent{_counter['agent']}"

        busy = new_handoff("E-1006")
        first_sid = agent_sid()
        check("customer waits in a hold window of the configured length", 'timeLimit="40"' in busy.hook("handoff", query="?spoken=1").text)
        mark = len(rest_calls)
        r = hook(client, f"/twilio/agent-status/{busy.id}", {"CallSid": first_sid, "CallStatus": "busy"})
        check("agent-status callback accepted", r.status_code == 204)
        moved = [d for u, d in rest_calls[mark:] if busy.sid in u]
        check("busy line -> the customer's call is pulled out to be told", len(moved) == 1 and moved[0]["Url"] == f"{PUBLIC}/twilio/handoff/{busy.id}?spoken=1&notice=busy", str(moved))
        check("HANDOFF_AGENT_UNAVAILABLE audited", "HANDOFF_AGENT_UNAVAILABLE" in busy.audit())
        n = len(created_calls())
        xml = busy.hook("handoff", query="?spoken=1&notice=busy").text
        check("customer hears the specialist is busy and to stay on the line", "busy with another call" in xml and "stay on the line" in xml and "connect you as soon as" in xml, xml[:260])
        check("...and stays in the conference (not hung up)", "<Conference" in xml and "<Hangup/>" not in xml)
        check("the agent is not hammered while still busy", len(created_calls()) == n)
        xml = busy.hook("handoff-ended").text
        check("when the hold window ends the agent is rung again", len(created_calls()) == n + 1 and created_calls()[-1]["To"] == AGENT_PHONE)
        check("customer keeps holding (not told to call back)", "<Conference" in xml and "couldn't reach" not in xml, xml[:200])
        check("the busy line is said once, then a gentler update", "busy with another call" not in xml)
        mark = len(rest_calls)
        hook(client, f"/twilio/agent-status/{busy.id}", {"CallSid": first_sid, "CallStatus": "busy"})
        check("a late callback from the first ring is ignored", not [1 for u, _ in rest_calls[mark:] if busy.sid in u])
        second_sid = agent_sid()
        hook(client, f"/twilio/agent-accept/{busy.id}", {"CallSid": second_sid, "Digits": "1"})
        hook(client, f"/twilio/conference/{busy.id}", {"CallSid": second_sid, "StatusCallbackEvent": "participant-join"})
        check("second ring: agent is free and joins the same call", "HANDOFF_AGENT_JOINED" in busy.audit())
        mark = len(rest_calls)
        hook(client, f"/twilio/agent-status/{busy.id}", {"CallSid": second_sid, "CallStatus": "completed"})
        check("the agent hanging up afterwards does not disturb anyone", not [1 for u, _ in rest_calls[mark:] if busy.sid in u])

        print("\n7b. Agent never picks up: rung up to the limit, then a callback promise")
        giveup = new_handoff("E-1006")
        for ring in (1, 2, 3):
            check(f"ring {ring} placed", giveup.audit().count("HANDOFF_AGENT_DIALLED") == ring)
            sid = agent_sid()
            mark = len(rest_calls)
            hook(client, f"/twilio/agent-status/{giveup.id}", {"CallSid": sid, "CallStatus": "no-answer"})
            target = [d["Url"] for u, d in rest_calls[mark:] if giveup.sid in u]
            if ring < 3:
                check(f"ring {ring} unanswered -> customer told to hold", target and target[0].endswith("notice=busy"), str(target))
                giveup.hook("handoff", query="?spoken=1&notice=busy")
                giveup.hook("handoff-ended")
            else:
                check("last ring unanswered -> straight to the outcome", target and target[0].endswith(f"/twilio/handoff-ended/{giveup.id}"), str(target))
        n = len(created_calls())
        xml = giveup.hook("handoff-ended").text
        check("after the last ring the customer gets the apology and a hangup", "couldn't reach a colleague" in xml and "<Hangup/>" in xml and "<Conference" not in xml, xml[:200])
        check("no fourth ring", len(created_calls()) == n and giveup.audit().count("HANDOFF_AGENT_DIALLED") == 3)
        check("HANDOFF_AGENT_NEVER_JOINED audited", "HANDOFF_AGENT_NEVER_JOINED" in giveup.audit())

        print("\n7c. Agent still ringing at the end of a window: keep holding, within reason")
        ringing = new_handoff("E-1006")
        n = len(created_calls())
        xml = ringing.hook("handoff-ended").text
        check("still ringing -> another hold window, no second ring", "<Conference" in xml and len(created_calls()) == n)
        for _ in range(8):
            xml = ringing.hook("handoff-ended").text
        check("but not forever", "couldn't reach a colleague" in xml and "<Hangup/>" in xml)

        print("\n7d. Agent declines or hangs up during the briefing: same as busy")
        decl = new_handoff("E-1006")
        sid = agent_sid()
        r = hook(client, f"/twilio/agent-accept/{decl.id}", {"CallSid": sid, "Digits": "2"})
        check("agent pressing something other than 1 is not connected", "<Conference" not in r.text)
        mark = len(rest_calls)
        hook(client, f"/twilio/agent-status/{decl.id}", {"CallSid": sid, "CallStatus": "completed"})
        check("customer is told to hold, not dropped", any(d["Url"].endswith("notice=busy") for u, d in rest_calls[mark:] if decl.sid in u))

        print("\n7e. Twilio refuses the agent's number (unverified trial number): a clear failure, not a retry loop")
        refuse_agent["on"] = True
        n = len(created_calls())
        bad = Phone(client, "E-1006")
        bad.dial()
        bad.answer()
        bad.say("Can you put me through to a person?")
        xml = bad.hook("handoff", query="?spoken=1").text
        refuse_agent["on"] = False
        check("customer is told a colleague could not be reached", "couldn't reach a colleague" in xml and "<Hangup/>" in xml and "<Conference" not in xml, xml[:200])
        failed = next(e["event_detail"] for e in bad.session()["audit_events"] if e["event_type"] == "HANDOFF_AGENT_FAILED")
        check("the audit trail says why, in Twilio's words", "unverified" in failed and "Trial accounts" in failed, failed)
        check("it was not treated as busy", "HANDOFF_AGENT_UNAVAILABLE" not in bad.audit() and "HANDOFF_HOLD_NOTICE" not in bad.audit())

        print("\n7f. The human agent can be rung again from the console")
        lonely = new_handoff("E-1006")
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
        drop.say("yes")  # the read-back: an unconfirmed date is not kept
        hook(client, f"/twilio/status/{drop.id}", {"CallSid": drop.sid, "CallStatus": "completed"})
        s = drop.session()
        check("session INCOMPLETE, outcome PHONE_HANG_UP", s["status"] == "INCOMPLETE" and s["outcome_detail"] == "PHONE_HANG_UP", f"{s['status']}/{s['outcome_detail']}")
        item = next(i for i in client.get("/leads").json() if i["lead"]["id"] == "E-1005")
        check("lead is back in the recovery queue", item["lead"]["status"] == "DROPPED_OFF" and item["latest_session_status"] == "INCOMPLETE")
        check("queue resumes after what was captured", item["resume_step"] == "energy_requirement", item["resume_step"])

        print("\n11b. Hanging up on the closing question does not undo a submitted journey")
        done = Phone(client, "E-1001")
        done.dial()
        done.answer()
        for said in ["Yes, go ahead.", "12 Test Street, Sydney NSW 2000", "yes", "1 December 2030", "yes", "gas", "yes",
                     "No", "No", "email", "yes that's all correct"]:
            done.say(said)
        check("submitted and waiting on the closing answer", done.session()["state"] == "CLOSING" and done.session()["submission"] is not None)
        hook(client, f"/twilio/status/{done.id}", {"CallSid": done.sid, "CallStatus": "completed"})
        s2 = done.session()
        check("customer hanging up leaves it COMPLETED (not INCOMPLETE)", s2["status"] == "COMPLETED" and s2["state"] == "COMPLETED", f"{s2['status']}/{s2['state']}")
        check("submission is intact", s2["submission"] is not None)

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

        stop = Phone(client, "E-1005")
        stop.dial()
        stop.answer()
        xml = stop.say("Please stop calling me.").text
        check("'stop calling me' on the phone: acknowledged, then a hangup", "Understood. I will record that request" in xml and "<Hangup/>" in xml, xml[:160])
        flagged = next(i for i in client.get("/leads").json() if i["lead"]["id"] == "E-1005")
        check("the lead is flagged Do-Not-Call", flagged["lead"]["dnc_status"] is True and flagged["can_start"] is False)
        n = len(created_calls())
        again = Phone(client, "E-1005", force=True)
        r = again.dial()
        check("no further call is ever placed, even forced", r.status_code == 409 and len(created_calls()) == n, r.text[:140])
        check("...and the refusal names the customer's own request", client.get(f"/calls/{again.id}").json()["outcome_detail"] == "DNC_CUSTOMER_REQUEST")

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
