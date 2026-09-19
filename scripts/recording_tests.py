"""Uploaded-recording tests.

Runs the FastAPI app in-process against a throwaway SQLite file, with the speech-to-text
vendor stubbed, so it needs no backend running, no API keys and spends no credits:

    python scripts/recording_tests.py

What it proves: a recording that covers the checklist completes the journey, one that does
not lands in the recovery queue (and the next recovery call resumes where it left off), a
second recording can supply what the first missed, and a refusal or a safety signal in the
audio is routed to DECLINED / HANDOFF instead of being submitted.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

_tmp = tempfile.mkdtemp(prefix="recording_tests_")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmp, 'test.db').as_posix()}"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import llm_service, recording_service  # noqa: E402
from app.services.llm_service import RulesOnlyLLM  # noqa: E402
from app.services.voice_service import voice_providers  # noqa: E402

# Rules engine only: deterministic, and never calls a real LLM.
llm_service._llm_singleton = RulesOnlyLLM()
recording_service.get_llm = lambda: llm_service._llm_singleton  # type: ignore[assignment]

PASS, FAIL = "PASS", "FAIL"
failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    failures += 0 if condition else 1
    print(f"  [{PASS if condition else FAIL}] {name}" + (f" — {detail}" if detail else ""))


AGENT, CUST = "0", "1"
# Agent lines are the script's own (docs/energy-agent-script-and-checklist.md), including the
# per-field read-backs ("I have X. Is that correct?"): the customer's "yes" to one of those
# must not be mistaken for an answer to the field it mentions.
OPENING = [
    (AGENT, "Hi Ava, this is the Energy Recovery Assistant from Energy Compare. You started comparing energy plans with us earlier. This call is recorded. Is now an okay time to continue where you left off?"),
    (CUST, "Yes, go ahead."),
]
ADDRESS = [
    (AGENT, "What is the full service address for the property you are connecting, including unit number if there is one, street number, street name, suburb, state, and postcode?"),
    (CUST, "12 Test Street, Sydney NSW 2000"),
    (AGENT, "I have 12 Test Street, Sydney NSW 2000. Is that correct?"),
    (CUST, "Yes."),
]
MOVE_IN = [
    (AGENT, "What date are you moving into the property, or when would you like the energy connection to begin?"),
    (CUST, "1 December 2030."),
    (AGENT, "I have 1 December 2030. Is that correct?"),
    (CUST, "Yes."),
]
ENERGY = [
    (AGENT, "Will you need electricity, gas, or both at this property?"),
    (CUST, "Electricity and gas please."),
    (AGENT, "I have Electricity and gas. Is that right?"),
    (CUST, "Yes."),
]
CONCESSION = [
    (AGENT, "Do you have an eligible concession card you would like noted for the account? You can answer yes, no, or not sure."),
    (CUST, "No."),
]
LIFE_SUPPORT_NO = [
    (AGENT, "Does anyone at the property use life-support equipment that needs a continuous electricity supply?"),
    (CUST, "No."),
]
CONTACT = [
    (AGENT, "For the next update, would you prefer we contact you by phone or email?"),
    (CUST, "Email is best."),
]


def recording(*parts: list[tuple[str, str]]) -> dict:
    turns = [turn for part in parts for turn in part]
    t = 0.0
    segments = []
    for speaker, text in turns:
        segments.append(
            {"speaker": int(speaker), "start": t, "end": t + 3.0, "text": text, "confidence": 0.97}
        )
        t += 3.5
    return {"text": " ".join(text for _, text in turns), "confidence": 0.97, "segments": segments, "provider": "stub"}


def upload(client: TestClient, stt: dict, **params):
    voice_providers.transcribe = lambda audio, mime="", recording=False: stt  # type: ignore[assignment]
    return client.post(
        "/recordings/upload",
        params={"filename": "call.wav", **params},
        content=b"RIFFfake-audio",
        headers={"Content-Type": "audio/wav"},
    )


def queue_item(client: TestClient, lead_id: str) -> dict:
    return next(i for i in client.get("/leads").json() if i["lead"]["id"] == lead_id)


def main() -> int:
    with TestClient(app) as client:
        full = recording(OPENING, ADDRESS, MOVE_IN, ENERGY, CONCESSION, LIFE_SUPPORT_NO, CONTACT)

        print("\n1. Every checklist item present -> completed journey")
        r = upload(client, full, lead_id="E-1001")
        body = r.json()
        check("200 OK", r.status_code == 200, r.text[:200])
        check("outcome COMPLETED", body["outcome"] == "COMPLETED", body["outcome"])
        check("nothing missing", body["missing_fields"] == [])
        session = body["session"]
        check("journey submitted", session["submission"] is not None)
        payload = (session["submission"] or {}).get("payload", {})
        check(
            "payload is the spoken answers",
            payload.get("energy_requirement") == "BOTH"
            and payload.get("move_in_date") == "2030-12-01"
            and payload.get("contact_preference") == "EMAIL"
            and payload.get("life_support") == "NO",
            str(payload),
        )
        speakers = {seg["speaker"] for seg in session["transcript"]}
        check("agent and customer told apart", speakers == {"AI_AGENT", "CUSTOMER"}, str(speakers))
        check("recording disclosure detected", session["recording_consent_disclosed"] is True)
        check("lead marked COMPLETED", queue_item(client, "E-1001")["lead"]["status"] == "COMPLETED")
        subs = client.get("/journey/submissions").json()
        check("listed under completed journeys", any(s["call_session_id"] == session["id"] for s in subs))

        print("\n2. Missing items -> recovery queue")
        partial = recording(OPENING, ADDRESS, MOVE_IN, ENERGY)
        r = upload(client, partial, lead_id="E-1004")
        body = r.json()
        check("outcome INCOMPLETE", body["outcome"] == "INCOMPLETE", body["outcome"])
        check(
            "missing = concession, life support, contact",
            body["missing_fields"] == ["concession_status", "life_support", "contact_preference"],
            str(body["missing_fields"]),
        )
        check("no submission", body["session"]["submission"] is None)
        item = queue_item(client, "E-1004")
        check("lead stays DROPPED_OFF", item["lead"]["status"] == "DROPPED_OFF", item["lead"]["status"])
        check("queue resumes at first missing step", item["resume_step"] == "concession_status", item["resume_step"])
        check("queue shows INCOMPLETE", item["latest_session_status"] == "INCOMPLETE")
        summary = client.get("/dashboard/summary").json()
        check("dashboard counts it as dropped off", summary["counts"]["dropped_off"] >= 1)

        print("\n3. The recovery call resumes there and keeps what the recording found")
        started = client.post("/calls/start/E-1004", json={}).json()
        call = started["session"]
        check("call resumes at concession_status", call["resume_step"] == "concession_status", call["resume_step"])
        check(
            "captured fields carried into the call",
            call["known_fields"].get("property_address") == "12 Test Street, Sydney NSW 2000"
            and call["known_fields"].get("energy_requirement") == "BOTH",
            str(call["known_fields"]),
        )
        check("carry is audited", any(e["event_type"] == "RECORDING_FIELDS_CARRIED" for e in call["audit_events"]))

        print("\n4. A second recording supplies what the first missed")
        # A new recording for a lead whose latest session is an incomplete recording.
        upload(client, recording(OPENING, ADDRESS, MOVE_IN), lead_id="E-1006")
        r = upload(client, recording(CONCESSION, LIFE_SUPPORT_NO, CONTACT, ENERGY), lead_id="E-1006")
        body = r.json()
        check("address and move-in date carried over from the first recording", body["missing_fields"] == [], str(body["missing_fields"]))
        check("second upload COMPLETES the journey", body["outcome"] == "COMPLETED", body["outcome"])

        print("\n5. Customer declines -> declined, never submitted")
        declined = recording(OPENING[:1], [(CUST, "Not interested, please stop calling me.")], ADDRESS)
        r = upload(client, declined, lead_id="E-1007")
        body = r.json()
        check("outcome DECLINED", body["outcome"] == "DECLINED", body["outcome"])
        check("no submission", body["session"]["submission"] is None)
        check("lead DECLINED", queue_item(client, "E-1007")["lead"]["status"] == "DECLINED")
        check("'stop calling me' in a recording flags the lead Do-Not-Call too", queue_item(client, "E-1007")["lead"]["dnc_status"] is True and body["session"]["outcome_detail"] == "DO_NOT_CALL_REQUESTED", str(body["session"]["outcome_detail"]))
        check("...so a recovery call for that lead is refused", client.post("/calls/start/E-1007", json={}).json()["blocked"] is True)

        print("\n6. Life support declared -> human handoff, never submitted")
        life_yes = [
            (AGENT, "Does anyone at the property use life-support equipment that needs a continuous electricity supply?"),
            (CUST, "Yes, my husband uses an oxygen concentrator."),
        ]
        r = upload(client, recording(OPENING, ADDRESS, MOVE_IN, ENERGY, CONCESSION, life_yes, CONTACT), lead_id="E-1002")
        body = r.json()
        check("outcome HANDOFF_REQUESTED", body["outcome"] == "HANDOFF_REQUESTED", body["outcome"])
        check("no submission", body["session"]["submission"] is None)
        check("handoff record created", body["session"]["handoff"] is not None)
        check("handoff reason SENSITIVE_TOPIC", body["session"]["handoff_reason"] == "SENSITIVE_TOPIC")

        print("\n7. Card data is redacted, not stored")
        card = [(CUST, "My card is 4111 1111 1111 1111.")]
        r = upload(client, recording(OPENING, card), lead_id="E-1005")
        text = " ".join(seg["text"] for seg in r.json()["session"]["transcript"])
        check("card number absent from transcript", "4111 1111 1111 1111" not in text)
        check("routed to a human", r.json()["outcome"] == "HANDOFF_REQUESTED")

        print("\n8. No speaker labels (single block of text), no LLM -> nothing to go on, queued")
        blob = {"text": "Twelve Test Street Sydney, both, phone.", "confidence": 0.9, "provider": "stub"}
        r = upload(client, blob, lead_id="E-1004")
        check("outcome INCOMPLETE, not a guess", r.json()["outcome"] == "INCOMPLETE", r.json().get("outcome", r.text))

        print("\n9. New lead created from the upload")
        before = len(client.get("/leads").json())
        r = upload(
            client, recording(OPENING, ADDRESS), new_first_name="Nia", new_phone="+61491570999",
            new_email="nia@example.com",
        )
        leads = client.get("/leads").json()
        check("lead added to the queue", len(leads) == before + 1, f"{before} -> {len(leads)}")
        check("new lead is in the recovery queue", r.json()["outcome"] == "INCOMPLETE")
        check("new lead id follows the sequence", r.json()["session"]["lead_id"] == "E-1008", r.json()["session"]["lead_id"])

        print("\n10. Request validation")
        r = upload(client, full)
        check("no lead given -> 422", r.status_code == 422, str(r.status_code))
        r = upload(client, full, lead_id="E-9999")
        check("unknown lead -> 404", r.status_code == 404, str(r.status_code))
        r = upload(client, {"text": "", "segments": []}, lead_id="E-1001")
        check("silent audio -> 422", r.status_code == 422, str(r.status_code))
        n = len(client.get("/leads").json())
        upload(client, {"text": "", "segments": []}, new_first_name="Zed", new_phone="1", new_email="z@example.com")
        check("failed transcription leaves no new lead behind", len(client.get("/leads").json()) == n)
        r = client.post("/recordings/upload", params={"lead_id": "E-1001"}, content=b"", headers={"Content-Type": "audio/wav"})
        check("empty body -> 400", r.status_code == 400, str(r.status_code))
        saved = dict(voice_providers.server_stt)
        voice_providers.server_stt.clear()
        r = upload(client, full, lead_id="E-1001")
        voice_providers.server_stt.update(saved)
        check("no STT provider configured -> 503", r.status_code == 503, str(r.status_code))

        print("\n11. Real-STT quirks: split year, and a one-word reply tagged as the agent")
        quirky = recording(
            OPENING, ADDRESS,
            [(AGENT, "What date are you moving into the property, or when would you like the energy connection to begin?"), (CUST, "The 1st of December 20 30."), (AGENT, "I have 1 December 2030. Is that correct?"), (CUST, "Yes.")],
            ENERGY, CONCESSION,
            [
                (AGENT, "Does anyone at the property use life-support equipment that needs a continuous electricity supply?"),
                (AGENT, "No."),  # diarization slip: the customer's reply, labelled as the agent
            ],
            CONTACT,
        )
        r = upload(client, quirky, lead_id="E-1003")
        body = r.json()
        payload = (body["session"]["submission"] or {}).get("payload", {})
        check("split year rejoined: 20 30 -> 2030", payload.get("move_in_date") == "2030-12-01", str(payload.get("move_in_date")))
        check("agent-tagged 'No.' read as the customer's answer", payload.get("life_support") == "NO", str(payload.get("life_support")))
        check("that recording completes", body["outcome"] == "COMPLETED", body["outcome"])

        print("\n12. An agent asking twice is not read as answering itself")
        two_questions = recording(
            OPENING,
            [(AGENT, "What date are you moving into the property, or when would you like the energy connection to begin?"), (AGENT, "Is that soon?")],
        )
        r = upload(client, two_questions, lead_id="E-1007")
        check("no field invented from agent-only speech", "move_in_date" in r.json()["missing_fields"])

    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
