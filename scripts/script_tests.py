"""Energy dropout-recovery script tests.

The script in docs/energy-agent-script-and-checklist.md is the single source of truth for
what the agent says and collects. These tests hold the code to it:

  * every quoted line of that document is in backend/app/data/energy_scripts.json, word for
    word, and nothing the agent says at any point comes from anywhere else;
  * the seven scenarios the checklist cares about behave as it says.

Runs the FastAPI app in-process against a throwaway SQLite file, rules engine only, so it
needs no backend running, no keys and no network:

    python scripts/script_tests.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

_tmp = tempfile.mkdtemp(prefix="script_tests_")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmp, 'test.db').as_posix()}"

from fastapi.testclient import TestClient  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import AuditEvent, CallSession, JourneyField, Speaker, TranscriptSegment  # noqa: E402
from app.services import llm_service  # noqa: E402
from app.services.llm_service import RulesOnlyLLM  # noqa: E402
from app.services.script_service import script_service  # noqa: E402

llm_service._llm_singleton = RulesOnlyLLM()  # deterministic: no LLM, no network

PASS, FAIL = "PASS", "FAIL"
failures = 0
DOC = BACKEND.parent / "docs" / "energy-agent-script-and-checklist.md"


def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    failures += 0 if condition else 1
    print(f"  [{PASS if condition else FAIL}] {name}" + (f" — {detail}" if detail else ""))


class Call:
    """One simulated customer on the console channel."""

    def __init__(self, client: TestClient, lead_id: str) -> None:
        self.client = client
        started = client.post(f"/calls/start/{lead_id}", json={}).json()
        self.session = started["session"]
        self.said: list[str] = list(started.get("agent_messages") or [])  # last turn's agent lines
        self.turns: list[list[str]] = [list(self.said)]
        self.last: dict = {}

    def say(self, text: str) -> dict:
        turn = self.client.post(
            f"/calls/{self.id}/utterance", json={"text": text, "source": "SIMULATED"}
        ).json()
        self.session = turn["session"]
        self.said = list(turn.get("agent_messages") or [])
        self.turns.append(list(self.said))
        self.last = turn
        return turn

    def run(self, *replies: str) -> "Call":
        for reply in replies:
            self.say(reply)
        return self

    @property
    def id(self) -> str:
        return self.session["id"]

    @property
    def status(self) -> str:
        return self.session["status"]

    @property
    def state(self) -> str:
        return self.session["state"]

    @property
    def text(self) -> str:
        return " ".join(self.said)

    @property
    def handoff(self) -> dict:
        return self.session.get("handoff") or {}

    def field(self, name: str) -> dict:
        return next(f for f in self.session["journey_fields"] if f["field_name"] == name)

    def events(self) -> list[str]:
        return [e["event_type"] for e in self.session["audit_events"]]


ADDRESS = "12 Test Street, Sydney NSW 2000"
YES = "Yes, that's correct."
# A whole clean journey up to the final read-back (address, date and energy are each confirmed).
CLEAN = ["Yes, go ahead.", ADDRESS, YES, "1 October 2030", YES, "Both electricity and gas.", YES, "No.", "No.", "Email please."]


def main() -> int:
    with TestClient(app) as client:
        templates = script_service.spoken_templates()

        # ------------------------------------------------------------------ #
        print("\n0. The document is the source of truth")
        doc_lines = [line[2:].strip() for line in DOC.read_text(encoding="utf-8").splitlines() if line.startswith("> ")]
        missing = [line for line in doc_lines if line not in templates]
        check(f"all {len(doc_lines)} quoted lines of the document are in the script file, word for word", not missing, str(missing[:2]))
        script_json = json.dumps(script_service.raw)
        for banned in ("Have I got all of that right", "I'll arrange one callback", "tomorrow between", "Great. What is the service address"):
            check(f"obsolete line removed: '{banned}'", banned not in script_json)
        check("no broadband content in the script", not re.search(r"broadband|\bnbn\b|internet|mobile plan", script_json, re.I))
        check("nothing the agent can say asks for payment details", not re.search(r"card number|cvv|expiry|bank account|bsb|payment", " ".join(templates), re.I))
        check("the milestone step is never spoken", "prompt" not in script_service.step("customer_continue"))
        check("the seven script fields are the only ones collected", script_service.required_fields == [
            "property_address", "move_in_date", "energy_requirement", "concession_status", "life_support", "contact_preference"])

        sessions: list[str] = []

        def call(lead: str) -> Call:
            c = Call(client, lead)
            sessions.append(c.id)
            return c

        # ------------------------------------------------------------------ #
        print("\n1. Successful complete recovery")
        c = call("E-1001")
        check("opens with the recording disclosure and the permission question", "This call is recorded." in c.text and c.text.endswith("continue where you left off?"), c.text[:120])
        check("before any data collection", not c.session["collected_fields"])
        c.say("Yes, go ahead.")
        check("permission -> straight to the first question, no lead-in", len(c.said) == 1 and c.said[0].startswith("What is the full service address") and c.said[0].count("?") == 1, str(c.said)[:200])
        check("the two removed lead-in lines are gone from the whole call", not any(x in " ".join(t.get("text", "") for t in c.session["transcript"]) for x in ("Great, thank you", "I can see you had reached", "I cannot give advice")))
        c.say(ADDRESS)
        check("address is read back before it counts", c.state == "CONFIRMING_FIELD" and c.text == f"I have {ADDRESS}. Is that correct?", c.text)
        check("...and is not collected yet", "property_address" not in c.session["collected_fields"])
        c.say(YES)
        check("confirmed -> collected, then the date is asked", c.session["collected_fields"].get("property_address") == ADDRESS and "What date are you moving into the property" in c.text)
        c.say("1 October 2030")
        check("date is read back in words", c.text == "I have 1 October 2030. Is that correct?", c.text)
        c.say(YES)
        c.say("Both electricity and gas.")
        check("energy is read back", c.text == "I have Electricity and gas. Is that right?", c.text)
        c.say(YES)
        check("concession is asked exactly as scripted", c.text.startswith("Do you have an eligible concession card you would like noted for the account?") and "yes, no, or not sure" in c.text)
        c.say("No.")
        check("then life support", c.text.startswith("Does anyone at the property use life-support equipment that needs a continuous electricity supply?"))
        c.say("No.")
        check("then contact preference", c.text == "For the next update, would you prefer we contact you by phone or email?", c.text)
        c.say("Email please.")
        check("final read-back lists all six details", c.state == "CONFIRMING_DETAILS" and all(x in c.text for x in [ADDRESS, "1 October 2030", "Electricity and gas", "concession status is No", "life-support status is No", "contact preference is Email"]), c.text[:160])
        check("...and asks once for the whole thing", c.text.endswith("Is all of that correct?"))
        c.say("Yes, that's all correct.")
        check("submitted: the scripted closing question", c.text.startswith("Thank you. I have submitted these details") and c.text.endswith("Is there anything else you need from a team member?"))
        check("journey COMPLETED with a receipt, call still open for the answer", c.status == "COMPLETED" and c.state == "CLOSING" and c.session["submission"] is not None, f"{c.status}/{c.state}")
        payload = c.session["submission"]["payload"]
        check("payload is exactly the confirmed answers", payload == {
            "lead_id": "E-1001", "vertical": "ENERGY", "property_address": ADDRESS, "move_in_date": "2030-10-01",
            "energy_requirement": "BOTH", "concession_status": "NO", "life_support": "NO", "contact_preference": "EMAIL"}, str(payload))
        c.say("No, that's everything, thanks.")
        check("'nothing else' -> scripted goodbye, call ended", c.text == "Thank you for your time. Goodbye." and c.state == "COMPLETED" and c.last["terminal"] is True)
        check("one question at a time: no agent turn asks two", all(" ".join(t).count("?") <= 1 for t in c.turns), str([t for t in c.turns if " ".join(t).count("?") > 1]))
        check("nothing said after the call ended is acted on", "ignored" in " ".join(c.say("hello?").get("system_notes", [])))

        print("\n1b. Only missing fields are asked (address already on file)")
        c = call("E-1002")
        c.say("Yeah, fine.")
        check("starts at the first missing field, the date", "What date are you moving into the property" in c.text and "What is the full service address" not in c.text, c.text[-140:])
        c.run("1 October 2030", YES, "Electricity only.", YES, "No.", "No.", "Phone is fine.")
        check("the address on file is not asked, but is read back at the end", "service address is 12 Test Street" in c.text)
        c.say("yes")
        check("and the journey submits with it", c.status == "COMPLETED" and c.session["submission"]["payload"]["property_address"] == "12 Test Street, Sydney NSW 2000")

        # ------------------------------------------------------------------ #
        print("\n2. Decline / stop-calling")
        c = call("E-1004")
        c.say("Stop calling me. I'm not interested.")
        check("acknowledged once with the scripted line", c.text == "Understood. I will record that request. Thank you for your time. Goodbye.", c.text)
        check("DECLINED, logged as a do-not-call request", c.status == "DECLINED" and c.session["outcome_detail"] == "DO_NOT_CALL_REQUESTED", str(c.session["outcome_detail"]))
        check("request audited", "DNC_REQUEST_LOGGED" in c.events() and "CALL_DECLINED" in c.events())
        n = len(c.session["transcript"])
        c.say("Actually, wait.")
        check("no further ask: the call is over", len(c.session["transcript"]) == n and c.status == "DECLINED")

        print("\n2b. A stop-calling request blocks every future call")
        item = next(i for i in client.get("/leads").json() if i["lead"]["id"] == "E-1004")
        check("the lead is flagged Do-Not-Call and cannot be started from the queue", item["lead"]["dnc_status"] is True and item["can_start"] is False)
        check("the flag is audited", "DNC_REGISTER_UPDATED" in c.events())
        again = client.post("/calls/start/E-1004", json={}).json()
        check("a new recovery call is blocked by the DNC gate", again["blocked"] is True and again["blocked_reason"] == "DNC_CUSTOMER_REQUEST", str(again.get("blocked_reason")))
        check("...before anything is spoken or dialled", not again["agent_messages"] and "DIAL_ATTEMPT" not in [e["event_type"] for e in again["session"]["audit_events"]])
        forced = client.post("/calls/start/E-1004", json={"force": True}).json()
        check("the demo override cannot bypass a customer's own request", forced["blocked"] is True and forced["blocked_reason"] == "DNC_CUSTOMER_REQUEST")
        register = client.post("/calls/start/E-1003", json={"force": True}).json()
        check("(a lead merely on the register can still be forced, for demos)", register["blocked"] is False)
        sessions_for_lead = client.get("/leads/E-1004/sessions").json()
        check("the blocked attempts are recorded, not silent", sum(1 for s in sessions_for_lead if s["status"] == "DNC_BLOCKED") == 2)

        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, YES)
        c.say("Not interested, thanks.")
        check("mid-journey refusal: same scripted line, no re-ask", c.text.startswith("Understood.") and "date" not in c.text.lower() and c.status == "DECLINED")
        check("'not interested' is a decline, not a do-not-call request", c.session["outcome_detail"] == "CUSTOMER_DECLINED")
        item = next(i for i in client.get("/leads").json() if i["lead"]["id"] == "E-1001")
        check("...so the lead is NOT flagged and can be called again", item["lead"]["dnc_status"] is False and item["can_start"] is True)
        for phrase in ("I already have a contact at another provider", "I'm not interested in the list of plans"):
            c2 = call("E-1005")
            c2.run("Yes, go ahead.", phrase)
            check(f"'{phrase}' is never read as a do-not-call request", c2.session.get("outcome_detail") != "DO_NOT_CALL_REQUESTED" and not next(i for i in client.get("/leads").json() if i["lead"]["id"] == "E-1005")["lead"]["dnc_status"])

        # ------------------------------------------------------------------ #
        print("\n3. Customer asks for a human")
        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, YES)
        c.say("Can I talk to a real person please?")
        check("scripted line", c.text == "Of course. I will connect you with a team member and pass along what we have so you do not need to repeat yourself.", c.text)
        check("HANDOFF_REQUESTED / CUSTOMER_REQUEST", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "CUSTOMER_REQUEST")
        h = c.handoff
        check("handoff carries the current step", h.get("current_step") == "move_in_date", str(h.get("current_step")))
        check("...every collected field", h["collected_fields"]["property_address"] == ADDRESS)
        check("...the last thing the customer said", h["last_customer_message"] == "Can I talk to a real person please?")
        check("...the escalation reason and flags", h["reason"] == "CUSTOMER_REQUEST" and "CUSTOMER_REQUEST" in h["safety_flags"])
        check("...what is still outstanding", h["outstanding_fields"] == ["move_in_date", "energy_requirement", "concession_status", "life_support", "contact_preference"], str(h["outstanding_fields"]))
        check("...how sure we are of each captured value, and its source", h["field_details"]["property_address"]["source"] == "CUSTOMER_SPOKEN" and h["field_details"]["property_address"]["confidence"] >= 0.8, str(h["field_details"].get("property_address")))
        check("...the lead id and contact details", h["lead_id"] == "E-1001" and h["known_contact"]["phone"])
        check("...and that the recording was disclosed", h["recording_disclosed"] is True and "recording" in h["context_summary"].lower())

        c = call("E-1001")
        c.say("Can you put me through to a person?")
        check("also at the very first reply", c.status == "HANDOFF_REQUESTED" and c.text.startswith("Of course."))

        print("\n3b. A value heard but not yet confirmed still reaches the human")
        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS)
        c.say("This is ridiculous, I already told you this.")
        check("frustration -> handoff", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "FRUSTRATION")
        check("the unconfirmed address is in the summary", ADDRESS in c.handoff["context_summary"] and "not yet confirmed" in c.handoff["context_summary"], c.handoff["context_summary"][-150:])
        check("...and shown with its status in the field details", c.handoff["field_details"]["property_address"]["status"] == "PENDING")
        check("default handoff line (no 'Of course')", c.text == "I will connect you with a team member now and pass along what we have.", c.text)

        # ------------------------------------------------------------------ #
        print("\n4. Life-support handoff")
        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, YES, "1 October 2030", YES, "Electricity.", YES, "No.")
        c.say("Yes, my father uses a ventilator.")
        check("scripted life-support line, no medical questions", c.text == "Thank you for letting me know. I will connect you with a specialist team member so this is handled appropriately. You will not need to repeat the details you have already given." and "?" not in c.text, c.text)
        check("SENSITIVE_TOPIC handoff", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "SENSITIVE_TOPIC")
        check("life support recorded as a bare yes", c.session["collected_fields"].get("life_support") == "YES")
        check("everything collected so far travels with it", c.handoff["collected_fields"]["move_in_date"] == "2030-10-01" and c.handoff["collected_fields"]["energy_requirement"] == "ELECTRICITY")
        check("nothing was submitted", c.session["submission"] is None)

        # ------------------------------------------------------------------ #
        print("\n5. Card / payment data: redacted and handed off")
        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, YES)
        c.say("Can I just pay now? My card is 4111 1111 1111 1111, expiry 05/28, cvv 123.")
        check("SENSITIVE_TOPIC handoff", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "SENSITIVE_TOPIC")
        stored = json.dumps(c.session)
        check("no card digits, expiry or CVV stored anywhere", "4111" not in stored and "05/28" not in stored and "123" not in c.session["last_customer_message"])
        check("redaction is audited, without the number", "CARD_DATA_REDACTED" in c.events() and "4111 1111" not in json.dumps(c.session["audit_events"]))
        check("the agent never repeats or asks for card details", "4111" not in c.text and "card" not in c.text.lower())
        db = SessionLocal()
        raw = " ".join(t.text for t in db.query(TranscriptSegment).filter_by(call_session_id=c.id))
        db.close()
        check("the database transcript is redacted too", "4111" not in raw and "REDACTED" in raw)

        # ------------------------------------------------------------------ #
        print("\n6. Unclear answer -> three follow-ups, each simpler -> handoff on the fourth failure")
        c = call("E-1001")
        c.run("Yes, go ahead.", "Just Queens Street")
        check("follow-up 1: the scripted re-ask of the address", c.text.startswith("Sorry, I missed that.") and c.state == "COLLECTING_FIELD", c.text[:80])
        c.say("Queens Street")
        check("follow-up 2: simpler wording, still ACTIVE", c.text.startswith("Please say it in this format:") and "For example: 12 Test Street, Sydney NSW 2000." in c.text and c.status == "ACTIVE", c.text[:80])
        c.say("Main Road")
        check("follow-up 3: simpler again, still ACTIVE", c.text.startswith("One more try, like this: 12 Test Street") and c.status == "ACTIVE", c.text[:80])
        check("no follow-up wording is repeated", len({t[0] for t in c.turns[-3:]}) == 3)
        c.say("Queens Street")
        check("still unclear after the third follow-up -> handoff (REPEATED_FAILURE)", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "REPEATED_FAILURE")
        check("four attempts in all: the question + three follow-ups", c.field("property_address")["attempts"] == 4, str(c.field("property_address")["attempts"]))
        check("three follow-ups were logged", c.events().count("CLARIFICATION_REQUESTED") == 3, str(c.events().count("CLARIFICATION_REQUESTED")))
        check("the handoff says how many attempts it took", "4 attempts" in json.dumps(c.session["audit_events"]))

        c = call("E-1001")
        c.run("Yes, go ahead.", "Queens Street", "Main Road", "still not sure")
        check("a clear answer on the 3rd follow-up still works: no handoff", c.status == "ACTIVE" and c.state == "COLLECTING_FIELD")
        c.say(ADDRESS)
        check("...it is read back and continues normally", c.state == "CONFIRMING_FIELD" and c.field("property_address")["attempts"] == 4)
        c.say(YES)
        check("...and is collected", c.session["collected_fields"].get("property_address") == ADDRESS)

        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, YES, "sometime soon")
        check("date follow-up 1: the document's own wording", c.text == "To make sure I record it correctly, could you give the day, month, and year?", c.text)
        c.say("whenever")
        check("date follow-up 2", c.text.startswith("Please give a date from today onwards") and "For example: 1 October 2030." in c.text and c.status == "ACTIVE", c.text[:60])
        c.say("any day")
        check("date follow-up 3", c.text == "One more try, like this: 1 October 2030." and c.status == "ACTIVE", c.text[:60])
        c.say("dunno")
        check("date still unclear -> handoff", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "REPEATED_FAILURE")

        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, "hmm maybe")
        check("unclear read-back answer: asked again", c.text == f"I have {ADDRESS}. Is that correct?" and c.state == "CONFIRMING_FIELD", c.text)
        c.run("banana", "purple monkey")
        check("still asked, up to the third follow-up", c.status == "ACTIVE" and c.state == "CONFIRMING_FIELD")
        c.say("asdf")
        check("still unclear after the third -> handoff (LOW_CONFIDENCE)", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "LOW_CONFIDENCE")

        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, "No, that's not right")
        check("a 'no' to the read-back starts follow-up 1", c.state == "COLLECTING_FIELD" and c.text.startswith("Sorry, I missed that."), c.text[:60])
        c.say("14 Sample Road, Parramatta NSW 2150")
        check("the corrected address is read back", c.text == "I have 14 Sample Road, Parramatta NSW 2150. Is that correct?", c.text)
        c.run("No", "14 Sample Road, Parramatta NSW 2150", "No", "14 Sample Road, Parramatta NSW 2150")
        check("rejected three times: still a read-back, not yet a handoff", c.status == "ACTIVE" and c.state == "CONFIRMING_FIELD")
        c.say("No")
        check("rejected a fourth time -> handoff, no loop", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "REPEATED_FAILURE")

        c = call("E-1001")
        c.say("mmm")
        check("opening unclear: follow-up 1 re-asks the question", c.text == "Sorry, I didn't catch that. Is now a good time to continue?", c.text)
        c.say("erm")
        check("opening follow-up 2", c.text.startswith("Please answer yes or no. For example") and c.status == "ACTIVE")
        c.say("zzz")
        check("opening follow-up 3", c.text == "Just yes or no, please." and c.status == "ACTIVE")
        c.say("qqq")
        check("opening still unclear -> handoff", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "LOW_CONFIDENCE")

        print("\n6b. The follow-ups are the same for every question")
        for lead, step_id, junk in (("E-1001", "energy_requirement", "maybe"), ("E-1001", "concession_status", "hmm"),
                                    ("E-1001", "life_support", "erm"), ("E-1001", "contact_preference", "dunno")):
            c = call(lead)
            state = {
                "energy_requirement": ["Yes, go ahead.", ADDRESS, YES, "1 October 2030", YES],
                "concession_status": ["Yes, go ahead.", ADDRESS, YES, "1 October 2030", YES, "Gas.", YES],
                "life_support": ["Yes, go ahead.", ADDRESS, YES, "1 October 2030", YES, "Gas.", YES, "No."],
                "contact_preference": ["Yes, go ahead.", ADDRESS, YES, "1 October 2030", YES, "Gas.", YES, "No.", "No."],
            }[step_id]
            c.run(*state)
            seen = []
            for _ in range(3):
                c.say(junk)
                seen.append(c.text)
            check(f"{step_id}: 3 different follow-ups, still on the same question", len(set(seen)) == 3 and c.status == "ACTIVE" and c.session["current_step"] == step_id, str(seen)[:120])
            c.say(junk)
            check(f"{step_id}: handed over on the fourth failure", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "REPEATED_FAILURE", c.status)

        # ------------------------------------------------------------------ #
        print("\n6c. Follow-ups are short, and the format they teach is one the validator really accepts")
        from app.conversation import safety_engine
        from app.conversation.validators import (
            detect_correction_fields, validate_address, validate_confirmation, validate_contact_preference,
            validate_energy_requirement, validate_move_in_date, validate_yes_no, validate_yes_no_unsure)

        accepts = {
            "recording_consent": lambda t: safety_engine.classify_gate_intent(t) in ("CONTINUE", "BUSY"),
            "property_address": lambda t: validate_address(t).ok,
            "move_in_date": lambda t: validate_move_in_date(t).ok,
            "energy_requirement": lambda t: validate_energy_requirement(t).ok,
            "concession_status": lambda t: validate_yes_no_unsure(t).ok,
            "life_support": lambda t: validate_yes_no(t).ok,
            "contact_preference": lambda t: validate_contact_preference(t).ok,
            "confirmation": lambda t: validate_confirmation(t).reason == "confirmation_declined" and detect_correction_fields(t) == ["move_in_date"],
        }
        for step_id, accepted in accepts.items():
            f1, f2, f3 = (script_service.fallback_prompt(step_id, n) for n in (1, 2, 3))
            words = [len(f.split()) for f in (f1, f2, f3)]
            check(f"{step_id}: follow-ups stay short ({words[0]}/{words[1]}/{words[2]} words)", words[0] <= 17 and words[1] <= 30 and words[2] <= 12, str(words))
            check(f"{step_id}: follow-up 2 gives an example", "for example" in f2.lower(), f2)
            quoted = re.findall(r'"([^"]+)"', f2)
            example = quoted[-1] if quoted else f2.split("For example:")[-1].strip().rstrip(".")
            check(f"{step_id}: the example in follow-up 2 is accepted ({example!r})", accepted(example), example)
            check(f"{step_id}: the three follow-ups are different", len({f1, f2, f3}) == 3)
        for step_id in ("property_address", "move_in_date"):
            f3 = script_service.fallback_prompt(step_id, 3)
            example = f3.split("like this:")[-1].strip().rstrip(".")
            check(f"{step_id}: follow-up 3's example is accepted too ({example!r})", accepts[step_id](example))
        check("follow-up 1 is a brief re-ask, not the whole question again", all(
            len(script_service.fallback_prompt(s["step_id"], 1)) < len(s["prompt"]) for s in script_service.steps() if s.get("fallback_prompts") and s["step_id"] != "move_in_date"))
        check("the other derived lines are short too", all(len(script_service.message(n).split()) <= 20 for n in (
            "busy_callback_question_again", "busy_callback_closing", "busy_later_closing", "handoff_default", "correction_prompt", "goodbye")))

        print("\n7. Final confirmation and submission validation")
        c = call("E-1001")
        c.run(*CLEAN)
        c.say("No, the date is wrong")
        check("a correction re-asks only the named field", c.state == "COLLECTING_FIELD" and c.session["current_step"] == "move_in_date" and c.text.startswith("What date are you moving into the property"), c.text[:60])
        c.say("5 November 2030")
        c.say(YES)
        check("then the whole read-back is repeated, once", c.state == "CONFIRMING_DETAILS" and "connection date is 5 November 2030" in c.text and ADDRESS in c.text)
        check("nothing else changed", c.session["collected_fields"]["energy_requirement"] == "BOTH" and c.session["collected_fields"]["contact_preference"] == "EMAIL")
        c.say("Yes, all correct.")
        check("confirmed -> submitted with the corrected date", c.status == "COMPLETED" and c.session["submission"]["payload"]["move_in_date"] == "2030-11-05")

        c = call("E-1001")
        c.run(*CLEAN)
        c.say("No, it's 5 November 2030")
        check("a new date given in the same breath is applied and read back", c.state == "CONFIRMING_DETAILS" and "5 November 2030" in c.text and "FIELD_CORRECTED" in c.events())

        c = call("E-1001")
        c.run(*CLEAN)
        c.say("No, that's not right.")
        check("no field named -> asks which detail", c.text == "Which detail would you like me to correct?", c.text)
        c.say("The contact preference.")
        check("that one field is asked again", c.session["current_step"] == "contact_preference" and c.text.startswith("For the next update"))
        c.say("Phone please")
        check("...and the read-back repeats", c.state == "CONFIRMING_DETAILS" and "contact preference is Phone" in c.text)
        c.say("No, still wrong")
        check("not confirmed a second time -> a person, never a loop", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "REPEATED_FAILURE" and c.session["submission"] is None)

        c = call("E-1001")
        c.run(*CLEAN)
        c.say("hmm")
        check("unclear final answer: follow-up 1", c.text == "Sorry, is all of that correct?", c.text)
        c.say("dunno")
        check("follow-up 2 is simpler", c.text.startswith('Please say "yes" if everything is right') and "no, the date is wrong" in c.text and c.status == "ACTIVE", c.text[:60])
        c.say("eh")
        check("follow-up 3 is simpler still", c.text == "Just yes or no, please." and c.status == "ACTIVE", c.text[:60])
        c.say("zzz")
        check("still unclear after the third -> a person, nothing submitted", c.status == "HANDOFF_REQUESTED" and c.session["submission"] is None)

        print("\n7b. Nothing is submitted unless every field is valid and confirmed")
        def tamper(lead: str, rows: dict[str, str], extra_state: str = "CONFIRMING_DETAILS") -> Call:
            """A call sitting at the final read-back whose stored fields are exactly `rows`."""
            c = call(lead)
            db = SessionLocal()
            s = db.get(CallSession, c.id)
            s.state, s.current_step = extra_state, "confirmation"
            for name, value in rows.items():
                row = next((r for r in db.query(JourneyField).filter_by(call_session_id=c.id, field_name=name)), None)
                if row is None:
                    row = JourneyField(call_session_id=c.id, field_name=name)
                    db.add(row)
                row.value, row.status, row.source, row.confidence = value, "VALID", "CUSTOMER_SPOKEN", 0.95
            db.commit()
            db.close()
            return c

        good = {"property_address": ADDRESS, "move_in_date": "2030-10-01", "energy_requirement": "GAS",
                "concession_status": "NO", "life_support": "NO", "contact_preference": "EMAIL"}
        c = tamper("E-1005", {k: v for k, v in good.items() if k != "contact_preference"})
        c.say("Yes, that's correct.")
        check("a missing required field blocks submission", c.session["submission"] is None and c.status == "HANDOFF_REQUESTED" and "SUBMISSION_BLOCKED" in c.events())
        check("...and is said plainly in the handoff", "contact_preference" in json.dumps(c.session["audit_events"]))

        c = tamper("E-1005", {**good, "energy_requirement": "COAL"})
        c.say("Yes, that's correct.")
        check("a payload that fails the journey's own validation is not submitted", c.session["submission"] is None and c.status == "HANDOFF_REQUESTED" and "SUBMISSION_BLOCKED" in c.events())

        c = tamper("E-1005", {**good, "move_in_date": "next friday"})
        c.say("Yes, that's correct.")
        check("a non-ISO date is not submitted", c.session["submission"] is None and c.status == "HANDOFF_REQUESTED")

        c = tamper("E-1005", {**good, "life_support": "YES"})
        c.say("Yes, that's correct.")
        check("life support = YES is never submitted", c.session["submission"] is None and c.session["handoff_reason"] == "SENSITIVE_TOPIC")

        c = tamper("E-1005", good)
        c.say("Yes, that's correct.")
        check("a complete, valid, confirmed set IS submitted", c.status == "COMPLETED" and c.session["submission"] is not None)
        c.say("Yes, I have a question actually.")
        check("'anything else?' answered yes -> a person, journey stays submitted", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "CUSTOMER_REQUEST" and c.session["submission"] is not None)
        check("...with the human-request line", c.text.startswith("Of course."))

        c = call("E-1006")
        c.run(*CLEAN, "Yes, that's all correct.")
        c.say("Please don't call me again.")
        check("a do-not-call request after submission is logged, journey stays COMPLETED", c.status == "COMPLETED" and c.session["outcome_detail"] == "DO_NOT_CALL_REQUESTED" and "DNC_REQUEST_LOGGED" in c.events() and c.text.startswith("Understood."))
        item = next(i for i in client.get("/leads").json() if i["lead"]["id"] == "E-1006")
        check("...and the lead is flagged Do-Not-Call so no one calls again", item["lead"]["dnc_status"] is True and item["can_start"] is False)

        # ------------------------------------------------------------------ #
        print("\n7c. Handoff: the human agent fills the remaining details and completes the journey")

        def capture(sid: str, field: str, value: str, agent: str = "Aarav"):
            return client.post(f"/calls/{sid}/field", json={"field_name": field, "value": value, "agent_name": agent})

        def submit(sid: str, confirmed: bool = True, agent: str = "Aarav", life: bool = False):
            return client.post(f"/calls/{sid}/handoff/submit", json={
                "agent_name": agent, "customer_confirmed": confirmed, "life_support_validated": life})

        def live(sid: str) -> dict:
            return client.get(f"/calls/{sid}").json()

        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, YES)
        c.say("Can I talk to a real person please?")
        check("handed off with five details outstanding", c.status == "HANDOFF_REQUESTED" and len(c.handoff["outstanding_fields"]) == 5)

        r = capture(c.id, "move_in_date", "1 October 2030")
        check("the agent can capture a field after the handoff", r.status_code == 200, r.text[:120])
        row = next(f for f in r.json()["session"]["journey_fields"] if f["field_name"] == "move_in_date")
        check("...stored as HUMAN_AGENT, validated (ISO date)", row["source"] == "HUMAN_AGENT" and row["value"] == "2030-10-01" and row["status"] == "VALID", str(row))
        check("...the AI stays silent: nothing is spoken", r.json()["agent_messages"] == [] and r.json()["session"]["status"] == "HANDOFF_REQUESTED")
        check("...and it is audited with the agent's name", any(e["event_type"] == "FIELD_CAPTURED_BY_HUMAN" and "Aarav" in e["event_detail"] for e in r.json()["session"]["audit_events"]))
        bad = capture(c.id, "energy_requirement", "coal")
        check("a human cannot push an invalid value in either (same validator)", bad.status_code == 422, bad.text[:100])
        check("'still outstanding' shrinks live as details are filled", "move_in_date" not in live(c.id)["handoff"]["outstanding_fields"] and len(live(c.id)["handoff"]["outstanding_fields"]) == 4)
        check("the snapshot of what was captured BEFORE the handoff is unchanged", live(c.id)["handoff"]["collected_fields"]["move_in_date"] is None)

        r = submit(c.id, confirmed=False)
        check("submitting needs the read-back confirmation", r.status_code == 422 and "Read the details back" in r.json()["detail"], r.text[:120])
        r = submit(c.id)
        check("submitting with details missing is refused, naming them", r.status_code == 422 and "Supply needed" in r.json()["detail"] and "Contact preference" in r.json()["detail"], r.text[:160])
        check("...and audited", "HUMAN_SUBMISSION_BLOCKED" in [e["event_type"] for e in live(c.id)["audit_events"]])
        check("...nothing was submitted", live(c.id)["submission"] is None and live(c.id)["status"] == "HANDOFF_REQUESTED")

        for field, value in (("energy_requirement", "Electricity"), ("concession_status", "No"), ("life_support", "No"), ("contact_preference", "Email")):
            capture(c.id, field, value)
        check("every detail filled: nothing outstanding", live(c.id)["handoff"]["outstanding_fields"] == [])
        r = submit(c.id)
        body = r.json()
        check("submit -> 200, journey submitted", r.status_code == 200 and body["journey_submitted"] is True, r.text[:160])
        check("call is COMPLETED, no longer a handoff", body["session"]["status"] == "COMPLETED" and body["session"]["state"] == "COMPLETED")
        check("payload = what the AI captured + what the agent filled in", body["session"]["submission"]["payload"] == {
            "lead_id": "E-1001", "vertical": "ENERGY", "property_address": ADDRESS, "move_in_date": "2030-10-01",
            "energy_requirement": "ELECTRICITY", "concession_status": "NO", "life_support": "NO", "contact_preference": "EMAIL"}, str(body["session"]["submission"]["payload"]))
        submitted = next(e for e in body["session"]["audit_events"] if e["event_type"] == "JOURNEY_SUBMITTED")
        check("audited as human-submitted, with who and the confirmation", "origin=HUMAN_AGENT" in submitted["event_detail"] and "by=Aarav" in submitted["event_detail"] and "customer_confirmed=true" in submitted["event_detail"])
        check("the handoff records who took it", body["session"]["handoff"]["accepted_by"] == "Aarav")
        check("lead is COMPLETED", next(i for i in client.get("/leads").json() if i["lead"]["id"] == "E-1001")["lead"]["status"] == "COMPLETED")
        check("it leaves the handoff queue...", c.id not in [h["session_id"] for h in client.get("/handoffs").json()])
        check("...and shows under completed journeys", any(x["call_session_id"] == c.id for x in client.get("/journey/submissions").json()))
        check("a submitted journey cannot be edited again", capture(c.id, "contact_preference", "Phone").status_code == 422)
        check("...or submitted twice", submit(c.id).status_code == 409)

        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, YES, "1 October 2030", YES, "Electricity.", YES, "No.")
        c.say("Yes, my father uses a ventilator.")
        capture(c.id, "contact_preference", "Phone")
        check("life support: the agent can still record the remaining details", live(c.id)["handoff"]["outstanding_fields"] == [])
        check("the AI itself never submitted it (only a person can, and only after validating)", live(c.id)["submission"] is None)
        r = submit(c.id)
        check("life support declared: submitting without validating it is refused", r.status_code == 422 and "Life-support" in r.json()["detail"] and "tick the life-support validation" in r.json()["detail"], r.text[:200])
        check("...even with the read-back confirmed and every detail filled", live(c.id)["submission"] is None and live(c.id)["status"] == "HANDOFF_REQUESTED")
        check("...and the refusal is audited", "HUMAN_SUBMISSION_BLOCKED" in [e["event_type"] for e in live(c.id)["audit_events"]])
        r = submit(c.id, confirmed=False, life=True)
        check("validating life support does not replace the read-back confirmation", r.status_code == 422 and "Read the details back" in r.json()["detail"])
        r = submit(c.id, life=True)
        body = r.json()
        check("once the agent validates it with the customer, the journey submits", r.status_code == 200 and body["journey_submitted"] is True, r.text[:200])
        check("...with life support recorded as YES in the payload", body["session"]["submission"]["payload"]["life_support"] == "YES")
        events = {e["event_type"]: e["event_detail"] for e in body["session"]["audit_events"]}
        check("...the validation is audited with who did it", "LIFE_SUPPORT_VALIDATED_BY_HUMAN" in events and "by=Aarav" in events["LIFE_SUPPORT_VALIDATED_BY_HUMAN"] and "vulnerable-customer" in events["LIFE_SUPPORT_VALIDATED_BY_HUMAN"])
        check("...and the submission records that it was validated", "life_support_validated=true" in events["JOURNEY_SUBMITTED"])
        check("...no medical detail was ever asked for or stored", "ventilator" not in json.dumps(body["session"]["journey_fields"]).lower())
        check("call is COMPLETED and off the handoff queue", body["session"]["status"] == "COMPLETED" and c.id not in [h["session_id"] for h in client.get("/handoffs").json()])

        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, YES, "1 October 2030", YES, "Electricity.", YES, "No.")
        c.say("Yes, my father uses a ventilator.")
        capture(c.id, "contact_preference", "Phone")
        capture(c.id, "life_support", "No")
        check("customer says it was misheard: the agent corrects Life support to No", live(c.id)["collected_fields"]["life_support"] == "NO")
        r = submit(c.id)
        check("...and then no life-support validation is needed", r.status_code == 200 and r.json()["session"]["submission"]["payload"]["life_support"] == "NO", r.text[:160])
        check("...the validation flag is not recorded when life support is NO", "LIFE_SUPPORT_VALIDATED_BY_HUMAN" not in [e["event_type"] for e in r.json()["session"]["audit_events"]])

        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, YES, "1 October 2030", YES, "Electricity.", YES, "No.")
        c.say("Yes, my father uses a ventilator.")
        r = submit(c.id, life=True)
        check("validating life support cannot submit an incomplete journey", r.status_code == 422 and "Contact preference" in r.json()["detail"] and live(c.id)["submission"] is None, r.text[:160])

        c = call("E-1001")
        c.run(*CLEAN, "Yes, that's all correct.")
        check("the validation flag is ignored on a call with no life support (nothing to validate)", submit(c.id, life=True).status_code == 409)

        c = call("E-1001")
        check("an ACTIVE call cannot be completed this way", submit(c.id).status_code == 409)
        c.say("Not interested, thanks.")
        check("a DECLINED call cannot be edited", capture(c.id, "property_address", ADDRESS).status_code == 422)
        check("...or submitted", submit(c.id).status_code == 409)
        check("an unknown call is a 404", submit("CS-NOPE").status_code == 404)

        # ------------------------------------------------------------------ #
        print("\n8. The rest of the script's rules")
        c = call("E-1001")
        c.run("Yes, go ahead.", ADDRESS, YES, "1 October 2030", YES, "Gas.", YES)
        c.say("Do I qualify for a concession?")
        check("eligibility: no advice, a person is offered", c.text == "I cannot advise on eligibility, but a team member can help. Would you like me to connect you now?" and c.state == "AWAITING_ELIGIBILITY_CHOICE", c.text)
        c.say("No, I'll just answer.")
        check("declined -> the concession question again", c.text.startswith("Do you have an eligible concession card") and c.state == "COLLECTING_FIELD")
        c.say("Am I eligible for one?")
        c.say("Yes please")
        check("accepted -> a person, with what was collected", c.status == "HANDOFF_REQUESTED" and c.handoff["collected_fields"]["energy_requirement"] == "GAS")

        c = call("E-1001")
        c.run("Yes, go ahead.", "Which plan is the cheapest for me?")
        check("advice request -> handoff, no advice", c.status == "HANDOFF_REQUESTED" and c.session["handoff_reason"] == "OFF_SCRIPT" and "$" not in c.text and "cheap" not in c.text.lower())

        c = call("E-1001")
        c.run("Yes, go ahead.", "Do you also sell NBN internet?")
        check("energy only: a broadband question is not answered", c.status == "HANDOFF_REQUESTED" and "nbn" not in c.text.lower() and "internet" not in c.text.lower())

        c = call("E-1001")
        c.say("I'm at work right now")
        check("busy -> the scripted callback question, once", c.text == "No problem. Would you prefer a callback, or should I leave the journey for you to continue later?")
        c.say("Call me tomorrow.")
        check("logged and ended, no persuasion", c.status == "DECLINED" and c.session["outcome_detail"] == "CALLBACK_REQUESTED" and "?" not in c.text)

        blocked = client.post("/calls/start/E-1003", json={}).json()
        check("Do-Not-Call lead: blocked before any dial or prompt", blocked["blocked"] is True and not blocked["agent_messages"])

        # ------------------------------------------------------------------ #
        print("\n9. Nothing is ever spoken that is not in the script file")
        db = SessionLocal()
        rows = db.query(TranscriptSegment).filter(TranscriptSegment.speaker == Speaker.AI_AGENT.value).all()
        patterns = [re.compile("^" + re.sub(r"\\\{[a-z_]+\\\}", ".+?", re.escape(t)) + "$", re.S) for t in templates]
        strangers = sorted({r.text for r in rows if not any(p.match(r.text) for p in patterns)})
        check(f"all {len(rows)} agent lines across every scenario match a script template", not strangers, str(strangers[:3]))
        db.close()

        # ------------------------------------------------------------------ #
        print("\n10. Reset restores the seeded leads")
        reset = client.post("/demo/reset").json()
        check("reset reports the leads it restored", reset["leads"].get("leads_restored", 0) >= 7, str(reset))
        leads = {i["lead"]["id"]: i for i in client.get("/leads").json()}
        check("E-1004 and E-1006 are callable again", leads["E-1004"]["lead"]["dnc_status"] is False and leads["E-1006"]["can_start"] is True)
        check("the lead genuinely on the register is still blocked", leads["E-1003"]["lead"]["dnc_status"] is True and leads["E-1003"]["can_start"] is False)

    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
