"""Robustness scenario tests.

Exercises the messy-human-behaviour paths the handout calls out — mishears, ambiguity,
"not interested", off-script questions, frustration, and card data — plus the live
WebSocket feed. Run against a live backend:

    python scripts/scenario_tests.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

import httpx

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))
    print(f"  [{PASS if condition else FAIL}] {name}" + (f" — {detail}" if detail else ""))


class Call:
    """Small driver that keeps the session snapshot in sync."""

    def __init__(self, client: httpx.Client, lead_id: str) -> None:
        self.client = client
        started = client.post(f"/calls/start/{lead_id}", json={}).json()
        self.session = started["session"]
        self.agent_messages: list[str] = list(started.get("agent_messages") or [])
        self.last_turn: dict[str, Any] = {"safety_flags": [], "system_notes": []}

    def say(self, text: str) -> dict[str, Any]:
        turn = self.client.post(
            f"/calls/{self.session['id']}/utterance",
            json={"text": text, "source": "SIMULATED"},
        ).json()
        self.session = turn["session"]
        self.agent_messages = list(turn.get("agent_messages") or [])
        self.last_turn = turn
        return turn

    @property
    def step(self) -> str:
        return self.session["current_step"]

    @property
    def status(self) -> str:
        return self.session["status"]

    @property
    def agent_said(self) -> str:
        return " ".join(self.agent_messages)

    def advance_to(self, step: str, answers: dict[str, str]) -> None:
        """Drive the call forward until `step` is active, using canned replies.

        The consent gate is always answered with CONSENT unless the caller overrides it.
        """
        replies = {"recording_consent": CONSENT, **answers}
        guard = 0
        while self.step != step and self.status == "ACTIVE" and guard < 10:
            guard += 1
            reply = replies.get(self.step)
            if reply is None:
                raise AssertionError(f"No canned reply for step '{self.step}'")
            self.say(reply)


CONSENT = "Yes, go ahead."
ADDRESS = "12 Test Street, Sydney NSW 2000"


def scenario_relative_date(client: httpx.Client) -> None:
    print("\nSCENARIO 1 — ambiguous relative date gets exactly one clarification")
    call = Call(client, "E-1001")
    call.say(CONSENT)
    call.say(ADDRESS)
    check("advanced to move_in_date", call.step == "move_in_date", call.step)

    turn = call.say("I'm moving in next Friday.")
    check("still on move_in_date", call.step == "move_in_date", call.step)
    check(
        "asked the approved fallback prompt",
        "day, month, and year" in call.agent_said,
        call.agent_said[:70],
    )
    check("session still ACTIVE", call.status == "ACTIVE", call.status)
    attempt = next(
        f for f in call.session["journey_fields"] if f["field_name"] == "move_in_date"
    )
    check("field not marked VALID", attempt["status"] != "VALID", attempt["status"])

    call.say("1 October 2026")
    check("accepted the complete date", call.step == "energy_requirement", call.step)
    row = next(f for f in call.session["journey_fields"] if f["field_name"] == "move_in_date")
    check("stored as ISO date", row["value"] == "2026-10-01", str(row["value"]))


def scenario_repeated_failure(client: httpx.Client) -> None:
    print("\nSCENARIO 2 — same field fails twice -> REPEATED_FAILURE handoff")
    call = Call(client, "E-1001")
    call.say(CONSENT)
    call.say(ADDRESS)
    call.say("sometime soon")
    check("clarification asked once", call.step == "move_in_date", call.step)
    call.say("whenever works for me")
    check("handed off", call.status == "HANDOFF_REQUESTED", call.status)
    check(
        "reason is REPEATED_FAILURE",
        call.session.get("handoff_reason") == "REPEATED_FAILURE",
        str(call.session.get("handoff_reason")),
    )
    check(
        "escalation signal CONFUSION",
        (call.session.get("handoff") or {}).get("escalation_signal") == "CONFUSION",
        str((call.session.get("handoff") or {}).get("escalation_signal")),
    )
    check(
        "only two attempts were made",
        next(f for f in call.session["journey_fields"] if f["field_name"] == "move_in_date")[
            "attempts"
        ]
        == 2,
    )


def scenario_advice(client: httpx.Client) -> None:
    print("\nSCENARIO 3 — advice request -> OFF_SCRIPT, no advice given")
    call = Call(client, "E-1001")
    call.advance_to("energy_requirement", {"property_address": ADDRESS, "move_in_date": "1 October 2026"})
    call.say("Which plan is cheapest for me?")
    check("handed off", call.status == "HANDOFF_REQUESTED", call.status)
    check(
        "reason is OFF_SCRIPT",
        call.session.get("handoff_reason") == "OFF_SCRIPT",
        str(call.session.get("handoff_reason")),
    )
    check(
        "agent refused to advise",
        "not able to give advice" in call.agent_said.lower(),
        call.agent_said[:80],
    )
    check(
        "no price or plan name invented",
        not any(token in call.agent_said.lower() for token in ["$", "cents", "cheapest plan"]),
    )


def scenario_mid_journey_decline(client: httpx.Client) -> None:
    print("\nSCENARIO 4 — mid-journey refusal -> DECLINED, no pressure loop")
    call = Call(client, "E-1001")
    call.advance_to("energy_requirement", {"property_address": ADDRESS, "move_in_date": "1 October 2026"})
    call.say("Just take me off your list please.")
    check("declined", call.status == "DECLINED", call.status)
    check(
        "no retry of the same question",
        "Electricity" not in call.agent_said and "electricity, gas" not in call.agent_said,
        call.agent_said[:70],
    )
    events = [e["event_type"] for e in call.session["audit_events"]]
    check("decline audited", "CALL_DECLINED" in events, ",".join(events[-3:]))


def scenario_off_script_question(client: httpx.Client) -> None:
    print("\nSCENARIO 5 — unrelated question at a field -> OFF_SCRIPT handoff")
    call = Call(client, "E-1001")
    call.say(CONSENT)
    call.say("What's the weather like in Sydney today?")
    check("handed off", call.status == "HANDOFF_REQUESTED", call.status)
    check(
        "reason is OFF_SCRIPT",
        call.session.get("handoff_reason") == "OFF_SCRIPT",
        str(call.session.get("handoff_reason")),
    )
    check(
        "address was NOT fabricated",
        call.session["collected_fields"].get("property_address") is None,
        str(call.session["collected_fields"].get("property_address")),
    )


def scenario_frustration(client: httpx.Client) -> None:
    print("\nSCENARIO 6 — frustration -> FRUSTRATION handoff with context")
    call = Call(client, "E-1001")
    call.say(CONSENT)
    call.say(ADDRESS)
    call.say("This is the second call today and I already told you this.")
    check("handed off", call.status == "HANDOFF_REQUESTED", call.status)
    check(
        "reason is FRUSTRATION",
        call.session.get("handoff_reason") == "FRUSTRATION",
        str(call.session.get("handoff_reason")),
    )
    handoff = call.session.get("handoff") or {}
    check(
        "collected address travels with the handoff",
        handoff.get("collected_fields", {}).get("property_address") == ADDRESS,
        str(handoff.get("collected_fields")),
    )
    check("last customer message preserved", bool(handoff.get("last_customer_message")))
    check("context summary present", bool(handoff.get("context_summary")))


def scenario_card_boundary(client: httpx.Client) -> None:
    print("\nSCENARIO 7 — card data is redacted, never stored, never echoed")
    call = Call(client, "E-1001")
    call.advance_to(
        "energy_requirement",
        {"property_address": ADDRESS, "move_in_date": "1 October 2026"},
    )
    call.say("My card is 4111 1111 1111 1111, expiry 05/28.")
    check("handed off as SENSITIVE_TOPIC", call.session.get("handoff_reason") == "SENSITIVE_TOPIC",
          str(call.session.get("handoff_reason")))
    stored = json.dumps(call.session)
    check("no card digits anywhere in the stored session", "4111" not in stored)
    check("redaction token present in transcript", "[REDACTED_CARD_DATA]" in stored)
    check(
        "agent never repeated the digits",
        "4111" not in call.agent_said,
        call.agent_said[:70],
    )
    events = [e["event_type"] for e in call.session["audit_events"]]
    check("redaction audited", "CARD_DATA_REDACTED" in events)


def scenario_busy_callback(client: httpx.Client) -> None:
    print("\nSCENARIO 8 — busy customer gets exactly one callback offer")
    call = Call(client, "E-1001")
    call.say("I'm driving, can you call back later?")
    check("call ended", call.status == "DECLINED", call.status)
    check(
        "outcome detail is BUSY_CALLBACK_SCHEDULED",
        call.session.get("outcome_detail") == "BUSY_CALLBACK_SCHEDULED",
        str(call.session.get("outcome_detail")),
    )
    check("exactly one callback offered", call.agent_said.lower().count("callback") == 1,
          call.agent_said)
    check(
        "no field was collected",
        all(v is None for v in call.session["collected_fields"].values()),
    )


async def scenario_websocket(client: httpx.Client, ws_base: str) -> None:
    print("\nSCENARIO 9 — live WebSocket feed pushes session updates")
    try:
        import websockets
    except ImportError:
        check("websockets library available", False, "pip install websockets")
        return

    started = client.post("/calls/start/E-1001", json={}).json()
    session_id = started["session"]["id"]

    uri = f"{ws_base}/calls/{session_id}/stream"
    async with websockets.connect(uri) as socket:
        first = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))
        check("received initial snapshot", first.get("type") == "session")
        check("snapshot has transcript", "transcript" in first.get("session", {}))

        client.post(
            f"/calls/{session_id}/utterance",
            json={"text": CONSENT, "source": "SIMULATED"},
        )
        second = json.loads(await asyncio.wait_for(socket.recv(), timeout=6))
        check(
            "pushed the new turn",
            len(second["session"]["transcript"]) > len(first["session"]["transcript"]),
            f"{len(first['session']['transcript'])} -> {len(second['session']['transcript'])}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    # Derive the WebSocket origin from --base-url so a backend on another port is honoured.
    ws_base = args.base_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")

    print("=" * 78)
    print("ROBUSTNESS SCENARIOS")
    print("=" * 78)

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        scenario_relative_date(client)
        scenario_repeated_failure(client)
        scenario_advice(client)
        scenario_mid_journey_decline(client)
        scenario_off_script_question(client)
        scenario_frustration(client)
        scenario_card_boundary(client)
        scenario_busy_callback(client)
        asyncio.run(scenario_websocket(client, ws_base))

    failed = [row for row in results if row[1] == FAIL]
    print("\n" + "=" * 78)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} — {detail}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
