"""Tests for the Agent-Assisted (Mode A) seam and the telephony adapter wiring.

Run against a live backend:
    python scripts/mode_a_tests.py
"""

from __future__ import annotations

import argparse
import sys

import httpx

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def audit_types(client: httpx.Client, sid: str) -> list[str]:
    return [e["event_type"] for e in client.get(f"/calls/{sid}").json()["audit_events"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    print("=" * 78)
    print("MODE A / TELEPHONY ADAPTER TESTS")
    print("=" * 78)

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        # ------------------------------------------------------------------ #
        print("\nSCENARIO 1 — dial goes through the telephony adapter after DNC clears")
        started = client.post("/calls/start/E-1001", json={}).json()
        sid = started["session"]["id"]
        check("call started", started["session"]["status"] == "ACTIVE", started["session"]["status"])
        check(
            "dial provider recorded on the session",
            bool(started["session"]["dial_provider"]),
            str(started["session"]["dial_provider"]),
        )
        events = audit_types(client, sid)
        check("DIAL_ATTEMPT audited", "DIAL_ATTEMPT" in events, ",".join(events[:4]))
        order = [e for e in events if e in {"DNC_CHECK_PASSED", "DIAL_ATTEMPT", "RECORDING_CONSENT_DISCLOSED"}]
        check(
            "order is DNC -> dial -> disclosure",
            order == ["DNC_CHECK_PASSED", "DIAL_ATTEMPT", "RECORDING_CONSENT_DISCLOSED"],
            " -> ".join(order),
        )

        print("\nSCENARIO 2 — DNC-blocked lead is never dialled")
        blocked = client.post("/calls/start/E-1003", json={}).json()
        check("blocked", blocked["blocked"] is True, str(blocked["blocked_reason"]))
        bevents = audit_types(client, blocked["session"]["id"])
        check("no DIAL_ATTEMPT was made", "DIAL_ATTEMPT" not in bevents, ",".join(bevents))

        # ------------------------------------------------------------------ #
        print("\nSCENARIO 3 — human agent captures a field the AI could not get")
        client.post(f"/calls/{sid}/utterance", json={"text": "Yes, go ahead.", "source": "SIMULATED"})
        # The AI fails twice on the address, which would normally hand off.
        client.post(f"/calls/{sid}/utterance", json={"text": "hmm", "source": "SIMULATED"})
        after_fail = client.get(f"/calls/{sid}").json()
        check(
            "address still outstanding after one failed attempt",
            after_fail["current_step"] == "property_address" and after_fail["status"] == "ACTIVE",
            f"{after_fail['status']}/{after_fail['current_step']}",
        )
        # A human steps in before the second failure.
        turn = client.post(
            f"/calls/{sid}/field",
            json={
                "field_name": "property_address",
                "value": "12 Test Street, Sydney NSW 2000",
                "agent_name": "Aarav",
            },
        ).json()
        session = turn["session"]
        row = next(f for f in session["journey_fields"] if f["field_name"] == "property_address")
        check("value stored", row["value"] == "12 Test Street, Sydney NSW 2000", str(row["value"]))
        check("source is HUMAN_AGENT", row["source"] == "HUMAN_AGENT", row["source"])
        check("status VALID", row["status"] == "VALID", row["status"])
        check(
            "journey advanced past the stuck step",
            session["current_step"] == "move_in_date",
            session["current_step"],
        )
        check(
            "capture audited with the agent's name",
            any(
                e["event_type"] == "FIELD_CAPTURED_BY_HUMAN" and "Aarav" in e["event_detail"]
                for e in session["audit_events"]
            ),
        )
        check("agent was given the next approved prompt", len(turn["agent_messages"]) == 1)

        # ------------------------------------------------------------------ #
        print("\nSCENARIO 4 — the human cannot bypass validation")
        bad = client.post(
            f"/calls/{sid}/field",
            json={"field_name": "move_in_date", "value": "whenever", "agent_name": "Aarav"},
        )
        check("rejected with 422", bad.status_code == 422, str(bad.status_code))
        check(
            "rejection explains why",
            "not a valid" in bad.json().get("detail", "") or "date" in bad.json().get("detail", ""),
            bad.json().get("detail", "")[:70],
        )
        check(
            "rejection audited",
            "HUMAN_CAPTURE_REJECTED" in audit_types(client, sid),
        )
        still = client.get(f"/calls/{sid}").json()
        check(
            "no invalid value reached journey state",
            next(
                f for f in still["journey_fields"] if f["field_name"] == "move_in_date"
            )["status"]
            != "VALID",
        )

        print("\nSCENARIO 5 — human captures the rest and completes the journey")
        for field, value in [
            ("move_in_date", "1 October 2030"),
            ("energy_requirement", "Electricity"),
            ("concession_status", "No"),
            ("life_support", "No"),
            ("contact_preference", "Email"),
        ]:
            resp = client.post(
                f"/calls/{sid}/field",
                json={"field_name": field, "value": value, "agent_name": "Aarav"},
            )
            if resp.status_code != 200:
                check(f"captured {field}", False, resp.text[:120])
        final = client.get(f"/calls/{sid}").json()
        check(
            "all six fields now VALID",
            len(final["collected_fields"]) == 6,
            str(len(final["collected_fields"])),
        )
        check(
            "human-captured fields carry HUMAN_AGENT source",
            all(
                f["source"] == "HUMAN_AGENT"
                for f in final["journey_fields"]
                if f["field_name"] in final["collected_fields"]
            ),
        )

        print("\nSCENARIO 6 — human correction of an already-valid field")
        corrected = client.post(
            f"/calls/{sid}/field",
            json={"field_name": "contact_preference", "value": "Phone", "agent_name": "Aarav"},
        ).json()
        check(
            "corrected value stored",
            corrected["session"]["collected_fields"]["contact_preference"] == "PHONE",
            str(corrected["session"]["collected_fields"]["contact_preference"]),
        )
        check(
            "correction audited separately from capture",
            "FIELD_CORRECTED_BY_HUMAN" in audit_types(client, sid),
        )

        # ------------------------------------------------------------------ #
        print("\nSCENARIO 7 — a submitted journey cannot be silently rewritten")
        client.post(f"/calls/{sid}/utterance", json={"text": "Yes, that's all correct.", "source": "SIMULATED"})
        submitted = client.get(f"/calls/{sid}").json()
        check("journey completed", submitted["status"] == "COMPLETED", submitted["status"])
        check("receipt issued", submitted["submission"] is not None)
        late = client.post(
            f"/calls/{sid}/field",
            json={"field_name": "property_address", "value": "99 Changed Road, Sydney", "agent_name": "Aarav"},
        )
        check("late edit refused", late.status_code in (409, 422), str(late.status_code))
        unchanged = client.get(f"/calls/{sid}").json()
        check(
            "submitted payload is untouched",
            unchanged["submission"]["payload"]["property_address"] == "12 Test Street, Sydney NSW 2000",
            unchanged["submission"]["payload"]["property_address"],
        )

        print("\nSCENARIO 8 — warm transfer attempt is recorded on handoff")
        h = client.post("/calls/start/E-1002", json={}).json()
        hid = h["session"]["id"]
        client.post(f"/calls/{hid}/utterance", json={"text": "Yeah, fine.", "source": "SIMULATED"})
        client.post(
            f"/calls/{hid}/utterance",
            json={"text": "Let me talk to a person.", "source": "SIMULATED"},
        )
        hevents = audit_types(client, hid)
        check("handoff created", "HANDOFF_CREATED" in hevents)
        check("warm transfer attempted", "WARM_TRANSFER_ATTEMPT" in hevents, ",".join(hevents[-3:]))

    failed = [r for r in results if not r[1]]
    print("\n" + "=" * 78)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} — {detail}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
