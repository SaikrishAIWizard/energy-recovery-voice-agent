"""End-to-end demo runner.

Drives every synthetic lead through the real API — no mocks, no shortcuts. This is the
same script you run in front of judges; it prints the transcript, the handoff package, and
the submitted payload for each lead.

Usage (backend must be running):
    python scripts/demo_run.py
    python scripts/demo_run.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    print("httpx is required: pip install httpx", file=sys.stderr)
    raise

BAR = "=" * 78
THIN = "-" * 78


def pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def run_lead(client: httpx.Client, lead_id: str, verbose: bool = True) -> dict[str, Any]:
    detail = client.get(f"/leads/{lead_id}").json()
    lead = detail["lead"]
    turns: dict[str, str] = detail.get("scripted_turns") or {}

    print(BAR)
    print(f"LEAD {lead_id} — {lead['first_name']} {lead.get('last_name') or ''}")
    print(THIN)
    print(f"  last completed step : {lead['last_completed_step']}")
    print(f"  DNC status          : {lead['dnc_status']}")
    print(f"  resume at           : {detail['resume_step']}")
    print(f"  expected outcome    : {detail.get('expected_outcome')}")
    if detail.get("scenario_notes"):
        print(textwrap.fill(f"  scenario            : {detail['scenario_notes']}", 100,
                            subsequent_indent=" " * 24))

    started = client.post(f"/calls/start/{lead_id}", json={"mode": "AGENT_DRIVEN"}).json()

    if started.get("blocked"):
        print(f"\n  >> BLOCKED: {started['blocked_reason']}")
        print(f"     {started['session']['outcome_detail']}")
        print(f"     No call placed. No prompts spoken. Transcript segments: "
              f"{len(started['session']['transcript'])}")
        return {"lead_id": lead_id, "outcome": "DNC_BLOCKED", "turns": 0}

    session = started["session"]
    print(f"\n  call session: {session['id']}")
    for message in started["agent_messages"]:
        print(f"\n  AI AGENT : {textwrap.fill(message, 96, subsequent_indent=' ' * 13)}")

    guard = 0
    # CLOSING = journey submitted, agent still asking "anything else?".
    while (session["status"] == "ACTIVE" or session["state"] == "CLOSING") and guard < 30:
        guard += 1
        step = session["current_step"]
        # Yes/no moments (read-backs, the busy question...) are keyed by state, the rest by step.
        reply = turns.get(f"state:{session['state']}") or turns.get(step)

        if reply is None:
            # No scripted line for this step: the scenario already ended the call.
            print(f"\n  (no scripted customer turn for '{step}' — stopping here)")
            break

        print(f"\n  CUSTOMER : {reply}")
        turn = client.post(
            f"/calls/{session['id']}/utterance",
            json={"text": reply, "source": "SIMULATED"},
        ).json()
        session = turn["session"]

        for message in turn["agent_messages"]:
            print(f"\n  AI AGENT : {textwrap.fill(message, 96, subsequent_indent=' ' * 13)}")
        for note in turn.get("system_notes", []):
            print(f"  [system] {note}")
        if turn.get("safety_flags"):
            print(f"  [safety] {', '.join(turn['safety_flags'])}")

    # --- outcome --------------------------------------------------------- #
    final = client.get(f"/calls/{session['id']}").json()
    print(THIN)
    print(f"  OUTCOME   : {final['status']}  (state={final['state']}, step={final['current_step']})")
    if final.get("outcome_detail"):
        print(f"  DETAIL    : {final['outcome_detail']}")
    print(f"  COLLECTED : {pretty(final['collected_fields'])}")

    if final.get("handoff"):
        print(f"\n  WARM HANDOFF PACKAGE")
        print(textwrap.indent(pretty(final["handoff"]), "    "))

    if final.get("submission"):
        print(f"\n  JOURNEY SUBMITTED")
        print(textwrap.indent(pretty(final["submission"]), "    "))

    if verbose:
        print(f"\n  AUDIT TRAIL ({len(final['audit_events'])} events)")
        for event in final["audit_events"]:
            detail_text = event["event_detail"][:96]
            print(f"    · {event['event_type']:<26} {detail_text}")

    return {
        "lead_id": lead_id,
        "outcome": final["status"],
        "detail": final.get("outcome_detail"),
        "turns": len([s for s in final["transcript"] if s["speaker"] == "CUSTOMER"]),
        "submission_id": (final.get("submission") or {}).get("submission_id"),
        "handoff_reason": (final.get("handoff") or {}).get("reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full Energy recovery demo.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--leads", nargs="*", default=None)
    parser.add_argument("--reset", action="store_true", help="Clear call history first.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        health = client.get("/health").json()
        print(BAR)
        print(f"{health['app']} v{health['version']}")
        print(f"leads={health['leads']}  call_sessions={health['call_sessions']}  "
              f"LLM={health['llm']['active_adapter']}")
        print(BAR)

        if args.reset:
            print("Resetting demo data…\n")
            client.post("/demo/reset")

        queue = client.get("/leads").json()
        for row in queue:
            # A "stop calling me" flags the lead itself, so it stays blocked until a reset.
            if row["lead"]["dnc_status"] and row.get("expected_outcome") != "DNC_BLOCKED":
                print(f"NOTE: {row['lead']['id']} was flagged Do-Not-Call by an earlier stop-calling "
                      "request, so it will be blocked. Run with --reset to restore it.\n")
        lead_ids = args.leads or [row["lead"]["id"] for row in queue]
        results = [run_lead(client, lead_id, verbose=not args.quiet) for lead_id in lead_ids]

        print("\n" + BAR)
        print("SUMMARY")
        print(BAR)
        print(f"{'LEAD':<8} {'OUTCOME':<20} {'DETAIL':<26} {'TURNS':<6} RECEIPT")
        print(THIN)
        for row in results:
            print(
                f"{row['lead_id']:<8} {row['outcome']:<20} "
                f"{(row.get('detail') or '-'):<26} {row['turns']:<6} "
                f"{row.get('submission_id') or row.get('handoff_reason') or '-'}"
            )

        summary = client.get("/dashboard/summary").json()
        counts = summary["counts"]
        print(THIN)
        print(
            f"dropped={counts['dropped_off']} active={counts['active_calls']} "
            f"completed={counts['completed']} handoffs={counts['handoffs']} "
            f"declined={counts['declined']} dnc_blocked={counts['dnc_blocked']}"
        )
        print(
            f"calls placed={summary['calls_placed']}  "
            f"completion rate={summary['completion_rate']:.0%}  "
            f"handoff rate={summary['handoff_rate']:.0%}"
        )
        print(
            f"fields auto-captured={summary['fields_captured_automatically']}  "
            f"≈{summary['estimated_manual_minutes_saved']} manual minutes saved"
        )
        print(BAR)

    expected = {
        "E-1001": "COMPLETED",
        "E-1002": "HANDOFF_REQUESTED",
        "E-1003": "DNC_BLOCKED",
        "E-1004": "DECLINED",
    }
    failures = [
        row
        for row in results
        if row["lead_id"] in expected and row["outcome"] != expected[row["lead_id"]]
    ]
    if failures:
        print("\nFAILED EXPECTATIONS:")
        for row in failures:
            print(f"  {row['lead_id']}: expected {expected[row['lead_id']]}, got {row['outcome']}")
        return 1

    print("\nAll spec'd demo expectations matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
