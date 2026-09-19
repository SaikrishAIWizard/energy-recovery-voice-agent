"""Script library service.

Loads `app/data/energy_scripts.json` and is the **single source of truth** for what the
agent is allowed to say. The state machine can only speak `prompt` for the active step,
or its `fallback_prompt` as a one-time clarification.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import DATA_DIR, settings
from app.conversation.validators import humanise_iso_date

SCRIPTS_PATH = DATA_DIR / "energy_scripts.json"
LEADS_PATH = DATA_DIR / "synthetic_leads.json"

_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")

_DISPLAY_LABELS = {
    "ELECTRICITY": "Electricity",
    "GAS": "Gas",
    "BOTH": "Electricity and gas",
    "YES": "Yes",
    "NO": "No",
    "UNSURE": "Unsure",
    "PHONE": "Phone",
    "EMAIL": "Email",
}


class ScriptService:
    def __init__(self, path: Path = SCRIPTS_PATH) -> None:
        self.path = path
        self._raw: dict[str, Any] | None = None
        self._steps: dict[str, dict[str, Any]] = {}

    # -- loading ----------------------------------------------------------- #
    @property
    def raw(self) -> dict[str, Any]:
        if self._raw is None:
            with self.path.open(encoding="utf-8") as handle:
                self._raw = json.load(handle)
            self._steps = {step["step_id"]: step for step in self._raw["steps"]}
        return self._raw

    def reload(self) -> None:
        self._raw = None
        self._steps = {}
        _ = self.raw

    # -- lookups ----------------------------------------------------------- #
    def steps(self) -> list[dict[str, Any]]:
        return list(self.raw["steps"])

    def step(self, step_id: str) -> dict[str, Any] | None:
        _ = self.raw
        return self._steps.get(step_id)

    @property
    def resume_order(self) -> list[str]:
        return list(self.raw["resume_order"])

    @property
    def data_fields(self) -> list[str]:
        return list(self.raw["data_fields"])

    @property
    def required_fields(self) -> list[str]:
        return list(self.raw["required_fields"])

    def next_step(self, step_id: str) -> str | None:
        step = self.step(step_id)
        if not step:
            return None
        nxt = step.get("next_step")
        return None if nxt == "SUBMIT" else nxt

    def resume_step_after(self, last_completed_step: str | None) -> str:
        """Where a returning lead picks up.

        Consent and the continue-gate are always re-run on a fresh call (two-party
        consent is per-call), so the resume point is the step *after* the last completed
        data-collection step.
        """
        order = self.resume_order
        if not last_completed_step or last_completed_step not in order:
            return "property_address"
        index = order.index(last_completed_step)
        for candidate in order[index + 1:]:
            step = self.step(candidate)
            if step and step.get("kind") in {"FIELD", "CONFIRMATION"}:
                return candidate
        return "confirmation"

    def skip_known(self, step_id: str, known_fields: set[str]) -> str:
        """Advance past leading FIELD steps whose value is already known.

        Used when an earlier uploaded recording already captured part of the journey, so
        the recovery call only asks for what is still missing.
        """
        order = self.resume_order
        if step_id not in order:
            return step_id
        for candidate in order[order.index(step_id):]:
            step = self.step(candidate)
            if step and step.get("kind") == "FIELD" and step.get("field_name") in known_fields:
                continue
            return candidate
        return "confirmation"

    def demo_replies(self, step_id: str) -> list[str]:
        step = self.step(step_id)
        return list(step.get("demo_replies", [])) if step else []

    def demo_replies_for(self, state: str, step_id: str) -> list[str]:
        """Quick replies for the console: state-specific when the agent is waiting on a
        yes/no (a read-back, the busy question...), otherwise the active step's."""
        by_state = self.raw.get("state_demo_replies", {})
        if state in by_state:
            return list(by_state[state])
        return self.demo_replies(step_id)

    def scripted_state_replies(self) -> dict[str, str]:
        return dict(self.raw.get("state_scripted_replies", {}))

    # -- named lines ------------------------------------------------------- #
    def message(self, name: str, **context: Any) -> str:
        """An APPROVED non-step line (closing, handoff, busy question...). Every line the
        agent speaks is in the script file; there are no strings hidden in code."""
        entry = self.raw.get("messages", {}).get(name)
        if entry is None:
            raise KeyError(f"No script message '{name}'")
        return self._render(entry["text"], self._with_defaults(context))

    def handoff_message(self, reason: str, *, life_support: bool = False) -> str:
        """What the customer hears as they are handed to a person."""
        if life_support:
            return self.message("handoff_life_support")
        if reason == "CUSTOMER_REQUEST":
            return self.message("handoff_human_request")
        return self.message("handoff_default")

    def readback_prompt(self, step_id: str, **context: Any) -> str | None:
        """The per-field read-back ("I have X. Is that correct?"), for steps that have one."""
        step = self.step(step_id)
        if not step or not step.get("readback_prompt"):
            return None
        return self._render(step["readback_prompt"], self._with_defaults(context))

    def spoken_templates(self) -> list[str]:
        """Every line the agent can say, unrendered. Used by the tests to prove that nothing
        outside this file is ever spoken."""
        lines: list[str] = []
        for step in self.steps():
            for key in ("prompt", "fallback_prompt", "readback_prompt"):
                if step.get(key):
                    lines.append(step[key])
            lines.extend(step.get("fallback_prompts", []))
        lines.extend(entry["text"] for entry in self.raw.get("messages", {}).values())
        return lines

    def _with_defaults(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"brand": settings.agent_brand_name, **context}

    # -- rendering --------------------------------------------------------- #
    def render_prompt(self, step_id: str, **context: Any) -> str:
        """Render the APPROVED prompt for a step.

        Unknown placeholders are left untouched, and missing values render as
        "not provided" — the agent never invents a value to fill a gap.
        """
        step = self.step(step_id)
        if step is None:
            raise KeyError(f"No script step '{step_id}'")
        if "prompt" not in step:
            raise KeyError(f"Step '{step_id}' is a journey milestone and is never spoken")
        return self._render(step["prompt"], self._with_defaults(context))

    def fallback_prompt(self, step_id: str, follow_up: int = 1) -> str:
        """The follow-up question for a step. `follow_up` is 1 for the first re-ask, 2 for the
        second...; each is worded more simply than the last, and the final one is reused if
        the number of follow-ups is raised past what the script provides."""
        step = self.step(step_id)
        if step is None:
            raise KeyError(f"No script step '{step_id}'")
        ladder = step.get("fallback_prompts")
        if ladder:
            return ladder[min(max(follow_up, 1), len(ladder)) - 1]
        return step.get("fallback_prompt") or step["prompt"]

    def _render(self, template: str, context: dict[str, Any]) -> str:
        def substitute(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in context:
                return match.group(0)
            value = context[key]
            if value in (None, ""):
                return "not provided"
            if key.endswith("_date") and isinstance(value, str):
                return humanise_iso_date(value)
            return _DISPLAY_LABELS.get(str(value), str(value))

        return _PLACEHOLDER.sub(substitute, template)

    # -- preexisting lead data -------------------------------------------- #
    @lru_cache(maxsize=1)
    def _lead_overrides(self) -> dict[str, dict[str, Any]]:
        if not LEADS_PATH.exists():
            return {}
        with LEADS_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return {lead["id"]: lead for lead in payload["leads"]}

    def preexisting_fields_for_lead(self, lead_id: str) -> dict[str, Any]:
        lead = self._lead_overrides().get(lead_id, {})
        return dict(lead.get("preexisting_fields") or {})

    def scripted_turns_for_lead(self, lead_id: str) -> dict[str, str]:
        """Canned customer replies, by step, for "Play scripted call". Replies for the
        yes/no moments (read-backs, the closing question...) are keyed `state:<STATE>`."""
        lead = self._lead_overrides().get(lead_id, {})
        turns = {f"state:{state}": reply for state, reply in self.scripted_state_replies().items()}
        turns.update(lead.get("scripted_turns") or {})
        return turns

    def scenario_notes_for_lead(self, lead_id: str) -> str | None:
        lead = self._lead_overrides().get(lead_id, {})
        return lead.get("scenario_notes")

    def expected_outcome_for_lead(self, lead_id: str) -> str | None:
        lead = self._lead_overrides().get(lead_id, {})
        return lead.get("expected_outcome")

    def field_prompts(self) -> dict[str, str]:
        """field_name -> approved prompt, so the console can show the agent what to ask."""
        return {
            step["field_name"]: step["prompt"]
            for step in self.steps()
            if step.get("kind") == "FIELD"
        }


script_service = ScriptService()
