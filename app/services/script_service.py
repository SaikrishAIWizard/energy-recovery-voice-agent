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

from app.config import DATA_DIR
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

    def demo_replies(self, step_id: str) -> list[str]:
        step = self.step(step_id)
        return list(step.get("demo_replies", [])) if step else []

    # -- rendering --------------------------------------------------------- #
    def render_prompt(self, step_id: str, **context: Any) -> str:
        """Render the APPROVED prompt for a step.

        Unknown placeholders are left untouched, and missing values render as
        "not provided" — the agent never invents a value to fill a gap.
        """
        step = self.step(step_id)
        if step is None:
            raise KeyError(f"No script step '{step_id}'")
        return self._render(step["prompt"], context)

    def fallback_prompt(self, step_id: str) -> str:
        step = self.step(step_id)
        if step is None:
            raise KeyError(f"No script step '{step_id}'")
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
        lead = self._lead_overrides().get(lead_id, {})
        return dict(lead.get("scripted_turns") or {})

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
