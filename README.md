# Energy Recovery Voice Agent

An AI-assisted **voice recovery system for dropped Energy comparison leads**.

A customer starts an energy comparison journey, enters their phone and email, then abandons a
later step. Today a human agent rings them back, reads a script, and types the answers in by
hand. This prototype lets an AI voice agent place that call, hold the conversation, collect the
missing fields, submit the journey — **and hand the caller to a human the moment things turn**,
with every collected detail travelling across so the customer never repeats themselves.

Built for the **CIMET · econnex Jaipur 12-hour hackathon** — Energy vertical only, dropout
recovery only, one journey, done properly.

> **Runs fully offline. No API keys, no telephony account, no cloud.** Browser voice, simulated
> transcript turns, and a rules engine cover 100% of the demo. LLM / STT / TTS / Twilio adapters
> exist behind interfaces and switch on only if you supply credentials.

---

## 1. What the demo proves

A live run shows one complete recovery path end to end:

| # | Requirement | Where it happens |
|---|---|---|
| 1 | Select a synthetic dropped Energy lead | Recovery Queue dashboard |
| 2 | Do-Not-Call check **before** dialling | `dnc_service` — runs before a session can exist |
| 3 | Recording disclosure **before** collecting data | the opening line (`recording_consent`): disclosure, then *"Is now an okay time to continue?"*, always first |
| 4 | Resume from the lead's last completed step | `script_service.resume_step_after()`; fields already valid are never re-asked, and the agent goes straight to the first missing question |
| 5 | Ask scripted questions one field at a time | `energy_scripts.json`, which follows `docs/energy-agent-script-and-checklist.md` line for line and is the only source of agent speech |
| 6 | Extract and validate answers | `field_extractor` + `validators` (Python is authoritative) |
| 7 | Save collected data in journey state | `journey_fields` table |
| 8 | Submit a valid mock Energy payload | `POST /journey/submit` → receipt `SUB-1001` |
| 9 | Handle "I'm busy", "not interested", misunderstandings, frustration | `safety_engine` + gate/field handlers |
| 10 | Warm-transfer with full context | `handoffs` table + Handoff Console |

All seven seeded leads are exercised by `backend/scripts/demo_run.py`, the messy conversational
paths by `backend/scripts/scenario_tests.py` (**49/49 checks pass**), the script itself (every
quoted line of the document, and the seven checklist scenarios) by `backend/scripts/script_tests.py`
(**222/222 checks pass**), the Agent-Assisted mode
plus telephony wiring by `backend/scripts/mode_a_tests.py` (**27/27 checks pass**), and the
optional provider adapters — including the proof that a vendor transcript cannot bypass the
guardrails — by `backend/scripts/provider_tests.py` (**51/51 checks pass**).

---

## 2. Quick start

Two terminals. Python 3.11+ and Node 18+.

### Windows (cmd) — fastest path

Double-click **`start-demo.bat`** in the project root. It finds a Python that already has the
dependencies, installs the frontend packages if needed, opens the backend and frontend each in
their own window, and launches the console. Close both windows to stop.

To do it by hand instead, open **two** Command Prompt windows:

```cmd
:: Terminal 1 - backend
cd /d "C:\Users\vijay\OneDrive\Desktop\References_Codes\Teachthephone\energy-recovery-voice-agent\backend"
venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

```cmd
:: Terminal 2 - frontend
cd /d "C:\Users\vijay\OneDrive\Desktop\References_Codes\Teachthephone\energy-recovery-voice-agent\frontend"
npm run dev
```

Then open <http://localhost:5173>. `npm install` only needs running once — `node_modules` is
already present.

> `backend\venv` already has the dependencies installed, so use its `python.exe` rather than a
> bare `python`. On this machine `python` is **not** on PATH and the Anaconda install does not
> have the dependencies. A short alias makes it painless:
>
> ```cmd
> doskey pyenv="C:\Users\vijay\OneDrive\Desktop\References_Codes\Teachthephone\energy-recovery-voice-agent\backend\venv\Scripts\python.exe" $*
> pyenv -m uvicorn app.main:app --reload --port 8000
> ```

**Optional — enable a real provider.** Copy the template and paste a key (see
[section 11](#11-optional-provider-adapters--and-which-ones-are-worth-signing-up-for)):

```cmd
cd /d "C:\Users\vijay\OneDrive\Desktop\References_Codes\Teachthephone\energy-recovery-voice-agent\backend"
copy .env.example .env
notepad .env
```

Restart the backend afterwards. The startup banner tells you whether the key was picked up —
if it is not listed there, it was not read.

### Backend (fresh clone / any OS)

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

`start-demo.bat` probes `backend\.venv` first, then `backend\venv`, then falls back to a known
interpreter — so a project-local virtualenv is picked up automatically.

Optional providers live in `backend/.env`, which is git-ignored:

```bash
cp .env.example .env      # then fill in whichever keys you have
```

`.env` is loaded on startup (real environment variables still win over the file, so CI or a
shell profile can override it). Nothing in it is required.

The SQLite database is created and seeded automatically on first start — seven synthetic Energy
leads, no migration step, no manual seed command. API docs at <http://127.0.0.1:8000/docs>.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

If the backend is not on `127.0.0.1:8000`, create `frontend/.env`:

```
VITE_API_BASE_URL=http://127.0.0.1:9000
```

---

## 3. Demo script

Start the backend, then in a second terminal:

```bash
cd backend
python scripts/demo_run.py          # full transcript, handoff packages, submitted payloads
python scripts/scenario_tests.py    # 49 robustness assertions across the messy paths
python scripts/script_tests.py      # 222 assertions: the script doc is the source of truth + the 7 checklist scenarios
python scripts/mode_a_tests.py      # 27 assertions: telephony wiring + human-in-the-loop capture
python scripts/recording_tests.py   # uploaded-recording checks (in-process, STT stubbed, no keys needed)
python scripts/twilio_tests.py      # live phone call + handoff-conference checks (in-process, Twilio stubbed)
python scripts/provider_tests.py    # 51 assertions: .env loading, provider wiring, guardrail integrity
```

`provider_tests.py` needs no credentials and no running server — it exercises the provider paths
with stubs, so the assertions hold whether or not you have configured keys. Add `--base-url` to
the first three if your backend is not on `127.0.0.1:8000`.

`demo_run.py` drives the real HTTP API — no mocks. It prints each transcript, the warm-handoff
package, the submitted payload, and the audit trail.

### The seven leads

| Lead | DNC | Resumes at | What the customer does | Expected outcome |
|---|---|---|---|---|
| **E-1001** Ava Thompson | clear | `property_address` | Agrees, gives clean answers, confirms each read-back and the final one, then says nothing else is needed | **COMPLETED** → `SUB-1001` |
| **E-1002** Marcus Lee | clear | `move_in_date` | *"I already told someone my details. Let me talk to a person."* | **HANDOFF_REQUESTED** (`CUSTOMER_REQUEST` / `ASKS`) |
| **E-1003** Priya Nair | **on register** | — | never called | **DNC_BLOCKED** |
| **E-1004** Daniel Okafor | clear | — | *"Stop calling me. I'm not interested."* | **DECLINED** (`DO_NOT_CALL_REQUESTED`), then flagged Do-Not-Call |
| E-1005 Sofia Ramirez | clear | `move_in_date` | Life-support equipment at the property | **HANDOFF_REQUESTED** (`SENSITIVE_TOPIC`) |
| E-1006 Tom Becker | clear | `property_address` | Offers a card number mid-journey | **HANDOFF_REQUESTED** (`SENSITIVE_TOPIC`, digits redacted) |
| E-1007 Hannah Wells | clear | — | *"Now's not great"*, then *"a callback please"* | **DECLINED** (`CALLBACK_REQUESTED`) |

### In the browser

1. **Recovery Queue** (`/`) — click **Start recovery** on E-1001. E-1003's button is disabled:
   the DNC gate refuses to dial and no call session is created.
2. **Live Recovery Console** (`/calls/…`) — the agent delivers the recording disclosure. Either
   speak into the mic, type a reply, click a *simulated customer reply* chip, or press
   **Run scripted call** to play the whole scenario hands-free. Watch the journey stepper, the
   field table (with confidence and attempt counts), and the audit trail fill in live over
   WebSocket.
3. **Warm handoff** — run E-1002 or E-1005, then open the **Handoff Console**. The human agent
   sees the reason, escalation signal, collected fields, last customer message, safety flags and
   full transcript — then clicks **Human agent accepted**. The **Complete the journey** panel lists
   the six details; the agent asks the customer for whatever is missing (the approved question is
   shown as they edit), types each answer, ticks *"I read these details back and they confirmed"*, and
   presses **Submit journey**. Every value goes through the same validators as a spoken answer and is
   saved as `HUMAN_AGENT`. If life support was declared, the panel shows an amber
   vulnerable-customer notice: the agent confirms it with the customer (or corrects Life support to No if it was
   misheard) and ticks a separate *"I confirmed with the customer… no medical details recorded"* validation before
   Submit unlocks. The AI itself never submits such a journey.
4. **Completed Journey** (`/completed/…`) — the accepted payload, receipt ID, timestamp, and
   every audit event.

The **Agent-driven / Agent-assisted** toggle mirrors the handout's Mode A / Mode B duality. In
agent-assisted mode each AI-drafted line is held for human approval before it is spoken.

**Reset between runs:** the sidebar's *Reset demo data*, or `curl -X POST localhost:8000/demo/reset`. This
matters more now: a stop-calling request flags the lead Do-Not-Call permanently, so E-1004 stays blocked
until a reset, which restores every seeded lead's Do-Not-Call flag and status along with clearing the calls.

### Screens

| Screen | What it shows |
|---|---|
| ![Recovery queue](docs/screenshots/02-recovery-queue-wide.png) | **Recovery Queue** — the seven leads with DNC state, call status, and the efficiency strip. E-1003's Start button is disabled: the DNC gate refuses to dial. |
| ![Live console](docs/screenshots/03-live-recovery-console.png) | **Live Recovery Console** mid-call — recording disclosed, resumed at `move_in_date`, stepper showing 3/9 complete, live transcript, per-field confidence, and the audit trail. |
| ![Handoff console](docs/screenshots/04-warm-handoff-console.png) | **Handoff Console** — reason, escalation signal, caller identity, captured fields, last customer message, safety flags, and the full transcript. |
| ![Completed journey](docs/screenshots/05-completed-journey.png) | **Completed Journey** — receipt `SUB-1001`, the accepted payload, 9/9 steps, and all 28 audit events. |
| ![Agent-assisted capture](docs/screenshots/06-agent-assisted-capture.png) | **Mode A** — the human agent captures the address the AI missed. Note `Dialled via browser_microphone`, the `human agent` source, and the journey advancing past the stuck step. |
| ![Server-side STT](docs/screenshots/07-server-stt-upload.png) | **Server-side STT** — with `DEEPGRAM_API_KEY` set, the Runtime panel reads `STT: deepgram` and the **Transcribe audio file** control appears next to the mic. With no key both revert to the browser, and the button is absent rather than broken. |

---

## 4. Architecture

```
                       React Agent Console  (Vite · TS · Tailwind · Router · Lucide)
                        │  REST  +  WebSocket  /calls/{id}/stream
                        ▼
                 FastAPI Orchestration API
                        │
   ┌────────────────────┼──────────────────────────────────────────────┐
   │                    │                                              │
 Lead Service      Conversation Engine                            Mock Journey Service
 DNC Service       ├── state_machine.py   ← the only decision maker  POST /journey/submit
 Script Library    ├── safety_engine.py   ← runs before extraction
 Call Sessions     ├── field_extractor.py ← rules-first, LLM optional
 Voice Providers   └── validators.py      ← Python validates everything
 Handoff Service
 Redaction Service
 LLM Service (optional)
   │
   └── SQLite  (leads · call_sessions · journey_fields · transcript_segments ·
                handoffs · audit_events · journey_submissions)
```

### The one architectural decision that matters

The reference implementations in this space (Pipecat-style pipelines) let an LLM drive the
conversation. **This system deliberately does not.**

| Concern | Owner |
|---|---|
| What to ask next | Deterministic state machine |
| Whether an answer is valid | Python validators |
| Whether to escalate | Python safety engine |
| Whether to submit | Python, only when every required field is VALID |
| Structured extraction | Rules first, optional LLM second — always re-validated in Python |
| Wording of an agent turn | Approved script library only — the LLM has no say in phrasing |

The LLM is an *input adapter*, never an authority. That is what makes the guardrails auditable
and impossible to prompt-inject. With no API key, the app is fully functional; `RulesOnlyLLM`
simply returns `None` and the rules engine answers.

### Data flow of one turn

```
customer utterance (RAW — not yet redacted)
   └─► safety_engine.evaluate(text)   ──► decline? handoff? (pure regex, never an LLM)
   │      It owns redaction, so it must see the raw text: redacting first
   │      would strip the digits and destroy the escalation signal.
   │      A decline or handoff short-circuits here — extraction never runs.
   │      └─► safe_text  ──► transcript_segments  (card data masked before storage)
        └─► state machine dispatch on session.state
             └─► field_extractor.extract_field()   [rules → optional LLM]
                  └─► validators.validate_field()  [authoritative]
                       ├─ VALID & conf ≥ 0.80  → save, advance step
                       ├─ low confidence      → up to 3 follow-ups, then LOW_CONFIDENCE handoff
                       └─ no match            → up to 3 follow-ups, then REPEATED_FAILURE handoff
                            └─► audit_events  +  WebSocket push
```

---

## 5. The conversation state machine

`backend/app/conversation/state_machine.py`

```
INIT → DNC_CHECK ─┬─► DNC_BLOCKED                                    (terminal, no call placed)
                  └─► AWAITING_CONTINUE_RESPONSE   "…This call is recorded. Is now an okay time…?"
                        ├─► DECLINED                                 (refusal / do-not-call: acknowledged once, logged)
                        ├─► AWAITING_BUSY_PREFERENCE ─► DECLINED     (busy: callback or later? asked once, logged)
                        └─► COLLECTING_FIELD ⇄ VALIDATING_FIELD
                              ├─► CONFIRMING_FIELD                   (address / date / energy: "I have X. Is that correct?")
                              ├─► AWAITING_ELIGIBILITY_CHOICE        (concession: "Do I qualify?" → a person is offered)
                              ├─► HANDOFF_REQUESTED                  (the escalation triggers)
                              └─► CONFIRMING_DETAILS                 (final read-back; one correction of one field allowed)
                                    ├─► SUBMITTING_JOURNEY → CLOSING → COMPLETED   ("anything else you need from a team member?")
                                    │                          └─► HANDOFF_REQUESTED  ("yes")
                                    └─► HANDOFF_REQUESTED            (still not confirmed after one correction)
```

Rules the engine enforces:

- Everything the agent **says** comes from the script file. There are no spoken strings in the code;
  `script_tests.py` checks every agent line in every scenario against the file's templates.
- **One question at a time**, and **one customer answer per field**. An unclear answer gets up to
  **three follow-up questions**, each worded more simply than the last (three are written for every
  question); if it is still unclear after the third, a person takes over.
- **Only missing fields are asked.** Values already valid on the lead record (or captured earlier) are
  skipped, and start-up resumes at the first missing one.
- Address, date and energy are **read back** and only count once the customer confirms. Everything
  is read back again at the end; a correction changes **only the field named**, then the read-back
  repeats **once**. Still not confirmed → a person, never a loop.
- Four failed attempts at the same field (the question + three follow-ups) → `REPEATED_FAILURE` handoff. An
  unclear answer to a read-back is asked again up to three times, then `LOW_CONFIDENCE`.
- Confidence below **0.80** after the follow-ups → `LOW_CONFIDENCE` handoff.
- A service address needs a **street number, street name, suburb and a state or postcode**
  (`validators.validate_address`); *"Queens Street"* is asked for again, once.
- Unknown required values are **never** submitted and **never** invented. Submission also refuses
  a payload that fails the journey's own validation, and any journey with `life_support = YES` (only a human agent can submit that, after validating it with the customer).
- The customer's "no", "I'm busy" or "stop calling" is acknowledged **once**, logged
  (`CUSTOMER_DECLINED` / `DO_NOT_CALL_REQUESTED` / `CALLBACK_REQUESTED` / `LEFT_FOR_LATER`), and the call
  ends. The one question the script asks on the busy branch (*callback, or later?*) is not repeated.

### Script library

`docs/energy-agent-script-and-checklist.md` is the source of truth for **what the agent says and
collects**, and `backend/app/data/energy_scripts.json` is that document in machine-readable form:
one section per journey step, plus the named lines (closing, handoff, busy question...). The
document's quoted lines are stored **verbatim** and `script_tests.py` compares them, so the two
cannot drift apart. To change the script, edit the document, then the JSON, and the tests say if
they disagree.

```json
{
  "step_id": "move_in_date",
  "kind": "FIELD",
  "section": "4. Move-in / connection date",
  "prompt": "What date are you moving into the property, or when would you like the energy connection to begin?",
  "fallback_prompt": "To make sure I record it correctly, could you give the day, month, and year?",
  "readback_prompt": "I have {move_in_date}. Is that correct?",
  "validation_type": "date",
  "max_retries": 4,
  "fallback_prompts": ["…follow-up 1…", "…follow-up 2, simpler…", "…follow-up 3, simpler still…"],
  "max_clarifications": 3,
  "next_step": "energy_requirement"
}
```

- `{braces}` are filled from the lead record or the journey (`{brand}` from `AGENT_BRAND_NAME` in
  `.env`); a missing value renders as *"not provided"*, never a guess.
- Follow-ups stay short. Follow-up 1 is a brief re-ask; follow-up 2 states the format that will be accepted with a
  simple example (*"Please say it in this format: unit (if any), street number, street name, suburb, state, postcode. For
  example: 12 Test Street, Sydney NSW 2000."*); follow-up 3 is the shortest nudge. `script_tests.py` checks that every
  example the agent gives is actually accepted by the validator.
- A line marked `derived` is the smallest acknowledgement the document needs but does not word
  (re-asking a question once, saying goodbye). Nothing else can be spoken.
- `customer_continue` is a journey **milestone** only: the customer's permission is asked once, in
  the opening, and that step is never spoken.
- `max_retries` is the total capture attempts: the question plus three follow-ups (`FIELD_FOLLOW_UPS` in `.env`). The JSON also carries
  `demo_replies` / `state_demo_replies` (console quick-replies) which the agent can never speak.

### Agent-Assisted mode (handout Mode A)

The handout's two modes are both real here, not a label:

- **Agent-driven (Mode B)** — the state machine runs the call. This is the default.
- **Agent-assisted (Mode A)** — a human agent works the call alongside the AI. Every journey
  field in the console has an edit control, so the agent can capture a value the AI misheard or
  correct one it got wrong, mid-call, via `POST /calls/{id}/field`.

The important part is that **the human cannot bypass a guardrail either**. A human-captured value
goes through the same Python validator as a spoken answer, and is stored with
`source = HUMAN_AGENT` so the audit trail always shows who captured what:

```
422  'whenever' is not a valid move_in_date (not_a_date).
     The same validator that checks spoken answers applies here.
```

A submitted journey is frozen — a late edit is refused rather than silently rewriting a payload
that has already been sent. This is also the escape hatch that makes `REPEATED_FAILURE` less
costly: the agent can rescue the field before the last failed attempt forces a handoff.

### Where the DNC check sits, and how the call is placed

The order is enforced in `start_call()` and asserted by `mode_a_tests.py`:

```
CALL_START_REQUESTED → DNC_CHECK_PASSED → DIAL_ATTEMPT → RECORDING_CONSENT_DISCLOSED → CALL_STARTED
```

Dialling goes **through the telephony adapter**, so the provider seam is a real call path rather
than a decorative interface. With no credentials the `browser_microphone` provider answers and
the console runs the call locally; with Twilio configured the same call actually dials and the
warm handoff transfers the live call. A DNC-blocked lead never reaches `DIAL_ATTEMPT` at all.

---

## 6. Safety and warm-handoff policy

### Escalation taxonomy

The handout's six signals map onto the six stored handoff reasons:

| Handout signal | Handoff reason | Detected by |
|---|---|---|
| **ASKS** | `CUSTOMER_REQUEST` | *"talk to a person"*, *"transfer me"*, *"are you a bot"* |
| **ANGER** | `FRUSTRATION` | profanity, *"stop wasting my time"*, *"already told you"*, *"second call today"* |
| **CONFUSION** | `REPEATED_FAILURE` | the same field still unclear after the question and three follow-ups |
| **SENSITIVE** | `SENSITIVE_TOPIC` | card data, life support, hardship, vulnerability, disputes |
| **OFF-SCRIPT** | `OFF_SCRIPT` | advice requests, non-Energy topics, unrelated questions |
| **LOW CONF** | `LOW_CONFIDENCE` | field confidence still below 0.80 after clarification |

### Priority order (deliberate, and tested)

1. **Explicit request for a person** — the customer has told us what they want.
2. **Refusal** — *"not interested"*, *"stop calling"*, *"take me off your list"*. Acknowledged once,
   logged, ended. No retry, no re-ask, no pressure loop. Respecting "no" outranks de-escalating.
   An explicit **stop-contact request** (*"stop calling me"*, *"don't call me again"*, *"take me off your
   list"*, *"unsubscribe"*) also **flags the lead Do-Not-Call** (`lead.dnc_status = true`), on a live call,
   a phone call or an uploaded recording. From then on every dial is refused by the DNC gate
   (`DNC_CUSTOMER_REQUEST`) and, unlike a lead that is merely on the register, the demo `force`
   override cannot bypass it. A plain *"not interested"* is only a decline and flags nothing.
3. **Payment card data** — never captured, stored, displayed, or repeated.
4. **Vulnerability / life support / dispute** — vulnerable-customer cases go to a human.
5. **Anger and profanity.**
6. **Advice requests** — the agent states it can collect but not advise, then offers a handoff.

Every signal that fires is recorded as a safety flag, so a call can carry
`["CUSTOMER_REQUEST", "FRUSTRATION"]` while escalating for the highest-priority reason.

### Warm handoff package

```json
{
  "lead_id": "E-1002",
  "handoff_reason": "CUSTOMER_REQUEST",
  "current_step": "move_in_date",
  "collected_fields": {
    "property_address": "12 Test Street, Sydney NSW 2000",
    "move_in_date": null, "energy_requirement": null, "concession_status": null,
    "life_support": null, "contact_preference": null
  },
  "last_customer_message": "I already told someone my details. Let me talk to a person.",
  "conversation_summary": "Recording consent was disclosed before any data collection (AU two-party consent). Journey resumed from the lead's last completed step 'property_address' at 'move_in_date'. Already captured on this call: Service address. Still outstanding: Move-in date, Supply needed, Concession, Life support, Contact preference. The customer asked to speak with a person. Handed over while at step 'move_in_date'.",
  "safety_flags": ["CUSTOMER_REQUEST", "FRUSTRATION"],
  "escalation_signal": "ASKS",
  "known_contact": { "first_name": "Marcus", "phone": "+61 400 000 002", "email": "marcus.lee@example.com" },
  "outstanding_fields": ["move_in_date", "energy_requirement", "concession_status", "life_support", "contact_preference"],
  "field_details": { "property_address": { "value": "12 Test Street, Sydney NSW 2000", "status": "VALID", "confidence": 1.0, "source": "PREEXISTING" } },
  "recording_disclosed": true
}
```

`field_details` carries each value's confidence and source, and a value heard but **not yet confirmed**
(mid read-back) is listed with status `PENDING` and named in the summary, so the human never has to
re-ask what the customer already said.

The Handoff Console renders this in full, then accepting the handoff writes the human agent into
the transcript — *"Hi Marcus, this is Aarav — I can see everything so far, no need to repeat
anything."*

---

## 7. Guardrails — and where each one lives in code

| Guardrail | Enforcement point |
|---|---|
| Synthetic data only | `app/data/synthetic_leads.json` — ACMA test range numbers, `example.com` emails |
| Consent before data collection | the opening line discloses the recording and asks permission first; `recording_consent_disclosed` is set before any field is asked |
| DNC check before call creation | `dnc_service.check()` runs inside `start_call()` before journey fields are seeded |
| Never capture card data | `redaction_service.redact_text()` on **every** utterance, before storage and before the LLM |
| Never give advice | `_ADVICE_REQUEST` / `is_eligibility_question` → handoff (or the scripted *"I cannot advise on eligibility… connect you now?"*); no advice text exists in the script |
| Respect "no" | `_DECLINE` → `_decline()`, which speaks *"Understood. I will record that request…"* once, logs it and ends. A stop-contact request also sets `lead.dnc_status` (`_flag_do_not_call`), so the DNC gate refuses every later call, including the phone dial, and no override applies |
| Energy only | no broadband content in the script; `_OFF_SCRIPT_TOPIC` hands NBN / internet / loans questions to a person |
| Audit everything important | `audit_events` written at every transition (`CALL_STARTED`, `FIELD_CAPTURED`, `CARD_DATA_REDACTED`, `HANDOFF_CREATED`, `JOURNEY_SUBMITTED`, …) |
| Redact card-like sequences | `redaction_service` — 13–19 digit PANs, CVV, expiry; AU phone numbers are preserved |
| Follow up (up to 3 times), then hand off | `MAX_FOLLOW_UPS = 3` (`FIELD_FOLLOW_UPS` in `.env`) → `MAX_FIELD_ATTEMPTS = 4`; `MIN_FIELD_CONFIDENCE = 0.80` |
| Never fabricate or falsely submit | `_submit()` refuses (`SUBMISSION_BLOCKED`) when a field is missing, the payload fails the journey's validation, or life support is YES (the AI never submits that; a human agent can, after validating it, see below); it only runs after a positive final confirmation |
| Human agents cannot bypass validation | `capture_field_by_human()` runs the same validator; a submitted payload is frozen |

Card redaction is verified in `scenario_tests.py`: the stored session contains **no** card
digits, the transcript carries `[REDACTED_CARD_DATA]`, and the agent never repeats them.

---

## 8. API overview

Base URL `http://127.0.0.1:8000`. Interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Status, seeded counts, active adapters, guardrail summary |
| `GET` | `/leads` | Recovery queue with latest call state and resume point |
| `GET` | `/leads/{lead_id}` | Lead detail + DNC verdict + scripted demo turns |
| `GET` | `/leads/{lead_id}/sessions` | The lead's whole call history, newest first (the queue only shows the latest call) |
| `GET` | `/handoffs` | Every call waiting on a human, even if a newer call has started for the same lead |
| `POST` | `/calls/start/{lead_id}` | DNC check, then start the session and deliver the disclosure |
| `POST` | `/calls/{id}/utterance` | One customer turn → full updated session snapshot |
| `POST` | `/calls/{id}/field` | **Mode A** — a human agent captures or corrects one field live |
| `POST` | `/calls/{id}/audio` | Server-side STT — raw audio body → transcript → same turn pipeline (503 if unconfigured) |
| `POST` | `/calls/{id}/dial` | **Start call by agent** — the assistant phones the customer through Twilio (503 if Twilio / `PUBLIC_BASE_URL` unset, 409 on a DNC lead) |
| `POST` | `/calls/{id}/handoff/dial-agent` | Ring the human agent again on a live phone handoff |
| `POST` | `/twilio/voice\|turn\|status\|handoff\|handoff-ended\|agent\|agent-accept\|agent-status\|conference/{id}` | Twilio's webhooks — public, **signature-checked** |
| `POST` | `/recordings/upload` | **Upload a finished call recording** — raw audio body + `lead_id` (or `new_first_name`/`new_phone`/`new_email`) → transcript → checklist → journey submitted or lead left in the recovery queue (503 if unconfigured) |
| `GET` | `/calls/{id}` | Full session (transcript, fields, handoff, submission, audit, stepper) |
| `GET` | `/calls/{id}/transcript` | Transcript segments |
| `GET` | `/calls/{id}/handoff` | Warm-handoff package |
| `POST` | `/calls/{id}/handoff` | Console-initiated handoff |
| `POST` | `/calls/{id}/handoff/accept` | Human agent accepts |
| `POST` | `/calls/{id}/handoff/submit` | **A human agent completes a handed-off journey** — after filling the gaps with `/calls/{id}/field` and reading the details back (`customer_confirmed: true`). When life support is YES it also needs `life_support_validated: true` (the agent confirmed it with the customer; audited as `LIFE_SUPPORT_VALIDATED_BY_HUMAN`). 422 if anything is missing/invalid or unvalidated, 409 if the call is not waiting on a human |
| `POST` | `/calls/{id}/end` | End the call from the console |
| `WS` | `/calls/{id}/stream` | Live session feed |
| `POST` | `/journey/submit` | **Mock Energy journey endpoint** |
| `GET` | `/journey/submissions` | Submission receipts |
| `GET` | `/dashboard/summary` | Counts, queue rows, completion/handoff rates, effort saved |
| `POST` | `/demo/reset` | Clear call history and reseed *(demo helper)* |

**Turn responses return the entire session**, so the console never reconstructs state or replays
events to stay in sync — it just renders the latest snapshot.

### Journey submission

```http
POST /journey/submit
{
  "lead_id": "E-1001",
  "vertical": "ENERGY",
  "property_address": "12 Test Street, Sydney NSW 2000",
  "move_in_date": "2026-10-01",
  "energy_requirement": "ELECTRICITY",
  "concession_status": "NO",
  "life_support": "NO",
  "contact_preference": "EMAIL"
}
```

```json
{ "success": true, "submission_id": "SUB-1001", "status": "COMPLETED" }
```

Required fields are validated by Pydantic; `move_in_date` must be a real ISO date; enums are
constrained. Idempotency is scoped to the **call session** — re-submitting the same call returns
the same receipt, while a fresh call for the same lead gets its own (`SUB-1001`, then
`SUB-1001-2`, …) so re-running the demo never orphans a submission.

---

## 9. Database schema

The seven tables in the spec, plus one addition:

| Table | Notes |
|---|---|
| `leads` | Synthetic Energy leads. `dnc_status` gates dialling. |
| `call_sessions` | Adds `state` (state machine), `outcome_detail` (`BUSY_CALLBACK_SCHEDULED`, `DNC_REGISTER_HIT`) so the status enum stays clean, and `dial_provider` / `telephony_reference` for the adapter that placed the call. |
| `journey_fields` | One row per field. Adds `attempts` to enforce the two-strike rule. `source` is `PREEXISTING` / `CUSTOMER_SPOKEN` / `HUMAN_AGENT` — all three are actually written, so the audit trail shows who captured what. |
| `transcript_segments` | Adds `redacted` so the console can badge a redacted turn. |
| `handoffs` | Adds `last_customer_message`, `safety_flags_json`, `collected_fields_json`. |
| `audit_events` | Every meaningful transition. |
| `journey_submissions` | **Addition.** The mock endpoint's own receipt (submission id, payload, timestamp) so the Completed Journey screen and dashboard can show it directly instead of parsing it back out of the audit log. |

---

## 10. Project structure

```
energy-recovery-voice-agent/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI app, lifespan seeding, /health, /demo/reset
│   │   ├── config.py               Settings + capability report (no key required)
│   │   ├── database.py             Engine, session, SQLite FK pragma
│   │   ├── models.py               8 SQLAlchemy models
│   │   ├── schemas.py              Pydantic contracts
│   │   ├── seed.py                 Idempotent auto-seeding
│   │   ├── routers/                leads · calls · journey · dashboard
│   │   ├── services/               lead · dnc · script · voice · handoff · redaction
│   │   │                           · llm · journey · session · recording
│   │   ├── conversation/           state_machine · safety_engine · field_extractor · validators
│   │   └── data/                   energy_scripts.json · synthetic_leads.json
│   ├── scripts/                    demo_run.py · scenario_tests.py · mode_a_tests.py · recording_tests.py
│   │                               · provider_tests.py
│   ├── .env.example                Optional providers (copy to .env)
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── api/client.ts           Typed API client + WebSocket + presentation helpers
│       ├── lib/                    types.ts · speech.d.ts
│       ├── components/             StatusBadge · JourneyStepper · TranscriptViewer
│       │                           · VoiceControls · HandoffCard · JourneyFieldEditor
│       │                           · AppShell
│       └── pages/                  DashboardPage · RecoveryConsolePage
│                                   · HandoffPage · CompletedJourneyPage
├── docs/screenshots/               Verified console screens
└── README.md
```

Additions to the spec's file list: `journey_service.py` and `session_service.py` (the mock
submission and the session serialiser — each substantial enough to own a module), plus
`AppShell.tsx`, `JourneyFieldEditor.tsx`, `config.py`, `seed.py`, `speech.d.ts`, and the four
demo scripts.

---

## 11. Optional provider adapters — and which ones are worth signing up for

**You do not need any of them.** The demo runs end-to-end with no keys: the rules engine does
all extraction, the browser does speech in and out, and the warm handoff queues on screen.

If you *do* want a real provider, copy the template and fill in what you want:

```cmd
cd backend
copy .env.example .env
```

Then restart the backend. The startup banner prints exactly what it picked up:

```
Configuration file: backend/.env
LLM adapter: {'active_adapter': 'RULES_ONLY', 'enabled': False, ...}
Voice providers: {'stt': 'browser_web_speech', ...}
Server-side STT: none (the browser transcribes)
```

If your key does not show up there, it was not read. That is the fastest way to diagnose a
misconfigured `.env`.

### Which services actually have a free tier

| Service | Free tier | What it adds | Worth it? |
|---|---|---|---|
| **OpenAI** | Paid (no free tier) | LLM field extraction (`gpt-4o-mini` by default) | ✅ the default in `.env.example`. The same key also enables Whisper server-side STT as a fallback after Deepgram/AssemblyAI |
| **Groq** | Free, no credit card | LLM field extraction | ✅ best free alternative — fast, OpenAI-compatible (`LLM_BASE_URL=https://api.groq.com/openai/v1`) |
| **Google Gemini** | Free tier | Same, alternative vendor | ✅ if you prefer Google |
| **Deepgram** | $200 credit, no card, no expiry | Server-side STT | ✅ generous, and simplest to wire |
| **AssemblyAI** | $50 credit | Server-side STT, alternative | ➖ redundant if you have Deepgram |
| **OpenAI** | ❌ none (prepaid) | LLM + Whisper STT | ➖ only if you already have credit |
| **Twilio** | ~$15 trial credit | Real outbound dialling + warm transfer | ❌ trial is restricted — see below |

Notes on the two that need care:

- **Twilio's trial is genuinely limited.** Outbound calls only reach numbers you verify in the
  console, and you must buy an Australian number out of the trial credit. It is the only adapter
  that needs a paid number, so it is the least useful for a local demo. Without it the console
  drives the call locally and queues the handoff — which is what the demo actually shows.
- **There is no TTS provider, on purpose.** The browser speaks every agent turn for free. Adding
  a paid TTS vendor would be a dependency with no demo benefit, so `/health` reports
  `tts.external_provider: false` rather than pretending otherwise.

### Adapter matrix

| Adapter | Interface | Enable with |
|---|---|---|
| Any OpenAI-compatible LLM (Groq, Gemini, OpenRouter, DeepSeek, OpenAI) | `LLMAdapter` | `LLM_API_KEY` + `LLM_BASE_URL` + `LLM_MODEL` (`OPENAI_API_KEY` also accepted) |
| Deepgram STT | `STTProvider` | `DEEPGRAM_API_KEY` |
| AssemblyAI STT | `STTProvider` | `ASSEMBLYAI_API_KEY` |
| OpenAI Whisper STT | `STTProvider` | shares the LLM key |
| Twilio telephony | `TelephonyProvider` | `TWILIO_ACCOUNT_SID` + `TWILIO_AUTH_TOKEN` + `TWILIO_FROM_NUMBER` (+ `HANDOFF_TRANSFER_NUMBER` for warm transfer) |
| Browser Web Speech API | `STTProvider` / `TTSProvider` | always on — no key |
| Simulated turns | `STTProvider` | always on — no key |

Because the LLM adapter speaks the plain OpenAI Chat Completions wire format, pointing it at a
free provider is a base-URL and model-name swap — no code change.

### Server-side speech-to-text

`POST /calls/{id}/audio` accepts a **raw audio body** (no multipart, so no extra dependency),
transcribes it with the configured provider, and feeds the text into the same
`handle_utterance` pipeline as any other turn. It returns `503` with a fix-it message when no
provider is configured. The console exposes this as **Transcribe audio file** — useful in a
browser without the Web Speech API.

The transcript from a vendor is **untrusted input**. It is redacted, screened by the safety
engine, and validated by the Python validators exactly like something the customer typed — a
vendor cannot bypass the guardrails by returning a clever transcript. `provider_tests.py`
asserts this: a card number in a vendor transcript still escalates, and the PAN never reaches
the transcript.

Telephony is genuinely invoked: `start_call()` dials through `voice_providers.dial()` after the
DNC gate, and `_handoff()` calls `warm_transfer()`. Only the *provider* is optional — the call
path is always exercised, which is what makes the seam trustworthy.

### Live phone calls (Twilio)

On an agent-driven recovery, the console's **Call channel** panel has a **Start call by agent**
button. It makes the assistant phone the customer through Twilio and hold the real conversation.
Starting a recovery does *not* ring anyone by itself — the browser console is still the default,
and the button is the explicit step that reaches Twilio.

```
Start call by agent → POST /calls/{id}/dial ──► Twilio rings the customer
customer answers    → /twilio/voice/{id}   ──► we speak the disclosure (already in the transcript)
customer speaks     → /twilio/turn/{id}    ──► Twilio's speech recognition → the SAME state machine
                                               (safety engine, validators, 0.80 floor) → we speak its reply
handoff             → /twilio/handoff/{id} ──► customer waits in a conference (hold music);
                                               the human agent's phone is dialled, hears a short
                                               briefing, presses 1, and joins THE SAME call
```

Twilio does the listening and speaking; **the state machine still decides everything**. The agent
only ever says what the engine returns, and Twilio's transcript is untrusted input like any other.
Silence is a failed capture (three follow-ups, then a human). A customer hanging up mid-call
leaves the lead in the queue as *Recovery needed* with what was captured kept, so the next call
resumes rather than restarts. Console **End call** hangs up the phone; console **Handoff to human**
moves the live call into the conference.

**Set-up** (all in `backend/.env`; the template explains each one):

| Variable | What it is |
|---|---|
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | From the Twilio console |
| `TWILIO_FROM_NUMBER` | A Twilio number that can place calls (+E.164) |
| `PUBLIC_BASE_URL` | A public **https** address for this backend. Locally: `ngrok http 8000`, paste the URL it prints, restart the backend when it changes |
| `HANDOFF_TRANSFER_NUMBER` | The human agent's phone. Blank → a phone handoff ends with "we'll call you back". On a trial account it must be a **verified** number too, and a different phone from the customer's |
| `HANDOFF_RETRY_ATTEMPTS`, `HANDOFF_WAIT_SECONDS` | Agent busy/unanswered: the customer is told the specialist is busy, stays on hold, and the agent is rung again (3 rings, 40 s of hold each by default) before a callback is promised |
| `TWILIO_DIAL_OVERRIDE_NUMBER` | Testing: ring this number instead of the lead's. **Required for a trial account** — it can only call numbers you have verified, and the seeded leads have synthetic numbers |
| `TWILIO_SAY_VOICE`, `TWILIO_SPEECH_LANGUAGE` | Voice and recogniser language (`Polly.Nicole`, `en-AU`) |
| `TWILIO_VALIDATE_SIGNATURE` | Keep `true`. Every webhook is public and one can dial a phone; unsigned requests get a 403 |

The startup banner says `Phone calls: ON` or lists exactly which variable is missing. With nothing
set, the button is disabled and says what to add; nothing else changes.

**Guardrails that hold on the phone:** the Do-Not-Call register is re-checked immediately before
dialling and can not be overridden for a real call (even for a session started with the demo
override); card numbers are redacted and hand off; recording disclosure is the first thing spoken.

**What is and isn't verified.** `twilio_tests.py` drives a whole journey, a silence, a hang-up, a
handoff conference and the guardrails through the real webhooks with Twilio's REST API stubbed and
requests signed exactly as Twilio signs them (checked against Twilio's published test vector). It
cannot prove Twilio itself: speech-recognition accuracy on a real line, ring timing, or that Twilio
accepts every TwiML attribute. Make one real call to your own phone before relying on it. Not
handled: voicemail detection (an answering machine is treated as a customer), and a browser-based
softphone for the human agent (the agent joins from a phone).

### Uploading a call recording

**Upload recording** in the sidebar (or on the Recovery Queue) takes a finished call —
MP3, WAV, M4A, OGG, FLAC or WebM, up to 40 MB — and says whose call it was (an existing lead,
or a new one created on the spot). The backend transcribes it (`POST /recordings/upload`),
separates the agent from the customer where the STT vendor supports speaker labels
(Deepgram and AssemblyAI do), and checks the six journey fields:

| Result | What happens |
|---|---|
| All six found | The journey is submitted through the same mock endpoint → **Completed journeys** |
| Anything missing | The lead stays in the **recovery queue** as `Recovery needed`. The next recovery call resumes at the first missing step and does not re-ask what the recording already captured; a second upload can supply the rest |
| Customer declines | `DECLINED`, nothing submitted — same as a live call |
| Safety signal (life support, card data, frustration, asks for a person…) | `HANDOFF_REQUESTED`, nothing submitted — same as a live call |

Nothing here is new policy. Fields go through the same extractor, Python validators and 0.80
confidence floor as a live turn, every customer segment goes through the safety engine, and
card data is redacted before storage. Values already on file for the lead count towards the
checklist. Whether the recording disclosure was heard is reported but does not gate the result.

Recordings are transcribed with `punctuate` + `numerals` rather than `smart_format`:
`smart_format` rewrites a spoken date into US-order numerals ("3rd of April" → `04/03`),
which an Australian reader would take as 4 March. The audio itself is not stored, only the
transcript. `STT_UPLOAD_TIMEOUT_SECONDS` (default 180) bounds the transcription.

### The capability report does not lie

`GET /health` reports exactly which adapters are live. Every flag is backed by a call path that
exists: a provider is only reported available if something will actually call it. There is no
aspirational reporting, because a capability report that claims a provider is live when nothing
calls it makes the entire guardrail story untrustworthy.

### LLM safety contract

The LLM has exactly **one** job: propose a structured extraction for the field currently being
collected. It may **never** decide to continue, submit, decline, or hand off; invent customer
data; give advice; or see card data. Enforced three ways:

1. A system prompt that states the rules as critical failures.
2. `RulesOnlyLLM` returns `None` and everything still works.
3. Any LLM output is re-validated by `validators.py` and its confidence is capped by the
   model's own, so a hesitant model can never inflate a weak capture into a valid one.

**Why the interface is this narrow.** An earlier sketch of the adapter also carried
`rephrase()` (re-word an approved script line before speaking) and `label_intent()` (tag
sentiment for a QA log). Both were implemented; neither was ever called. They were removed
rather than wired up:

- The script copy is already written to be spoken aloud, and re-wording it would add a round
  trip to every agent turn — real latency in a voice agent, for copy that is already compliant.
- Intent classification already belongs to the deterministic safety engine. A second, opaque
  source of intent truth sitting beside the one that actually drives escalation is a liability
  in a regulated flow, not a feature.

`provider_tests.py` now asserts that every method on every adapter protocol has a caller in
`app/`, so this cannot silently regress. The single documented exception — `TTSProvider.synthesize`
— is listed in that test with its reason: it is an extension point, and `/health` reports
`tts.external_provider: false` rather than claiming it is live.

### Verified live (not just offline-tested)

The `provider_tests.py` suite uses `TestClient` + stubs so it runs without network access. The
following integrations were also exercised end-to-end against real credentials, so a regression
that breaks the wire format (header, body, response path) is caught here too — not only by the
offline tests:

- **Groq LLM** (`openai/gpt-oss-20b`, `api.groq.com/openai/v1`) — three live `extract_field`
  calls returned clean JSON, and the relative-date case correctly refused to invent a value
  (`value=null`, `needs_clarification=true`), proving the no-invention rule still holds with the
  real model.
- **Deepgram STT** (`api.deepgram.com/v1/listen`, `model=nova-2`, `language=en-AU`) — a 1-second
  silence WAV posted to `POST /calls/{id}/audio` returned the expected `422 empty transcript`,
  which means auth, upload, response parse, and the Python guardrail on empty transcripts all
  ran against the live service.

Neither check is automated in CI — both depend on outbound network and real keys. They are the
proof, not the regression net.

---

## 12. Non-goals

Not built, by design: real CRM or dialler integration, real payments, real financial or product
recommendations, cloud deployment, Docker/Kubernetes/CI, vector databases, RAG, LangGraph, model
training, and the other six CIMET verticals. The scope is one Energy journey, one mock
submission endpoint, one warm-handoff path, one local browser demo — done reliably.

---

## 13. Where the efficiency gain comes from

A human agent reads the script, types each answer, and re-keys it into the journey iframe.
At roughly 45 seconds per field of manual effort, each automated call that captures six fields
removes about 4.5 minutes of operator work — and it happens without the agent being on the call
at all. The dashboard reports fields auto-captured and the equivalent manual minutes avoided.

The quality gain matters more than the time: every field is validated before it is saved, every
guardrail fires deterministically, and when the agent is unsure it escalates to a human with the
context attached instead of guessing. That is what turns a recovery queue into something you can
let run unattended.
