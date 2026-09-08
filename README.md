# Claim Auditor

A two-pass AI pipeline that catches hallucinations in agent output before they
reach the user. One pass drafts an answer from live web search; a second,
independent pass breaks that draft into individual claims and checks each one
against the raw search results — flagging or removing anything the sources
don't actually support.

**Status:** Stages 1–3 complete (skeleton → agent → auditor). The pipeline is
fully working end-to-end. Stage 4 (running it against an adversarial eval set
and recording precision/recall) and Stage 5 (polish) are next — this README
will get real numbers added once Stage 4 is done.

## Why this exists

I worked at Mercor as an AI Training Specialist, checking LLM outputs against
ground truth for model training. This project automates the core of that job:
instead of a human comparing a model's claims to source material by hand, the
auditor pass does it, systematically, for every claim in every answer.

## How it works

```
question ──▶ [Agent]  Tavily search ──▶ evidence ──▶ GPT-5 Mini draft
                                            │
                                            ▼
                                       [Auditor]  decompose draft into
                                       atomic claims ──▶ verify each claim
                                       against evidence, independently ──▶
                                       rebuild final answer from only what
                                       survived
                                            │
                                            ▼
                        { draft_answer, final_answer, per-claim report }
```

The single rule the whole design is built around: **the auditor never sees
the agent's reasoning.** It gets one isolated claim and the raw evidence —
never the question, never the draft, never why the agent believed what it
believed. Sharing more than that would let the auditor inherit the agent's
assumptions and rubber-stamp its own mistakes.

## Design decisions worth knowing about

- **Evidence travels with the draft, not just the answer.** Auditing a
  summary against another summary verifies nothing — the auditor needs the
  original source text, not the agent's paraphrase of it.
- **Three verdicts, not a boolean.** `SUPPORTED`, `UNSUPPORTED` (evidence is
  silent), and `CONTRADICTED` (evidence says the opposite) are different
  failure modes with different causes — collapsing them into "false" throws
  away the diagnostic signal.
- **A `SUPPORTED` verdict requires a verbatim quote — and we check it
  ourselves.** The model can assert "yes, supported" for free, but it can't
  fabricate a quote that survives a mechanical substring search against the
  real evidence. We don't just trust the auditor's own citation — see the
  example below, where this caught the auditor being wrong about itself.
- **One LLM call per claim, run concurrently.** Batching all claims into one
  prompt is cheaper, but the model anchors on its own prior answers — after
  several `SUPPORTED` verdicts in a row it tends to agree with the next one
  on momentum rather than re-examining the evidence. Independent calls buy
  independent judgment; `asyncio.gather` claws back the latency that would
  otherwise cost, so N independent calls take roughly one call's wall time
  instead of N.
- **The final answer is a plain-Python join, not an LLM rewrite.** Rewriting
  the surviving claims into smoother prose would risk the rewrite
  introducing wording no verified claim actually supports — which would
  mean the rewrite itself needs auditing. Deterministic composition is what
  makes "everything in the final answer was checked" a guarantee instead of
  a hope. The cost is choppier prose, which is the right trade here.

## A concrete example

Asked for Nvidia's exact quarterly revenue "down to the dollar" — a prompt
designed to bait a model into manufacturing false precision — the agent
produced several claims. One was:

> **Claim:** "$68,127 million equals $68,127,000,000."
> **Model's own verdict:** `SUPPORTED`
> **Model's cited quote:** `"(In millions, except per share data)\n\nRevenue$68,127"`

The model was confident enough to mark this claim `SUPPORTED` and cite a
quote to back it up. But the auditor doesn't take that self-report at face
value: it independently searched the raw evidence text for that exact quote
(whitespace-normalized), and it wasn't there as a contiguous span — most
likely because the source's table layout puts other content between
"Revenue" and the dollar figure that a naive reading skips past.

Result: `quote_verified=False`, and the claim was silently excluded from the
final answer — even though the *model itself* said it was fine. This is the
project's core premise working exactly as intended: don't just check the
agent's claims, don't even fully trust the auditor's own citations either.
Verify mechanically, wherever mechanical verification is possible.

## Architecture

| File | Role |
|---|---|
| `models.py` | Shared Pydantic contracts — the only things that cross module boundaries |
| `prompts.py` | System prompts + evidence formatting, shared by both passes so they read evidence identically |
| `agent.py` | Pass 1 — Tavily search, GPT-5 Mini draft |
| `auditor.py` | Pass 2 — claim decomposition, independent per-claim verification, final answer rebuild |
| `main.py` | FastAPI app (`POST /ask`) wiring both passes together |
| `test_questions.py` | Adversarial eval set (false-premise, thin-evidence, numeric, multi-hop, control questions) |

`agent.py` and `auditor.py` both import from `models.py` and `prompts.py`,
but never from each other — that's the isolation rule enforced at the import
level, not just in a docstring.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in OPENAI_API_KEY and TAVILY_API_KEY
uvicorn main:app --reload
```

Then `POST /ask` with `{"question": "..."}`, or use the interactive docs at
`http://127.0.0.1:8000/docs`.

## Stack

Python · FastAPI · OpenAI (GPT-5 Mini, Responses API with structured output)
· Tavily Search API
