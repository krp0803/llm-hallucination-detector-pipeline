# Claim Auditor

A two-pass AI pipeline that catches hallucinations in agent output before they
reach the user. One pass drafts an answer from live web search; a second,
independent pass breaks that draft into individual claims and checks each one
against the raw search results — flagging or removing anything the sources
don't actually support.

**Status:** Stages 1–4 complete. The pipeline works end-to-end and has been
measured against a hand-labeled adversarial eval set (see
[Evaluation results](#evaluation-results-stage-4) below) — precision and
recall are currently mediocre, but every major error has a confirmed,
specific cause, which is what Stage 5 targets next.

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
  otherwise cost. Measured across the Stage 4 eval run (44 claims, 10
  questions): verifying sequentially would have taken 368s of combined call
  time; running concurrently took 103s wall time — a **3.6x** speedup for
  free, just from not awaiting one at a time.
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

## Evaluation results (Stage 4)

The eval set in `test_questions.py` (10 adversarial questions, 5 categories)
was run once through the full pipeline and frozen as a snapshot
(`evals/run_20260908T030703Z.json`), producing 44 claims. Every claim was
then hand-labeled — genuinely `supported` / `unsupported` / `contradicted` /
`n/a` (not a checkable factual claim) — against its actual retrieved
evidence, with the auditor's own verdict hidden until after each label was
recorded, to avoid anchoring the judgment on it. This mirrors the actual
Mercor workflow this project automates, just applied to the auditor itself
instead of to a model being trained.

**Framing:** the auditor is a detector for "claims that should be removed."
It flags a claim whenever `verdict != SUPPORTED OR quote_verified == False` —
the mechanical quote check counts as part of the detector, since a
genuinely-true claim whose citation fails that check gets removed by the
real system regardless.

|                    | truly unsupported | truly supported |
|--------------------|:---:|:---:|
| **system flagged** | 2 (TP) | 5 (FP) |
| **system kept**    | 3 (FN) | 34 (TN) |

**Precision: 0.29 · Recall: 0.40 · F1: 0.33**  (N = 44 checkable claims, 0 excluded as n/a)

Agent hallucination rate: 5/44 (11%) of claims produced by the agent weren't
genuinely supported by the evidence retrieved for their question — a modest,
honest number, not an inflated one. The interesting part isn't the headline
score, though; it's *why* it landed there, since both sides of the confusion
matrix trace to specific, confirmed causes rather than vague model error:

### The 5 false positives (good claims wrongly removed)

Four of five trace directly to bugs in the mechanical quote-checker, each
confirmed by inspecting the actual evidence text:

1. **Citing the rendered `[n] Title (url)` header instead of body text.** For
   the claim *"OpenAI is a well-known rival of Anthropic,"* the model's
   "quote" was literally `OpenAI focuses on business users amid competition
   with rival Anthropic | PBS News (https://...)` — the citation line
   `format_evidence()` renders above each snippet. `_quote_is_present()`
   only searches `SearchResult.content`, so a quote lifted from the header
   can never match. Model judgment was fine; the checker was blind to where
   the text actually came from.
2. **Splicing two non-adjacent lines into one fake-contiguous quote.** For a
   Sun Devils attendance claim, the source table has a `2025: 381,109...`
   row sitting between the header and the `2024: 293,901...` row the model
   cited — it skipped that row and presented the two as one contiguous
   span. Every word is real; the contiguity isn't.
3. **Silently dropping words mid-quote, no ellipsis.** A quote about
   Anthropic's stance on open-weight models was accurate except it dropped
   "as a category," from the middle — a sneakier version of the
   "paraphrased quote" risk flagged back in Stage 3, since there wasn't
   even an ellipsis to signal the cut.

The fifth was a genuine model-judgment disagreement, not a checker bug — the
model marked a claim `UNSUPPORTED` that the human labeler considered
reasonable to infer from the evidence.

### The 3 false negatives (real hallucinations that got through)

One has a confirmed, specific cause: the claim *"Many GPT-3 and GPT-3.5
models were scheduled to be shut off on January 4, 2024"* was verified
`SUPPORTED` against a source reading *"...older GPT-3 and GPT-3.5 models...
We also announced the upcoming retirement of our first-generation text
embedding models. **They** will be shut down on January 04, 2024."*
Grammatically, "They" refers to the embedding models mentioned immediately
before it, not the GPT-3.5 models two sentences back — the verifier resolved
an ambiguous pronoun toward the reading that happened to confirm the claim
it was checking. A real judgment error, not a quoting one.

The other two are less clear-cut. They're plausibly connected to the
project's own isolation rule: `verify_claim()` never sees the question, on
purpose, so it structurally cannot recognize when a textually-accurate claim
is being used to imply support for a false premise (both came from
false-premise questions). That's a real architectural tradeoff, not
obviously a bug to patch. It's also possible some labeling noise is mixed
in here — the labeling tool originally truncated evidence to 300 characters,
which sent the labeler to the actual source URLs for some claims to find
the relevant text; that's now fixed (`label.py` shows full content), but
this run's labels predate the fix.

### Per-category

| Category | N | Precision | Recall |
|---|---|---|---|
| control | 5 | n/a | n/a |
| false_premise | 14 | n/a | 0.00 |
| multi_hop | 5 | n/a | n/a |
| numeric | 6 | n/a | n/a |
| thin_evidence | 14 | 0.29 | 0.67 |

`n/a` means the category had no claims on that side of the ratio (no false
positives, or no genuinely-unsupported claims to catch) — most of the
signal in this run happened to land in `thin_evidence`. Notably, zero false
positives came from the `control` category, which is the one specifically
designed to catch an over-aggressive auditor mangling clean, well-documented
answers.

### What this means for Stage 5

Every finding above is a concrete lever, not a vague "improve accuracy":
check quotes against titles too, tolerate a bounded number of dropped words
or a small line-gap instead of demanding a perfect contiguous match, and add
an explicit instruction against resolving ambiguous references in the
claim's favor. The two possibly-architectural false negatives are a genuine
open question rather than a bug — worth a second labeled run with the fixed
tooling before deciding whether they're a pattern or noise.

## Architecture

| File | Role |
|---|---|
| `models.py` | Shared Pydantic contracts — the only things that cross module boundaries |
| `prompts.py` | System prompts + evidence formatting, shared by both passes so they read evidence identically |
| `agent.py` | Pass 1 — Tavily search, GPT-5 Mini draft |
| `auditor.py` | Pass 2 — claim decomposition, independent per-claim verification, final answer rebuild |
| `main.py` | FastAPI app (`POST /ask`) wiring both passes together |
| `test_questions.py` | Adversarial eval set data (false-premise, thin-evidence, numeric, multi-hop, control questions) |
| `evaluate.py` | Runs the eval set through the live pipeline, freezes the result as a timestamped JSON snapshot |
| `label.py` | Interactive hand-labeling CLI over a frozen snapshot — the auditor's own verdict is hidden until after each label is recorded |
| `metrics.py` | Precision/recall/F1 + per-category breakdown from a snapshot and its labels |
| `evals/` | Committed run snapshots and hand labels — the actual evidence behind the numbers above |

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

**Reproducing the evaluation:**

```bash
python evaluate.py              # runs the eval set, writes evals/run_<id>.json
python label.py evals/run_<id>.json   # hand-label the claims (resumable)
python metrics.py evals/run_<id>.json # precision/recall/F1 report
```

## Stack

Python · FastAPI · OpenAI (GPT-5 Mini, Responses API with structured output)
· Tavily Search API
