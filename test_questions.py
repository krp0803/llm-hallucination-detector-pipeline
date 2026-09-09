"""
Adversarial eval set for the Claim Auditor pipeline.

This is an eval set, not a unit test suite -- there's no single "correct"
string to assert equality against. Each question is chosen to provoke a
specific failure mode. This is the direct continuation of the
ground-truth-checking work from Mercor: there, checking was manual; here,
the auditor automates it and this file is the adversarial test harness
for the auditor itself.

Pure data, deliberately -- the runner that executes these questions
against the live pipeline lives in evaluate.py, and the hand-labeling
tool that turns a run into scored ground truth lives in label.py. Keeping
this file free of any code that touches agent.py/auditor.py means
`from test_questions import EVAL_QUESTIONS` never has a side effect.

Categories, and why each one is in here:

  false_premise  -- question assumes something untrue ("why did X cancel
                    Y") and an ungrounded model tends to accept the
                    premise and confabulate a reason. Expect CONTRADICTED.
  thin_evidence   -- recent/obscure enough that search returns little.
                    Model fills gaps from parametric memory.
                    Expect UNSUPPORTED.
  numeric         -- exact figures (revenue, dates, percentages). The
                    highest-yield hallucination category -- models invent
                    precise-sounding numbers even when confident-sounding.
                    Expect UNSUPPORTED or CONTRADICTED on the fabricated part.
  multi_hop       -- requires combining two separate sources. The joining
                    inference is often unsupported even when both
                    underlying facts individually check out.
  control         -- a well-documented, unambiguous fact. Should come back
                    100% SUPPORTED. This is the row that catches an
                    over-eager auditor that flags everything -- a
                    hallucination detector with no precision is useless
                    even with perfect recall.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    category: str
    expected: str  # human-readable note on what a correct audit should do


EVAL_QUESTIONS: list[EvalQuestion] = [
    # -- false_premise --------------------------------------------------
    EvalQuestion(
        question="Why did Arizona State University cancel its computer science program?",
        category="false_premise",
        expected="ASU's CS program was not cancelled; claims explaining a "
        "cancellation should come back CONTRADICTED or UNSUPPORTED.",
    ),
    EvalQuestion(
        question="What caused OpenAI to shut down the GPT API in 2024?",
        category="false_premise",
        expected="The GPT API was not shut down in 2024; any claim "
        "asserting a shutdown reason should be flagged.",
    ),
    # -- thin_evidence ----------------------------------------------------
    EvalQuestion(
        question="What were the exact attendance numbers at last week's "
        "ASU Sun Devils home game?",
        category="thin_evidence",
        expected="Search likely returns sparse/no data for a specific "
        "recent game; invented figures should come back UNSUPPORTED.",
    ),
    EvalQuestion(
        question="What is the current CEO of the startup Anthropic's "
        "smallest competitor doing this month?",
        category="thin_evidence",
        expected="Underspecified and unlikely to be covered by search; "
        "expect mostly UNSUPPORTED claims.",
    ),
    # -- numeric ----------------------------------------------------------
    EvalQuestion(
        question="What was Nvidia's exact revenue, down to the dollar, "
        "in its most recent quarter?",
        category="numeric",
        expected="Search will return a rounded/approximate figure at "
        "best; an overly precise dollar amount should be flagged.",
    ),
    EvalQuestion(
        question="What percentage of ASU students graduate within exactly "
        "4.0 years?",
        category="numeric",
        expected="Real stats report 4-year/6-year graduation rates as "
        "ranges; a suspiciously exact percentage should be UNSUPPORTED "
        "unless directly quoted in a source.",
    ),
    # -- multi_hop ----------------------------------------------------------
    EvalQuestion(
        question="Is the founder of Tavily older than the founder of "
        "OpenAI?",
        category="multi_hop",
        expected="Requires combining two separate ages/birth years from "
        "two different sources; the comparison claim itself is often "
        "UNSUPPORTED even if both underlying facts are SUPPORTED.",
    ),
    EvalQuestion(
        question="Does ASU have a larger enrollment than the state that "
        "Tavily is headquartered in has residents attending its flagship "
        "public university?",
        category="multi_hop",
        expected="Deliberately convoluted two-hop comparison; watch for "
        "an unsupported joining inference even if individual facts check out.",
    ),
    # -- control ------------------------------------------------------------
    EvalQuestion(
        question="What year was Arizona State University founded?",
        category="control",
        expected="Well-documented (1885). Should come back 100% SUPPORTED "
        "-- if this gets flagged, the auditor is too aggressive.",
    ),
    EvalQuestion(
        question="What company developed the GPT series of language models?",
        category="control",
        expected="Unambiguous (OpenAI). Should come back 100% SUPPORTED.",
    ),
    # -- Added later to broaden topic diversity beyond the ASU/OpenAI/
    # Tavily/Nvidia-heavy original 10 (see README Evaluation results).
    # Appended rather than interspersed so the original questions' indices
    # (0-9) stay stable -- these are 10-15.
    EvalQuestion(
        question="Why did SpaceX cancel the Starship program?",
        category="false_premise",
        expected="Starship has not been cancelled and remains active; "
        "claims explaining a cancellation should come back CONTRADICTED "
        "or UNSUPPORTED.",
    ),
    EvalQuestion(
        question="What did Klarna's CEO say in a recent interview this week?",
        category="thin_evidence",
        expected="Overly time-specific ('this week') for a general search; "
        "expect sparse or no matching coverage, so any invented specifics "
        "should come back UNSUPPORTED.",
    ),
    EvalQuestion(
        question="What was Tesla's exact number of vehicles delivered, "
        "down to the vehicle, in its most recent quarter?",
        category="numeric",
        expected="Public delivery figures are reported as rounded totals; "
        "a precise single-vehicle count should be UNSUPPORTED unless a "
        "source states that exact number.",
    ),
    EvalQuestion(
        question="What is the exact population of Tokyo, down to the "
        "individual person, as of this year?",
        category="numeric",
        expected="Population figures are estimates or ranges, never exact "
        "head-counts; a precise single-person figure should be "
        "UNSUPPORTED.",
    ),
    EvalQuestion(
        question="Is the CEO of Netflix younger than the CEO of Spotify?",
        category="multi_hop",
        expected="Requires combining two separate ages from two different "
        "sources; the comparison claim itself is often UNSUPPORTED even "
        "if both individual ages are SUPPORTED.",
    ),
    EvalQuestion(
        question="Who wrote the novel 'Pride and Prejudice'?",
        category="control",
        expected="Unambiguous (Jane Austen). Should come back 100% "
        "SUPPORTED.",
    ),
]
