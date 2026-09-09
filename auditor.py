"""
Pass 2: decompose the draft into atomic claims, verify each one against
the raw evidence, rebuild a final answer from what survives.

The rule this whole file enforces: the auditor never sees the agent's
reasoning, question phrasing, or chain of thought -- only one isolated
claim at a time plus the evidence list. Sharing more context than that
would let the auditor inherit the agent's assumptions and rubber-stamp
its own mistakes. Concretely, that means verify_claim() takes a bare
`str` claim, not the AgentResult, and never sees `draft_answer` at all.

decompose() is the one exception, and on purpose: it takes the question
too, because resolving a pronoun ("it was founded in 1885") into a
self-contained claim ("ASU was founded in 1885") requires knowing what
"it" refers to. That's context that helps make a claim CHECKABLE, not
context that could bias the CHECK -- which is why it stops at decompose()
and never reaches verify_claim().
"""

import asyncio
import re
from difflib import SequenceMatcher
from functools import lru_cache

from openai import AsyncOpenAI
from pydantic import BaseModel

from models import AgentResult, AuditReport, ClaimCheck, SearchResult, Verdict
from prompts import (
    DECOMPOSE_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    format_evidence,
    source_text,
)

MODEL = "gpt-5-mini"

# Separate from MODEL (used by decompose()) so verify_claim()'s model can
# be tuned independently of the drafting model. Tested against the full
# gpt-5 model as a diagnostic for the confirmed pronoun-resolution miss
# (Stage 5): gpt-5 reached the identical verdict on the identical claim,
# which is evidence the miss is genuine ambiguity in the source text, not
# a gpt-5-mini capability gap -- so gpt-5-mini stays, at roughly 1/8th
# the measured cost for the same accuracy on this eval set.
VERIFY_MODEL = "gpt-5-mini"

# Cost ceiling: verify_claim() below is one LLM call per claim, fanned
# out concurrently. Without a cap, a sufficiently rambling draft could
# turn one /ask request into an unbounded number of paid calls.
MAX_CLAIMS = 25

# Quote-matching tuning -- see _quote_is_present() for what these gate.
# Both values were chosen empirically against the real Stage 4 labeled
# dataset, not guessed: every confirmed-bad quote scored 1.000 coverage
# once the haystack included the title, with wide margin above 0.9.
QUOTE_MATCH_THRESHOLD = 0.9
MIN_FUZZY_QUOTE_LENGTH = 20


# -- Local response envelopes -------------------------------------------
#
# These two models are wire formats for a single LLM call -- they never
# leave this file. That's different from models.py, which holds
# contracts that cross module boundaries (agent.py <-> main.py <->
# auditor.py). Not every Pydantic model is a shared contract; keeping
# these local keeps models.py from turning into a dumping ground for
# call-shaped structs that only auditor.py ever touches.


class ClaimList(BaseModel):
    """decompose()'s structured-output envelope."""

    claims: list[str]


class VerificationResult(BaseModel):
    """
    verify_claim()'s structured-output envelope.

    Deliberately NOT the same shape as ClaimCheck in models.py, and that
    gap is the point:
    - It omits `claim`. We already have the claim text; asking the model
      to echo it back risks it coming back subtly reworded, which would
      make the final report display a claim the agent never actually made.
    - It omits `quote_verified`. That field is OUR mechanical check of
      the model's own citation -- letting the model self-report it would
      defeat the reason it exists.
    ClaimCheck gets assembled afterward from: the claim we already had +
    this result + our own independent quote check.
    """

    verdict: Verdict
    supporting_quote: str | None
    reasoning: str
    source_url: str | None


@lru_cache(maxsize=1)
def _openai() -> AsyncOpenAI:
    """
    Lazy + cached, same reasoning as agent.py's client: AsyncOpenAI()
    raises immediately with no API key, so building it at import time
    would break `import auditor` for anyone without a .env.
    """
    return AsyncOpenAI()


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase, for tolerant substring matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _fuzzy_coverage(needle: str, haystack: str) -> float:
    """
    What fraction of `needle`'s characters can be matched, in order
    (allowing gaps), somewhere in `haystack`.

    Deliberately NOT SequenceMatcher.ratio(): ratio() is 2*matched /
    (len(needle)+len(haystack)), which tanks for a short quote against a
    long page purely because the page is long, not because the quote is
    fake. Dividing by len(needle) alone measures the thing that actually
    matters here -- how much of the (possibly short, possibly gappy)
    quote is genuinely findable in the source -- independent of how much
    other text surrounds it.
    """
    if not needle:
        return 0.0
    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(needle)


def _quote_is_present(quote: str | None, evidence: list[SearchResult]) -> bool:
    """
    Mechanically check whether `quote` actually appears in the evidence,
    independent of whatever the model claimed. This is the second half
    of the "citation, not opinion" design in ClaimCheck: the model can be
    made to cite a quote, but it can't be made to cite one that survives
    this check unless the quote is real.

    Checked against source_text() -- title, url, AND content -- not just
    content alone. A Stage 4 measurement found the auditor flagging a
    perfectly good claim because the model quoted a source's title
    (which format_evidence() shows it) and the checker only ever searched
    the body text, so the citation could never be found.

    Below MIN_FUZZY_QUOTE_LENGTH, falls back to an exact whitespace-
    normalized substring check -- fuzzy coverage is meaningless for a
    very short quote (a handful of characters can trivially "cover"
    almost anything), so this floor exists specifically so the fix below
    doesn't introduce a new way to rubber-stamp a fabricated citation.

    At or above that length, use fuzzy coverage (see _fuzzy_coverage())
    rather than an exact contiguous substring. Stage 4 found two real
    but non-exact quotes get unfairly flagged this way: one skipped a
    row in a table (splicing two real, non-adjacent lines together), one
    silently dropped a few words mid-sentence with no ellipsis. In both
    cases every character the model quoted was real and in the right
    order -- just not perfectly contiguous -- so coverage scores 1.0
    while a strict substring check scores 0. QUOTE_MATCH_THRESHOLD=0.9
    was chosen with margin: every confirmed-good case scores 1.000, and
    the confirmed-bad case (checked without the title fix) scored 0.589.
    """
    if not quote:
        return False
    needle = _normalize(quote)
    if len(needle) < MIN_FUZZY_QUOTE_LENGTH:
        return any(needle in _normalize(source_text(result)) for result in evidence)
    return any(
        _fuzzy_coverage(needle, _normalize(source_text(result))) >= QUOTE_MATCH_THRESHOLD
        for result in evidence
    )


async def decompose(question: str, draft: str) -> list[str]:
    """
    Split `draft` into atomic, self-contained claims -- one factual
    assertion per string, no pronouns or dangling references, and no
    meta-commentary about the sources themselves (see
    DECOMPOSE_SYSTEM_PROMPT for why that last part matters).

    Sentence-level checking isn't granular enough: "Acme, founded in
    1997, employs 400 people" can be correct about the year and invented
    about the headcount. Splitting to one assertion per claim gives
    per-fact resolution.
    """
    user_input = f"Question: {question}\n\nDraft answer:\n{draft}"
    response = await _openai().responses.parse(
        model=MODEL,
        instructions=DECOMPOSE_SYSTEM_PROMPT,
        input=user_input,
        text_format=ClaimList,
    )
    claims = response.output_parsed.claims
    return claims[:MAX_CLAIMS]


async def verify_claim(claim: str, evidence: list[SearchResult]) -> ClaimCheck:
    """
    Check a single claim against the evidence and return a verdict.

    Called once per claim rather than batching all claims into one prompt.
    Batching is cheaper, but the model anchors on its own prior answers --
    after five SUPPORTED verdicts in a row it tends to agree with the
    sixth on momentum rather than re-examining the evidence. Independent
    calls buy independent judgments, at the cost of N calls' worth of
    latency, which is why audit() below fans these out concurrently
    instead of awaiting them one at a time.

    The model's verdict is never taken at face value: `supporting_quote`
    gets mechanically checked against the raw evidence text below, and a
    `source_url` that doesn't match any actual evidence URL is dropped
    rather than passed through as a fabricated citation.
    """
    user_input = f"Claim: {claim}\n\nSearch results:\n{format_evidence(evidence)}"
    response = await _openai().responses.parse(
        model=VERIFY_MODEL,
        instructions=VERIFY_SYSTEM_PROMPT,
        input=user_input,
        text_format=VerificationResult,
    )
    result = response.output_parsed

    quote_verified = _quote_is_present(result.supporting_quote, evidence)

    valid_urls = {source.url for source in evidence}
    source_url = result.source_url if result.source_url in valid_urls else None

    return ClaimCheck(
        claim=claim,
        verdict=result.verdict,
        supporting_quote=result.supporting_quote,
        reasoning=result.reasoning,
        source_url=source_url,
        quote_verified=quote_verified,
    )


async def audit(agent_result: AgentResult) -> AuditReport:
    """
    Run the full auditor pass: decompose the draft, verify every claim
    concurrently, and rebuild a final answer from the claims that
    survived.

    `asyncio.gather` (not a loop of sequential awaits) is what makes
    "one LLM call per claim" affordable -- N independent calls in
    parallel cost roughly one call's wall-clock time instead of N.

    "Rebuild" is deliberate phrasing: the final answer is a plain-Python
    join of surviving claim strings, not a third LLM call asked to smooth
    them into prose. An LLM rewrite could introduce wording that no
    verified claim actually supports -- which would mean the rewrite
    itself needs auditing, an infinite regress. Deterministic composition
    is what makes "everything in the final answer was verified" an actual
    guarantee instead of a hope. The cost is choppier prose; that's the
    right trade for this project.
    """
    claims = await decompose(agent_result.question, agent_result.draft_answer)

    if not claims:
        return AuditReport(
            checks=[],
            final_answer="No claims in the draft could be verified against the sources.",
            claims_total=0,
            claims_removed=0,
        )

    checks = await asyncio.gather(
        *(verify_claim(claim, agent_result.evidence) for claim in claims)
    )

    surviving = [
        check.claim
        for check in checks
        if check.verdict == Verdict.SUPPORTED and check.quote_verified
    ]
    final_answer = (
        " ".join(surviving)
        if surviving
        else "No claims in the draft could be verified against the sources."
    )

    return AuditReport(
        checks=list(checks),
        final_answer=final_answer,
        claims_total=len(checks),
        claims_removed=len(checks) - len(surviving),
    )
