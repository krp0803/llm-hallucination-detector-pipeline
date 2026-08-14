"""
Pass 2: decompose the draft into atomic claims, verify each one against
the raw evidence, rebuild a final answer from what survives.

The rule this whole file enforces: the auditor never sees the agent's
reasoning, question phrasing, or chain of thought -- only one isolated
claim at a time plus the evidence list. Sharing more context than that
would let the auditor inherit the agent's assumptions and rubber-stamp
its own mistakes. Concretely, that means verify_claim() takes a bare
`str` claim, not the AgentResult, and never sees `draft_answer` at all.
"""

import asyncio

from models import AgentResult, AuditReport, ClaimCheck, SearchResult


async def decompose(draft: str) -> list[str]:
    """
    Split `draft` into atomic, self-contained claims -- one factual
    assertion per string, no pronouns or dangling references.

    Sentence-level checking isn't granular enough: "Acme, founded in
    1997, employs 400 people" can be correct about the year and invented
    about the headcount. Splitting to one assertion per claim gives
    per-fact resolution.

    The "self-contained" part matters just as much as the splitting.
    Because verify_claim() sees each claim in isolation (see module
    docstring), "it was founded in 1997" is unverifiable on its own --
    this function has to resolve it to "Acme was founded in 1997" before
    handing it off. Expect this prompt to need the most iteration of
    anything in the project.
    """
    raise NotImplementedError


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

    A SUPPORTED verdict must come with a verbatim `supporting_quote`
    lifted from `evidence` -- see ClaimCheck in models.py for why that
    requirement is there.
    """
    raise NotImplementedError


async def audit(agent_result: AgentResult) -> AuditReport:
    """
    Run the full auditor pass: decompose the draft, verify every claim
    concurrently, and rebuild a final answer from the claims that
    survived.

    `asyncio.gather` (not a loop of sequential awaits) is what makes
    "one LLM call per claim" affordable -- N independent calls in
    parallel cost roughly one call's wall-clock time instead of N.

    "Rebuild" is deliberate phrasing: the final answer is composed fresh
    from SUPPORTED claims, not produced by deleting spans out of
    `draft_answer`. Excising text from the original would leave broken
    grammar and dangling references behind.
    """
    raise NotImplementedError
