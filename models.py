"""
Shared data contracts for the Claim Auditor pipeline.

Nothing in agent.py or auditor.py should define its own ad-hoc dicts for
passing data around -- everything that crosses a function boundary is
modeled here, once, as a Pydantic class. Three payoffs:

1. FastAPI turns these into request/response validation and an interactive
   /docs page for free -- no separate schema to maintain.
2. Import direction becomes a design constraint we can actually enforce:
   agent.py and auditor.py both import FROM models.py, but never from each
   other. That's what keeps the auditor blind to the agent's reasoning.
3. Bugs where a field is silently missing or the wrong type turn into a
   validation error at the boundary, not a mystery three calls later.
"""

from enum import Enum

from pydantic import BaseModel


class SearchResult(BaseModel):
    """
    One normalized Tavily search hit.

    This -- not the agent's draft answer -- is the ground truth the auditor
    checks claims against. `content` is the raw snippet text Tavily
    returned; we never let the agent's paraphrase of it stand in for it.
    """

    title: str
    url: str
    content: str
    score: float


class AgentResult(BaseModel):
    """
    Output of the agent pass: a draft answer PLUS the evidence it was
    (supposedly) grounded in.

    Bundling `evidence` here instead of dropping it after drafting is the
    single most important modeling decision in this project. If the
    auditor only received `draft_answer`, it would have nothing to check
    the claims against except its own reading of the question -- which
    just reproduces whatever bias or gap produced the hallucination in the
    first place. The evidence has to survive the handoff.
    """

    question: str
    draft_answer: str
    evidence: list[SearchResult]


class Verdict(str, Enum):
    """
    Three outcomes, not a boolean.

    SUPPORTED     -- evidence contains this claim (and cites a quote proving it)
    UNSUPPORTED   -- evidence is silent; the model likely filled the gap
                     from parametric memory rather than the search results
    CONTRADICTED  -- evidence actively says something different

    Collapsing UNSUPPORTED and CONTRADICTED into a single "false" verdict
    would throw away the most useful diagnostic signal the auditor
    produces: "the model doesn't know" and "the model misread the source"
    are different bugs with different fixes.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class ClaimCheck(BaseModel):
    """
    The result of auditing a single atomic claim.

    `supporting_quote` is required to be a verbatim span from the evidence
    whenever verdict == SUPPORTED. This is deliberate: asserting "yes,
    supported" costs the model nothing, but producing a quote that
    actually appears in the source text is falsifiable -- a later stage
    can mechanically substring-check it. It turns the auditor's opinion
    into a citation.

    `quote_verified` is that mechanical check's result -- whether WE found
    `supporting_quote` inside the evidence, independent of the model's own
    verdict. It's not the model's self-report; a model asserting
    SUPPORTED with a quote that doesn't actually appear anywhere in the
    evidence gets `quote_verified=False` here, which is exactly the case
    this field exists to catch.
    """

    claim: str
    verdict: Verdict
    supporting_quote: str | None
    reasoning: str
    source_url: str | None
    quote_verified: bool


class AuditReport(BaseModel):
    """
    The full audit trail for one draft answer: every claim, its verdict,
    and the reconstructed answer built only from claims that survived.

    We keep `claims_total` / `claims_removed` as explicit counts (rather
    than making callers len() the checks list) because they're the numbers
    that turn into the eventual resume bullet -- e.g. "flagged N% of
    claims across an adversarial eval set."
    """

    checks: list[ClaimCheck]
    final_answer: str
    claims_total: int
    claims_removed: int


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str


class AskResponse(BaseModel):
    """
    Response body for POST /ask.

    Returns draft_answer, final_answer, AND the report together -- not
    just the cleaned-up final answer. The diff between draft and final,
    with the report explaining why each claim was kept or cut, is the
    actual demo. Hiding it behind a single "clean" string would throw away
    the part of the project worth showing an interviewer.
    """

    question: str
    draft_answer: str
    final_answer: str
    report: AuditReport
