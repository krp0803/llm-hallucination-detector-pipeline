"""
Pass 1: search the web, draft an answer grounded in what came back.

This is the half of the pipeline that's ALLOWED to hallucinate. We prompt
it to answer only from the evidence it was given, and in Stage 2 it will
still occasionally invent a fact anyway -- that gap between "grounded
prompt" and "grounded output" is the entire reason auditor.py needs to
exist. If prompting alone were reliable, there'd be nothing to audit.

Client construction (OpenAI, Tavily) belongs at module level here, built
once at import time -- not inside these functions -- so a request doesn't
pay connection-setup cost on every call. That wiring lands in Stage 2
alongside the real logic; for now these are signatures only.
"""

from models import AgentResult, SearchResult


async def search(question: str) -> list[SearchResult]:
    """
    Query Tavily for `question` and normalize the response into our own
    SearchResult model.

    This function owns the ONLY copy of "what the evidence looked like"
    that the rest of the pipeline will ever see -- auditor.py never talks
    to Tavily directly. That keeps ground truth defined in exactly one
    place.
    """
    raise NotImplementedError


async def draft_answer(question: str, evidence: list[SearchResult]) -> str:
    """
    Ask GPT-5 Mini to answer `question` using only `evidence`.

    The prompt built here in Stage 2 will explicitly instruct the model to
    rely solely on the provided search results and to avoid filling gaps
    from its own training data. Expect it to do that imperfectly -- see
    module docstring.
    """
    raise NotImplementedError


async def run_agent(question: str) -> AgentResult:
    """
    Orchestrate the agent pass: search, then draft, then bundle both into
    an AgentResult.

    Evidence is carried forward into the return value rather than
    discarded after drafting -- see AgentResult's docstring in models.py
    for why that's the load-bearing design decision of this whole project.
    """
    raise NotImplementedError
