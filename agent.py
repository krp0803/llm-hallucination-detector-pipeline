"""
Pass 1: search the web, draft an answer grounded in what came back.

This is the half of the pipeline that's ALLOWED to hallucinate. We prompt
it to answer only from the evidence it was given, and it will still
occasionally invent a fact anyway -- that gap between "grounded prompt"
and "grounded output" is the entire reason auditor.py needs to exist. If
prompting alone were reliable, there'd be nothing to audit.

Clients are built lazily (see _openai/_tavily below), not at module
level. AsyncOpenAI() raises immediately if no API key is available --
confirmed by testing it directly against the installed SDK -- so eager
construction would make `import agent` crash for anyone who clones this
repo before setting up a .env. Lazy + cached gets the same "build once
per process" behavior without paying that cost at import time.
"""

from functools import lru_cache

from openai import AsyncOpenAI
from tavily import AsyncTavilyClient

from models import AgentResult, SearchResult
from prompts import AGENT_SYSTEM_PROMPT, format_evidence

MODEL = "gpt-5-mini"
SEARCH_DEPTH = "advanced"
MAX_RESULTS = 5


@lru_cache(maxsize=1)
def _openai() -> AsyncOpenAI:
    """
    Constructed on first use, not at import time -- see module docstring.
    lru_cache(maxsize=1) on a zero-argument function is just a memoized
    singleton: the first call builds the client, every call after returns
    the same instance.
    """
    return AsyncOpenAI()


@lru_cache(maxsize=1)
def _tavily() -> AsyncTavilyClient:
    return AsyncTavilyClient()


async def search(question: str) -> list[SearchResult]:
    """
    Query Tavily for `question` and normalize the response into our own
    SearchResult model.

    This function owns the ONLY copy of "what the evidence looked like"
    that the rest of the pipeline will ever see -- auditor.py never talks
    to Tavily directly. That keeps ground truth defined in exactly one
    place.
    """
    response = await _tavily().search(
        query=question,
        search_depth=SEARCH_DEPTH,
        max_results=MAX_RESULTS,
    )
    return [
        SearchResult(
            title=result.get("title", ""),
            url=result.get("url", ""),
            content=result.get("content", ""),
            score=result.get("score", 0.0),
        )
        for result in response.get("results", [])
    ]


async def draft_answer(question: str, evidence: list[SearchResult]) -> str:
    """
    Ask GPT-5 Mini to answer `question` using only `evidence`.

    AGENT_SYSTEM_PROMPT instructs the model to rely solely on the
    provided search results and to avoid filling gaps from its own
    training data. Expect it to do that imperfectly -- see module
    docstring.
    """
    user_input = f"Question: {question}\n\nSearch results:\n{format_evidence(evidence)}"
    response = await _openai().responses.create(
        model=MODEL,
        instructions=AGENT_SYSTEM_PROMPT,
        input=user_input,
    )
    return response.output_text.strip()


async def run_agent(question: str) -> AgentResult:
    """
    Orchestrate the agent pass: search, then draft, then bundle both into
    an AgentResult.

    Evidence is carried forward into the return value rather than
    discarded after drafting -- see AgentResult's docstring in models.py
    for why that's the load-bearing design decision of this whole project.

    Deliberately does NOT short-circuit when search returns no results.
    An obscure question that comes back with thin or empty evidence, gets
    drafted anyway, and ends up flagged wholesale as UNSUPPORTED by the
    auditor is the "thin_evidence" eval category working as intended --
    bailing out early here would hide that behavior instead of surfacing it.
    """
    evidence = await search(question)
    draft = await draft_answer(question, evidence)
    return AgentResult(question=question, draft_answer=draft, evidence=evidence)
