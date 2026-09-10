"""
FastAPI entry point. Wires the two passes together: POST /ask runs the
agent, then hands its output straight to the auditor, then returns both
the raw and cleaned answers plus the full audit trail.

load_dotenv() runs here because this is an entry point: the rule for the
whole project is that entry points configure the environment (this file,
test_questions.py) and library modules just consume it (agent.py,
auditor.py). agent.py's OpenAI/Tavily clients are constructed lazily on
first use rather than at import time, so -- unlike in Stage 1 -- import
order here no longer matters for correctness. load_dotenv() still goes
first on principle: env should be loaded before anything downstream might
read it.
"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402

from agent import run_agent  # noqa: E402
from auditor import audit  # noqa: E402
from models import AskRequest, AskResponse  # noqa: E402

app = FastAPI(
    title="LLM Hallucination Detection Pipeline",
    description="Two-pass pipeline that catches hallucinations in agent output.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check -- confirms the app booted and env vars loaded."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """
    Run both passes and return everything: draft, final, and the
    per-claim report.

    Returning only `final_answer` would hide the actual point of the
    project -- the draft-vs-final diff, backed by the report explaining
    why each claim was kept or cut, is what makes this demonstrable.
    """
    agent_result = await run_agent(request.question)
    report = await audit(agent_result)
    return AskResponse(
        question=request.question,
        draft_answer=agent_result.draft_answer,
        final_answer=report.final_answer,
        report=report,
    )
