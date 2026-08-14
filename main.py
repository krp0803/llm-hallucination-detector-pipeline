"""
FastAPI entry point. Wires the two passes together: POST /ask runs the
agent, then hands its output straight to the auditor, then returns both
the raw and cleaned answers plus the full audit trail.

Env vars (OPENAI_API_KEY, TAVILY_API_KEY) load once at import time via
python-dotenv, before agent.py's module-level clients are constructed --
load_dotenv() has to run first or those clients would initialize with
missing keys.
"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from agent import run_agent
from auditor import audit
from models import AskRequest, AskResponse

app = FastAPI(
    title="Claim Auditor",
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
