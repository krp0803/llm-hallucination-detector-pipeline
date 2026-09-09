"""
Stage 4 evaluation runner.

Runs every question in EVAL_QUESTIONS through the live pipeline and
writes a frozen JSON snapshot to evals/run_<id>.json. This snapshot --
not "the pipeline" -- is what gets hand-labeled (label.py) and scored
(metrics.py). Tavily results and model outputs vary run to run
(confirmed directly: the same question returned different evidence and a
different claim count minutes apart during Stage 3 testing), so ground
truth has to attach to one frozen run, not to live behavior that keeps
moving underneath it.

Calls agent.search()/draft_answer() and auditor.decompose()/verify_claim()
directly rather than the composed run_agent()/audit() wrappers, purely to
record per-phase timing (search vs draft, decompose vs the verify
fan-out) -- which puts an actual number on what Stage 3's asyncio.gather
decision bought. The final-answer composition below mirrors what
auditor.audit() does internally; if that logic changes, this needs to
stay in sync with it.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent import draft_answer, search
from auditor import decompose, verify_claim
from models import AgentResult, ClaimCheck, Verdict
from test_questions import EVAL_QUESTIONS, EvalQuestion

EVALS_DIR = Path(__file__).parent / "evals"


async def _run_question(eval_q: EvalQuestion) -> dict:
    """Run one question through the full pipeline with per-phase timing."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    evidence = await search(eval_q.question)
    timings["search"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    draft = await draft_answer(eval_q.question, evidence)
    timings["draft"] = time.perf_counter() - t0

    agent_result = AgentResult(
        question=eval_q.question, draft_answer=draft, evidence=evidence
    )

    t0 = time.perf_counter()
    claims = await decompose(agent_result.question, agent_result.draft_answer)
    timings["decompose"] = time.perf_counter() - t0

    async def _timed_verify(claim: str) -> tuple[ClaimCheck, float]:
        start = time.perf_counter()
        check = await verify_claim(claim, agent_result.evidence)
        return check, time.perf_counter() - start

    t0 = time.perf_counter()
    timed_results = (
        await asyncio.gather(*(_timed_verify(c) for c in claims)) if claims else []
    )
    timings["verify_fanout_wall"] = time.perf_counter() - t0
    timings["verify_call_durations"] = [d for _, d in timed_results]
    timings["verify_call_sum"] = sum(d for _, d in timed_results)

    # Same survival rule as auditor.audit(): a claim only makes the final
    # answer if the model said SUPPORTED *and* our own quote check agreed.
    checks = [check for check, _ in timed_results]
    surviving = [
        c.claim for c in checks if c.verdict == Verdict.SUPPORTED and c.quote_verified
    ]
    final_answer = (
        " ".join(surviving)
        if surviving
        else "No claims in the draft could be verified against the sources."
    )

    return {
        "question": eval_q.question,
        "category": eval_q.category,
        "expected": eval_q.expected,
        "evidence": [e.model_dump(mode="json") for e in evidence],
        "draft_answer": draft,
        "final_answer": final_answer,
        "claims_total": len(checks),
        "claims_removed": len(checks) - len(surviving),
        "checks": [c.model_dump(mode="json") for c in checks],
        "timings": timings,
    }


async def _run_all(questions: list[EvalQuestion]) -> list[dict]:
    results = []
    for i, eval_q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {eval_q.category}: {eval_q.question}")
        results.append(await _run_question(eval_q))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 4 eval set.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N questions."
    )
    parser.add_argument(
        "--question",
        type=int,
        nargs="+",
        default=None,
        help="Run only the question(s) at these indices (0-based) in "
        "EVAL_QUESTIONS, e.g. --question 10 11 12. All land in one run "
        "artifact together.",
    )
    args = parser.parse_args()

    if args.question is not None:
        questions = [EVAL_QUESTIONS[i] for i in args.question]
    elif args.limit is not None:
        questions = EVAL_QUESTIONS[: args.limit]
    else:
        questions = EVAL_QUESTIONS

    # Entry point -- loads its own env, same rule as main.py.
    load_dotenv()
    question_results = asyncio.run(_run_all(questions))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "questions": question_results,
    }

    EVALS_DIR.mkdir(exist_ok=True)
    out_path = EVALS_DIR / f"run_{run_id}.json"
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path} ({len(question_results)} questions)")


if __name__ == "__main__":
    main()
