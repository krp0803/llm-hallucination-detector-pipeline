"""
Re-verify: re-run ONLY the verification step against a frozen run's
existing claims and evidence -- not search, not draft, not decompose.

Exists so a fix to verify_claim() or _quote_is_present() (Stage 5) can be
measured against the exact same hand-labeled claims from a Stage 4 run,
without a new labeling pass. This is valid specifically because ground
truth -- "is this claim, as written, actually supported by its
evidence" -- doesn't depend on which system prediction it's being
compared against. Reusing the exact claim text is what makes the old
labels still apply to new predictions.

Deliberately does NOT call decompose() again: a fresh decomposition
could reword the claims, which would break the claim-text match against
the existing labels entirely -- metrics.py's mismatch guard would
(correctly) refuse to score that as nonsense, and it would force a whole
new labeling pass just to test a verification-only fix.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from auditor import verify_claim
from models import SearchResult, Verdict

EVALS_DIR = Path(__file__).parent / "evals"


async def _reverify_question(q: dict) -> dict:
    evidence = [SearchResult(**e) for e in q["evidence"]]
    claims = [check["claim"] for check in q["checks"]]

    checks = (
        await asyncio.gather(*(verify_claim(claim, evidence) for claim in claims))
        if claims
        else []
    )

    # Same survival rule as auditor.audit() -- kept in sync deliberately,
    # see evaluate.py's docstring for the same tradeoff.
    surviving = [
        c.claim for c in checks if c.verdict == Verdict.SUPPORTED and c.quote_verified
    ]
    final_answer = (
        " ".join(surviving)
        if surviving
        else "No claims in the draft could be verified against the sources."
    )

    new_q = dict(q)
    new_q["checks"] = [c.model_dump(mode="json") for c in checks]
    new_q["final_answer"] = final_answer
    new_q["claims_total"] = len(checks)
    new_q["claims_removed"] = len(checks) - len(surviving)
    return new_q


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python reverify.py evals/run_<id>.json")
        raise SystemExit(1)

    src_path = Path(sys.argv[1])
    run = json.loads(src_path.read_text(encoding="utf-8"))

    async def _run_all() -> list[dict]:
        return await asyncio.gather(*(_reverify_question(q) for q in run["questions"]))

    load_dotenv()
    new_questions = asyncio.run(_run_all())

    new_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-reverify"
    new_run = {
        "run_id": new_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reverified_from": run["run_id"],
        "questions": new_questions,
    }

    EVALS_DIR.mkdir(exist_ok=True)
    out_path = EVALS_DIR / f"run_{new_run_id}.json"
    out_path.write_text(json.dumps(new_run, indent=2), encoding="utf-8")

    # Carry the existing labels over under the new run_id. The labels
    # themselves are untouched -- same claim text, same human judgment --
    # only the run_id pointer changes, so metrics.py can score the new
    # predictions against them immediately.
    old_labels_path = EVALS_DIR / f"labels_{run['run_id']}.json"
    old_labels = json.loads(old_labels_path.read_text(encoding="utf-8"))
    new_labels = {"run_id": new_run_id, "labels": old_labels["labels"]}
    new_labels_path = EVALS_DIR / f"labels_{new_run_id}.json"
    new_labels_path.write_text(json.dumps(new_labels, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Carried over labels -> {new_labels_path}")


if __name__ == "__main__":
    main()
