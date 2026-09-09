"""
Interactive hand-labeling tool for a Stage 4 run artifact.

Ground truth has to come from a human -- that's the whole point of Stage
4, and not coincidentally the actual job this project automates: checking
model output against ground truth, the way Mercor's AI Training
Specialists do it by hand. This script shows one claim at a time and
records a judgment; metrics.py later compares that judgment against what
the system actually did.

The system's own verdict is deliberately withheld until AFTER a label is
recorded, for the same anchoring reason verify_claim() checks claims
independently instead of in a batch (see auditor.py): seeing "the system
said SUPPORTED" before judging it yourself biases you toward agreeing
with it, which would make the resulting metric measure agreement with
the auditor instead of correctness against the evidence.

Resumable: progress is written to the labels file after every single
claim, so quitting mid-session and re-running against the same run file
picks up exactly where you left off.
"""

import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).parent / "evals"

VALID_LABELS = {"s": "supported", "u": "unsupported", "c": "contradicted", "n": "na"}


def _claim_key(q_index: int, c_index: int) -> str:
    return f"{q_index}:{c_index}"


def _labels_path_for(run_id: str) -> Path:
    return EVALS_DIR / f"labels_{run_id}.json"


def _load_labels(labels_path: Path, run_id: str) -> dict:
    if not labels_path.exists():
        return {"run_id": run_id, "labels": {}}
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    if data.get("run_id") != run_id:
        raise ValueError(
            f"{labels_path} holds labels for run {data.get('run_id')!r}, "
            f"not {run_id!r}. Refusing to mix runs."
        )
    return data


def _save_labels(labels_path: Path, labels_data: dict) -> None:
    labels_path.write_text(json.dumps(labels_data, indent=2), encoding="utf-8")


def _flatten_claims(run: dict) -> list[tuple[int, int, dict, dict]]:
    """(q_index, c_index, question_dict, check_dict) for every claim in the run."""
    out = []
    for qi, q in enumerate(run["questions"]):
        for ci, check in enumerate(q["checks"]):
            out.append((qi, ci, q, check))
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python label.py evals/run_<id>.json")
        raise SystemExit(1)

    run_path = Path(sys.argv[1])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    labels_path = _labels_path_for(run["run_id"])
    labels_data = _load_labels(labels_path, run["run_id"])

    all_claims = _flatten_claims(run)
    total = len(all_claims)
    # Fixed at session start: claims not yet labeled when we launched.
    # "back" navigates within this list and can revisit/overwrite a claim
    # labeled earlier in THIS session -- it must not re-check labels_data
    # each iteration, or revisiting one would just skip past it again.
    remaining = [
        item for item in all_claims if _claim_key(item[0], item[1]) not in labels_data["labels"]
    ]
    already = total - len(remaining)

    print(f"Run {run['run_id']}: {total} claims, {already} already labeled.")
    print("Labels: [s]upported  [u]nsupported  [c]ontradicted  [n]/a (not a checkable claim)")
    print("Commands: back, skip, quit\n")

    i = 0
    while i < len(remaining):
        qi, ci, q, check = remaining[i]
        key = _claim_key(qi, ci)

        existing = labels_data["labels"].get(key)
        marker = f" (currently: {existing['label']})" if existing else ""
        print(f"--- Claim {already + i + 1} of {total}  [{q['category']}]{marker} ---")
        print(f"Question: {q['question']}")
        print(f"Claim:    {check['claim']}")
        print("Evidence:")
        for n, ev in enumerate(q["evidence"], start=1):
            print(f"  [{n}] {ev['title']} ({ev['url']})")
            # Full content, not a truncated snippet: a 300-char preview
            # (the original version of this tool) can cut off before the
            # actual relevant sentence in a long page, forcing a manual
            # trip to the source URL to find it -- exactly what happened
            # during the first real labeling pass on this eval set.
            for line in ev["content"].splitlines():
                print(f"      {line}")
        print()

        raw = input("Your label (s/u/c/n, or back/skip/quit): ").strip().lower()

        if raw == "quit":
            break
        if raw == "skip":
            i += 1
            continue
        if raw == "back":
            i = max(0, i - 1)
            continue
        if raw not in VALID_LABELS:
            print(f"'{raw}' not recognized -- try s/u/c/n/back/skip/quit.\n")
            continue

        labels_data["labels"][key] = {
            "claim": check["claim"],
            "label": VALID_LABELS[raw],
        }
        _save_labels(labels_path, labels_data)

        # Only now, after the label is locked in, reveal what the system did.
        print(f"  (system said: {check['verdict']}, quote_verified={check['quote_verified']})\n")
        i += 1

    labeled_now = sum(
        1 for qi, ci, _, _ in all_claims if _claim_key(qi, ci) in labels_data["labels"]
    )
    print(f"\nSaved. {labeled_now}/{total} claims labeled -> {labels_path}")


if __name__ == "__main__":
    main()
