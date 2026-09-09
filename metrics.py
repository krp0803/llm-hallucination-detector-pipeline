"""
Stage 4 metrics: precision/recall for the auditor as a detector of claims
that should be removed.

Ground truth is your hand label from label.py: was this claim genuinely
supported by the evidence retrieved for its question? The system's own
prediction is "flagged" whenever verdict != SUPPORTED or quote_verified
is False -- the quote check counts as part of the detector on purpose. A
claim that's actually true but whose cited quote failed the mechanical
substring check still gets removed by the real system, so it has to
count against precision here too, not get waved away as a technicality.

                     truly unsupported   truly supported
  system flagged            TP                 FP  <- removed a true claim
  system kept                FN <- hallucin-        TN
                                 ation got through

precision = TP / (TP + FP)   -- of what it removed, how much deserved it
recall    = TP / (TP + FN)   -- of what deserved removal, how much it caught

"na"-labeled claims (not checkable facts -- e.g. "$68,127 million equals
$68,127,000,000") are excluded from this matrix entirely and reported as
a separate count, since forcing them into supported/unsupported would
distort both numbers either way.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

EVALS_DIR = Path(__file__).parent / "evals"


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)


def is_flagged(check: dict) -> bool:
    """The system's own prediction: did it remove this claim from the final answer?"""
    return not (check["verdict"] == "supported" and check["quote_verified"])


def is_truly_unsupported(label: str) -> bool:
    """Ground truth: does this claim deserve removal? Caller excludes 'na' first."""
    return label in ("unsupported", "contradicted")


def compute_confusion(pairs: list[tuple[bool, bool]]) -> ConfusionMatrix:
    """
    pairs: (flagged, truly_unsupported) booleans, one per labeled non-'na'
    claim. Pure function, no I/O -- this is the piece that gets unit
    tested directly, on the same instinct as Stage 3's quote-checker
    test: a bug in this specific function would silently put a wrong
    number in front of an interviewer.
    """
    cm = ConfusionMatrix()
    for flagged, truly_unsupported in pairs:
        if flagged and truly_unsupported:
            cm.tp += 1
        elif flagged and not truly_unsupported:
            cm.fp += 1
        elif not flagged and truly_unsupported:
            cm.fn += 1
        else:
            cm.tn += 1
    return cm


def _load(run_path: Path) -> tuple[dict, dict]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    labels_path = EVALS_DIR / f"labels_{run['run_id']}.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"No labels file at {labels_path}. Run label.py first.")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if labels.get("run_id") != run["run_id"]:
        raise ValueError(
            f"{labels_path} holds labels for run {labels.get('run_id')!r}, "
            f"not {run['run_id']!r}. Refusing to mix runs."
        )
    return run, labels


def _iter_labeled_claims(run: dict, labels: dict):
    """
    Yield (question_dict, check_dict, label_str) for every labeled claim.

    Re-checks that the claim text stored in the labels file still matches
    the claim text in the run at that position -- belt-and-suspenders on
    top of the run_id check in _load(), in case a labels file was ever
    hand-edited or applied to a run that happens to share an id.
    """
    for qi, q in enumerate(run["questions"]):
        for ci, check in enumerate(q["checks"]):
            entry = labels["labels"].get(f"{qi}:{ci}")
            if entry is None:
                continue
            if entry["claim"] != check["claim"]:
                raise ValueError(
                    f"Label/run mismatch at {qi}:{ci}: label was recorded for "
                    f"claim {entry['claim']!r}, but the run has "
                    f"{check['claim']!r} there."
                )
            yield q, check, entry["label"]


def _fmt(x: float | None) -> str:
    return f"{x:.2f}" if x is not None else "n/a"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python metrics.py evals/run_<id>.json")
        raise SystemExit(1)

    run, labels = _load(Path(sys.argv[1]))

    overall_pairs: list[tuple[bool, bool]] = []
    category_pairs: dict[str, list[tuple[bool, bool]]] = {}
    na_count = 0
    unsupported_total = 0
    fp_quote_check_caused = 0
    labeled_total = 0

    for q, check, label in _iter_labeled_claims(run, labels):
        labeled_total += 1
        if label == "na":
            na_count += 1
            continue

        flagged = is_flagged(check)
        truly_unsupported = is_truly_unsupported(label)
        if truly_unsupported:
            unsupported_total += 1

        overall_pairs.append((flagged, truly_unsupported))
        category_pairs.setdefault(q["category"], []).append((flagged, truly_unsupported))

        if flagged and not truly_unsupported and check["verdict"] == "supported":
            # The model's own verdict agreed the claim was fine; only our
            # mechanical quote check disagreed and caused this removal.
            fp_quote_check_caused += 1

    checkable = len(overall_pairs)
    cm = compute_confusion(overall_pairs)

    print(f"Run: {run['run_id']}")
    print(f"Labeled claims: {labeled_total}  (checkable: {checkable}, n/a: {na_count})\n")

    print("Confusion matrix (detector = 'system removed this claim'):")
    print("                    truly unsupported   truly supported")
    print(f"  system flagged        {cm.tp:>4}                {cm.fp:>4}")
    print(f"  system kept           {cm.fn:>4}                {cm.tn:>4}")
    print()

    print(f"Precision: {_fmt(cm.precision)}   Recall: {_fmt(cm.recall)}   F1: {_fmt(cm.f1)}")
    print(f"(N = {checkable} checkable claims, {na_count} excluded as n/a)\n")

    if checkable:
        rate = unsupported_total / checkable
        print(
            f"Agent hallucination rate: {unsupported_total}/{checkable} "
            f"({rate:.0%}) of checkable claims were not genuinely supported "
            f"by the evidence retrieved for their question.\n"
        )

    if cm.fp:
        print(
            f"Of {cm.fp} false positive(s), {fp_quote_check_caused} were claims "
            f"the model itself verdicted SUPPORTED -- removed only because the "
            f"mechanical quote check disagreed.\n"
        )

    print("Per-category:")
    for cat, pairs in sorted(category_pairs.items()):
        sub_cm = compute_confusion(pairs)
        print(
            f"  {cat:<15} N={len(pairs):<3} "
            f"precision={_fmt(sub_cm.precision):<5} recall={_fmt(sub_cm.recall)}"
        )


if __name__ == "__main__":
    main()
