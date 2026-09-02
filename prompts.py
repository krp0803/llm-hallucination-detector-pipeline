"""
Prompts and evidence formatting shared across the pipeline.

This exists as its own module -- not folded into agent.py -- for one
reason: the auditor has to read evidence formatted the exact same way the
agent did, or "claim isn't supported" and "claim isn't supported in the
format I checked" become impossible to tell apart. If format_evidence()
lived in agent.py, auditor.py would need `from agent import
format_evidence`, which is precisely the import the whole project is
built to avoid -- see the module docstrings in models.py and auditor.py.
A third module both sides depend on keeps that boundary intact.

This is also, realistically, the file that gets iterated on the most.
Prompt wording is where the actual behavior of both passes lives.
"""

from models import SearchResult

AGENT_SYSTEM_PROMPT = """\
You are a research assistant. You will be given a question and a numbered \
list of search results. Answer the question using ONLY information found \
in those search results.

Rules:
- Do not use any outside knowledge, even if you're confident it's correct.
- If the search results don't fully answer the question, say what they do \
cover and be explicit about what's missing. Don't fill the gap yourself.
- Be concise. Answer in a few sentences, not a report.
- Don't mention "search results" or cite source numbers in your answer -- \
write it as a direct answer to the question, in your own words.
"""


def format_evidence(evidence: list[SearchResult]) -> str:
    """
    Render search results as a numbered list for a prompt.

    Numbering gives claims a way to trace back to a source later (the
    auditor's ClaimCheck.source_url in Stage 3), and keeping each entry
    to title + url + snippet -- no raw page content -- matters for cost:
    whatever goes in here gets re-sent on every per-claim audit call in
    Stage 3, so it's multiplied by claim count, not paid once.
    """
    blocks = [
        f"[{i}] {result.title} ({result.url})\n{result.content}"
        for i, result in enumerate(evidence, start=1)
    ]
    return "\n\n".join(blocks)
