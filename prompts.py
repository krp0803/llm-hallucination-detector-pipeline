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


DECOMPOSE_SYSTEM_PROMPT = """\
You are a claim extraction tool. You will be given a question and a \
draft answer to it. Break the draft into a list of atomic, factual \
claims about the world.

Rules:
- One factual assertion per claim. If a sentence bundles two facts \
("Acme, founded in 1997, employs 400 people"), split it into two claims.
- Every claim must be self-contained. Resolve every pronoun and \
reference using the question and the rest of the draft -- a claim will \
be checked completely on its own, with no other context available. \
"It was founded in 1997" is not acceptable; "Acme was founded in 1997" is.
- Extract only claims about the world. Do NOT extract the draft's own \
meta-commentary about its sources -- e.g. a sentence like "the sources \
don't specify the exact date" is not a factual claim to check, it's the \
draft describing a gap in its evidence. Skip sentences like that entirely.
- If the draft contains no checkable factual claims at all, return an \
empty list.
"""

VERIFY_SYSTEM_PROMPT = """\
You are a fact-checker. You will be given ONE claim and a numbered list \
of search results. Decide whether the claim is supported by those \
results -- nothing else. You have no other knowledge of the world for \
this task; if you happen to know the claim is true, that does not count \
unless the provided results also say so.

If confirming the claim depends on resolving an ambiguous pronoun or \
reference in the evidence (e.g. "it", "they", "this"), and the most \
natural reading -- typically the nearest antecedent -- does not clearly \
support the claim, treat the claim as "unsupported" rather than resolving \
the ambiguity in the claim's favor.

Return:
- verdict:
  - "supported" if the results state or clearly entail the claim.
  - "unsupported" if the results simply don't address the claim one way \
or another.
  - "contradicted" if the results state something that conflicts with \
the claim.
- supporting_quote: for "supported" or "contradicted", a short quote \
copied VERBATIM from the results -- exact wording, not a paraphrase or \
summary -- that justifies the verdict. For "unsupported", null.
- source_url: the URL of the result the quote came from, exactly as \
given in the numbered list. Null if supporting_quote is null.
- reasoning: one sentence explaining the verdict.
"""


def source_text(result: SearchResult) -> str:
    """
    The full text of one source as shown to the model: title, url, and
    content -- not just content alone.

    This exists because a Stage 4 measurement found the auditor's quote
    checker only searched `.content`, so a model that quoted a source's
    TITLE (which format_evidence() renders right above the content) could
    never have that citation verified, even though the title text was
    genuinely something the model was shown. auditor.py's quote check
    uses this same function as its search target, so the prompt-rendering
    and the verification code can never again disagree about what the
    model actually saw.
    """
    return f"{result.title} ({result.url})\n{result.content}"


def format_evidence(evidence: list[SearchResult]) -> str:
    """
    Render search results as a numbered list for a prompt.

    Numbering gives claims a way to trace back to a source later (the
    auditor's ClaimCheck.source_url), and keeping each entry to title +
    url + snippet -- no raw page content -- matters for cost: whatever
    goes in here gets re-sent on every per-claim audit call, so it's
    multiplied by claim count, not paid once.
    """
    blocks = [f"[{i}] {source_text(result)}" for i, result in enumerate(evidence, start=1)]
    return "\n\n".join(blocks)
