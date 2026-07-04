"""Guardrails — the Safety ring of the harness.

A guardrail is a constraint checked at a boundary the system cannot cross without
passing through it. They come in three rings (docs/harness/HARNESS_ENGINEERING.md P4):

  - INPUT  — is the request well-formed?  (already held: typed Pydantic tool args)
  - ACTION — can the loop run away?        (held: round-budget + oscillation guard;
             this module adds `spend_cap_tripped`, denominated in TOKENS not rounds)
  - OUTPUT — does the result hold up?       (held: the abstention path; this module
             adds `check_citation_grounding`, a deterministic no-fabricated-cites check)

These are the *decision* half of a guardrail (a pure predicate); the *control* half
(what the loop does about it — stop, flag) stays in the loop where sequencing lives.
Keeping the decisions pure is what makes them unit-testable with no API and no loop.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Set

if TYPE_CHECKING:  # annotations only — avoids importing chromadb (Hit) / the sdks (Usage) at runtime
    from agentic_rag.llm.provider import Usage
    from agentic_rag.rag.vector_store import Hit


# ───────────────────────────── ACTION ring: spend-cap ─────────────────────────────

def spend_cap_tripped(usage: "Usage", cap_tokens: int) -> bool:
    """True once cumulative token spend reaches the cap — the circuit breaker in its TRUE unit.

    `max_rounds` bounds ITERATIONS, but batching (DD-041) lets one round fan out many calls, so
    rounds are a loose proxy for cost. This caps the resource we actually pay for (tokens). We pass
    the CONTROLLER usage (the part of the loop that can spin); the single generator call is bounded
    already. `cap_tokens <= 0` disables it (same off-convention as answer_char_budget).
    """
    if cap_tokens <= 0:
        return False
    return usage.total_tokens >= cap_tokens


# ───────────────────────────── OUTPUT ring: citation grounding ─────────────────────────────

@dataclass
class GroundingResult:
    """Whether an answer's citations point at evidence the loop actually retrieved.

    `ungrounded` = filenames the answer cites that are NOT in the retrieved set — i.e. fabricated
    citations, the thing this guard exists to catch. Empty `ungrounded` == grounded.
    """
    cited: Set[str]        # source filenames the answer cites
    available: Set[str]    # source filenames actually in the retrieved evidence
    ungrounded: Set[str]   # cited − available (fabricated / unsupported citations)

    @property
    def is_grounded(self) -> bool:
        return not self.ungrounded


# A citation looks like `[filename.ext]` (prompt rule 3). Match bracketed content; a later filter
# keeps only tokens that actually look like filenames, so prose like `[note]` or `[1]` is ignored.
# The filename filter allows: an optional `upload:` source prefix (bring-your-own-doc ids look like
# `upload:My File.pdf`), spaces in the name, and requires the extension to START with a letter so a
# bracketed decimal like `[see 2.1]` isn't mistaken for a citation. Keep it in sync with UPLOAD_PREFIX.
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_FILENAME_RE = re.compile(r"^(?:upload:)?[\w ./-]+\.[A-Za-z][A-Za-z0-9]*$")


def extract_citations(answer: str) -> Set[str]:
    """Pull the source filenames an answer cites in `[...]` markers.

    Tolerant on the ways a model varies the format: strips a leading `source:` label (the CONTEXT
    passages are shown as `[source: file]`, and the model sometimes echoes that), and splits a
    multi-file bracket like `[a.md, b.md]`. Only tokens shaped like a filename are kept.
    """
    cited: Set[str] = set()
    for inner in _BRACKET_RE.findall(answer):
        for token in re.split(r"[,;]", inner):
            name = token.strip()
            if name.lower().startswith("source:"):
                name = name[len("source:"):].strip()
            if _FILENAME_RE.match(name):
                cited.add(name)
    return cited


def check_citation_grounding(answer: str, retrieved: List["Hit"]) -> GroundingResult:
    """Deterministically check every cited filename against what was actually retrieved.

    Cheap by design (a set-diff, no second model call): the eval's LLM faithfulness judge measures
    grounding offline; this is the inline, mechanical version that catches the narrow, checkable
    failure — a citation naming a document the loop never retrieved. `retrieved` only needs `.source`.
    """
    cited = extract_citations(answer)
    available = {hit.source for hit in retrieved}
    return GroundingResult(cited=cited, available=available, ungrounded=cited - available)


# The valid citation targets are known exactly (the retrieved sources), so a cited name that isn't
# among them and is *almost* one is a generator typo (e.g. `PRINCIPALS` for `PRINCIPLES`), not a real
# fabrication. difflib.get_close_matches is the trusted commodity for "nearest string in a set"; the
# 0.8 cutoff is tight enough that a genuinely-invented citation (far from every source) is NOT snapped
# and stays flagged by check_citation_grounding — we correct typos without masking hallucinations.
_SNAP_CUTOFF = 0.8


def snap_citations(answer: str, retrieved: List["Hit"]) -> str:
    """Correct near-miss cited filenames to the exact retrieved source they typo, in the answer text.

    Only touches citations that are NOT already grounded (a name absent from the retrieved set) and
    only when a single close match exists above the cutoff — so it strictly repairs, never invents.
    Run this BEFORE check_citation_grounding so the grounding verdict reflects the corrected text.
    """
    available = {hit.source for hit in retrieved}
    if not available:
        return answer
    corrected = answer
    for name in extract_citations(answer):
        if name in available:
            continue
        match = difflib.get_close_matches(name, available, n=1, cutoff=_SNAP_CUTOFF)
        if match:
            corrected = corrected.replace(name, match[0])
    return corrected
