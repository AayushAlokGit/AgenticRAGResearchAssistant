"""No-API unit tests for the citation guardrails (harness/guardrails.py).

Runnable directly — ``python tests/test_citations.py`` prints a pass/fail summary and exits non-zero
on failure. Also pytest-compatible (``test_*`` functions with asserts) if pytest is ever added. Kept
dependency-free on purpose, mirroring the guardrails themselves: pure decision logic, no API, no
heavy deps (a tiny ``FakeHit`` stands in for the store's Hit, which the guards only read ``.source`` of).
"""
from dataclasses import dataclass

from agentic_rag.harness.guardrails import (check_citation_grounding, extract_citations,
                                            snap_citations)


@dataclass
class FakeHit:
    source: str


# ── extract_citations ────────────────────────────────────────────────────────────────────

def test_extract_recognizes_seed_corpus_filenames():
    text = "See [CONTEXT_ENGINEERING.md] and [01-Architecture-with-azure-cognitive-search.md]."
    assert extract_citations(text) == {
        "CONTEXT_ENGINEERING.md", "01-Architecture-with-azure-cognitive-search.md"}


def test_extract_recognizes_uploaded_source_ids():
    # upload ids carry a `upload:` colon-prefix, spaces, and real-filename punctuation
    for name in ["upload:Aayush Alok Resume.pdf", "upload:Resume (1).pdf",
                 "upload:Q3 Report & Notes.pdf", "upload:Bob's Notes.txt"]:
        assert extract_citations(f"cited [{name}] here") == {name}, name


def test_extract_recognizes_unicode_filename():
    assert extract_citations("[upload:R\xe9sum\xe9.pdf]") == {"upload:R\xe9sum\xe9.pdf"}


def test_extract_ignores_prose_and_numeric_refs():
    for prose in ["[see 2.1]", "[1]", "[note]", "[e.g. this]", "[Note: see other.md]", "[step 3]"]:
        assert extract_citations(prose) == set(), prose


def test_extract_strips_source_label():
    assert extract_citations("[source: CONTEXT_ENGINEERING.md]") == {"CONTEXT_ENGINEERING.md"}
    assert extract_citations("[source: upload:My File.pdf]") == {"upload:My File.pdf"}


def test_extract_splits_multi_file_bracket():
    assert extract_citations("[a.md, b.md]") == {"a.md", "b.md"}


# ── check_citation_grounding ─────────────────────────────────────────────────────────────

def test_grounding_true_when_cited_is_retrieved():
    g = check_citation_grounding("Skills [upload:Aayush Alok Resume.pdf].",
                                 [FakeHit("upload:Aayush Alok Resume.pdf")])
    assert g.is_grounded and not g.ungrounded


def test_grounding_flags_uploaded_citation_not_retrieved():
    # the regression this fixes: an uploaded-doc citation is now VERIFIED, not trivially grounded
    g = check_citation_grounding("Claims from [upload:Aayush Alok Resume.pdf].",
                                 [FakeHit("CONTEXT_ENGINEERING.md")])
    assert not g.is_grounded
    assert g.ungrounded == {"upload:Aayush Alok Resume.pdf"}


def test_grounding_flags_fabricated_citation():
    g = check_citation_grounding("See [TOTALLY_MADE_UP.md].", [FakeHit("CONTEXT_ENGINEERING.md")])
    assert g.ungrounded == {"TOTALLY_MADE_UP.md"}


# ── snap_citations ───────────────────────────────────────────────────────────────────────

def test_snap_corrects_near_miss_typo():
    retrieved = [FakeHit("EVALUATION_PRINCIPLES.md"), FakeHit("AGENT_ROADMAP.md")]
    fixed = snap_citations("As in [EVALUATION_PRINCIPALS.md].", retrieved)
    assert "[EVALUATION_PRINCIPLES.md]" in fixed
    assert check_citation_grounding(fixed, retrieved).is_grounded


def test_snap_leaves_genuine_fabrication_flagged():
    retrieved = [FakeHit("EVALUATION_PRINCIPLES.md")]
    fixed = snap_citations("See [COMPLETELY_UNRELATED.md].", retrieved)
    assert "[COMPLETELY_UNRELATED.md]" in fixed                       # unchanged
    assert not check_citation_grounding(fixed, retrieved).is_grounded


def test_snap_noop_on_correct_citation_and_empty_evidence():
    retrieved = [FakeHit("CONTEXT_ENGINEERING.md")]
    assert snap_citations("[CONTEXT_ENGINEERING.md]", retrieved) == "[CONTEXT_ENGINEERING.md]"
    assert snap_citations("[anything.md]", []) == "[anything.md]"     # no evidence -> no-op


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    raise SystemExit(0 if passed == len(tests) else 1)
