"""Backfill chunk `text` into existing eval-run JSONs that stored only (source, chunk_index).

Why this exists: the agentic-eval harness historically persisted each retrieved chunk as
``{source, chunk_index, score}`` only — enough to score recall, but NOT enough to *read* the
passage offline (the chunk-level diagnosis the other harnesses already support via a stored
``text`` field). agentic_eval.py now writes ``text`` going forward; this script repairs the
runs written before that fix.

How the text is recovered: each ``(source, chunk_index)`` is looked back up in the live
vector store (the same trick scripts/regrade.py uses to rebuild a generator's context). This
is EXACT when the run used ``parent_expansion=false`` — the hit text was the stored chunk
verbatim — which is every agentic run to date (DD-028). If a run used
``parent_expansion=true`` the hit text was a merged neighbour *window*, so this restores the
raw child chunk instead (an approximation); the script WARNS when it sees that flag.

Idempotent and safe: entries that already carry ``text`` are left untouched, and a file is
only rewritten if something actually changed — so you can point it at any or all run files,
including the ones that were never missing text.

Usage:
    python -m scripts.backfill_chunk_text                       # default: eval_runs/agentic/**/*.json
    python -m scripts.backfill_chunk_text path/to/run.json ...  # explicit files
"""
from __future__ import annotations

import glob
import json
import sys

from agentic_rag.config import load_config
from scripts.regrade import build_text_lookup

DEFAULT_GLOB = "eval_runs/agentic/**/*.json"


def fill_entry(entry: dict, text_lookup: dict) -> tuple[dict, str]:
    """Return (possibly-rebuilt entry, status) where status is already|filled|missing.

    On a fill we rebuild the dict so ``text`` lands right after ``score``, matching the
    key order the live harness writes (purely cosmetic, keeps diffs clean)."""
    if "text" in entry:
        return entry, "already"

    key = (entry.get("source"), entry.get("chunk_index"))
    text = text_lookup.get(key)
    if text is None:
        return entry, "missing"

    rebuilt: dict = {}
    for k, v in entry.items():
        rebuilt[k] = v
        if k == "score":
            rebuilt["text"] = text
    if "text" not in rebuilt:  # no score key to anchor after — append at the end
        rebuilt["text"] = text
    return rebuilt, "filled"


def backfill_file(path: str, text_lookup: dict) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    per_question = data.get("per_question")
    if not isinstance(per_question, list):
        print(f"  skip (no per_question): {path}")
        return

    # Honesty check: a parent-expanded run's hit text was a merged window, not a single
    # stored chunk — flag it so the recovered text isn't mistaken for byte-exact.
    parent_expansion = data.get("config", {}).get("retrieval", {}).get("parent_expansion")
    if parent_expansion:
        print(f"  WARN parent_expansion=true — recovered text is the raw child chunk, "
              f"not the merged window: {path}")

    filled = already = missing = 0
    for pq in per_question:
        retrieved = pq.get("retrieved")
        if not isinstance(retrieved, list):
            continue
        for i, entry in enumerate(retrieved):
            new_entry, status = fill_entry(entry, text_lookup)
            retrieved[i] = new_entry
            if status == "filled":
                filled += 1
            elif status == "already":
                already += 1
            else:
                missing += 1

    label = path.replace("\\", "/")
    if filled == 0:
        print(f"  {label}  (no change — filled 0, already {already}, missing {missing})")
        return

    # Match the harness write exactly: json.dumps(record, indent=2), utf-8, no trailing newline.
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2))
    note = f", MISSING {missing} (not in store)" if missing else ""
    print(f"  {label}  filled {filled}, already had {already}{note}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob(DEFAULT_GLOB, recursive=True))
    if not paths:
        print(f"No files matched (default glob: {DEFAULT_GLOB}).")
        return

    config = load_config()
    text_lookup = build_text_lookup(config)
    print(f"Backfilling chunk text from the store ({len(text_lookup)} chunks in lookup) "
          f"into {len(paths)} file(s):\n")
    for path in paths:
        backfill_file(path, text_lookup)


if __name__ == "__main__":
    main()
