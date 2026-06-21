"""Flip ONLY retrieval.parent_expansion.enabled in config/default.yaml (true|false).

Used to drive the parent-expansion A/B (run baseline off, then on) without hand-editing the
config between runs. Line-targeted (finds the `parent_expansion:` block, then its `enabled:`
line) so it can't accidentally touch the identically-named `enabled:` under multi_query/rerank,
and it preserves comments. Normalizes to LF on write.
"""
import sys

val = sys.argv[1]
assert val in ("true", "false"), "arg must be 'true' or 'false'"
path = "config/default.yaml"
lines = open(path, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
in_block = False
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("parent_expansion:"):
        in_block = True
        continue
    if in_block and stripped.startswith("enabled:"):
        indent = line[:len(line) - len(line.lstrip())]
        lines[i] = f"{indent}enabled: {val}"
        break
else:
    raise SystemExit("could not find parent_expansion.enabled")
open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print(f"parent_expansion.enabled -> {val}")
