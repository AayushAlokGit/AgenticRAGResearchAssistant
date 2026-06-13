# Prompts

Prompts live here as versioned files (not inline string literals in code) so that a
single prompt change can be A/B tested against the eval set, and so the git history is
an honest record of how the prompts evolved.

## Convention

- One prompt per file, named by role: `agent_system.md`, `answer_with_citations.md`,
  `judge_faithfulness.md`, etc.
- Code loads prompts by name; it does not embed prompt text.
- When you change a prompt meaningfully, run the eval set before/after and note the
  score delta — a prompt change is a measurable change like any other (DD-001).

Prompts are loaded **by name** from this dir (config `prompts.dir`). Templates use
`string.Template` `$placeholders`, filled at runtime by `render_prompt` in code.

Status: **`answer_system.md` + `answer_user.md` exist** — the naive RAG answer pipeline's
system instructions (grounding + source citations + exact-phrase abstention) and the
`$context` / `$question` user template (DD-010). Judge and agent-system prompts are added
as those layers are built.
