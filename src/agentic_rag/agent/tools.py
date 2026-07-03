"""The agent's TOOLS and the tool REGISTRY (Module 2, increment A1).

WHY THIS EXISTS — the general idea
----------------------------------
An LLM agent's *capability* is exactly its ACTION SPACE: the set of tools it may invoke.
The naive loop had one real action (search) + finish, which makes it a workflow, not an
agent. The moment there are SEVERAL tools, three new problems appear — and they are the
heart of agent engineering:

  1. tool SELECTION   — given the state, which tool? (a routing/policy problem)
  2. argument SYNTHESIS — fill the tool's parameters correctly (a structured-output problem)
  3. COMPOSITION      — chain tools so one's output feeds the next (a planning problem)

A TOOL bundles four things. Two face the MODEL — `name`, `description`, and a typed
parameter `schema` (this is ALL the model knows about a tool, so writing them well is
prompt engineering) — and one faces our CODE — the `run` function that actually executes.
The key principle this enforces is the separation of INTENT from EXECUTION: the model only
*names* an action and *proposes* arguments; the harness validates and runs it. That seam is
what later makes native tool-calling (A2), guardrails (A3), and delegation (D1) drop-in
rather than rewrites.

WHAT WE HAND-ROLL vs BORROW
---------------------------
The control loop stays hand-rolled (that's the learning artifact). For the COMMODITY part —
typed argument schemas + validation — we use **Pydantic**: it validates the controller's
proposed args for free, and `model_json_schema()` gives us the exact JSON Schema we'll hand
to the provider's tool-use API in A2. So this isn't a shortcut around learning; it's the
same object the production API wants.

A note on `expand_document` vs the retrieval-stage parent-expansion (DD-020): they look
similar but live at different layers. Parent-expansion auto-pulls ±1 neighbour on EVERY
search hit (a fixed retrieval policy). `expand_document` gives the CONTROLLER discretion to
pull a whole document when it judges it needs the full thing — agency, not a fixed rule.
Whether that discretion earns its place is what the A/B measures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Type

from pydantic import BaseModel, Field

from agentic_rag.rag.vector_store import Hit

logger = logging.getLogger(__name__)


# ───────────────────────────── tool result + shared context ─────────────────────────────

@dataclass
class ToolResult:
    """What a tool returns: an OBSERVATION for the controller, and optional EVIDENCE.

    The split is the lesson: not everything the agent observes belongs in the final answer.
      - `observation` — short text shown back to the CONTROLLER so it can pick the next action
        (e.g. "search X -> 5 hits", or a directory listing). Routing info.
      - `hits` — chunks that become EVIDENCE feeding the final answer's context. The retrieval
        tools (search, expand_document) produce these; `list_sources` emits the corpus MAP as a
        synthetic hit (so the generator can answer enumeration FROM it); `finish` produces none.
    """
    observation: str
    hits: List[Hit] = field(default_factory=list)


@dataclass
class ToolContext:
    """Shared dependencies a tool's `run` may need. Passed to every tool so tools stay pure
    functions of (args, ctx) — no globals, easy to test, and trivial to extend later."""
    retriever: object   # has .query(text, k) -> List[Hit]
    store: object       # ChromaVectorStore: .fetch_source_chunks(source), .all_chunks()
    top_k: int


# ───────────────────────────── typed arg schemas (Pydantic) ─────────────────────────────
# One model per tool. The Field descriptions are surfaced to the model (they document the
# argument), and the model class is what we validate the controller's proposed args against.

class SearchArgs(BaseModel):
    query: str = Field(..., description="A focused search query for ONE fact or sub-question, "
                                        "phrased the way the documentation would state it.")


class ExpandDocumentArgs(BaseModel):
    source: str = Field(..., description="The exact source filename (as shown in the evidence, "
                                         "e.g. 'chromadb_overview.md') to pull in FULL.")


# How many same-source chunks on EACH side of the target chunk expand_around_chunk pulls in.
# ±2 => up to a 5-chunk contiguous window. Sized from the a16 failure (the answer-bearing
# neighbour sat ~2 chunks from the retrieved one); a fixed constant, not an agent arg, so the
# controller only has to name WHERE to expand, not tune HOW WIDE (keeps tool-selection simple).
NEIGHBOUR_WINDOW = 2


class ExpandAroundChunkArgs(BaseModel):
    source: str = Field(..., description="The exact source filename of the chunk to expand around "
                                         "(as shown in the evidence, e.g. 'normalizer.md').")
    chunk_id: int = Field(..., description="The chunk index to expand around — the number after "
                                           "'#' in the evidence (e.g. 7 for '[normalizer.md #7]').")


class NoArgs(BaseModel):
    """For tools that take no arguments (list_sources, finish)."""
    pass


# ───────────────────────────── the Tool abstraction ─────────────────────────────

@dataclass
class Tool:
    name: str
    description: str
    args_model: Type[BaseModel]
    run: Callable[[BaseModel, ToolContext], ToolResult]
    terminal: bool = False   # finish is terminal: choosing it ends the loop (no execution)

    def args_example(self) -> Dict[str, str]:
        """A placeholder args dict for the prompt, e.g. {"query": "<query>"} — built from the
        Pydantic field names so the prompt's action grammar can't drift from the real schema."""
        return {name: f"<{name}>" for name in self.args_model.model_fields}

    def validate_args(self, raw_args: dict) -> BaseModel:
        """Coerce/validate the controller's proposed args into the typed model.

        Raises pydantic.ValidationError if the args are missing/ill-typed — the loop treats
        that as a controller formatting error and re-asks (A1). In A3 we'll instead feed the
        error back as an observation so the agent can self-correct mid-trajectory.
        """
        return self.args_model(**(raw_args or {}))


# ───────────────────────────── the four tool implementations ─────────────────────────────

def _run_search(args: SearchArgs, ctx: ToolContext) -> ToolResult:
    """Retrieve the top_k chunks for a reformulated query (the original, core action)."""
    hits = ctx.retriever.query(args.query, ctx.top_k)
    located = ", ".join(f"{h.source}#{h.chunk_index}" for h in hits) or "nothing"
    return ToolResult(observation=f'search "{args.query}" -> {len(hits)} hit(s): {located}', hits=hits)


def _run_expand_document(args: ExpandDocumentArgs, ctx: ToolContext) -> ToolResult:
    """Pull a whole source document's chunks in order — controller-chosen small-to-big.

    Use after a search reveals a clearly-relevant source whose retrieved fragment looks
    partial (a split table, a section continued below). Returns every chunk of that source
    as evidence; the answer-time char budget trims if it's huge.
    """
    by_index = ctx.store.fetch_source_chunks(args.source)
    if not by_index:
        return ToolResult(observation=f'expand_document "{args.source}" -> no such source in the corpus')
    hits = []
    for index in sorted(by_index):
        # score=0.0: these chunks weren't ranked by a retriever — they were pulled wholesale.
        hits.append(Hit(source=args.source, chunk_index=index, text=by_index[index], score=0.0))
    return ToolResult(observation=f'expand_document "{args.source}" -> {len(hits)} chunk(s) (full document)',
                      hits=hits)


def _run_expand_around_chunk(args: ExpandAroundChunkArgs, ctx: ToolContext) -> ToolResult:
    """Pull the ±NEIGHBOUR_WINDOW same-source neighbours of a specific chunk — surgical coverage.

    The middle ground between search (re-hits the same neighbourhood) and expand_document (dumps
    the whole doc): when the RIGHT source is found but the needed fact sits just outside the
    retrieved snippet, this fetches only the few adjacent chunks. Reuses the same per-source
    fetch expand_document uses, sliced to a window. Scratchpad dedup drops any already held.
    """
    by_index = ctx.store.fetch_source_chunks(args.source)
    if not by_index:
        return ToolResult(observation=f'expand_around_chunk "{args.source}" -> no such source in the corpus')
    lo, hi = args.chunk_id - NEIGHBOUR_WINDOW, args.chunk_id + NEIGHBOUR_WINDOW
    indices = [index for index in sorted(by_index) if lo <= index <= hi]
    if not indices:
        return ToolResult(observation=f'expand_around_chunk "{args.source}" #{args.chunk_id} '
                                      f'-> no chunks near #{args.chunk_id} in that source')
    # score=0.0: pulled wholesale by position, not ranked by a retriever (as in expand_document).
    hits = [Hit(source=args.source, chunk_index=index, text=by_index[index], score=0.0)
            for index in indices]
    return ToolResult(observation=f'expand_around_chunk "{args.source}" #{args.chunk_id} -> '
                                  f'{len(hits)} neighbour chunk(s) (#{indices[0]}–#{indices[-1]})',
                      hits=hits)


def _load_doc_tags(store) -> dict:
    """Load the per-doc FREE-FORM tags persisted next to the store (rag/tagging.py).

    Read straight from the store's persist dir so the tool needs no extra wiring; returns
    {source: {doc_type, topics: [...], entities: [...]}} — empty if the corpus hasn't been tagged.
    """
    import json
    from pathlib import Path

    base = Path(getattr(store, "persist_directory", "") or "")
    tags_file = base / "doc_tags.json"
    if tags_file.exists():
        return json.loads(tags_file.read_text(encoding="utf-8"))
    return {}


def _format_doc_tags(tag: dict) -> str:
    """Render one doc's free-form tags compactly for the corpus map (agent-readable, no fixed vocab)."""
    if not tag:
        return "untagged"
    parts = []
    if tag.get("doc_type"):
        parts.append(f"type={tag['doc_type']}")
    if tag.get("topics"):
        parts.append("topics: " + ", ".join(tag["topics"]))
    if tag.get("entities"):
        parts.append("entities: " + ", ".join(tag["entities"]))
    return " | ".join(parts) or "untagged"


def _run_list_sources(args: NoArgs, ctx: ToolContext) -> ToolResult:
    """List every source document WITH its metadata tags — the corpus map (no evidence).

    This is what answers "which/what documents are X" / "list all docs that…" enumeration
    questions: the controller reads each doc's tags and picks the matching set directly, instead
    of hoping similarity-search surfaces every member (which it can't — see DD-035). Also serves
    plain orientation before searching. Pure routing info: an observation, no hits.
    """
    counts: Dict[str, int] = {}
    for chunk in ctx.store.all_chunks():
        source = chunk["source"]
        counts[source] = counts.get(source, 0) + 1

    tags = _load_doc_tags(ctx.store)
    lines = []
    for source in sorted(counts):
        lines.append(f"  - {source} ({counts[source]} chunk(s)) [{_format_doc_tags(tags.get(source, {}))}]")

    # Free-form tags (open vocabulary): the controller READS each doc's type/topics/entities and
    # judges relevance — there is no fixed legend to filter on (retrieval stays soft, DD-045-style).
    corpus_map = "corpus sources (with free-form tags):\n" + "\n".join(lines)

    # Emit the map BOTH as an observation (for the controller's routing) AND as an evidence hit,
    # so it lands in the scratchpad and the GENERATOR can answer enumeration questions FROM it.
    # Without the hit, the map reaches only the controller and the final answer is written from a
    # different (often empty -> naive-fallback) context — the two-model split (DD-035).
    map_hit = Hit(source="corpus_map", chunk_index=0, text=corpus_map, score=0.0)
    return ToolResult(observation=corpus_map, hits=[map_hit])


def _run_finish(args: NoArgs, ctx: ToolContext) -> ToolResult:
    """No-op: choosing finish ends the loop. Modelled as a tool so the action grammar is
    uniform (every action is a tool), which is exactly the shape A2's tool-use API expects."""
    return ToolResult(observation="(finishing — answering from gathered evidence)")


# The catalogue of every tool the agent COULD have. Which ones are actually available in a
# run is chosen by config (agent.tools), so the tool set is an eval-gated knob like any other.
_TOOL_CATALOGUE: Dict[str, Tool] = {
    "search": Tool(
        name="search",
        description="Search the corpus for evidence with a focused query. Use a SEPARATE search "
                    "for each distinct fact or sub-question of a multi-hop question.",
        args_model=SearchArgs, run=_run_search,
    ),
    "expand_document": Tool(
        name="expand_document",
        description="Fetch a whole source document in full. Use when a search returned a clearly "
                    "relevant source but only a partial fragment of it (a split table/section).",
        args_model=ExpandDocumentArgs, run=_run_expand_document,
    ),
    "expand_around_chunk": Tool(
        name="expand_around_chunk",
        description="Fetch the immediate same-document NEIGHBOURS (a few chunks before and after) "
                    "of a specific chunk you already found. Use when a search surfaced the RIGHT "
                    "document but the specific fact you need is just outside the retrieved "
                    "snippet — more focused and cheaper than pulling the whole document.",
        args_model=ExpandAroundChunkArgs, run=_run_expand_around_chunk,
    ),
    "list_sources": Tool(
        name="list_sources",
        description="List every source document with its metadata TAGS (topic/role per document). "
                    "Use this to answer 'which/what documents are X' or 'list all docs that…' "
                    "questions — read the tags and pick the matching documents — and to orient on "
                    "what exists before searching. One call shows the whole corpus.",
        args_model=NoArgs, run=_run_list_sources,
    ),
    "finish": Tool(
        name="finish",
        description="Stop searching and answer from the evidence gathered so far. Choose this as "
                    "soon as the evidence is sufficient — do not waste rounds.",
        args_model=NoArgs, run=_run_finish, terminal=True,
    ),
}

DEFAULT_TOOLS = ["search", "finish"]  # reproduces the pre-A1 champion (search + finish only)


def known_tool_names() -> List[str]:
    """Every tool the agent COULD have (the full catalogue, across all arms).

    Used by the eval-set loader to validate ``trajectory.expects_tool`` so a renamed/typo'd
    tool fails loudly rather than silently never-matching (DD-027). Validates against the
    catalogue, NOT a single arm's registry — a tool legitimately absent from one A/B arm
    still names a real tool.
    """
    return list(_TOOL_CATALOGUE)


# ───────────────────────────── the registry ─────────────────────────────

class ToolRegistry:
    """The agent's available action space: name -> Tool, plus prompt rendering.

    Single source of truth for BOTH dispatch (the loop looks tools up here) and the prompt
    (the tool list is rendered from here), so adding/removing a tool can't desync the two.
    """

    def __init__(self, tools: List[Tool]):
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def names(self) -> List[str]:
        return list(self.tools)

    def render_for_prompt(self) -> str:
        """Render the tool list into the controller prompt: for each tool, its description AND
        the exact JSON action the controller must emit to call it. Generated from the schema so
        the documented grammar always matches what `validate_args` will accept."""
        import json

        blocks = []
        for tool in self.tools.values():
            action = {"thought": "<why you are taking this action>", "action": tool.name}
            example = tool.args_example()
            action["args"] = example  # always present (an empty {} for no-arg tools)
            blocks.append(f"- {tool.name}: {tool.description}\n    {json.dumps(action)}")
        return "\n".join(blocks)


def build_registry(tool_names: List[str]) -> ToolRegistry:
    """Build a registry from a list of tool names (config agent.tools), preserving order.

    Unknown names raise — a typo in config should fail loudly, not silently drop a tool.
    `finish` is always included even if omitted (the loop must always be able to stop).
    """
    names = list(tool_names)
    if "finish" not in names:
        names.append("finish")
    tools = []
    for name in names:
        tool = _TOOL_CATALOGUE.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool {name!r} in agent.tools "
                             f"(known: {sorted(_TOOL_CATALOGUE)})")
        tools.append(tool)
    logger.debug("tool registry built with %d tool(s): %s", len(tools), [t.name for t in tools])
    return ToolRegistry(tools)
