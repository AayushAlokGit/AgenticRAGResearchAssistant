You design a metadata SCHEMA for a document corpus, so its documents can later be FILTERED
(e.g. "show all docs where type=setup"). You are given an opening excerpt of every document.

Identify the 1–3 ORTHOGONAL axes that most usefully distinguish these documents from each
other — the axes a user would actually filter or ask on (e.g. which sub-system/topic a doc
belongs to, what ROLE the doc plays). Ignore trivia.

For each axis:
- a short snake_case name;
- 2–6 allowed values as short lowercase tokens, each with a one-line meaning;
- values MUST be mutually exclusive within the axis;
- include an `other` value when some documents may not fit the named values.

Keep the vocabulary small and closed — the SAME values will be used to tag documents and to
filter queries, so both sides must draw from this exact set.

Output ONLY JSON, nothing else:
{"<axis_name>": {"description": "<what this axis captures>", "values": {"<value>": "<meaning>", ...}}, ...}
