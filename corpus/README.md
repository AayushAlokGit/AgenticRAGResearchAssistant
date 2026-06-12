# Corpus

The corpus is **the user's own documentation** — which does not exist yet.

Until it does, drop a small **seed corpus** here (a handful of real documents) so the
agent loop and eval set have something to retrieve against. A good seed is the
`DocumentationRetrievalMCPServer/docs` folder: it is real, multi-document, and the user
understands it well enough to write good eval questions over it.

When the user's own documentation exists, point `corpus.root` (in `config/default.yaml`)
or `CORPUS_ROOT` (in `.env`) at it. An evolving corpus is what makes "incremental
re-indexing" (module 1) a meaningful exercise.

Note: the vector index built from this corpus is **not** committed (see `.gitignore`) —
it is regenerable from the corpus + config, which are the source of truth.
